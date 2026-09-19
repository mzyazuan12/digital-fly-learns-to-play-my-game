"""Raw-data, fail-closed neural milestone; never loads the full simulation graph."""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from flybrain.loader import ANN_FILE, NT_FILE, EDGE_FILE, DEFAULT_DATA
from flybrain.lif_sanity import assert_lif_sanity, _spiking_models, _pair_psp
from flybrain.network import MixedDynamicsNetwork
from flybrain.neurons import SHIU_LIF_SANITY_MODEL, nt_sign, shiu_lif_params, voltage_is_physiological, voltages_finite
from organism.roi_innervation import (CORE_CPG_TYPES, load_annotations,
    scan_syn_points, rois_to_row, build_cpg_mapping, validate_cpg_mapping)
from organism.cpg_rhythm import rhythmicity_score
from experiment.restricted_cpg import restricted_cpg_graph


def raw_subgraph():
    """Both MaleCNS DNg100 plus core CPG types and downstream motor readouts."""
    return restricted_cpg_graph()


def run(out: Path, *, steps=2000, warmup=100, current=40.0, seed=1):
    out.mkdir(parents=True, exist_ok=False)
    report = {'model_id': SHIU_LIF_SANITY_MODEL, 'valid_for_rhythm_analysis':False, 'allow_lesions':False,
              'dominant_frequency':None, 'rhythmicity_score':None, 'fft_executed':False,
              'full_malecns_allowed':False, 'status':'running'}
    def save(): (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    try:
        report['sanity'] = assert_lif_sanity()
        print('LIF sanity PASS', flush=True)
        anns = load_annotations()
        counts = scan_syn_points({r['malecns_body_id'] for r in anns}, progress=True)
        mapping = build_cpg_mapping([{**r, **rois_to_row(counts[r['malecns_body_id']])} for r in anns])
        validate_cpg_mapping(mapping)
        mapping_bytes = (json.dumps(mapping, sort_keys=True, indent=2)+'\n').encode()
        (out/'cpg_mapping.json').write_bytes(mapping_bytes)
        (out/'roi_counts.json').write_text(json.dumps(counts, indent=2)+'\n')
        report['mapping_sha256'] = hashlib.sha256(mapping_bytes).hexdigest()
        report['identity_mapping_valid'] = True
        report['mapping_unassigned'] = {role:len(mapping[typ]['unassigned']) for role,typ in CORE_CPG_TYPES.items()}
        print('Raw identity and ROI mapping PASS', flush=True)
        graph = raw_subgraph()
        # This milestone uses the LIF sanity model for every cell. Graded VNC
        # assumptions belong to a separate model comparison, not this identifier.
        params = shiu_lif_params(dt=1.0)
        np.savez_compressed(out/'subgraph.npz', model_id=SHIU_LIF_SANITY_MODEL,
                            source_dataset='MaleCNS_v1.0', malecns_body_ids=graph.neuron_ids,
                            pre_ptr=graph.pre_ptr,post=graph.post,anatomical=graph.anatomical,
                            cell_type=graph.cell_type.astype(str),nt=graph.neurotransmitter.astype(str))
        edge_lookup={(i,int(graph.post[e])):int(graph.anatomical[e]) for i in range(graph.n) for e in range(graph.pre_ptr[i],graph.pre_ptr[i+1])}
        pairs=[(i,j,n) for (i,j),n in edge_lookup.items() if i != j and edge_lookup.get((j,i),0)==0 and graph.neurotransmitter[i]=='acetylcholine' and n<=20]
        if not pairs: raise RuntimeError('No suitable asymmetric anatomical ACh edge for isolated orientation probe')
        i,j,n=max(pairs,key=lambda item:item[2])
        probe=_pair_psp('acetylcholine',synapse_count=n)
        reverse=_pair_psp('acetylcholine',synapse_count=n,reverse=True)
        report['anatomical_edge_orientation']={'source_dataset':'MaleCNS_v1.0','pre_malecns_body_id':int(graph.neuron_ids[i]),'post_malecns_body_id':int(graph.neuron_ids[j]),'forward_synapse_count':n,'reverse_synapse_count':0,'matrix_convention':'W[pre, post], outgoing CSR','isolated_forward':probe,'isolated_reverse':reverse,'ok':bool(probe['g_post']>0 and reverse['g_post']==0)}
        if not report['anatomical_edge_orientation']['ok']: raise RuntimeError('Anatomical edge orientation failed')
        report["source_files"] = {name:{"bytes":(DEFAULT_DATA/name).stat().st_size,"mtime_ns":(DEFAULT_DATA/name).stat().st_mtime_ns} for name in (ANN_FILE,NT_FILE,EDGE_FILE,"syn-points-male-cns-v1.0-minconf-0.5.feather")}
        report.update(dataset='MaleCNS_v1.0', n_neurons=graph.n, n_edges=graph.n_edges,
                      neural_parameters=asdict(params), seed=seed, dt_ms=1.0, steps=steps, warmup_steps=warmup,
                      current_mv_drive=current, voltage_unit='mV', weight_transformation='anatomical count × presynaptic NT sign × 0.275 mV',
                      nt_sign_rules={str(t):nt_sign(t) for t in graph.neurotransmitter},
                      git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                      working_tree_diff=subprocess.check_output(['git','diff'], text=True),
                      software={'python':platform.python_version(),'numpy':np.__version__}, subset=graph.report)
        conditions = {}
        roles = {'DNg100':'DNg100', **CORE_CPG_TYPES}
        indices = {role:np.flatnonzero(graph.cell_type == typ) for role,typ in roles.items()}
        indices['MN'] = np.flatnonzero(np.char.find(graph.superclass.astype(str), 'motor') >= 0)
        for body in (10045,10056):
            net=MixedDynamicsNetwork(graph, params=params, seed=seed, models=_spiking_models(graph.n))
            report["neuron_model_assignment"] = net.models.snapshot()
            net.intrinsic_noise_std=0.0
            net.reset()
            v=np.zeros((steps,graph.n)); activity=np.zeros_like(v); spikes=np.zeros_like(v,dtype=bool)
            for t in range(steps):
                if t == warmup: net.add_drive([graph.index_of(body)],current,source='validation.DNg100')
                net.step(1)
                v[t]=net.v; spikes[t]=net.last_spikes
                activity[t]=np.where(net.is_graded,net.graded_output,net.last_spikes)
            dng=v[:,indices['DNg100']]
            invalid = (~np.isfinite(v)) | (v < -100) | (v > 40)
            violations = []
            for idx in np.flatnonzero(np.any(invalid,axis=0)):
                first=int(np.flatnonzero(invalid[:,idx])[0])
                violations.append({'source_dataset':'MaleCNS_v1.0','malecns_body_id':int(graph.neuron_ids[idx]),'type':str(graph.cell_type[idx]),'first_invalid_time_ms':first+1,'min_mv':float(v[:,idx].min()),'max_mv':float(v[:,idx].max())})
            finite=voltages_finite(dng); phys=voltage_is_physiological(dng)
            network_valid=voltage_is_physiological(v)
            census={}
            for role,idx in indices.items():
                a=activity[warmup:,idx]
                census[role]={'n':len(idx),'active_neuron_count':int(np.any(a>1e-8,axis=0).sum()),
                              'mean_activity':float(a.mean()) if a.size else 0.0,
                              'spikes':int(spikes[warmup:,idx].sum()),
                              'mean_firing_rate_hz':float(spikes[warmup:,idx].mean()*1000) if a.size else 0.0,
                              'recruited':bool(a.size and np.any(a>1e-8))}
            allowed=bool(finite and phys and network_valid and all(census[r]['recruited'] for r in ('DNg100','E1','E2','I1','I2','MN')))
            row={'stimulated_malecns_body_ids':[body], 'dng_finite':finite,'dng_physiological':phys,
                 'dng_dynamics_valid':finite and phys,'network_dynamics_valid':network_valid,
                 'all_dynamics_valid':finite and phys and network_valid,
                 'valid_for_rhythm_analysis':allowed,'allow_lesions':allowed,'census':census,
                 'v_min_mv':float(v.min()),'v_max_mv':float(v.max()),'dominant_frequency':None,'rhythmicity_score':None,
                 'fft_executed':False, 'voltage_violations':violations}
            np.savez_compressed(out/f'DNg100_{body}.npz',model_id=SHIU_LIF_SANITY_MODEL,
                                malecns_body_ids=graph.neuron_ids, neuron_kind=net.models.kind.astype(str), cell_type=graph.cell_type.astype(str), source_dataset='MaleCNS_v1.0',v_mv=v, activity=activity, spikes=spikes, dt_ms=1.0)
            if allowed:
                row['rhythm']={role:rhythmicity_score(activity[warmup:,idx].mean(axis=1),1.0) for role,idx in indices.items() if len(idx)}
                row['fft_executed']=any(m['fft_executed'] for m in row['rhythm'].values())
            conditions[str(body)]=row
            print(f'DNg100 {body}: voltage_valid={network_valid}, recruitment='+str({r:c['recruited'] for r,c in census.items()}),flush=True)
        report['conditions']=conditions
        report['valid_for_rhythm_analysis']=all(r['valid_for_rhythm_analysis'] for r in conditions.values())
        report['allow_lesions']=report['valid_for_rhythm_analysis']
        report['status']='tiny_gate_passed' if report['valid_for_rhythm_analysis'] else 'blocked_at_tiny_circuit'
        report['next_step']='Review tiny-circuit gate and separately establish Pugliese reference before full graph or embodiment.'
        save()
        return report
    except Exception as exc:
        report.update(status='error',error=f'{type(exc).__name__}: {exc}',valid_for_rhythm_analysis=False,allow_lesions=False)
        save()
        raise

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs')/('neural_validation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    args=parser.parse_args()
    result=run(args.out)
    print(args.out.resolve())
    raise SystemExit(0 if result['valid_for_rhythm_analysis'] else 2)
