"""Matched lesions only on a validated, saved biological tiny-circuit control."""
from pathlib import Path
import json
import numpy as np
from flybrain.lif_sanity import _spiking_models, assert_lif_sanity
from flybrain.network import MixedDynamicsNetwork
from flybrain.neurons import SHIU_LIF_SANITY_MODEL, shiu_lif_params, voltage_is_physiological
from organism.roi_innervation import CORE_CPG_TYPES
from organism.cpg_rhythm import rhythmicity_score
from experiment.validate_neural_milestone import raw_subgraph


def run(control_dir: Path, reference_metrics: Path, *, body: int, out: Path):
    parent=json.loads((control_dir/'report.json').read_text())
    reference=json.loads(reference_metrics.read_text())
    control=parent['conditions'][str(body)]
    if not control.get('allow_lesions') or parent['model_id'] != SHIU_LIF_SANITY_MODEL:
        raise ValueError('Lesions require a validated control for the same model and stimulated body')
    if not reference.get('rhythm_reproduced'):
        raise ValueError('Pugliese reference has not passed')
    assert_lif_sanity()
    graph=raw_subgraph()
    saved=np.load(control_dir/f'DNg100_{body}.npz')
    if not np.array_equal(saved['malecns_body_ids'],graph.neuron_ids) or not voltage_is_physiological(saved['v_mv']):
        raise ValueError('Control trace does not match the valid biological graph')
    out.mkdir(parents=True,exist_ok=False)
    roles={'DNg100':'DNg100',**CORE_CPG_TYPES}
    idx={r:np.flatnonzero(graph.cell_type==typ) for r,typ in roles.items()}
    idx['MN']=np.flatnonzero(np.char.find(graph.superclass.astype(str),'motor')>=0)
    conditions={'intact':np.array([],dtype=int),**{'lesion_'+r:idx[r] for r in ('E1','E2','I1','I2','E3')}}
    conditions['lesion_I1_I2']=np.concatenate([idx['I1'],idx['I2']])
    candidates=np.setdiff1d(np.arange(graph.n),idx['DNg100'])
    conditions['random_count_matched']=np.random.default_rng(parent['seed']).choice(candidates,len(idx['E1']),replace=False)
    report={'model_id':SHIU_LIF_SANITY_MODEL,'source_dataset':'MaleCNS_v1.0',
            'control_report':str((control_dir/'report.json').resolve()),'reference_metrics':str(reference_metrics.resolve()),
            'mapping_sha256':parent['mapping_sha256'],'configuration':{k:parent[k] for k in ('seed','steps','warmup_steps','dt_ms','current_mv_drive','neural_parameters','weight_transformation','nt_sign_rules')},
            'stimulated_malecns_body_ids':[body],'conditions':{},'is_pugliese_reproduction':False,
            'random_control':'cell-count matched only; not degree matched'}
    steps=parent['steps']; warmup=parent['warmup_steps']
    for name,lesion in conditions.items():
        net=MixedDynamicsNetwork(graph,params=shiu_lif_params(dt=parent['dt_ms']),seed=parent['seed'],models=_spiking_models(graph.n))
        net.intrinsic_noise_std=0;net.reset();net.lesion(lesion,silent=True)
        v=np.zeros((steps,graph.n));spikes=np.zeros_like(v,dtype=bool)
        for t in range(steps):
            if t==warmup:net.add_drive([graph.index_of(body)],parent['current_mv_drive'],source='validation.DNg100')
            net.step(1);v[t]=net.v;spikes[t]=net.last_spikes
        if name=='intact' and not np.array_equal(v,saved['v_mv']):
            raise ValueError('Control replay differs; refusing unmatched lesion comparison')
        valid=voltage_is_physiological(v)
        row={'lesioned_malecns_body_ids':graph.neuron_ids[lesion].astype(int).tolist(),
             'network_dynamics_valid':valid,'valid_for_rhythm_analysis':valid,'fft_executed':False,
             'dominant_frequency':None,'rhythmicity_score':None,'v_min_mv':float(v.min()),'v_max_mv':float(v.max()),'populations':{}}
        signals={r:spikes[warmup:,ii].mean(axis=1) for r,ii in idx.items() if len(ii)}
        for role,signal in signals.items():
            pop={'n':len(idx[role]),'active_neuron_count':int(np.any(spikes[warmup:,idx[role]],axis=0).sum()),
                 'mean_firing_rate_hz':float(signal.mean()*1000/parent['dt_ms']),
                 'amplitude':float(np.ptp(signal)), 'dominant_frequency':None,'rhythmicity_score':None}
            if valid:
                pop.update(rhythmicity_score(signal,parent['dt_ms']))
            row['populations'][role]=pop
        row['fft_executed']=any(p.get('fft_executed',False) for p in row['populations'].values())
        # Relative Fourier phase at the intact E1 spectral peak; a readout, not
        # evidence that spike trains implement the published mechanism.
        row['phase_relative_to_E1_rad']=None
        if valid:
            target=(row if name=='intact' else report['conditions']['intact'])['populations']['E1']['dominant_frequency']
            if target:
                freqs=np.fft.rfftfreq(steps-warmup,d=parent['dt_ms']/1000)
                k=int(np.argmin(abs(freqs-target)))
                spectra={r:np.fft.rfft(x-x.mean())[k] for r,x in signals.items()}
                e1=spectra['E1']
                row['phase_relative_to_E1_rad']={r:float(np.angle(z*np.conj(e1))) if abs(z)>1e-8 and abs(e1)>1e-8 else None for r,z in spectra.items()}
                row['phase_frequency_hz']=target
        report['conditions'][name]=row
        np.savez_compressed(out/f'{name}.npz',model_id=SHIU_LIF_SANITY_MODEL,source_dataset='MaleCNS_v1.0',malecns_body_ids=graph.neuron_ids,v_mv=v,spikes=spikes,dt_ms=parent['dt_ms'])
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(name,'voltage_valid=',valid,flush=True)
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--body',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.control,a.reference,body=a.body,out=a.out)
