import json
import pytest
from experiment import tiny_cpg_lesions


def test_lesions_refused_before_graph_load_when_control_invalid(tmp_path, monkeypatch):
    (tmp_path/'report.json').write_text(json.dumps({'model_id':'shiu_lif_sanity_v1','conditions':{'10045':{'allow_lesions':False}}}))
    reference=tmp_path/'reference.json'
    reference.write_text(json.dumps({'rhythm_reproduced':True}))
    monkeypatch.setattr(tiny_cpg_lesions,'raw_subgraph',lambda:pytest.fail('loaded graph despite failed control'))
    with pytest.raises(ValueError,match='validated control'):
        tiny_cpg_lesions.run(tmp_path,reference,body=10045,out=tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()
