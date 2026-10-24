import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import pytest

torch = pytest.importorskip("torch", reason="Install config/requirements-resnet.txt for the optional feature experiment")
pytest.importorskip("torchvision")

from src import deep_features, transfer
from src.baseline import training_pairs
from src.deep_features import frozen_encoder, fit_projection, pair_matrix, build_pair_tables
from src.features import stable_digest
from src.svr import load_svr_config
from src.utils import assign_splits, index_dataset, matched_cases
from tools.synthetic_data import create_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]


def unit(values):
    return values/np.linalg.norm(values, axis=1, keepdims=True)


def test_frozen_architecture_has_no_trainable_parameters_or_bn_updates():
    torch.set_num_threads(2)
    encoder = frozen_encoder(None)  # Architecture test only; not a pretrained accuracy experiment.
    before = encoder.bn1.running_mean.clone()
    with torch.inference_mode():
        values = encoder(torch.rand(2, 3, 224, 224))
    assert values.shape == (2, 512) and not encoder.training
    assert not any(p.requires_grad for p in encoder.parameters())
    assert torch.equal(before, encoder.bn1.running_mean)


def test_pca_uses_train_images_only_and_pair_features_are_symmetric():
    rng = np.random.default_rng(42)
    embeddings = pd.DataFrame({"split": ["train"]*12+["validation"]*4,
                               "embedding": list(unit(rng.normal(size=(16, 512))))})
    first = fit_projection(embeddings, 4)
    changed = embeddings.copy(deep=True)
    changed.loc[changed.split.eq("validation"), "embedding"] = pd.Series(list(unit(rng.normal(size=(4, 512)))), index=changed.index[-4:])
    second = fit_projection(changed, 4)
    np.testing.assert_equal(first.mean_, second.mean_)
    np.testing.assert_equal(first.components_, second.components_)
    q, c = np.stack(embeddings.embedding[:3]), np.stack(embeddings.embedding[3:6])
    np.testing.assert_allclose(pair_matrix(q,c,first),pair_matrix(c,q,first),atol=1e-12)
    same = pair_matrix(q,q,first)
    np.testing.assert_allclose(same[:,:4],0,atol=1e-12)
    np.testing.assert_allclose(same[:,-1],0,atol=1e-12)
    with pytest.raises(ValueError):
        fit_projection(embeddings, 12)


def test_embedding_cache_uses_content_and_never_downloads_implicitly(tmp_path, monkeypatch):
    cfg = json.loads((ROOT/'config/resnet.json').read_text())
    cfg.update(device="cpu",batch_size=2)
    monkeypatch.setattr(deep_features,"verify_weights",lambda *args: "f37072fd_fixture")
    calls=[]
    class TinyEncoder(torch.nn.Module):
        def forward(self, x):
            calls.append(len(x))
            return (x.mean(dim=(2,3)).abs()+1).repeat(1,171)[:,:512]
    monkeypatch.setattr(deep_features,"frozen_encoder",lambda *args:TinyEncoder().eval())
    path=tmp_path/'image.png'
    Image.fromarray(np.full((48,64,3),80,dtype=np.uint8)).save(path)
    first,info=deep_features.embed_paths([str(path)],tmp_path,cfg)
    second,again=deep_features.embed_paths([str(path)],tmp_path,cfg)
    assert calls==[1] and info['cache_hits']==0 and again['cache_hits']==1
    np.testing.assert_equal(first[str(path)],second[str(path)])
    Image.fromarray(np.full((48,64,3),120,dtype=np.uint8)).save(path)
    deep_features.embed_paths([str(path)],tmp_path,cfg)
    assert calls==[1,1]
    with pytest.raises(FileNotFoundError):
        deep_features.embed_paths(['missing.png'],tmp_path,cfg)


@pytest.fixture
def reference_data(tmp_path):
    data=create_synthetic_dataset(tmp_path/'data')
    cases,_=matched_cases(index_dataset(data))
    cases=assign_splits(cases)
    cases['human_verified']=False
    cases['label_basis']='filename_unverified'
    cfg={"seed":42,"negatives_per_query":10,"same_row_negatives":5,
         "score_tie_atol":1e-12,"score_tie_rtol":1e-10}
    pairs=training_pairs(cases,cfg)
    digest=stable_digest(pairs[["query_id","candidate_id","target","sample_weight"]].to_dict('records'))
    reference={"training_pairs":pairs,"summary":{"settings":cfg,"versions":{"fixture":"synthetic"},
               "training_pair_identity_hash":digest,"frozen_protocol_hashes":{}}}
    return tmp_path,cases,reference


def test_development_validation_rejects_changed_samples_and_rows(reference_data):
    root,cases,reference=reference_data
    bad=cases.copy()
    bad.loc[bad.split.eq('validation'),'row']=cases[cases.split.eq('train')].iloc[0].row
    with pytest.raises(ValueError,match='rows'):
        transfer.validate_development(bad,reference)
    ref={**reference,'training_pairs':reference['training_pairs'].copy()}
    ref['training_pairs'].loc[0,'sample_weight']*=2
    with pytest.raises(ValueError,match='changed'):
        transfer.validate_development(cases,ref)


def test_transfer_roundtrip_and_test_images_never_embedded(reference_data,monkeypatch):
    root,cases,reference=reference_data
    observed=[]
    def fake_embeddings(paths, root, settings):
        result={}
        for path in paths:
            observed.append(str((root/path).resolve()))
            seed=int(hashlib.sha256((root/path).read_bytes()).hexdigest()[:8],16)
            result[str(path)]=unit(np.random.default_rng(seed).normal(size=(1,512)))[0]
        return result,{"versions":{"fixture":"synthetic"},"signature":"synthetic_fixture",
                       "weights_sha256":"synthetic_fixture","cnn_parameters_trained":0}
    monkeypatch.setattr(transfer,'embed_paths',fake_embeddings)
    cfg=json.loads((ROOT/'config/resnet.json').read_text())
    cfg.update(pca_components=4,timing_repeats=2)
    svr_cfg=load_svr_config(ROOT)
    svr_cfg.update(C=[.1,1.],gamma=['scale'],epsilon=[.1])
    result=transfer.run_transfer(root,cases,reference,cfg,svr_cfg,render=False)
    test=cases[cases.split.eq('test')]
    assert not set(observed)&(set(test.summer_rgb)|set(test.winter_rgb))
    assert result['summary']['training_pairs']==198 and result['summary']['pca_training_images']==36
    assert result['summary']['test_images_embedded']==result['summary']['test_queries_scored']==0
    assert len(result['trials'])==2
    val=cases[cases.split.eq('validation')].sort_values('tree_id')
    query=val.iloc[0]
    gallery=val[['tree_id','winter_rgb']].rename(columns={'tree_id':'candidate_id','winter_rgb':'rgb'})
    for name in ['linear','svr']:
        bundle=joblib.load(result['output_dir']/f'models/{name}.joblib')
        ranked=transfer.match_resnet(query.summer_rgb,gallery,bundle,root)
        expected=pd.concat(result[name]['rankings'])
        expected=expected[expected.query_id.eq(query.tree_id)].set_index('candidate_id')
        np.testing.assert_allclose(ranked.score,expected.loc[ranked.candidate_id].score,atol=1e-8)
    old=(root/cfg['selection_file']).read_bytes()
    again=transfer.run_transfer(root,cases,reference,cfg,svr_cfg,render=False)
    assert again['summary']['selection_reused'] and (root/cfg['selection_file']).read_bytes()==old
    with pytest.raises(ValueError,match='selection lock'):
        transfer.run_transfer(root,cases,reference,dict(cfg,pca_components=3),svr_cfg,render=False)
