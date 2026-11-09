"""Export frozen fitted trees and paged benchmark data; never retrain or retune."""
from pathlib import Path
import sys, json, gzip, struct, hashlib
import numpy as np
import pandas as pd
import joblib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ml/src'))
from ids import read_data, feature_frame

def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False))

def write_gzip(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(data,compresslevel=6,mtime=0))

bundles = {name:joblib.load(ROOT/'ml/models'/f'{stem}.joblib') for name,stem in [('xgboost','xgb_depth7'),('randomforest','rf_depth24')]}
b = bundles['xgboost']
prep=b['preprocessor']
num=[c for c in b['feature_columns'] if c not in ['proto','service','state']]
cat=['proto','service','state']
metadata={'version':1,'recordBytes':20,'features':b['feature_columns'],'numeric':num,'categorical':cat,
    'medians':prep.named_transformers_['numeric'].named_steps['imputer'].statistics_.tolist(),
    'categoryModes':prep.named_transformers_['categorical'].named_steps['imputer'].statistics_.tolist(),
    'categories':[v.tolist() for v in prep.named_transformers_['categorical'].named_steps['onehot'].categories_], 'models':{}}
for name,bundle in bundles.items():
    roots=[]; nodes=[]
    if name=='randomforest':
        for tree in bundle['model'].estimators_:
            t=tree.tree_; offset=len(nodes); roots.append(offset)
            for i in range(t.node_count):
                if t.children_left[i] < 0:
                    v=t.value[i,0]; nodes.append((-1,-1,-2,float(v[1]/v.sum())))
                else:
                    nodes.append((int(t.children_left[i])+offset,int(t.children_right[i])+offset,int(t.feature[i]),float(t.threshold[i])))
        base=0
    else:
        raw=json.loads(bytes(bundle['model'].get_booster().save_raw(raw_format='json')))
        learner=raw['learner']; p=np.float32(json.loads(learner['learner_model_param']['base_score'])[0])
        base=float(np.float32(np.log(p/(np.float32(1)-p))))
        for tree in learner['gradient_booster']['model']['trees']:
            offset=len(nodes); roots.append(offset)
            for i,left in enumerate(tree['left_children']):
                if left < 0:
                    nodes.append((-1,-1,-2,float(np.float32(tree['split_conditions'][i]))))
                else:
                    # Dense imputed inputs contain no NaN. All splits are numeric.
                    assert tree['split_type'][i] == 0
                    nodes.append((left+offset,tree['right_children'][i]+offset,int(tree['split_indices'][i]),float(np.float32(tree['split_conditions'][i]))))
    data=bytearray(len(nodes)*20)
    for i,node in enumerate(nodes):struct.pack_into('<iiid',data,i*20,*node)
    destination=ROOT/'public/models'/f'{name}.bin.gz';write_gzip(destination,data)
    metadata['models'][name]={'name':bundle['name'],'roots':roots,'threshold':bundle['threshold'],'baseMargin':base,'nodes':len(nodes),
        'sourceSha256':hashlib.sha256((ROOT/'ml/models'/f"{bundle['name']}.joblib").read_bytes()).hexdigest(),
        'binarySha256':hashlib.sha256(data).hexdigest(),'compressedBytes':destination.stat().st_size,'treeCount':len(roots)}
write_json(ROOT/'public/models/manifest.json',metadata)

# Complete paged datasets. Each filter index covers the entire partition, not just the visible page.
datasets={}
for partition in ['train','test']:
    frame=read_data(ROOT/'ml/data/raw'/f'{partition}.csv')
    assert np.array_equal(frame.id,np.arange(1,len(frame)+1))
    filters={'all':frame,'Normal':frame[frame.label.eq(0)],'Attack':frame[frame.label.eq(1)]}
    filters.update({category:frame[frame.attack_cat.eq(category)] for category in sorted(frame.attack_cat.unique()) if category!='Normal'})
    datasets[partition]={'rows':len(frame),'columns':frame.columns.tolist(),'categories':frame.attack_cat.value_counts().to_dict(),'filters':{}}
    for key, subset in filters.items():
        datasets[partition]['filters'][key]={'count':len(subset),'chunks':int(np.ceil(len(subset)/500))}
        values=subset.to_numpy().tolist()
        for offset in range(0,len(values),500):
            payload=json.dumps(values[offset:offset+500],separators=(',',':'),allow_nan=False).encode()
            write_gzip(ROOT/'public/data'/partition/key/f'{offset//500}.json.gz',payload)
write_json(ROOT/'public/data/datasets.json',datasets)

samples=[]
for name,label in [('simple_sample','Starter'),('error_examples','Error case'),('demo_records','Explore')]:
    inputs=read_data(ROOT/'ml/data'/f'{name}.csv')
    answer_name={'simple_sample':'simple_sample_answers','error_examples':'error_examples_answers','demo_records':'demo_answers'}[name]
    answers=read_data(ROOT/'ml/data'/f'{answer_name}.csv').set_index('id')
    for row in inputs.to_dict('records'):
        a=answers.loc[row['id']]
        samples.append({'id':str(row['id'])+'-'+name,'title':f"{label} · record {row['id']}", 'source':label,
            'values':{c:row[c] for c in metadata['features']},'truth':int(a.label),'category':a.attack_cat,'recordId':row['id']})
write_json(ROOT/'public/data/samples.json',samples)
metrics=pd.read_csv(ROOT/'ml/results/test_metrics.csv')
metrics=metrics[(metrics.task=='binary') & metrics.model.isin(['xgb_depth7','rf_depth24'])].replace({np.nan:None}).to_dict('records')
importance={}
for name,stem in [('XGBoost','xgb_depth7'),('Random Forest','rf_depth24')]:
    path=ROOT/'ml/results'/f'feature_importance_{stem}.csv'
    if path.exists():importance[name]=pd.read_csv(path).head(10).to_dict('records')
write_json(ROOT/'public/data/analysis.json',{'metrics':metrics,'importance':importance,'trainingRows':175341,'testingRows':82332,'exactOverlap':8541,'fitRows':140273,'validationRows':35068})
# Full verification inputs stay outside the public site. Labels never enter the browser predictor.
test=read_data(ROOT/'ml/data/raw/test.csv')
x=prep.transform(feature_frame(test,b['feature_columns'])).astype('<f4')
work=ROOT/'.cache/model-verification';work.mkdir(parents=True,exist_ok=True)
x.tofile(work/'test-encoded.bin')
expected={name:bundle['model'].predict_proba(x)[:,1].tolist() for name,bundle in bundles.items()}
write_json(work/'expected.json',expected)
print(json.dumps({name:{'trees':m['treeCount'],'nodes':m['nodes'],'downloadMiB':round(m['compressedBytes']/1024**2,2)} for name,m in metadata['models'].items()},indent=2))
