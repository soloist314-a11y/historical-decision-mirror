#!/usr/bin/env python3
import json, pathlib, argparse, importlib.util
BASE=pathlib.Path(__file__).resolve().parent
PAIRS=BASE.parent/'data/contrast_pairs.jsonl'
CASES=BASE.parent/'data/decision_nodes.jsonl'

def load_pairs():
    return [json.loads(x) for x in PAIRS.read_text(encoding='utf-8').splitlines() if x.strip()]

def load_cases():
    return {r['case_id']:r for r in (json.loads(x) for x in CASES.read_text(encoding='utf-8').splitlines() if x.strip())}

def by_case(case_id, limit=5):
    cases=load_cases(); out=[]
    for p in load_pairs():
        if case_id not in (p['primary_case_id'],p['contrast_case_id']):
            continue
        other=p['contrast_case_id'] if case_id==p['primary_case_id'] else p['primary_case_id']
        orientation='PRIMARY_TO_CONTRAST' if case_id==p['primary_case_id'] else 'CONTRAST_TO_PRIMARY'
        out.append({
            **p,
            'orientation':orientation,
            'selected_case_id':case_id,
            'selected_case_title':cases.get(case_id,{}).get('case_title',''),
            'other_case_id':other,
            'other_case_title':cases.get(other,{}).get('case_title',''),
        })
    return out[:limit]

def from_query(query, topn=3):
    spec=importlib.util.spec_from_file_location('retrieval',BASE/'retrieve.py')
    retrieval=importlib.util.module_from_spec(spec); spec.loader.exec_module(retrieval)
    ranked=retrieval.rank(query,topn)
    primary=ranked['ranking'][0] if ranked['ranking'] else None
    confident=bool(primary and ranked['query_features'] and primary.get('structural_score',0)>0)
    pairs=by_case(primary['case_id']) if confident else []
    warning=None if confident else '未形成有效结构特征命中；不自动输出反案例，需先完成现实问题结构化。'
    return {'query':query,'primary':primary,'query_features':ranked['query_features'],'contrast_candidates':pairs,'contrast_confidence':'OK' if confident else 'LOW','warning':warning}

if __name__=='__main__':
    ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group(required=True); g.add_argument('--case-id'); g.add_argument('--query'); ap.add_argument('--limit',type=int,default=5)
    a=ap.parse_args(); res={'case_id':a.case_id,'contrasts':by_case(a.case_id,a.limit)} if a.case_id else from_query(a.query,a.limit)
    print(json.dumps(res,ensure_ascii=False,indent=2))
