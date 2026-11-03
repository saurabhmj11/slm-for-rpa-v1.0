"""Flow scorer: JSON-schema validity + every action/input exists in the catalog.
Usage: python flow_check.py PLATFORM_REPO catalog.jsonl
Why not only `ooa validate`: the Python reference lacks most toolkits (browser, desktop...), so it rejects valid flows.
"""
import json,glob,sys
import jsonschema
P=sys.argv[1]; C=sys.argv[2]
schema=json.load(open(P+'/contracts/flow-schema/flow.v1.json'))
cat={}
for l in open(C):
    e=json.loads(l); cat[e['id']]=e
def steps(n):
    for s in n:
        yield s
        for k in ('children','else'):
            v=s.get(k)
            if isinstance(v,list): yield from steps(v)
        for ei in s.get('elseIf',[]) or []:
            yield from steps(ei.get('children',[]))
        for c in s.get('catch',[]) or []:
            yield from steps(c.get('children',[]))
        if isinstance(s.get('finally'),list): yield from steps(s['finally'])
tot=sok=0; unk=bad_in=0; unk_ids=set(); ex=[]
for f in sorted(glob.glob(P+'/examples/**/*.flow.json',recursive=True)):
    d=json.load(open(f)); tot+=1
    try: jsonschema.validate(d,schema); sok+=1
    except Exception as e: ex.append((f.split('/')[-1],str(e)[:100]))
    for s in steps(d.get('steps',[])):
        a=s.get('action') or s.get('use') or s.get('type')
        if not a or not isinstance(a,str) or '/' not in a: continue
        if a not in cat: unk+=1; unk_ids.add(a); continue
        names={i['name'] for i in cat[a]['inputs']}
        for k in (s.get('args') or s.get('inputs') or {}):
            if k not in names: bad_in+=1
print('flows',tot,'schema_ok',sok,'unknown actions',unk,sorted(unk_ids)[:8],'unknown inputs',bad_in); print(ex[:3])
