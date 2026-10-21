import tempfile
from pathlib import Path
import json
import numpy as np
from dataset import topology,generate,FAMILIES
b=np.zeros((20,20),bool);b[2:18,2:18]=True
assert topology(b)==(1,0,1)
b[6:14,6:14]=False
assert topology(b)==(1,1,0)
b=np.zeros((20,20),bool);b[2:4,2:4]=1;b[10:12,10:12]=1
assert topology(b)==(2,0,2)
with tempfile.TemporaryDirectory() as d:
 a=generate(d,24,48,3);b=generate(d,24,48,3)
 assert a==b and len(a)==24
 m=json.loads((Path(d)/'dataset.json').read_text())
 assert set(r['family'] for r in m['selected'])==set(FAMILIES)
 assert len(set(r['sha256'] for r in m['selected']))==24
print('PASS: components/holes, reproducible selection, 20 family coverage, 24 distinct files')
