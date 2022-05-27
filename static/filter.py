import random
import pathlib
import shutil

d = pathlib.Path('hooktheory_test_all')
o = pathlib.Path('hooktheory_test')

random.seed(0)
uids = sorted(list(set([p.stem for p in d.rglob('*.mid') if p.parent.stem != 'user'])))
print(len(uids))

sample = set(random.sample(uids, 100))
print(sample)

for p in sorted(list(d.rglob('*.mid')) + list(d.rglob('*.mp3'))):
    if p.stem in sample:
        op = pathlib.Path(o, p.relative_to(d))
        op.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(str(p), str(op))

import gzip
import json

with gzip.open('0503_theorytab_simple.json.gz', 'r') as f:
    hooktheory = json.load(f)

output = []
for uid in sorted(list(sample)):
    youtube_id = hooktheory[uid]['youtube']['id']
    output.append((uid, youtube_id))

with open('web.json', 'w') as f:
    f.write(json.dumps(output[::-1], indent=2))
