import random
import pathlib
import shutil

d = pathlib.Path('hooktheory_test_all')
o = pathlib.Path('hooktheory_test')

random.seed(0)
uids = sorted(list(set([p.stem for p in d.rglob('*.mid')])))
print(len(uids))

sample = set(random.sample(uids, 100))
print(sample)

for p in sorted(list(d.rglob('*.mid')) + list(d.rglob('*.mp3'))):
    if p.stem in sample:
        op = pathlib.Path(o, p.relative_to(d))
        op.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(str(p), str(op))
