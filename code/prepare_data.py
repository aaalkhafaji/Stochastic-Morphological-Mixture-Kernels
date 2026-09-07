"""Download canonical inputs, record hashes and IDs, then freeze clean-image splits."""
from pathlib import Path
import argparse, gzip, hashlib, json, struct, urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URLS = {
    'mnist_train.gz': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
    'mnist_test.gz': 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
    'fashion_train.gz': 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz',
    'fashion_test.gz': 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz'}

def shapes(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    rr, cc = np.mgrid[:48, :48]
    for _ in range(n):
        x = np.zeros((48, 48), bool)
        # Independent primitives; overlaps are allowed and true topology is measured.
        for j in range(int(rng.integers(1, 6))):
            r, c = rng.integers(7, 41, size=2)
            a, b = rng.integers(3, 9, size=2)
            kind = int(rng.integers(0, 3))
            if kind == 0: obj = (abs(rr-r) <= a) & (abs(cc-c) <= b)
            else:
                dist = ((rr-r)/a)**2 + ((cc-c)/b)**2
                obj = dist <= 1
                if kind == 2: obj &= dist >= .35
            x |= obj
        out.append(x)
    return np.stack(out)

def main(cache):
    cache.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, url in URLS.items():
        p = cache / name
        if not p.exists():
            with urllib.request.urlopen(url, timeout=90) as f: p.write_bytes(f.read())
        raw = p.read_bytes()
        b = gzip.decompress(raw)
        magic, n, h, w = struct.unpack('>4I', b[:16])
        assert magic == 2051 and len(b) == 16+n*h*w
        records[name] = {'url': url, 'sha256': hashlib.sha256(raw).hexdigest(),
                         'md5': hashlib.md5(raw).hexdigest(), 'shape':[n,h,w]}
    for ds in ('mnist', 'fashion'):
        train = np.frombuffer(gzip.decompress((cache/f'{ds}_train.gz').read_bytes()), np.uint8, offset=16).reshape(-1,28,28)
        test = np.frombuffer(gzip.decompress((cache/f'{ds}_test.gz').read_bytes()), np.uint8, offset=16).reshape(-1,28,28)
        rng = np.random.default_rng(20260905)
        ix = rng.permutation(len(train))
        tx = rng.permutation(len(test))
        seen=set(); selected=[]
        for i in ix:
            key=np.packbits(train[i]>=128).tobytes()
            if key not in seen:
                selected.append(i);seen.add(key)
            if len(selected)==2500: break
        ix=np.asarray(selected)
        # Pilot test IDs and their binarized duplicates are quarantined.
        pilot=tx[:1000].copy()
        seen.update(np.packbits(test[i]>=128).tobytes() for i in pilot)
        selected=[]
        for i in tx[1000:]:
            key=np.packbits(test[i]>=128).tobytes()
            if key not in seen:
                selected.append(i);seen.add(key)
            if len(selected)==1000:break
        tx=np.asarray(selected)
        # No class labels used; threshold is fixed before inspecting results.
        np.savez_compressed(ROOT/'data'/f'{ds}_clean.npz',
            train=train[ix[:2000]] >= 128, val=train[ix[2000:2500]] >= 128,
            test=test[tx] >= 128, train_ids=ix[:2000], val_ids=ix[2000:2500], test_ids=tx,
            source_train=train[ix[:2500]], source_test=test[tx], pilot_test_ids=pilot)
    train_shapes=shapes(1000,61001);val_shapes=shapes(300,61002);test_shapes=shapes(500,64003)
    seen={np.packbits(x).tobytes() for x in np.concatenate([train_shapes,val_shapes,shapes(500,61003)])}
    excluded=[]
    for i,x in enumerate(test_shapes):
        key=np.packbits(x).tobytes()
        if key in seen:excluded.append(i)
        seen.add(key)
    np.savez_compressed(ROOT/'data'/'shapes_clean.npz', train=train_shapes,
        val=val_shapes, test=test_shapes, test_excluded=np.asarray(excluded,dtype=int),
        train_ids=np.arange(1000), val_ids=np.arange(300), test_ids=np.arange(500))
    (ROOT/'data'/'source_manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    print('Saved clean masks, source pixels, split IDs and source checksums.', flush=True)

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--cache', type=Path, default=ROOT/'downloads')
    main(ap.parse_args().cache)
