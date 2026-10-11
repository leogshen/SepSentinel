"""Resume a four-connection download of official public CinC training files."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import re
import threading
import time
import requests

OUT = Path(__file__).resolve().parents[2] / 'data_local/cinc2019_official'
LOCAL = threading.local()

def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    return LOCAL.session

def fetch(item):
    url, path = item
    if path.exists() and path.read_bytes().startswith(b'HR|O2Sat|Temp|'):
        return 'cached'
    for attempt in range(5):
        try:
            r = session().get(url, timeout=45)
            r.raise_for_status()
            assert r.content.startswith(b'HR|O2Sat|Temp|'), 'unexpected response'
            tmp = path.with_suffix('.part')
            tmp.write_bytes(r.content)
            tmp.replace(path)
            return 'downloaded'
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    # B first: the candidate external target; A is for provenance reconciliation.
    for site in ['B', 'A']:
        url = f'https://physionet.org/files/challenge-2019/1.0.0/training/training_set{site}/'
        directory = OUT / f'training_set{site}'
        directory.mkdir(exist_ok=True)
        r = session().get(url, timeout=45)
        r.raise_for_status()
        names = sorted(set(re.findall(r'href="(p[0-9]+\.psv)"', r.text)))
        assert len(names) >= 20000
        print(f'{site}: {len(names)} official files', flush=True)
        counts = {'cached': 0, 'downloaded': 0}
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            for i, result in enumerate(pool.map(fetch, [(url+n, directory/n) for n in names]), 1):
                counts[result] += 1
                if i % 500 == 0:
                    print(site, i, counts, flush=True)
        report[site] = {'url': url, 'files': len(names), **counts,
                        'listing_sha256': hashlib.sha256(r.content).hexdigest()}
        (OUT/'download_manifest.json').write_text(json.dumps(report, indent=2))
    print('Download complete', flush=True)

if __name__ == '__main__':
    main()
