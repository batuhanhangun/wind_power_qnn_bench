"""Copied original data/results must remain byte-identical across platforms."""
import hashlib
import json
from qnnbench.config import ROOT

def test_original_data_and_result_checksums():
    manifest=json.loads((ROOT/'results/reference/FILE_MANIFEST.json').read_text(encoding='utf-8'))
    checked=0
    for entry in manifest:
        path=entry['destination'].replace('\\','/')
        if not path.startswith(('data/seed_','data/raw/','results/reference/')):
            continue
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==entry['sha256'],path
        checked+=1
    assert checked>=400
