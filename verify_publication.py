"""Verify public research data and reproducibility without private keys."""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
NEW = ROOT / 'new code 19-10-2026'
TABLE7 = NEW / 'analysis' / 'QEF_All_Models_Source_Audit_20261019'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Optional destination for a JSON report')
    args = parser.parse_args()
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and not any(v in p.relative_to(ROOT).parts for v in
                               ('.git', '.venv', '__pycache__', 'local_runs')))
    assert not any('private_keys' in p.relative_to(ROOT).parts for p in files), 'Private keys must remain local.'
    python_files = [p for p in files if p.suffix == '.py']
    for path in python_files:
        compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
    json_files = [p for p in files if p.suffix == '.json']
    for path in json_files:
        json.loads(path.read_text(encoding='utf-8-sig'))

    publication = ROOT / 'publication_manifest.json'
    hashes_checked = 0
    if publication.exists():
        inventory = json.loads(publication.read_text(encoding='utf-8'))
        for record in inventory['files']:
            path = ROOT / record['path']
            assert path.is_file() and sha256(path) == record['sha256'], record['path']
            hashes_checked += 1

    manifest = json.loads((NEW / 'input_manifest.json').read_text(encoding='utf-8'))
    for row in manifest:
        path = NEW / Path(row['relative_input'].replace('\\', '/'))
        assert sha256(path) == row['file_sha256']
        image = np.asarray(Image.open(path).convert('RGB'))
        assert image.shape == (row['height'], row['width'], 3)
        assert hashlib.sha256(image.tobytes()).hexdigest() == row['rgb_pixel_sha256']
        original = ROOT / row['original_source']
        assert original.is_file() and sha256(original) == row['file_sha256']

    sys.path.insert(0, str(NEW))
    from image_metrics import compare
    import security_tests

    runs = json.loads((NEW / 'results' / 'all_runs.json').read_text(encoding='utf-8'))
    assert len(runs) == 24
    assert len(read_rows(NEW / 'results' / 'all_runs.csv')) == 24
    groups = {(row['profile'], row['image_id']) for row in runs}
    assert len(groups) == 8
    for profile, image_id in sorted(groups):
        folder = NEW / 'results' / profile
        original = np.asarray(Image.open(folder / (image_id + '_input.png')).convert('RGB'))
        decrypted = np.asarray(Image.open(folder / (image_id + '_decrypted.png')).convert('RGB'))
        preview = np.asarray(Image.open(folder / (image_id + '_diagnostic_preview.png')).convert('RGB'))
        assert np.array_equal(original, decrypted)
        digest = hashlib.sha256(original.tobytes()).hexdigest()
        matching = [row for row in runs if (row['profile'], row['image_id']) == (profile, image_id)]
        assert len(matching) == 3
        for row in matching:
            assert digest == row['input_rgb_sha256'] == row['decrypted_rgb_sha256']
            assert row['byte_identical_decryption'] is True
        first = next(row for row in matching if row['replicate'] == 1)
        for field, value in compare(original, preview).items():
            expected = first['preview_' + field]
            if math.isfinite(value):
                assert abs(value - expected) < 1e-10, (profile, image_id, field)

    saved = read_rows(TABLE7 / 'Table7_All_Models.csv')
    assert len(saved) == 128
    spec = importlib.util.spec_from_file_location('table7_publication_check', TABLE7 / 'calculate_all_models.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    with tempfile.TemporaryDirectory(prefix='qef-table7-') as directory:
        generator.ROOT = Path(directory)
        with contextlib.redirect_stdout(io.StringIO()):
            generator.main()
        assert read_rows(Path(directory) / 'Table7_All_Models.csv') == saved
        assert (Path(directory) / 'Table7_All_Models.tex').read_bytes() == (TABLE7 / 'Table7_All_Models.tex').read_bytes()

    dense = security_tests.production_dense_equivalence()
    legacy = security_tests.legacy_attack(ROOT / 'images encryption and decryption models' / 'code')
    result = {
        'passed': True,
        'published_file_hashes_checked': hashes_checked,
        'python_sources_parsed': len(python_files),
        'json_files_parsed': len(json_files),
        'source_images_hash_verified': len(manifest),
        'saved_profile_image_sets_checked': len(groups),
        'saved_run_records_checked': len(runs),
        'saved_decrypted_images_byte_identical': True,
        'saved_preview_metrics_match': True,
        'table7_detailed_rows_regenerated': len(saved),
        'table7_display_rows': len({(r['image'], r['model']) for r in saved}),
        'table7_latex_byte_identical': True,
        'independent_dense_operator_cases': len(dense),
        'legacy_array_regression_cases': legacy['cases'],
        'legacy_array_distinguishing_successes': legacy['correct'],
        'private_keys_published': False,
        'private_key_replay_performed': False,
        'full_image_experiments_rerun': False,
        'scope': 'Public data integrity, mathematical table regeneration, and small-system regression checks; not a source-security certification.',
        'python_version': sys.version.split()[0],
        'numpy_version': np.__version__,
    }
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
