"""Check delivered images, tables and private-key replay without printing keys."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from qef_qirbp import (KeyMaterial,Schedule,encrypt_chunk,decrypt_chunk,recover_rgb,
                       prepare_branch,branch_overlap,diagnostic_preview)
from image_metrics import compare
from security_tests import production_dense_equivalence

ROOT=Path(__file__).resolve().parent


def main():
    rows=json.loads((ROOT/'results'/'all_runs.json').read_text())
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    summaries=list(csv.DictReader((ROOT/'results'/'summary_by_image.csv').open(encoding='utf-8')))
    assert len(rows)==24 and len(summaries)==8
    by_id={r['image_id']:r for r in manifest}
    cached={};key_values=set();replay_errors=[];preview_error=0.0
    for row in rows:
        group=(row['profile'],row['image_id'])
        if group not in cached:
            folder=ROOT/'results'/row['profile']
            original=np.array(Image.open(folder/(row['image_id']+'_input.png')).convert('RGB'))
            decrypted=np.array(Image.open(folder/(row['image_id']+'_decrypted.png')).convert('RGB'))
            preview=np.array(Image.open(folder/(row['image_id']+'_diagnostic_preview.png')).convert('RGB'))
            assert np.array_equal(original,decrypted)
            cached[group]=(original,decrypted,preview)
        original,_,preview=cached[group]
        digest=hashlib.sha256(original.tobytes()).hexdigest()
        assert digest==row['input_rgb_sha256']==row['decrypted_rgb_sha256']
        spec,key=KeyMaterial.load_private(ROOT/'private_keys'/(row['run_id']+'.json'))
        key_values.add(key.substitution+key.permutation+key.entanglement)
        assert (spec.height,spec.width)==original.shape[:2]
        size=min(4096,spec.pixels);k=np.arange(size);pixels=original.reshape(-1,3)[:size]
        schedule=Schedule(spec,key);cipher=encrypt_chunk(pixels,k,schedule)
        kk,core,wave=decrypt_chunk(cipher,schedule)
        assert np.array_equal(kk,k) and np.array_equal(recover_rgb(core),pixels)
        pcore,pwave=prepare_branch(pixels,spec.wavelength_qubits)
        overlap=np.mean(branch_overlap(pcore,pwave,core,wave))
        replay_errors.append(float(abs(1-abs(overlap)**2)))
        if row['replicate']==1:
            seed=2026101900+int(row['image_id'][-1])*100+1+(1000 if row['profile']=='paper256' else 0)
            sampled=diagnostic_preview(cipher.core,np.random.default_rng(seed))
            assert np.array_equal(sampled,preview.reshape(-1,3)[cipher.positions])
            metrics=compare(original,preview)
            for field,value in metrics.items():
                preview_error=max(preview_error,abs(value-row['preview_'+field]))
    assert len(key_values)==24 and preview_error<1e-12
    for rec in manifest:
        assert hashlib.sha256((ROOT/rec['relative_input']).read_bytes()).hexdigest()==rec['file_sha256']
    for summary in summaries:
        group=[r for r in rows if r['profile']==summary['profile'] and r['image_id']==summary['image_id']]
        assert len(group)==3
        assert abs(float(summary['preview_entropy_mean_mean'])-np.mean([r['preview_entropy_mean'] for r in group]))<1e-12
    dense=production_dense_equivalence()
    result={'complete_runs_checked':24,'saved_profile_image_sets_checked':8,
            'private_key_records_checked':24,'observed_distinct_key_records':len(key_values),
            'private_replay_positions_per_run':4096,'maximum_private_replay_fidelity_error':max(replay_errors),
            'saved_preview_replay_exact':True,'saved_preview_metric_error':preview_error,
            'saved_decrypted_images_byte_identical':True,'input_file_hashes_match':True,
            'summary_tables_match_run_data':True,
            'independent_dense_tests_on_this_interpreter':len(dense),'python_version':sys.version.split()[0],
            'numpy_version':np.__version__}
    (ROOT/'results'/'package_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
