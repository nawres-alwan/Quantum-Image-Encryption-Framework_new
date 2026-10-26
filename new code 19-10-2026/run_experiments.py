"""Run full-resolution QIRBP encryption/decryption on the four original images."""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
from qef_qirbp import (VERSION,ImageSpec,KeyMaterial,Schedule,encrypt_chunk,decrypt_chunk,
                       prepare_branch,branch_overlap,recover_rgb,diagnostic_preview)
from image_metrics import compare,entropy_channels,adjacent_correlations,verify_ssim
from prepare_inputs import prepare
from security_tests import run_security_tests

ROOT=Path(__file__).resolve().parent
OLD_CODE=ROOT.parent/'images encryption and decryption models'/'code'


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value): return 'Infinity' if value>0 else '-Infinity'
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,list): return [json_safe(v) for v in value]
    return value


def source_integrity():
    paths=[OLD_CODE/'encryption.py',OLD_CODE/'decryption.py',OLD_CODE/'qirbp_model.py']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def experiment(image,record,profile,replicate,args):
    spec=ImageSpec(image.shape[0],image.shape[1],args.wavelength_qubits)
    run_id=f'{profile}_{record["image_id"]}_r{replicate}'
    keypath=ROOT/'private_keys'/(run_id+'.json')
    if keypath.exists():
        if not args.replay_recorded_keys: raise FileExistsError('Recorded key exists; use --replay-recorded-keys only for private simulation reproduction')
        recorded_spec,key=KeyMaterial.load_private(keypath)
        if recorded_spec!=spec: raise ValueError('Recorded key has different image dimensions')
    else:
        key=KeyMaterial.generate(spec);key.save_private(keypath,spec)
    schedule=Schedule(spec,key)
    pixels=image.reshape(-1,3);decrypted=np.empty_like(pixels);preview=np.empty_like(pixels)
    rng=np.random.default_rng(2026101900+int(record['image_id'][-1])*100+replicate+(1000 if profile=='paper256' else 0))
    overlap_sum=0j;max_norm_error=0.0;max_core_error=0.0;max_wave_error=0.0
    encrypt_seconds=0.0;decrypt_seconds=0.0;preview_seconds=0.0
    next_notice=0;start=time.perf_counter()
    for first in range(0,spec.pixels,args.chunk_size):
        last=min(first+args.chunk_size,spec.pixels);k=np.arange(first,last)
        t0=time.perf_counter();cipher=encrypt_chunk(pixels[first:last],k,schedule);encrypt_seconds+=time.perf_counter()-t0
        max_norm_error=max(max_norm_error,float(np.max(np.abs(cipher.norm_squared_per_position()-1))))
        t0=time.perf_counter();recovered_k,core,wave=decrypt_chunk(cipher,schedule);decrypt_seconds+=time.perf_counter()-t0
        assert np.array_equal(recovered_k,k)
        plain_core,plain_wave=prepare_branch(pixels[first:last],spec.wavelength_qubits)
        overlap_sum+=np.sum(branch_overlap(plain_core,plain_wave,core,wave))
        max_core_error=max(max_core_error,float(np.max(np.abs(core-plain_core))))
        if spec.wavelength_qubits: max_wave_error=max(max_wave_error,float(np.max(np.abs(wave-plain_wave))))
        decrypted[recovered_k]=recover_rgb(core)
        t0=time.perf_counter();preview[cipher.positions]=diagnostic_preview(cipher.core,rng);preview_seconds+=time.perf_counter()-t0
        if args.save_checkpoints:
            state_dir=ROOT/'private_simulator_states'/run_id;state_dir.mkdir(parents=True,exist_ok=True)
            cipher.save_simulator_checkpoint(state_dir/f'chunk_{first:09d}.npz')
        progress=int(100*last/spec.pixels)
        if progress>=next_notice:
            print(f'{run_id}: {progress}% ({last}/{spec.pixels} pixels)',flush=True)
            next_notice+=25
    decrypted=decrypted.reshape(image.shape);preview=preview.reshape(image.shape)
    raw_fidelity=float(abs(overlap_sum/spec.pixels)**2)
    assert np.array_equal(decrypted,image)
    assert max_norm_error<1e-11 and abs(raw_fidelity-1)<1e-10
    assert max_core_error<1e-11 and max_wave_error<1e-11
    original_hash=hashlib.sha256(image.tobytes()).hexdigest()
    decrypted_hash=hashlib.sha256(decrypted.tobytes()).hexdigest()
    before=compare(image,preview);after=compare(image,decrypted)
    ent=entropy_channels(preview);adj=adjacent_correlations(preview)
    if replicate==1:
        folder=ROOT/'results'/profile;folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(image).save(folder/(record['image_id']+'_input.png'))
        Image.fromarray(decrypted).save(folder/(record['image_id']+'_decrypted.png'))
        Image.fromarray(preview).save(folder/(record['image_id']+'_diagnostic_preview.png'))
    row={'run_id':run_id,'profile':profile,'image_id':record['image_id'],'image_name':record['title'],
         'replicate':replicate,'height':spec.height,'width':spec.width,'N':spec.pixels,
         'n':spec.position_qubits,'r':spec.feature_qubits,'Q':spec.total_qubits,
         'valid_dimension':spec.valid_dimension,'full_hilbert_dimension':1 << spec.total_qubits,
         'key_encoding_bits':384,'reserved_uniform_substitution_bits':spec.position_qubits+2*spec.feature_qubits,
         'uniform_offset_entropy_bits':math.log2(spec.pixels),
         'privacy_core_entropy_bits':spec.position_qubits+2*spec.feature_qubits+math.log2(spec.pixels),
         'ideal_privacy_epsilon':0.0,'ideal_optimal_guess_probability':0.5,
         'quantum_fidelity_raw':raw_fidelity,'quantum_fidelity_clipped':min(1.0,raw_fidelity),
         'max_branch_norm_squared_error':max_norm_error,'max_recovered_core_amplitude_error':max_core_error,
         'max_recovered_wavelength_amplitude_error':max_wave_error,
         'byte_identical_decryption':original_hash==decrypted_hash,
         'input_rgb_sha256':original_hash,'decrypted_rgb_sha256':decrypted_hash,
         'decrypt_mse':after['mse'],'decrypt_psnr_db':after['psnr_db'],'decrypt_ssim_7x7':after['ssim_7x7'],
         'decrypt_changed_pixel_percent':after['changed_pixel_percent'],
         'preview_entropy_R':ent[0],'preview_entropy_G':ent[1],'preview_entropy_B':ent[2],
         'preview_entropy_mean':float(np.mean(ent)),
         'preview_horizontal_cc':adj['horizontal'],'preview_vertical_cc':adj['vertical'],'preview_diagonal_cc':adj['diagonal'],
         **{'preview_'+k:v for k,v in before.items()},
         'encrypt_seconds':encrypt_seconds,'decrypt_seconds':decrypt_seconds,'preview_seconds':preview_seconds,
         'elapsed_seconds':time.perf_counter()-start}
    print(f'{run_id}: exact decryption; state fidelity {raw_fidelity:.15f}',flush=True)
    return row


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles',nargs='+',default=['native','paper256'],choices=['native','paper256'])
    p.add_argument('--replicates',type=int,default=3)
    p.add_argument('--chunk-size',type=int,default=2048)
    p.add_argument('--wavelength-qubits',type=int,default=5)
    p.add_argument('--save-checkpoints',action='store_true',help='Private state descriptions; potentially large; outside ciphertext security target')
    p.add_argument('--skip-security-tests',action='store_true')
    p.add_argument('--replay-recorded-keys',action='store_true',help='Reproduce this private simulation; never reuse keys for a real transmission')
    args=p.parse_args()
    if args.replicates<1 or args.chunk_size<1: p.error('Replicates and chunk size must be positive')
    (ROOT/'results').mkdir(exist_ok=True);(ROOT/'private_keys').mkdir(exist_ok=True)
    initial_sources=source_integrity();manifest=prepare(ROOT)
    ssim_error=verify_ssim()
    if not args.skip_security_tests:
        tests=run_security_tests(OLD_CODE)
        (ROOT/'results'/'security_verification.json').write_text(json.dumps(tests,indent=2),encoding='utf-8')
        write_csv(ROOT/'results'/'exhaustive_channel_table.csv',tests['exhaustive_private_channels'])
        write_csv(ROOT/'results'/'ablation_table.csv',tests['ablations'])
        write_csv(ROOT/'results'/'dense_equivalence_table.csv',tests['dense_production_equivalence'])
        print('Independent dense checks, exhaustive channel checks, and legacy-attack reproduction passed.',flush=True)
    rows=[]
    for profile in args.profiles:
        for record in manifest:
            with Image.open(ROOT/record['relative_input']) as im:
                if profile=='paper256': im=im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
                else: im=im.convert('RGB')
                image=np.array(im)
            for replicate in range(1,args.replicates+1):
                rows.append(experiment(image,record,profile,replicate,args))
                write_csv(ROOT/'results'/'all_runs.csv',rows)
                (ROOT/'results'/'all_runs.json').write_text(json.dumps(json_safe(rows),indent=2,allow_nan=False),encoding='utf-8')
    final_sources=source_integrity()
    assert initial_sources==final_sources
    for rec in manifest:
        original=Path(rec['original_source'])
        if not original.is_absolute(): original=ROOT.parent/original
        if original.exists(): assert hashlib.sha256(original.read_bytes()).hexdigest()==rec['file_sha256']
        assert hashlib.sha256((ROOT/rec['relative_input']).read_bytes()).hexdigest()==rec['file_sha256']
    verification={'algorithm':VERSION,'completed_utc':datetime.now(timezone.utc).isoformat(),
                  'profiles':args.profiles,'replicates':args.replicates,'runs_completed':len(rows),
                  'all_exact_decryptions':all(r['byte_identical_decryption'] for r in rows),
                  'max_fidelity_error':max(abs(1-r['quantum_fidelity_raw']) for r in rows),
                  'max_branch_norm_error':max(r['max_branch_norm_squared_error'] for r in rows),
                  'ssim_independent_window_check_error':ssim_error,
                  'original_sources_before':initial_sources,'original_sources_after':final_sources,
                  'original_sources_unchanged':initial_sources==final_sources,
                  'input_files_unchanged':True,
                  'randomness_provider':'Python secrets; OS randomness; ideal theorem assumes uniform independent reserved bits',
                  'not_executed_on_quantum_hardware':True,
                  'diagnostic_previews_are_not_classical_ciphertexts':True}
    (ROOT/'results'/'run_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if 'sources' not in k},indent=2),flush=True)


if __name__=='__main__': main()
