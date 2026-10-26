"""Independent dense-matrix checks and finite-enumeration privacy experiments."""
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from qef_qirbp import (ImageSpec,KeyMaterial,Schedule,encrypt_chunk,decrypt_chunk,
                      prepare_branch,branch_overlap,recover_rgb,materialize_small_state)

I=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],dtype=complex)
Z=np.diag([1,-1]).astype(complex)


def tensor(items):
    result=np.ones((1,1),dtype=complex)
    for item in items: result=np.kron(result,item)
    return result


def entangle_dense(matrix,count,r,choices):
    """Apply actual chronological CNOT/CZ gates to basis labels, independently."""
    d=1 << r;n=(count-1).bit_length();size=count*d
    t=np.repeat(np.arange(count),d);f=np.tile(np.arange(d),count)
    phase=np.ones(size,dtype=np.int8)
    for p in range(n):
        active=((t >> (n-1-p)) & 1).astype(bool)
        for j in range(r):
            bit=1 << (r-1-j)
            if choices[p,j]==0: f[active] ^= bit
            else: phase[active] *= 1-2*((f[active] >> (r-1-j)) & 1).astype(np.int8)
    result=np.empty_like(matrix)
    result[t*d+f]=phase[:,None]*matrix
    return result


def dense_from_parts(count,r,base,z,a,b,affine,shift,choices):
    d=1 << r
    local=[]
    for j in range(r):
        aj,bj=(a >> j)&1,(b >> j)&1
        local.append(((-1j)**(aj+bj))*np.linalg.matrix_power(Z,bj)@np.linalg.matrix_power(X,aj))
    v=tensor(local)
    result=np.zeros((count*d,count*d),dtype=complex)
    for k in range(count):
        t=(affine*k+shift) % count
        sign=(-1)**((z & k).bit_count()%2)
        result[t*d:(t+1)*d,k*d:(k+1)*d]=sign*v@base[k]
    return entangle_dense(result,count,r,choices)


def production_dense_equivalence():
    rng=np.random.default_rng(202610194)
    rows=[]
    for case,(height,width,wq) in enumerate([(2,2,1),(2,2,1),(1,3,0),(1,3,0)]):
        spec=ImageSpec(height,width,wq);key=KeyMaterial.generate(spec);s=Schedule(spec,key)
        image=rng.integers(0,256,(spec.pixels,3),dtype=np.uint8)
        k=np.arange(spec.pixels);local=s.local_rotations(k)
        base=[tensor(local[row]) for row in range(spec.pixels)]
        u=dense_from_parts(spec.pixels,spec.feature_qubits,base,s.phase_z,s.feature_x,s.feature_z,
                           s.affine_a,s.shift,s.entanglement_choices)
        core,wave=prepare_branch(image,wq)
        initial=np.concatenate([tensor_vector(core[row],wave[row]) for row in range(spec.pixels)])/math.sqrt(spec.pixels)
        encrypted=encrypt_chunk(image,k,s)
        dense_state=materialize_small_state(encrypted)[:spec.valid_dimension]
        output_error=float(np.max(np.abs(u@initial-dense_state)))
        unitary_error=float(np.max(np.abs(u.conj().T@u-np.eye(spec.valid_dimension))))
        restored_k,restored_core,restored_wave=decrypt_chunk(encrypted,s)
        recovered=np.concatenate([tensor_vector(restored_core[row],restored_wave[row]) for row in range(spec.pixels)])/math.sqrt(spec.pixels)
        inverse_error=float(np.max(np.abs(u.conj().T@dense_state-recovered)))
        reference_error=float(np.linalg.norm(recovered-initial))
        assert max(output_error,unitary_error,inverse_error,reference_error)<1e-11
        assert np.array_equal(restored_k,k) and np.array_equal(recover_rgb(restored_core),image)
        # Reordering serialized chunks must not be needed to decrypt them.
        order=rng.permutation(spec.pixels)
        from qef_qirbp import CipherChunk
        shuffled=CipherChunk(spec,encrypted.positions[order],encrypted.core[order],encrypted.wavelength[order])
        sk,sc,sw=decrypt_chunk(shuffled,s)
        assert np.array_equal(recover_rgb(sc),image[sk])
        rows.append({'case':case,'pixels':spec.pixels,'r':spec.feature_qubits,'valid_dimension':spec.valid_dimension,
                     'unitarity_max_error':unitary_error,'factorized_dense_max_error':output_error,
                     'inverse_dense_max_error':inverse_error,'recovered_state_l2_error':reference_error})
    return rows


def tensor_vector(core,wave):
    result=core
    for state in wave: result=np.kron(result,state)
    return result


def small_ensemble(count,r,ablation=None):
    n=(count-1).bit_length();rng=np.random.default_rng(1000+count*100+r)
    base=[]
    for k in range(count):
        local=[]
        for j in range(r):
            alpha,beta,gamma=rng.uniform(0,2*math.pi,3)
            rx=math.cos(alpha/2)*I-1j*math.sin(alpha/2)*X
            ry=np.array([[math.cos(beta/2),-math.sin(beta/2)],[math.sin(beta/2),math.cos(beta/2)]])
            rz=np.diag([np.exp(-0.5j*gamma),np.exp(0.5j*gamma)])
            local.append(rz@ry@rx)
        base.append(tensor(local))
    affine=next(a for a in range(max(1,count//2),count+1) if math.gcd(a,count)==1)
    choices=rng.integers(0,2,(n,r),dtype=np.uint8)
    zs=[0] if ablation=='no_position_phase' else range(1 << n)
    masks=[0] if ablation=='no_feature_mask' else range(1 << r)
    shifts=[0] if ablation=='no_uniform_shift' else range(count)
    return [dense_from_parts(count,r,base,z,a,b,affine,s,choices)
            for z,a,b,s in itertools.product(zs,masks,masks,shifts)]


def trace_distance(a,b):
    delta=(a-b+(a-b).conj().T)/2
    return float(np.sum(np.abs(np.linalg.eigvalsh(delta)))/2)


def analyze_ensemble(count,r,ablation=None):
    ensemble=small_ensemble(count,r,ablation)
    d=count*(1 << r)
    choi=np.zeros((d*d,d*d),dtype=complex)
    for u in ensemble:
        v=u.reshape(-1,order='F')/math.sqrt(d)
        choi += np.outer(v,v.conj())/len(ensemble)
    ideal=np.eye(d*d)/(d*d)
    residual=float(np.max(np.abs(choi-ideal)))
    choi_distance=trace_distance(choi,ideal)
    # Distinguish two arbitrary orthogonal valid states, not only classical pixels.
    psi0=np.zeros(d,dtype=complex);psi0[0]=1
    psi1=np.zeros(d,dtype=complex);psi1[-1]=1
    averaged=[]
    for psi in [psi0,psi1]:
        averaged.append(sum(np.outer(u@psi,(u@psi).conj()) for u in ensemble)/len(ensemble))
    pair_distance=trace_distance(*averaged)
    rng=np.random.default_rng(992+count+r)
    psi=rng.normal(size=(d,2))+1j*rng.normal(size=(d,2));psi/=np.linalg.norm(psi)
    rho_ref=psi.T@psi.conj()
    actual=np.zeros((2*d,2*d),dtype=complex)
    for u in ensemble:
        vector=(u@psi).reshape(-1)
        actual += np.outer(vector,vector.conj())/len(ensemble)
    expected=np.kron(np.eye(d)/d,rho_ref)
    reference_distance=trace_distance(actual,expected)
    if ablation is None:
        assert max(residual,choi_distance,pair_distance,reference_distance)<1e-11
    else: assert choi_distance>1e-3
    return {'N':count,'r':r,'n':(count-1).bit_length(),'dimension':d,'enumerated_keys':len(ensemble),
            'ablation':ablation or 'complete','choi_max_abs_residual':residual,
            'normalized_choi_trace_distance':choi_distance,
            'pair_ciphertext_trace_distance':pair_distance,'optimal_pair_guess_probability_numeric':0.5+pair_distance/2,
            'reference_decoupling_trace_distance':reference_distance,
            'choi_trace':float(np.trace(choi).real)}


def test_sampler_and_permutation():
    checks=[]
    for count in [1,2,3,7,16,50394,65536,786432]:
        spec=ImageSpec(1,count,5)
        for _ in range(8):
            key=KeyMaterial.generate(spec);s=Schedule(spec,key)
            probes=np.unique(np.array([0,count//2,count-1],dtype=np.int64))
            assert np.array_equal(s.inverse_permute(s.permute(probes)),probes)
        counts=np.bincount(np.arange((256//count)*count)%count,minlength=count) if count<=256 else None
        if counts is not None: assert np.all(counts==counts[0])
        limit=((1 << 64)//count)*count
        if limit < (1 << 64):
            bad=KeyMaterial(bytes(16),bytes(8)+limit.to_bytes(8,'big'),bytes(16))
            try: Schedule(spec,bad)
            except ValueError: pass
            else: raise AssertionError('Invalid offset accepted')
        checks.append({'N':count,'offset_acceptance_support':str(limit),
                       'roundtrip_checked':True,'exact_rejection_rule_checked':True})
    return checks


def wrong_phase_key_test():
    rng=np.random.default_rng(9201);rows=[]
    for height,width in [(2,2),(1,3)]:
        spec=ImageSpec(height,width,5);key=KeyMaterial.generate(spec)
        image=rng.integers(0,256,(spec.pixels,3),dtype=np.uint8)
        original_core,original_wave=prepare_branch(image,5)
        cipher=encrypt_chunk(image,np.arange(spec.pixels),Schedule(spec,key))
        wrong=KeyMaterial((int.from_bytes(key.substitution,'big') ^ 1).to_bytes(16,'big'),key.permutation,key.entanglement)
        k,core,wave=decrypt_chunk(cipher,Schedule(spec,wrong))
        overlap=np.sum(branch_overlap(original_core,original_wave,core,wave))/spec.pixels
        fidelity=float(abs(overlap)**2)
        expected=(0.0 if spec.pixels%2==0 else 1/spec.pixels**2)
        assert abs(fidelity-expected)<1e-12 and np.array_equal(recover_rgb(core),image)
        rows.append({'N':spec.pixels,'wrong_key_bit':'position-phase bit 0',
                     'quantum_fidelity':fidelity,'exact_expected_fidelity':expected,
                     'classical_pixel_mse':0.0,
                     'meaning':'image readout alone cannot detect every quantum phase error'})
    return rows


def legacy_attack(old_code):
    ns={'np':np,'math':math,'__name__':'isolated_old_qef'}
    modules=[('FRQI.py','FRQI'),('NEQR.py','NEQR'),('AFQIRHSI.py','AFQIRHSI'),('qirbp_model.py','QIRBP'),('qirwc_model.py','QIRWC')]
    for filename,name in modules:
        path=Path(old_code)/filename;tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    path=Path(old_code)/'encryption.py';tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='encrypt']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    correct=0;cases=0
    for side in [2,4]:
        for seed in range(32):
            for bit in [0,1]:
                image=np.zeros((side,side,3),dtype=np.uint8)
                if bit: image[0,0]=255
                cipher=ns['encrypt'](ns['QIRBP'](image),key=seed)
                guess=int(np.any(cipher['flat'][-1]!=0))
                correct+=guess==bit;cases+=1
    assert correct==cases
    return {'cases':cases,'correct':int(correct),'guess_probability':correct/cases,
            'scope':'old classical-array code, last-pixel checksum attack',
            'encryption_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def run_security_tests(old_code=None):
    result={'dense_production_equivalence':production_dense_equivalence(),
            'exact_sampler_and_permutation':test_sampler_and_permutation(),
            'wrong_phase_key_tests':wrong_phase_key_test(),
            'exhaustive_private_channels':[analyze_ensemble(n,r) for n,r in [(2,1),(3,1),(4,1),(2,2),(3,2),(4,2)]],
            'ablations':[analyze_ensemble(4,1,a) for a in ['no_position_phase','no_feature_mask','no_uniform_shift']]}
    if old_code: result['legacy_attack']=legacy_attack(old_code)
    return result


if __name__=='__main__':
    result=run_security_tests()
    out=Path('results');out.mkdir(exist_ok=True)
    (out/'security_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'dense_cases':len(result['dense_production_equivalence']),
                      'exhaustive_channel_cases':len(result['exhaustive_private_channels']),
                      'max_choi_distance':max(r['normalized_choi_trace_distance'] for r in result['exhaustive_private_channels'])},indent=2))
