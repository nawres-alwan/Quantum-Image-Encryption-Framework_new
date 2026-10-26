"""QEF-QIRBP v2: unitary reference implementation with a uniform privacy core.

Three macro layers: position-controlled substitution, reversible address
permutation, and position-controlled CNOT/CZ entanglement. This is a repaired
variant, not a reproduction of an unspecified HCS/HG-Phi random-number source.

The default key provider is Python secrets (OS cryptographic randomness).
The exact privacy theorem assumes its reserved key bits are uniform and
independent. SHAKE-derived per-pixel Euler rotations are an invertible mixing
component and contribute no assumption to that theorem.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import secrets
from pathlib import Path
import numpy as np

VERSION = 'QEF-QIRBP-v2-uniform-core'


@dataclass(frozen=True)
class ImageSpec:
    height: int
    width: int
    wavelength_qubits: int = 5

    def __post_init__(self):
        if self.height < 1 or self.width < 1 or not 0 <= self.wavelength_qubits <= 16:
            raise ValueError('Invalid image dimensions or wavelength register')
        if self.pixels > (1 << 31)-1:
            raise ValueError('Reference integer arithmetic is limited to 2^31-1 pixels')
        if self.position_qubits + 2*self.feature_qubits >= 128:
            raise ValueError('Privacy core does not fit in the substitution key field')

    @property
    def pixels(self): return self.height*self.width
    @property
    def position_qubits(self): return (self.pixels-1).bit_length()
    @property
    def feature_qubits(self): return 6+self.wavelength_qubits
    @property
    def total_qubits(self): return self.position_qubits+self.feature_qubits
    @property
    def valid_dimension(self): return self.pixels*(1 << self.feature_qubits)


@dataclass(frozen=True)
class KeyMaterial:
    substitution: bytes
    permutation: bytes
    entanglement: bytes

    def __post_init__(self):
        if any(len(v) != 16 for v in (self.substitution,self.permutation,self.entanglement)):
            raise ValueError('Each of the three key fields must contain exactly 128 bits')

    @classmethod
    def generate(cls,spec):
        # The offset word is rejection-sampled before reduction modulo N.
        # Its distribution modulo N is exactly uniform in the ideal bit model.
        limit=((1 << 64)//spec.pixels)*spec.pixels
        word=secrets.randbits(64)
        while word >= limit: word=secrets.randbits(64)
        return cls(secrets.token_bytes(16),secrets.token_bytes(8)+word.to_bytes(8,'big'),secrets.token_bytes(16))

    def save_private(self,path,spec):
        data={'algorithm':VERSION,'private_research_key_do_not_publish':True,
              'height':spec.height,'width':spec.width,'wavelength_qubits':spec.wavelength_qubits,
              'substitution':self.substitution.hex(),'permutation':self.permutation.hex(),
              'entanglement':self.entanglement.hex()}
        Path(path).write_text(json.dumps(data,indent=2),encoding='utf-8')

    @classmethod
    def load_private(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('algorithm') != VERSION: raise ValueError('Unsupported key format')
        spec=ImageSpec(data['height'],data['width'],data['wavelength_qubits'])
        key=cls(*(bytes.fromhex(data[k]) for k in ['substitution','permutation','entanglement']))
        Schedule(spec,key)
        return spec,key


def parity(v):
    v=np.asarray(v,dtype=np.uint64).copy()
    for shift in (32,16,8,4,2,1): v ^= v >> np.uint64(shift)
    return (v & 1).astype(np.int8)


class Schedule:
    def __init__(self,spec,key):
        self.spec,self.key=spec,key
        n,r=spec.position_qubits,spec.feature_qubits
        value=int.from_bytes(key.substitution,'big')
        self.phase_z=value & ((1 << n)-1)
        self.feature_x=(value >> n) & ((1 << r)-1)
        self.feature_z=(value >> (n+r)) & ((1 << r)-1)
        self.mixing_seed=(value >> (n+2*r)).to_bytes(16,'big')
        raw_a=int.from_bytes(key.permutation[:8],'big')
        word=int.from_bytes(key.permutation[8:],'big')
        limit=((1 << 64)//spec.pixels)*spec.pixels
        if word >= limit: raise ValueError('Permutation key violates exact-offset rejection rule')
        self.shift=word % spec.pixels
        a=raw_a % spec.pixels if spec.pixels>1 else 1
        if a==0: a=1
        while math.gcd(a,spec.pixels)!=1: a=(a+1) % spec.pixels
        self.affine_a=a
        self.inverse_a=pow(a,-1,spec.pixels) if spec.pixels>1 else 0
        count=n*r
        raw=hashlib.shake_256(b'QEF-v2-entanglement'+key.entanglement).digest((count+7)//8)
        self.entanglement_choices=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little')[:count].reshape(n,r)
        self.context=b'QEF-v2-Euler'+spec.height.to_bytes(8,'big')+spec.width.to_bytes(8,'big')+bytes([spec.wavelength_qubits])

    def permute(self,k):
        return (self.affine_a*np.asarray(k,dtype=np.int64)+self.shift) % self.spec.pixels

    def inverse_permute(self,t):
        return (self.inverse_a*(np.asarray(t,dtype=np.int64)-self.shift)) % self.spec.pixels

    def local_rotations(self,k):
        r=self.spec.feature_qubits
        raw=b''.join(hashlib.shake_256(self.context+self.mixing_seed+int(i).to_bytes(8,'big')).digest(r*3*8) for i in k)
        angles=((np.frombuffer(raw,dtype='<u8') >> 11).astype(np.float64)*(2*math.pi/(1 << 53))).reshape(len(k),r,3)
        ca,sa=np.cos(angles[...,0]/2),np.sin(angles[...,0]/2)
        cb,sb=np.cos(angles[...,1]/2),np.sin(angles[...,1]/2)
        em=np.exp(-0.5j*angles[...,2]);ep=np.conj(em)
        u=np.empty((len(k),r,2,2),dtype=np.complex128)
        u[...,0,0]=em*(cb*ca+1j*sb*sa)
        u[...,0,1]=em*(-sb*ca-1j*cb*sa)
        u[...,1,0]=ep*(sb*ca-1j*cb*sa)
        u[...,1,1]=ep*(cb*ca-1j*sb*sa)
        return u

    def entanglement_masks(self,t):
        n,r=self.entanglement_choices.shape
        xx=np.zeros((len(t),r),dtype=np.uint8);zz=np.zeros_like(xx)
        sign=np.zeros(len(t),dtype=np.uint8)
        for p in range(n):
            active=((t >> (n-1-p)) & 1).astype(np.uint8)
            for j in range(r):
                if self.entanglement_choices[p,j]==0:
                    sign ^= active & zz[:,j]
                    xx[:,j] ^= active
                else: zz[:,j] ^= active
        return xx,zz,sign


@dataclass
class CipherChunk:
    spec: ImageSpec
    positions: np.ndarray
    core: np.ndarray
    wavelength: np.ndarray

    def norm_squared_per_position(self):
        result=np.sum(np.abs(self.core)**2,axis=1)
        if self.spec.wavelength_qubits:
            result *= np.prod(np.sum(np.abs(self.wavelength)**2,axis=2),axis=1)
        return result

    def save_simulator_checkpoint(self,path):
        # Full classical state descriptions are private simulation artifacts,
        # not quantum ciphertexts covered by the one-state privacy theorem.
        np.savez_compressed(path,positions=self.positions,core=self.core,wavelength=self.wavelength,
                            shape=np.array([self.spec.height,self.spec.width,self.spec.wavelength_qubits]),
                            artifact_type='private simulator state description')


def prepare_branch(rgb,wavelength_qubits):
    rgb=np.asarray(rgb,dtype=np.uint8).reshape(-1,3)
    count=len(rgb);planes=np.arange(8,dtype=np.uint8)
    colour=(((rgb[:,0,None] >> planes)&1) << 2) | (((rgb[:,1,None] >> planes)&1) << 1) | ((rgb[:,2,None] >> planes)&1)
    core=np.zeros((count,64),dtype=np.complex128)
    core[np.arange(count)[:,None],colour*8+planes]=1/math.sqrt(8)
    wave=np.full((count,wavelength_qubits,2),1/math.sqrt(2),dtype=np.complex128)
    return core,wave


def single_gate_batch(core,u,j):
    shaped=core.reshape(len(core),1 << j,2,1 << (5-j))
    a,b=shaped[:,:,0,:],shaped[:,:,1,:]
    out0=u[:,0,0,None,None]*a+u[:,0,1,None,None]*b
    out1=u[:,1,0,None,None]*a+u[:,1,1,None,None]*b
    return np.stack((out0,out1),axis=2).reshape(len(core),64)


def apply_local_rotations(core,wave,u,inverse=False):
    if inverse: u=u.conj().swapaxes(-1,-2)
    for j in range(6): core=single_gate_batch(core,u[:,j],j)
    wave=np.einsum('nvij,nvj->nvi',u[:,6:],wave,optimize=False)
    return core,wave


def apply_pauli_masks(core,wave,xx,zz,sign=None,inverse=False):
    count=len(core);r=6+wave.shape[1]
    xx=np.broadcast_to(xx,(count,r));zz=np.broadcast_to(zz,(count,r))
    scalar=np.zeros(count,dtype=np.uint8) if sign is None else sign.copy()
    if inverse: scalar ^= np.sum(xx & zz,axis=1,dtype=np.uint64).astype(np.uint8) & 1
    shifts=np.arange(5,-1,-1,dtype=np.uint8)
    xmask=np.sum(xx[:,:6] << shifts,axis=1,dtype=np.uint64)
    zmask=np.sum(zz[:,:6] << shifts,axis=1,dtype=np.uint64)
    idx=np.arange(64,dtype=np.uint64)[None,:]
    core=np.take_along_axis(core,(idx ^ xmask[:,None]).astype(np.int64),axis=1)
    core *= 1-2*parity(idx & zmask[:,None])
    core *= (1-2*scalar.astype(np.int8))[:,None]
    if wave.shape[1]:
        slots=np.arange(2,dtype=np.int64)[None,None,:] ^ xx[:,6:,None].astype(np.int64)
        wave=np.take_along_axis(wave,slots,axis=2)
        wave[:,:,1] *= 1-2*zz[:,6:].astype(np.int8)
    return core,wave


def feature_masks(schedule):
    r=schedule.spec.feature_qubits
    xx=np.array([(schedule.feature_x >> j)&1 for j in range(r)],dtype=np.uint8)
    zz=np.array([(schedule.feature_z >> j)&1 for j in range(r)],dtype=np.uint8)
    mu=(-1j)**int(np.sum(xx)+np.sum(zz))
    return xx,zz,mu


def encrypt_chunk(rgb,k,schedule):
    k=np.asarray(k,dtype=np.int64)
    if len(rgb)!=len(k) or np.any(k<0) or np.any(k>=schedule.spec.pixels): raise ValueError('Invalid plaintext positions')
    core,wave=prepare_branch(rgb,schedule.spec.wavelength_qubits)
    core,wave=apply_local_rotations(core,wave,schedule.local_rotations(k))
    xx,zz,mu=feature_masks(schedule)
    core,wave=apply_pauli_masks(core,wave,xx,zz)
    core *= (mu*(1-2*parity(k.astype(np.uint64) & np.uint64(schedule.phase_z))))[:,None]
    t=schedule.permute(k)
    ex,ez,es=schedule.entanglement_masks(t)
    core,wave=apply_pauli_masks(core,wave,ex,ez,es)
    return CipherChunk(schedule.spec,t,core,wave)


def decrypt_chunk(cipher,schedule):
    if cipher.spec != schedule.spec: raise ValueError('Key dimensions differ from the ciphertext')
    t=cipher.positions
    core,wave=cipher.core.copy(),cipher.wavelength.copy()
    ex,ez,es=schedule.entanglement_masks(t)
    core,wave=apply_pauli_masks(core,wave,ex,ez,es,inverse=True)
    k=schedule.inverse_permute(t)
    xx,zz,mu=feature_masks(schedule)
    core *= np.conj(mu)*(1-2*parity(k.astype(np.uint64) & np.uint64(schedule.phase_z)))[:,None]
    core,wave=apply_pauli_masks(core,wave,xx,zz,inverse=True)
    core,wave=apply_local_rotations(core,wave,schedule.local_rotations(k),inverse=True)
    # Remove only the representational gauge between tensor factors. Multiplying
    # each wavelength ket by phase^-1 and the core by phase leaves the ket equal.
    if wave.shape[1]:
        pivot=np.argmax(np.abs(wave),axis=2)
        selected=np.take_along_axis(wave,pivot[:,:,None],axis=2)[:,:,0]
        phase=selected/np.abs(selected)
        wave/=phase[:,:,None]
        core*=np.prod(phase,axis=1)[:,None]
    return k,core,wave


def recover_rgb(core):
    colour=np.argmax(np.abs(core.reshape(-1,8,8))**2,axis=1)
    values=np.zeros((len(core),3),dtype=np.uint16)
    for channel in range(3):
        values[:,channel]=np.sum((((colour >> (2-channel))&1) << np.arange(8)),axis=1)
    return values.astype(np.uint8)


def diagnostic_preview(core,rng):
    """Sample conditional bit planes for inspection, not a public ciphertext."""
    probabilities=np.abs(core.reshape(-1,8,8))**2
    probabilities /= np.maximum(np.sum(probabilities,axis=1,keepdims=True),1e-300)
    cdf=np.cumsum(probabilities,axis=1)
    draws=rng.random((len(core),8))
    colour=np.sum(draws[:,None,:] > cdf,axis=1).clip(0,7)
    result=np.zeros((len(core),3),dtype=np.uint16)
    for channel in range(3):
        result[:,channel]=np.sum((((colour >> (2-channel))&1) << np.arange(8)),axis=1)
    return result.astype(np.uint8)


def branch_overlap(core0,wave0,core1,wave1):
    result=np.sum(np.conj(core0)*core1,axis=1)
    if wave0.shape[1]: result *= np.prod(np.sum(np.conj(wave0)*wave1,axis=2),axis=1)
    return result


def materialize_small_state(cipher):
    """Dense state for small independent checks; full padded register included."""
    spec=cipher.spec
    if spec.total_qubits>16: raise ValueError('Dense verification is limited to 16 qubits')
    result=np.zeros((1 << spec.position_qubits,1 << spec.feature_qubits),dtype=np.complex128)
    for row,t in enumerate(cipher.positions):
        feature=cipher.core[row]
        for local in cipher.wavelength[row]: feature=np.kron(feature,local)
        result[int(t)]=feature/math.sqrt(spec.pixels)
    return result.reshape(-1)
