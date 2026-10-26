import numpy as np
from numpy.fft import fft2, ifft2

def encrypt_image_ref3(rgb_image: np.ndarray, key1: int, key2: int) -> np.ndarray:
    """Encrypt an RGB image using space-frequency dual encryption (Ref [3])."""
    H, W, _ = rgb_image.shape
    # 1. RGB to HSI conversion
    rgb = rgb_image.astype(float) / 255.0
    H_chan, S_chan, I_chan = rgb_to_hsi(rgb)   # assume this function exists
    # 2. Double random-phase encryption on Hue and Saturation
    rng1 = np.random.RandomState(key1)
    phase1_H = np.exp(1j * 2*np.pi * rng1.rand(H, W))
    phase2_H = np.exp(1j * 2*np.pi * rng1.rand(H, W))
    F_H = fft2(H_chan * phase1_H)
    H_enc = np.real(ifft2(F_H * phase2_H))     # use real part of complex result as encrypted H
    phase1_S = np.exp(1j * 2*np.pi * rng1.rand(H, W))
    phase2_S = np.exp(1j * 2*np.pi * rng1.rand(H, W))
    F_S = fft2(S_chan * phase1_S)
    S_enc = np.abs(ifft2(F_S * phase2_S))      # use magnitude of result as encrypted S
    # 3. Intensity channel bit-plane diffusion
    rng2 = np.random.RandomState(key2)
    # Generate random permutation of 8 bit-plane indices
    perm = rng2.permutation(8)
    # Generate two random bit matrices for XOR and XNOR
    xor_mask = rng2.randint(0, 2, size=(H, W), dtype=np.uint8)
    xnor_mask = rng2.randint(0, 2, size=(H, W), dtype=np.uint8)
    # Split intensity into 8 bit-planes
    I_bytes = np.round(I_chan * 255).astype(np.uint8)
    planes = [((I_bytes >> b) & 1).astype(np.uint8) for b in range(8)]
    # Apply bit-plane permutation
    permuted_planes = [planes[b] for b in perm]
    # XOR and XNOR diffusion on each plane
    for i, plane in enumerate(permuted_planes):
        plane ^= xor_mask         # XOR: flip bits where mask=1
        # XNOR: flip bits where xnor_mask=0 (since b XNOR 0 = NOT b, b XNOR 1 = b)
        flip_idx = (xnor_mask == 0)
        plane[flip_idx] ^= 1
        permuted_planes[i] = plane
    # Inverse permute bit-planes back to original order
    inv_perm = np.argsort(perm)
    planes_enc = [permuted_planes[j] for j in inv_perm]
    # Reconstruct encrypted intensity from bit-planes
    I_enc_bytes = np.zeros((H, W), dtype=np.uint8)
    for b in range(8):
        I_enc_bytes |= (planes_enc[b] << b)
    I_enc = I_enc_bytes.astype(float) / 255.0
    # 4. Convert encrypted HSI back to RGB
    cipher_rgb = hsi_to_rgb(H_enc, S_enc, I_enc) * 255.0  # assume this function exists
    return cipher_rgb.astype(np.uint8)
