import numpy as np
import math
from FRQI import FRQI
from NEQR import NEQR
from AFQIRHSI import AFQIRHSI
from qirbp_model import QIRBP
from qirwc_model import QIRWC

def decrypt(qir_obj, encrypted_data, key=0):
    """
    Decrypt the encrypted quantum image data back into the given QIR object's representation.
    It reverses the entanglement, permutation, and substitution steps in that order.
    The qir_obj (newly instantiated) will be populated with the decrypted representation.
    """
    np.random.seed(key)  # re-seed RNG to regenerate any random values used in encryption
    if isinstance(qir_obj, FRQI):
        # Encrypted data contains flattened angles and permutation.
        theta_chain = encrypted_data['theta']
        perm_indices = encrypted_data['perm']
        shape = encrypted_data['shape']
        # Reverse Entanglement: subtract previous angle from current (reverse chain addition).
        theta_flat = theta_chain.copy()
        for i in range(theta_flat.size-1, 0, -1):
            theta_flat[i] = (theta_flat[i] - theta_flat[i-1]) % (2 * math.pi)
        # Reverse Permutation: invert the pixel shuffle.
        inv_perm = np.empty_like(perm_indices)
        inv_perm[perm_indices] = np.arange(perm_indices.size)
        theta_unperm = theta_flat[inv_perm]
        # At this point, theta_unperm = original θ + random θ (from substitution).
        # Reverse Substitution: subtract the same random angles added (regenerate and subtract).
        # We can regenerate random angles in the same order:
        np.random.seed(key)
        rand_angles = np.random.rand(theta_unperm.size) * 2 * math.pi
        theta_orig = (theta_unperm - rand_angles) % (2 * math.pi)
        # Reshape to original image shape and set in qir_obj
        qir_obj.theta = theta_orig.reshape(shape)
        qir_obj.shape = shape
        return qir_obj

    elif isinstance(qir_obj, NEQR):
        flat_chain = encrypted_data['flat']
        perm_indices = encrypted_data['perm']
        shape = encrypted_data['shape']
        # Reverse Entanglement: XOR backward through the chain.
        flat = flat_chain.copy().astype(np.uint8)
        for i in range(flat.size-1, 0, -1):
            flat[i] ^= flat[i-1]
        # Reverse Permutation: invert the shuffle of pixels.
        inv_perm = np.empty_like(perm_indices)
        inv_perm[perm_indices] = np.arange(perm_indices.size)
        flat_unperm = flat[inv_perm]
        # Reverse Substitution: XOR with the same random values (regenerated).
        np.random.seed(key)
        regen_sub_key = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        img_orig = np.bitwise_xor(flat_unperm.reshape(shape), regen_sub_key)
        qir_obj.data = img_orig.astype(np.uint8)
        qir_obj.shape = shape
        return qir_obj

    elif isinstance(qir_obj, AFQIRHSI):
        H_chain = encrypted_data['H']
        S_chain = encrypted_data['S']
        I_chain = encrypted_data['I']
        perm_indices = encrypted_data['perm']
        shape = encrypted_data['shape']
        # Reverse Entanglement: XOR backward in the flattened sequence for H, S, I.
        H_flat = H_chain.copy().astype(np.uint8)
        S_flat = S_chain.copy().astype(np.uint8)
        I_flat = I_chain.copy().astype(np.uint8)
        for i in range(H_flat.size-1, 0, -1):
            H_flat[i] ^= H_flat[i-1]
            S_flat[i] ^= S_flat[i-1]
            I_flat[i] ^= I_flat[i-1]
        # Reverse Permutation: (since we skipped permutation, this is effectively identity).
        inv_perm = np.empty_like(perm_indices)
        inv_perm[perm_indices] = np.arange(perm_indices.size)
        H_unperm = H_flat[inv_perm].reshape(shape)
        S_unperm = S_flat[inv_perm].reshape(shape)
        I_unperm = I_flat[inv_perm].reshape(shape)
        # Reverse Substitution: XOR with the same random keys for H, S, I.
        np.random.seed(key)
        keyH = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        keyS = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        keyI = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        H_orig = np.bitwise_xor(H_unperm, keyH)
        S_orig = np.bitwise_xor(S_unperm, keyS)
        I_orig = np.bitwise_xor(I_unperm, keyI)
        # Set decrypted H, S, I in the qir_obj
        qir_obj.H = H_orig.astype(np.uint8)
        qir_obj.S = S_orig.astype(np.uint8)
        qir_obj.I = I_orig.astype(np.uint8)
        qir_obj.shape = shape
        return qir_obj

    elif isinstance(qir_obj, QIRBP):
        flat_chain = encrypted_data['flat']
        perm_indices = encrypted_data['perm']
        shape = encrypted_data['shape']
        mask = encrypted_data['mask']
        # Reverse Entanglement: XOR backward through flattened array.
        flat = flat_chain.copy().astype(np.uint8)
        if flat.ndim == 2 and flat.shape[1] > 1:
            # Color (N x 3) or multi-channel
            for i in range(flat.shape[0]-1, 0, -1):
                flat[i, 0] ^= flat[i-1, 0]
                flat[i, 1] ^= flat[i-1, 1]
                flat[i, 2] ^= flat[i-1, 2]
        else:
            # Grayscale (N x 1)
            for i in range(flat.shape[0]-1, 0, -1):
                flat[i, 0] ^= flat[i-1, 0]
        # Reverse Permutation: invert pixel shuffle.
        inv_perm = np.empty_like(perm_indices)
        inv_perm[perm_indices] = np.arange(perm_indices.size)
        unperm_flat = flat[inv_perm]
        # Reshape to original image shape
        decrypted = unperm_flat.reshape(shape)
        # Reverse Substitution: flip the same bit-planes again (flipping twice restores original).
        if decrypted.ndim == 2:
            mask_bits = mask  # tuple of length 8 for grayscale
            for bit in range(8):
                if mask_bits[bit] == 1:
                    decrypted = np.bitwise_xor(decrypted, np.uint8(1 << bit))
        else:
            maskR, maskG, maskB = mask
            R, G, B = decrypted[...,0], decrypted[...,1], decrypted[...,2]
            for bit in range(8):
                if maskR[bit]: R[:] = np.bitwise_xor(R, np.uint8(1 << bit))
                if maskG[bit]: G[:] = np.bitwise_xor(G, np.uint8(1 << bit))
                if maskB[bit]: B[:] = np.bitwise_xor(B, np.uint8(1 << bit))
            decrypted[...,0], decrypted[...,1], decrypted[...,2] = R, G, B
        qir_obj.data = decrypted.astype(np.uint8)
        qir_obj.shape = shape
        return qir_obj

    elif isinstance(qir_obj, QIRWC):
        enc_image = encrypted_data['data'].astype(np.uint8)
        shape = encrypted_data['shape']
        # Reverse Entanglement: reverse horizontal XOR chain for each row.
        dec = enc_image.copy()
        H, W = shape[0], shape[1]
        if dec.ndim == 2:
            # Grayscale: iterate each row backwards
            for r in range(H):
                for c in range(W-1, 0, -1):
                    dec[r, c] ^= dec[r, c-1]
        else:
            # Color: iterate each row backwards for each channel
            for r in range(H):
                for c in range(W-1, 0, -1):
                    dec[r, c, 0] ^= dec[r, c-1, 0]
                    dec[r, c, 1] ^= dec[r, c-1, 1]
                    dec[r, c, 2] ^= dec[r, c-1, 2]
        # Reverse Substitution: XOR with same random values (regenerated by key).
        np.random.seed(key)
        if dec.ndim == 2:
            # Grayscale
            regen_vals = np.random.randint(0, 256, size=dec.shape, dtype=np.uint8)
            orig = np.bitwise_xor(dec, regen_vals)
        else:
            # Color
            regenR = np.random.randint(0, 256, size=(H, W), dtype=np.uint8)
            regenG = np.random.randint(0, 256, size=(H, W), dtype=np.uint8)
            regenB = np.random.randint(0, 256, size=(H, W), dtype=np.uint8)
            orig = np.empty_like(dec)
            orig[...,0] = np.bitwise_xor(dec[...,0], regenR)
            orig[...,1] = np.bitwise_xor(dec[...,1], regenG)
            orig[...,2] = np.bitwise_xor(dec[...,2], regenB)
        qir_obj.data = orig.astype(np.uint8)
        qir_obj.shape = shape
        return qir_obj

    else:
        raise TypeError("Unsupported QIR object type for decryption")
