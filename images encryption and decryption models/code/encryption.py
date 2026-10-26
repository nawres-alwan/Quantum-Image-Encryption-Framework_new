import numpy as np
import math
from FRQI import FRQI
from NEQR import NEQR
from AFQIRHSI import AFQIRHSI
from qirbp_model import QIRBP
from qirwc_model import QIRWC

def encrypt(qir_obj, key=0):
    """
    Encrypt the given quantum image representation using three layers: Substitution, Permutation, Entanglement.
    - Substitution: alters pixel data (angles or bits) using random rotations or XOR (based on key).
    - Permutation: shuffles pixel positions.
    - Entanglement: introduces inter-pixel dependencies (simulated via XOR or addition chaining):contentReference[oaicite:23]{index=23}.
    Returns a dictionary of encrypted data needed for decryption.
    """
    np.random.seed(key)  # seed the RNG for reproducibility
    # Encryption logic branches by representation type:
    if isinstance(qir_obj, FRQI):
        # FRQI encryption:
        # Substitution: add a random rotation to each pixel's angle θ.
        theta = qir_obj.theta
        rand_angles = np.random.rand(*theta.shape) * 2 * math.pi  # random angles [0, 2π)
        theta_sub = (theta + rand_angles) % (2 * math.pi)
        # Permutation: shuffle pixel positions (we flatten and permute the angle array).
        theta_flat = theta_sub.flatten()
        perm_indices = np.random.permutation(theta_flat.size)
        theta_perm = theta_flat[perm_indices]
        # Entanglement: chain addition of angles to entangle neighboring pixels.
        theta_chain = theta_perm.copy()
        for i in range(1, theta_chain.size):
            # Add previous angle to current (mod 2π) to diffuse information
            theta_chain[i] = (theta_chain[i] + theta_chain[i-1]) % (2 * math.pi)
        # Return encrypted angle list and permutation (for decryption).
        return {"theta": theta_chain, "perm": perm_indices, "shape": qir_obj.shape}

    elif isinstance(qir_obj, NEQR):
        # NEQR encryption:
        img = qir_obj.data  # 2D uint8 array
        # Substitution: XOR each pixel with a random 8-bit value.
        sub_key = np.random.randint(0, 256, size=img.shape, dtype=np.uint8)
        img_sub = np.bitwise_xor(img, sub_key)
        # Permutation: shuffle pixel positions.
        flat = img_sub.flatten()
        perm_indices = np.random.permutation(flat.size)
        perm_flat = flat[perm_indices]
        # Entanglement: XOR-chain diffusion - each pixel XORed with previous one.
        chain_flat = perm_flat.copy().astype(np.uint8)
        for i in range(1, chain_flat.size):
            chain_flat[i] ^= chain_flat[i-1]
        # Return encrypted flat pixel array, permutation, and substitution key.
        return {"flat": chain_flat, "perm": perm_indices, "shape": qir_obj.shape}

    elif isinstance(qir_obj, AFQIRHSI):
        # AFQIRHSI encryption:
        H, S, I = qir_obj.H, qir_obj.S, qir_obj.I  # each are 2D uint8 arrays
        h, w = H.shape
        # Substitution: XOR each H, S, I value with random 8-bit values.
        keyH = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
        keyS = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
        keyI = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
        H_sub = np.bitwise_xor(H, keyH)
        S_sub = np.bitwise_xor(S, keyS)
        I_sub = np.bitwise_xor(I, keyI)
        # Permutation: (AFQIRHSI might skip global permutation to preserve adjacency info).
        # We'll skip shuffling to maintain spatial relationships (adjacency-intensity entanglement will follow).
        perm_indices = np.arange(h * w)  # identity permutation
        H_perm = H_sub.flatten()[perm_indices]
        S_perm = S_sub.flatten()[perm_indices]
        I_perm = I_sub.flatten()[perm_indices]
        # Entanglement: entangle adjacent intensities (and also H, S for completeness) by XOR-chaining.
        H_chain = H_perm.copy().astype(np.uint8)
        S_chain = S_perm.copy().astype(np.uint8)
        I_chain = I_perm.copy().astype(np.uint8)
        for i in range(1, I_chain.size):
            # XOR current value with previous value to diffuse information along the sequence
            H_chain[i] ^= H_chain[i-1]
            S_chain[i] ^= S_chain[i-1]
            I_chain[i] ^= I_chain[i-1]
        # Return encrypted flattened H, S, I and keys.
        return {"H": H_chain, "S": S_chain, "I": I_chain, "perm": perm_indices, "shape": qir_obj.shape}

    elif isinstance(qir_obj, QIRBP):
        # QIRBP encryption:
        img = qir_obj.data  # could be grayscale or RGB
        H, W = img.shape[0], img.shape[1]
        out = np.copy(img)
        # Substitution: Flip random bit-planes across the image:contentReference[oaicite:24]{index=24}.
        if img.ndim == 2:
            # Grayscale image: decide which of the 8 bit-planes to flip
            mask = np.random.randint(0, 2, size=8)  # 8 bits
            for bit in range(8):
                if mask[bit] == 1:
                    out = np.bitwise_xor(out, np.uint8(1 << bit))
        else:
            # Color image: decide separately for R, G, B channels which planes to flip
            maskR = np.random.randint(0, 2, size=8)
            maskG = np.random.randint(0, 2, size=8)
            maskB = np.random.randint(0, 2, size=8)
            # Flip bits in each channel accordingly
            R, G, B = out[...,0], out[...,1], out[...,2]
            for bit in range(8):
                if maskR[bit]: R[:] = np.bitwise_xor(R, np.uint8(1 << bit))
                if maskG[bit]: G[:] = np.bitwise_xor(G, np.uint8(1 << bit))
                if maskB[bit]: B[:] = np.bitwise_xor(B, np.uint8(1 << bit))
            out[...,0], out[...,1], out[...,2] = R, G, B
            mask = (maskR, maskG, maskB)
        # Permutation: shuffle all pixels (maintaining their multi-bit or multi-channel values together).
        flat = out.reshape(-1, out.shape[-1] if out.ndim == 3 else 1)
        perm_indices = np.random.permutation(flat.shape[0])
        perm_flat = flat[perm_indices]
        # Entanglement: XOR-chain each channel across the flattened sequence to diffuse adjacent pixel info.
        chain_flat = perm_flat.copy().astype(np.uint8)
        if out.ndim == 2:
            # single channel
            for i in range(1, chain_flat.shape[0]):
                chain_flat[i, 0] ^= chain_flat[i-1, 0]
        else:
            # three channels
            for i in range(1, chain_flat.shape[0]):
                chain_flat[i, 0] ^= chain_flat[i-1, 0]
                chain_flat[i, 1] ^= chain_flat[i-1, 1]
                chain_flat[i, 2] ^= chain_flat[i-1, 2]
        # Return encrypted flattened array, permutation, and bit-plane flip mask.
        return {"flat": chain_flat, "perm": perm_indices, "shape": qir_obj.shape, "mask": mask}

    elif isinstance(qir_obj, QIRWC):
        # QIRWC encryption:
        img = qir_obj.data  # grayscale or RGB
        H, W = img.shape[0], img.shape[1]
        enc = np.copy(img)
        # Substitution: XOR each pixel (and each channel) with random values.
        if img.ndim == 2:
            # Grayscale
            rand_vals = np.random.randint(0, 256, size=enc.shape, dtype=np.uint8)
            enc = np.bitwise_xor(enc, rand_vals)
        else:
            # Color: separate random mask for R, G, B
            randR = np.random.randint(0, 256, size=(H, W), dtype=np.uint8)
            randG = np.random.randint(0, 256, size=(H, W), dtype=np.uint8)
            randB = np.random.randint(0, 256, size=(H, W), dtype=np.uint8)
            enc[...,0] = np.bitwise_xor(enc[...,0], randR)
            enc[...,1] = np.bitwise_xor(enc[...,1], randG)
            enc[...,2] = np.bitwise_xor(enc[...,2], randB)
        # Permutation: (QIRWC focuses on adjacency, so we avoid global permutation to keep neighbors intact).
        enc_perm = enc  # no shuffle to preserve spatial adjacency relationships
        # Entanglement: Entangle neighboring pixel colors by XORing each pixel with its left neighbor (horizontal chain per row).
        enc_chain = enc_perm.copy().astype(np.uint8)
        if enc_chain.ndim == 2:
            # Grayscale: iterate each row
            for r in range(H):
                for c in range(1, W):
                    enc_chain[r, c] ^= enc_chain[r, c-1]
        else:
            # Color: iterate each row, entangle each channel
            for r in range(H):
                for c in range(1, W):
                    enc_chain[r, c, 0] ^= enc_chain[r, c-1, 0]
                    enc_chain[r, c, 1] ^= enc_chain[r, c-1, 1]
                    enc_chain[r, c, 2] ^= enc_chain[r, c-1, 2]
        # Return the encrypted image data and (for completeness) the random sub-key used.
        # We include the random values for potential use in decryption, though we will regenerate them via the key.
        if img.ndim == 2:
            return {"data": enc_chain, "shape": qir_obj.shape}
        else:
            return {"data": enc_chain, "shape": qir_obj.shape}
    else:
        raise TypeError("Unsupported QIR object type for encryption")
