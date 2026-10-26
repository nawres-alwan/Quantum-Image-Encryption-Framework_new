import numpy as np

# Chaotic map functions: logistic map and sine map
def logistic_map(x, r=3.99):
    return r * x * (1 - x)  # logistic map in [0,1]

def sine_map(x):
    return np.sin(np.pi * x)  # sine map in [0,1]

# Generate chaotic sequences using 2D Logistic-Sine-Coupling Map (LSCM)
def generate_lscm_sequences(length, x0, y0, mu=3.99, alpha=0.8, skip=100):
    """Generate two chaotic sequences (X_seq, Y_seq) of given length using 2D-LSCM."""
    # Skip initial transient iterations for better randomness
    x, y = x0, y0
    for _ in range(skip):
        x_next = sine_map(x + y) % 1.0
        y_next = (mu * y * (1 - y) + alpha * x) % 1.0
        x, y = x_next, y_next
    # Produce sequences
    X_seq = np.zeros(length); Y_seq = np.zeros(length)
    X_seq[0], Y_seq[0] = x, y
    for i in range(1, length):
        X_seq[i] = sine_map(X_seq[i-1] + Y_seq[i-1]) % 1.0
        Y_seq[i] = (mu * Y_seq[i-1] * (1 - Y_seq[i-1]) + alpha * X_seq[i-1]) % 1.0
    return X_seq, Y_seq

# Encrypt an image (as a NumPy array of uint8 pixels) with chaotic keys
def encrypt_image(image_array, x0, y0, z0):
    """Return the cipher image array and a tuple of (perm_index, key_add, key_xor) for decryption."""
    h, w, ch = image_array.shape
    N = h * w
    # 1. Generate chaotic sequences
    X_seq, Y_seq = generate_lscm_sequences(N, x0, y0)          # from 2D-LSCM
    Z_seq = np.zeros(N)
    # Use a sine map (part of FCSM) for an extra sequence
    z = z0
    for _ in range(100):  # skip transient
        z = sine_map(z)
    for i in range(N):
        z = sine_map(z)
        Z_seq[i] = z
    # 2. Permutation: sort X_seq to get permutation indices
    perm_index = np.argsort(X_seq)            # chaotic sort order
    flat = image_array.reshape(N, ch)         # flatten pixels
    permuted = flat[perm_index]              # apply permutation
    # 3. Diffusion round 1: modular addition with Y_seq
    key_add = (Y_seq * 256).astype(np.uint8)  # scale [0,1) to [0,255]
    key_add = np.repeat(key_add.reshape(N,1), ch, axis=1)  # expand to (N,ch)
    diffused = (permuted + key_add) % 256
    # 4. Diffusion round 2: XOR with Z_seq
    key_xor = (Z_seq * 256).astype(np.uint8)
    key_xor = np.repeat(key_xor.reshape(N,1), ch, axis=1)
    cipher_flat = diffused ^ key_xor
    cipher_image = cipher_flat.reshape(h, w, ch).astype(np.uint8)
    return cipher_image, (perm_index, key_add[:,0], key_xor[:,0])

# Decrypt an image using the recorded keys (inverse of encryption)
def decrypt_image(cipher_array, perm_index, key_add, key_xor):
    h, w, ch = cipher_array.shape
    N = h * w
    cipher_flat = cipher_array.reshape(N, ch)
    # Inverse of diffusion 2: XOR
    diffused = cipher_flat ^ np.repeat(key_xor.reshape(N,1), ch, axis=1)
    # Inverse of diffusion 1: subtract mod 256
    permuted = (diffused - np.repeat(key_add.reshape(N,1), ch, axis=1)) % 256
    # Inverse of permutation: apply inverse index mapping
    original_flat = np.empty_like(permuted)
    original_flat[perm_index] = permuted  # place each element back to original index
    original_image = original_flat.reshape(h, w, ch).astype(np.uint8)
    return original_image

# Example usage on an image array `img_arr`:
# cipher_arr, keys = encrypt_image(img_arr, x0=0.1234, y0=0.5678, z0=0.9012)
# recovered_arr = decrypt_image(cipher_arr, *keys)
