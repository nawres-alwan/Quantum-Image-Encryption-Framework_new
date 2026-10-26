import numpy as np
import hashlib

def encrypt_image_ref2(image: np.ndarray, key: bytes) -> np.ndarray:
    """Encrypt image using 3-D hyperchaotic logistic-sine map (Ref [2])."""
    H, W = image.shape  # grayscale image assumed
    # 1. Split into 4-bit MSB and LSB images
    msb = image >> 4                        # top 4 bits 
    lsb = image & 0xF                       # bottom 4 bits
    # 2. Gray-code encode MSB bits for scrambling
    gray_msb = msb ^ (msb >> 1)
    # Simulate CNOT: flip bit2 if bit3 is 1, and flip bit0 if bit1 is 1
    mask_b2 = ((gray_msb & 0x8) >> 1)       # 0x8 (b3) >>1 => 0x4 mask
    mask_b0 = ((gray_msb & 0x2) >> 1)       # 0x2 (b1) >>1 => 0x1 mask
    gray_msb ^= mask_b2                    # flip b2 where b3 was 1
    gray_msb ^= mask_b0                    # flip b0 where b1 was 1
    # 3. Generate chaotic key stream using a 3D logistic-sine map 
    #    (here simplified to a logistic map for demonstration)
    # Use SHA-256 hash of MSB+LSB as seed for chaos
    combined_bytes = msb.tobytes() + lsb.tobytes()
    hash_bytes = hashlib.sha256(combined_bytes).digest()
    seed = int.from_bytes(hash_bytes[:8], 'little')  # 64-bit integer from hash
    x = (seed % 2**32) / 2**32                      # initial x in [0,1)
    r = 3.99
    chaotic_stream = np.zeros((H, W), dtype=np.uint8)
    for i in range(H*W):
        # 3D map simplified as iterative logistic map for demo
        x = r * x * (1 - x)  
        chaotic_stream.flat[i] = int(x * 256) & 0xFF
    # XOR diffuse both Gray-scrambled MSB and raw LSB with chaotic key
    cipher = ((gray_msb << 4) | lsb) ^ chaotic_stream
    return cipher

def decrypt_image_ref2(cipher: np.ndarray, key: bytes) -> np.ndarray:
    """Decrypt image (inverse of encrypt_image_ref2)."""
    H, W = cipher.shape
    # Regenerate the same chaotic key stream (identical to encryption side)
    # ... (same code as above to produce chaotic_stream from key) ...
    chaotic_stream = ...  # omitted for brevity
    combined = cipher ^ chaotic_stream
    # Inverse Gray + CNOT: (perform same operations as encryption to undo)
    gray_msb = combined >> 4
    lsb = combined & 0xF
    mask_b2 = ((gray_msb & 0x8) >> 1)
    mask_b0 = ((gray_msb & 0x2) >> 1)
    gray_msb ^= mask_b2
    gray_msb ^= mask_b0
    # Gray-code decode: iteratively XOR right-shifted bits
    msb = np.zeros_like(gray_msb)
    temp = gray_msb.copy()
    for _ in range(4):     # 4-bit Gray code
        msb ^= temp
        temp >>= 1
    # Recombine MSB and LSB to obtain plaintext
    plain = (msb << 4) | lsb
    return plain
