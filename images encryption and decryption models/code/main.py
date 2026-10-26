import cv2
import numpy as np
import matplotlib.pyplot as plt
from FRQI import FRQI
from NEQR import NEQR
from AFQIRHSI import AFQIRHSI
from qirbp_model import QIRBP
from qirwc_model import QIRWC
from encryption import encrypt
from decryption import decrypt
import math

def hsi_to_rgb(H, S, I):
    """
    Convert H, S, I components (each as 2D arrays, H in [0,255]→0-360°, 
    S in [0,255]→0-1, I in [0,255]) back to an RGB image (uint8).
    Uses standard HSI↔RGB conversion formulas:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.
    """
    # Convert HSI from uint8 ranges to floats
    H = H.astype(float) * (360.0 / 255.0)
    S = S.astype(float) / 255.0
    I_comp = I.astype(float)
    # Initialize output channels
    R = np.zeros_like(I_comp)
    G = np.zeros_like(I_comp)
    B = np.zeros_like(I_comp)
    # Grayscale pixels (S=0): R=G=B=I
    gray_mask = S == 0
    R[gray_mask] = I_comp[gray_mask]
    G[gray_mask] = I_comp[gray_mask]
    B[gray_mask] = I_comp[gray_mask]
    # Convert each colored pixel (vectorized by cases)
    H_deg = H  # hue in degrees
    # Case 0° ≤ H < 120°
    mask1 = (H_deg < 120) & (~gray_mask)
    cosH = np.cos(np.deg2rad(H_deg[mask1]))
    cos60_H = np.cos(np.deg2rad(60 - H_deg[mask1]))
    B_val = I_comp[mask1] * (1 - S[mask1])
    R_val = I_comp[mask1] + I_comp[mask1] * S[mask1] * cosH / cos60_H
    G_val = 3 * I_comp[mask1] - (R_val + B_val)
    R[mask1], G[mask1], B[mask1] = R_val, G_val, B_val
    # Case H = 120° (very rare exact match)
    mask2 = (np.isclose(H_deg, 120)) & (~gray_mask)
    R[mask2] = I_comp[mask2] - I_comp[mask2] * S[mask2]
    G[mask2] = I_comp[mask2] + 2 * I_comp[mask2] * S[mask2]
    B[mask2] = I_comp[mask2] - I_comp[mask2] * S[mask2]
    # Case 120° < H < 240°
    mask3 = (H_deg > 120) & (H_deg < 240) & (~gray_mask)
    H2 = H_deg[mask3] - 120
    cosH2 = np.cos(np.deg2rad(H2))
    cos60_H2 = np.cos(np.deg2rad(60 - H2))
    R_val = I_comp[mask3] * (1 - S[mask3])
    G_val = I_comp[mask3] + I_comp[mask3] * S[mask3] * cosH2 / cos60_H2
    B_val = 3 * I_comp[mask3] - (R_val + G_val)
    R[mask3], G[mask3], B[mask3] = R_val, G_val, B_val
    # Case H = 240°
    mask4 = np.isclose(H_deg, 240) & (~gray_mask)
    R[mask4] = I_comp[mask4] - I_comp[mask4] * S[mask4]
    G[mask4] = I_comp[mask4] - I_comp[mask4] * S[mask4]
    B[mask4] = I_comp[mask4] + 2 * I_comp[mask4] * S[mask4]
    # Case 240° < H < 360°
    mask5 = (H_deg > 240) & (~gray_mask)
    H3 = H_deg[mask5] - 240
    cosH3 = np.cos(np.deg2rad(H3))
    cos60_H3 = np.cos(np.deg2rad(60 - H3))
    G_val = I_comp[mask5] * (1 - S[mask5])
    B_val = I_comp[mask5] + I_comp[mask5] * S[mask5] * cosH3 / cos60_H3
    R_val = 3 * I_comp[mask5] - (G_val + B_val)
    R[mask5], G[mask5], B[mask5] = R_val, G_val, B_val
    # Clip and convert to uint8
    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)
    return np.stack([R, G, B], axis=2)

def main():
    # Load input image
    img_bgr = cv2.imread('Ashurbanipal.jpg')
    if img_bgr is None:
        raise FileNotFoundError("Input image 'Ashurbanipal.jpg' not found")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Prepare QIR objects for each model
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    qir_frqi = FRQI(img_gray)
    qir_neqr = NEQR(img_gray)  # using grayscale for NEQR
    qir_afqirhsi = AFQIRHSI(img_rgb)
    qir_qirbp = QIRBP(img_rgb)
    qir_qirwc = QIRWC(img_rgb)
    key = 0  # encryption key/seed
    # Encrypt and decrypt with each model
    results = []
    for name, qir in [("FRQI", qir_frqi), ("NEQR", qir_neqr), ("AFQIRHSI", qir_afqirhsi), ("QIRBP", qir_qirbp), ("QIRWC", qir_qirwc)]:
        enc_data = encrypt(qir, key=key)
        dec_obj = decrypt(type(qir)(), enc_data, key=key)
        results.append((name, qir, enc_data, dec_obj))
    # Visualization of original, encrypted, decrypted images
    plt.figure(figsize=(15, 10))
    idx = 1
    for (name, orig_qir, enc_data, dec_qir) in results:
        if name == "FRQI":
            # Original grayscale image
            orig_img = img_gray
            # Encrypted: interpret theta chain as grayscale intensities
            theta_chain = enc_data['theta'].reshape(orig_qir.shape)
            enc_img = np.uint8((theta_chain / (2 * math.pi)) * 255)
            # Decrypted: reconstruct grayscale from decrypted theta
            theta_dec = dec_qir.theta
            dec_img = np.uint8((theta_dec / (math.pi/2)) * 255)
            dec_img = np.clip(dec_img, 0, 255)
            cmap = 'gray'
        elif name == "NEQR":
            orig_img = img_gray
            flat_chain = enc_data['flat']
            enc_img = flat_chain.reshape(enc_data['shape']).astype(np.uint8)  # scrambled grayscale
            dec_img = dec_qir.data  # decrypted grayscale
            cmap = 'gray'
        elif name == "AFQIRHSI":
            orig_img = img_rgb
            H_enc = enc_data['H'].reshape(enc_data['shape']).astype(np.uint8)
            S_enc = enc_data['S'].reshape(enc_data['shape']).astype(np.uint8)
            I_enc = enc_data['I'].reshape(enc_data['shape']).astype(np.uint8)
            enc_img = hsi_to_rgb(H_enc, S_enc, I_enc)  # scrambled color
            dec_img = hsi_to_rgb(dec_qir.H, dec_qir.S, dec_qir.I)  # reconstructed color
            cmap = None
        elif name == "QIRBP":
            orig_img = img_rgb
            flat_chain = enc_data['flat']
            enc_img = flat_chain.reshape(enc_data['shape']).astype(np.uint8)  # permuted & bit-flipped
            dec_img = dec_qir.data  # decrypted original image
            cmap = None
        elif name == "QIRWC":
            orig_img = img_rgb
            enc_img = enc_data['data']  # encrypted image matrix
            dec_img = dec_qir.data      # decrypted original image
            cmap = None
        # Plot images
        plt.subplot(5, 3, idx);   plt.imshow(orig_img, cmap=cmap); plt.title(f"{name} Original");   plt.axis('off')
        plt.subplot(5, 3, idx+1); plt.imshow(enc_img, cmap=cmap);  plt.title(f"{name} Encrypted");  plt.axis('off')
        plt.subplot(5, 3, idx+2); plt.imshow(dec_img, cmap=cmap);  plt.title(f"{name} Decrypted");  plt.axis('off')
        idx += 3
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
