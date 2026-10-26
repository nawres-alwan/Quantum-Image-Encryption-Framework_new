QEF-QIRBP v2 -- repaired unitary implementation and complete experiment package
Date: 19 October 2026

This package implements a corrected variant of the three-layer QEF on QIRBP.
It does not relabel the original classical XOR-chain code as a quantum circuit.
It preserves complex amplitudes and relative position phases, uses a reversible
address permutation, and implements position-controlled CNOT/CZ gates exactly.

WHAT WAS CHANGED

1. A QIRBP image is a normalized superposition of positions and RGB bit planes.
   The input is an RGB image with no wavelength-dependent extra measurements.
   Five wavelength-label qubits are included as uniform, replicated labels, so
   r=11 as in the paper's QIRBP qubit allocation. They are not extra image data.

2. Substitution first applies per-position Rx/Ry/Rz mixing gates W_k. It then
   applies independent uniform feature Pauli masks and secret position phases.
   This privacy core is absorbed in the same controlled substitution macro layer.
   It is a substantive correction of the key law, not a claim that the old HCS
   samples already had the required distribution.

3. The position permutation is pi(k)=(a*k+s) mod N, with gcd(a,N)=1. Its offset
   s is exactly uniform on 0,...,N-1 by rejection sampling. The inverse is computed
   exactly. Padded invalid addresses are not part of the permitted input support.

4. Entanglement uses the chronological CNOT/CZ gate schedule. Its aggregated
   Pauli representation includes all commutation phases; decryption uses the
   inverse order. Dense independent tests verify this optimization.

KEYS AND RANDOMNESS

The key record contains three 128-bit fields (384 storage bits). The uniform
privacy core occupies n+2r substitution bits and the uniform permutation offset.
The remaining substitution bits seed SHAKE-derived per-pixel mixing angles;
the permutation coefficient and entangling choices have separate fields.
Conditioning the offset word on its rejection rule changes the accepted key
domain slightly for N not dividing 2^64; the analysis gives the exact domain.
Neither 384 storage bits nor a generic Grover formula is reported as a proven
192-bit attack-cost lower bound.

Python secrets supplies OS cryptographic randomness in this reference package.
No IBM quantum device, physical QRNG, nD-HCS implementation, or HG-Phi-QRNG is
claimed to have generated these test keys. SHAKE mixing is not called hyperchaos.
The ideal secrecy theorem assumes independent uniform reserved bits and offset.
External HCS mixing may replace W_k without changing the proof if it remains
independent of the reserved privacy bits. A physical QRNG requires a justified
extraction/entropy model before it can inherit the same quantitative guarantee.

EXACT RESISTANCE FORMULA

For every joint valid-input/reference state rho_PR, the ideal averaged channel is

  (E tensor I_R)(rho_PR) = [I_valid/(N*2^r)] tensor rho_R.

Consequently epsilon = 1/2 ||E - replacement_channel||_diamond = 0, optimal
equal-prior image guessing is 1/2, and fresh independent keys on every oracle
call give zero distinguishing advantage for any number of valid-input queries.
For a real conditional key distribution at total-variation distance Delta from
the specified ideal core distribution, epsilon <= Delta and a q-use game has
guessing excess at most min(1/2, q*Delta). Delta is not inferred from image metrics.

This is the guarantee of the repaired variant and its stated key law. It is not
a retroactive proof of the old code or the paper's unverified generator law.

SECURITY TARGET AND OUTPUTS

Checking every recovered RGB byte uses simulator access to the recovered
amplitudes. It does not claim complete classical-image readout from one physical
quantum state. Physical readout needs repeated preparations and measurements,
with a fresh key on every ciphertext preparation.

The public ciphertext in the theorem is a quantum state, not a classical file
containing its full amplitudes. Each physical ciphertext preparation needs a
fresh key. Reusing one key to supply multiple copies for tomography changes the
model and is outside this theorem. This construction provides confidentiality;
it is not a quantum authentication/error-correction code.

The diagnostic_preview PNG files are simulator inspection images assembled from
conditional bit-plane measurements. They are not reversible classical ciphertext
files and have no claimed secrecy guarantee. Their entropy/NPCR/SSIM statistics
are clearly separated from the quantum privacy result. Decrypted PNGs are the
exact recovered original RGB pixels obtained by unitary simulation.

The native profile preserves all four source resolutions. The paper256 profile
resizes the same source images with explicitly recorded Pillow LANCZOS sampling.
Each profile/image is tested with three fresh keys (24 complete image runs).
No original experiment or source-image file is modified.

The representation is exact for this RGB QIRBP subfamily: a 64-component
colour/bit-plane core and five two-component wavelength factors per position.
Batching changes storage only and does not perform a quantum measurement.
It avoids allocating the entire 2^Q state vector for large images.

REPRODUCTION

Dependencies: Python 3, NumPy, Pillow. No Qiskit, SciPy or scikit-image is needed.

  python security_tests.py
  python run_experiments.py --replicates 3 --chunk-size 2048

The second command is for an unused output directory. Existing private experiment
keys are never silently overwritten. To reproduce the delivered private run:

  python run_experiments.py --replicates 3 --chunk-size 2048 --replay-recorded-keys
  python analyze_results.py

The replay option is only for offline numerical reproduction. Real transmissions
must generate fresh keys. REPRODUCE.ps1 defaults to fresh experiment keys. Its optional
-ReplayRecordedKeys switch is for private offline replay only.
Its optional -PythonExecutable argument selects a specific installed interpreter.

The original offline package has private_keys/ for exact private replay;
that folder is excluded from this public repository. Exclude this folder from public supplements. Optional simulator
state checkpoints also remain private and are disabled by default.

RESULT FILES

results/all_runs.csv              Complete per-run data, including timings.
results/summary_by_image.csv      All eight profile/image summaries.
results/privacy_parameters.csv    Exact mathematical resource/security table.
results/exhaustive_channel_table.csv  Complete small-channel key enumeration.
results/ablation_table.csv        Which privacy guarantees fail if masks are removed.
results/dense_equivalence_table.csv   Production versus independent dense operators.
results/security_verification.json    Full diagnostic evidence.
results/run_verification.json         Full-image checks and source integrity.
input_manifest.json              Repository-relative source paths, hashes and native dimensions.
analysis/                        Mathematical proof, tables and final PDF.

All timings are local classical-simulator timings. No hardware fidelity,
device-independent randomness, 192-bit lower bound, or PNG confidentiality is
inferred from these experiments.
