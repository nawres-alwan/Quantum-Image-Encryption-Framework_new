QEF coherent verification oracle
===============================

This experiment builds a key-controlled QEF circuit, a coherent known-pair
verification oracle, a clean-ancilla phase reflection, and Grover diffusion.
It reports concrete logical gate counts in the basis {CX, U} and draws the
circuits with Qiskit's Matplotlib drawer.

Run from this directory:

  python -X utf8 test_qef_gsa_oracle.py
  python -X utf8 qef_gsa_oracle.py

The default output directory is results/. Dependencies are in requirements.txt.
Install pylatexenc from requirements.txt before drawing the circuits.

EXPERIMENT PARAMETERS

* QIRBP: two pixel addresses, n=1.
* Four original input photographs: a crop containing the first two RGB pixels
  of the first row is used. The full photographs are not resized or simulated.
* Six feature qubits with wavelength_qubits=0; eleven feature qubits with
  wavelength_qubits=5. The default run evaluates both configurations.
* Candidate register: 2 substitution bits, 1 permutation bit, 1 entanglement
  bit, giving 16 public test candidates. Target fixtures differ across images.
* Angle grid: angle_bits=12; step=2*pi/4096 radians. Each angle is rounded
  down on this grid. Floating-point circuit simulation is separately checked.
* One plaintext/ciphertext pair per target key. Different images are separate
  experiments and do not pool data from different refreshed keys.
* Ciphertext access: its complete classical state description is available
  in this benchmark, giving coherent state preparation and inverse preparation.
  Capturing one unknown quantum ciphertext does not grant this access.
* Gate model: arbitrary one-qubit U plus CX, with optimization_level=0.
  These are logical counts, not physical or fault-tolerant gate estimates.

RELATION TO THE EXISTING CODE AND PAPER

The main encryption modules, saved keys, original images, and paper are not
modified. No saved private key is read.

The existing qef_qirbp.py uses a repaired v2 privacy core and SHAKE-derived
parameter schedules. The experiment imports that module for QIRBP conventions
and public test parameter records; it does not identify SHAKE with HCS/HG-Phi.
The original generators and a full reversible 384-bit key schedule are not
implemented by this finite-bank experiment.

Two circuit families are explicit:

  --model paper-three-layer
    Default: position-controlled Rx/Ry/Rz substitution, address permutation,
    position-controlled CNOT/CZ entanglement. It omits the additional v2 privacy
    core. The local parameters are public finite fixtures, not measured HCS
    or HG-Phi outputs. This instantiates the paper's operator architecture.

  --model existing-qirbp-v2
    Also includes the existing code's feature Paulis, position phases and
    conditional global phases. The tests compare unquantized fixed-key
    circuits with the independent encrypt_chunk implementation, including
    phase. Quantization error is checked against a derived operator bound.

The public fixture records have 128-bit storage fields, but selection from
four/two/two records gives only FOUR candidate-key bits. Neither 128-bit nor
384-bit resistance is inferred from these tests. Parameter lookup is expanded
coherently into gates and is included in the counted programmable circuit.
A full-size reversible generator implementation would have its own cost.

VERIFIER CONSTRUCTION

For a candidate k, plaintext preparation P and ciphertext preparation C,

  A_k = C^dagger U_k P
  f_k = |<0|A_k|0>|^2.

All candidates in the finite public bank are enumerated to identify accepted
keys and evaluate the numerical gap Delta=min_rejected(1-f_k). Accepted keys
are also compared with the target encryption operator over its full valid
Hilbert space using tensor-block Hilbert-Schmidt overlaps. An experiment fails
if a known-pair false positive is not an equivalent encryption operator.

Repeat each test s times and let V perform those tests coherently. The oracle is

  O_tilde = V^dagger (I - 2|0><0|_tests) V.

The repeated-test oracle is approximate. With total query budget q_budget and
required accumulated state error tau=1e-6, choose

  s = ceil[(2/Delta) ln(2*q_budget/tau)].

Then the accumulated oracle error is at most

  2*q_budget * (max_rejected f_k)^(s/2) <= tau.

The experiments compute both this bound and the actual Grover probabilities.

EXACT RESOURCE ACCOUNTING FOR THIS CIRCUIT

The test block and its inverse are independently transpiled to CX+U.
For w=s*Q test qubits the zero-state reflection uses w-2 clean ancillas and

  2(w-2) Toffoli + 1 CZ + 2w X

for w>=2. Each Toffoli compiles to 6 CX + 9 U; CZ to 1 CX + 2 U.
Thus the reflection has

  CX = 12(w-2)+1,   U = 20w-34.

The full oracle count is the sum of s forward blocks, s inverse blocks and
this reflection. No inter-block cancellation is assumed. The tests verify
these composition counts by fully expanding s=1,2,3 circuits.

The attack count includes initial H gates on the candidate register and
q Grover iterations, each containing one oracle and one diffusion circuit.
It excludes final measurements, classical acquisition of state descriptions,
device routing, error correction and synthesis to Clifford+T.

SIMULATION WITHOUT EXPONENTIALLY MANY WORK AMPLITUDES

Let v_k=A_k^dagger|0>. For s repetitions the oracle block is a reflection
about v_k^(tensor s). Starting with zero work, Grover's evolution stays in

  span{|0>, v_0^(tensor s), ..., v_(M-1)^(tensor s)}.

Its dimension is at most M+1. The code computes its Gram matrix using
<v_k^(tensor s)|v_l^(tensor s)>=<v_k|v_l>^s, factors it, and evolves the
candidate register and this invariant subspace. This includes residual
verification work and is not an ideal Boolean-oracle substitution.

An independent Aer statevector simulation of the physical single-block
oracle and diffusion is compared with this method. This check includes all
clean phase ancillas and tests the whole candidate distribution.

OUTPUTS

  resource_summary.csv       Concrete resources for all image crops/configurations.
  SUMMARY_AR.txt             Arabic explanation and the complete results table.
  experiment_report.json    Full results, precision, access model and scope.
  test_report.json          Independent circuit and count validation.
  component_gate_counts.csv Compiled component sizes and depths.
  oracle_full.qpy           Actual sufficient-repetition oracle for the first case.
  oracle_single_block.qpy   Physical one-block example used in the detailed drawing.
  programmable_qef.qpy      Full key-controlled three-layer circuit.
  oracle_full_mpl.png/svg   Bundled-register overview of the full counted oracle.
  oracle_single_block_mpl.* Detailed actual circuit with s=1, clearly labelled.
  verification_test_mpl.*  State preparation and the three controlled QEF layers.
  programmable_qef_mpl.*    Actual programmable three-layer circuit.
  entanglement_selector_mpl.*  Actual b-controlled CNOT/CZ selector.
  grover_success.png/svg    Constructed-oracle and ideal reference probabilities.
  IM*_w*/                  Candidate fidelities, resource record and Grover trace.

These results establish operation and cost of the specified finite experiment.
They do not give a numerical full-384-bit oracle cost or a security lower bound
for the paper's entire key family.
