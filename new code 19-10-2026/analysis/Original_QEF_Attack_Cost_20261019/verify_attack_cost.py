"""Check attack-cost algebra against the ORIGINAL three-layer operator family.

No generator is replaced, no new encryption layer is introduced, and no test
is represented as executing an exponential attack on the paper's full keys.
Small systems verify algebra; the full-size estimates remain conditional.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np

OUT = Path(__file__).resolve().parent
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)


def kron_all(ops):
    ans = np.array([[1]], dtype=complex)
    for op in ops:
        ans = np.kron(ans, op)
    return ans


def rot(axis, angle):
    return math.cos(angle / 2) * I2 - 1j * math.sin(angle / 2) * axis


def local_e(t, choices):
    n, r = choices.shape
    ops = [I2.copy() for _ in range(r)]
    for p in range(n):
        if (t >> (n - 1 - p)) & 1:
            for j in range(r):
                ops[j] = (Z if choices[p, j] else X) @ ops[j]
    return ops


def circuit_matrix(local, perm, choices):
    """Dense gate multiplication, independent of the derived block formula."""
    N, r = local.shape[:2]
    f = 1 << r
    D = N * f
    R = np.zeros((D, D), complex)
    P = np.zeros_like(R)
    for k in range(N):
        R[k*f:(k+1)*f, k*f:(k+1)*f] = kron_all(local[k])
        P[perm[k]*f:(perm[k]+1)*f, k*f:(k+1)*f] = np.eye(f)
    E = np.eye(D, dtype=complex)
    for p in range(choices.shape[0]):
        for j in range(r):
            gate = np.zeros_like(E)
            for index in range(D):
                t, feature = divmod(index, f)
                active = (t >> (choices.shape[0] - 1 - p)) & 1
                if not active:
                    gate[index, index] = 1
                elif choices[p, j]:
                    gate[index, index] = -1 if (feature >> (r-1-j)) & 1 else 1
                else:
                    gate[t*f + (feature ^ (1 << (r-1-j))), index] = 1
            E = gate @ E
    return E @ P @ R


def block_matrix(local, perm, choices):
    N, r = local.shape[:2]
    f = 1 << r
    out = np.zeros((N*f, N*f), complex)
    for k, t in enumerate(perm):
        ops = [v @ a for v, a in zip(local_e(t, choices), local[k])]
        out[t*f:(t+1)*f, k*f:(k+1)*f] = kron_all(ops)
    return out


def dump_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng = np.random.default_rng(20261019)  # public test fixture, never an encryption key
    grover = []
    max_grover_error = 0.0
    for L in range(2, 11):
        M = 1 << L
        for t in [1, 2, min(5, M-1)]:
            state = np.full(M, 1/math.sqrt(M), complex)
            theta = math.asin(math.sqrt(t/M))
            for q in range(13):
                actual = float(np.sum(np.abs(state[:t])**2))
                expected = math.sin((2*q+1)*theta)**2
                error = abs(actual-expected)
                max_grover_error = max(max_grover_error, error)
                assert error < 2e-13
                assert actual <= min(1, (2*q+1)**2*t/M) + 2e-13
                grover.append(dict(L=L, M=M, accepted_keys=t, oracle_queries=q,
                                   probability=actual, analytic_probability=expected,
                                   absolute_error=error))
                state[:t] *= -1
                state = 2*state.mean() - state
    dump_csv('grover_small_system_checks.csv', grover)

    algebra = []
    for N, r in [(2, 1), (3, 2), (4, 3), (5, 1), (8, 2)]:
        n = (N-1).bit_length()
        for repetition in range(8):
            angles = rng.uniform(-4*math.pi, 4*math.pi, (N, r, 3))
            local = np.empty((N, r, 2, 2), complex)
            for k in range(N):
                for j in range(r):
                    a, b, c = angles[k, j]
                    local[k, j] = rot(Z, c) @ rot(Y, b) @ rot(X, a)
            perm = rng.permutation(N)
            e = rng.integers(0, 2, (n, r))
            other_e = e.copy()
            other_e[0, 0] ^= 1
            U = circuit_matrix(local, perm, e)
            direct = block_matrix(local, perm, e)
            equivalent_local = local.copy()
            for k, t in enumerate(perm):
                for j, (v, w) in enumerate(zip(local_e(t, e), local_e(t, other_e))):
                    equivalent_local[k, j] = w.conj().T @ v @ local[k, j]
            same_U = block_matrix(equivalent_local, perm, other_e)
            same_error = float(np.max(np.abs(U-same_U)))
            determinant_error = float(np.max(np.abs(np.linalg.det(equivalent_local)-1)))
            # On an all-zero feature state, a common right Z rotation changes
            # the operator but adds only a common phase to this plaintext.
            modified_local = local.copy()
            delta = 0.731
            for k in range(N):
                modified_local[k, 0] = local[k, 0] @ rot(Z, delta)
            modified_U = block_matrix(modified_local, perm, e)
            plain = np.zeros(N*(1 << r), complex)
            plain[::(1 << r)] = 1/math.sqrt(N)
            overlap = abs(np.vdot(U @ plain, modified_U @ plain))**2
            row = dict(N=N, r=r, repetition=repetition,
                       block_formula_error=float(np.max(np.abs(U-direct))),
                       unitarity_error=float(np.max(np.abs(U.conj().T@U-np.eye(len(U))))),
                       alternative_gate_choices_same_operator_error=same_error,
                       replacement_local_SU2_det_error=determinant_error,
                       different_operators_distance=float(np.linalg.norm(U-modified_U, 2)),
                       same_plaintext_cipher_fidelity=float(overlap))
            assert row['block_formula_error'] < 1e-12
            assert row['unitarity_error'] < 1e-12
            assert same_error < 1e-12 and determinant_error < 1e-12
            assert row['different_operators_distance'] > 0.1
            assert abs(overlap-1) < 1e-12
            algebra.append(row)
    dump_csv('original_operator_checks.csv', algebra)

    # A normalized QIRBP image with zero RGB data and three plane-index qubits.
    N, r = 4, 6
    n = 2
    local = np.empty((N, r, 2, 2), complex)
    for k in range(N):
        for j in range(r):
            a, b, c = rng.uniform(-math.pi, math.pi, 3)
            local[k, j] = rot(Z, c) @ rot(Y, b) @ rot(X, a)
    perm = rng.permutation(N)
    e = rng.integers(0, 2, (n, r))
    modified = local.copy()
    for k in range(N):
        modified[k, 0] = local[k, 0] @ rot(Z, 0.731)
    U = block_matrix(local, perm, e)
    U2 = block_matrix(modified, perm, e)
    feature = np.zeros(64, complex)
    feature[:8] = 1/math.sqrt(8)
    image = np.kron(np.full(N, 1/math.sqrt(N)), feature)
    qirbp = dict(N=N, r=r, norm=float(np.vdot(image, image).real),
                 same_black_QIRBP_cipher_fidelity=float(abs(np.vdot(U@image, U2@image))**2),
                 operators_differ=float(np.linalg.norm(U-U2, 'fro')))
    assert abs(qirbp['same_black_QIRBP_cipher_fidelity']-1) < 1e-12
    assert qirbp['operators_differ'] > 0.1

    estimates = []
    for description, L, t_log2 in [
        ('three independent uniform 128-bit fields, unique accepted key', 384, 0),
        ('same nominal space, 2^128 accepted keys', 384, 128),
        ('two shared-generator 128-bit seeds, unique accepted key', 256, 0),
        ('one 128-bit seed for all three outputs, unique accepted key', 128, 0),
        ('legacy scalar 32-bit seed space, unique accepted seed', 32, 0),
    ]:
        exponent = (L-t_log2)/2
        scale = 2.0**exponent
        estimates.append(dict(scenario=description, uniform_candidate_bits=L,
                              log2_accepted_keys=t_log2, grover_exponent=exponent,
                              square_root_scale=scale,
                              near_certain_leading_queries=math.pi/4*scale,
                              status='CONDITIONAL black-box baseline; not proved QEF security'))
    dump_csv('conditional_search_costs.csv', estimates)

    dimension_rows = []
    for name, N in [('IM1_native', 65536), ('IM2_native', 50394),
                    ('IM3_native', 65536), ('IM4_native', 786432)]:
        n = (N-1).bit_length()
        r = 11
        D = N*(1 << r)
        dimension_rows.append(dict(image=name, N=N, n=n, r=r, Q=n+r,
                                   valid_dimension=D, candidate_key_bits_assumed=384,
                                   one_known_pure_input_key_guess_upper=D/2.0**384,
                                   log2_upper=math.log2(D)-384,
                                   entangled_chosen_input_key_guess_upper=D*D/2.0**384,
                                   log2_entangled_upper=2*math.log2(D)-384,
                                   scope='exact full-key identification only; no verifier; not plaintext secrecy'))
    dump_csv('one_state_exact_key_bounds.csv', dimension_rows)

    # Tight dimension-bound examples: M distinct labels share D basis states.
    # A measurement identifies the state exactly but not the full key label.
    dimension_checks = []
    for D, M in [(2, 8), (4, 16), (6, 24), (8, 64)]:
        effects = np.zeros((M, D, D), complex)
        for label in range(D):
            effects[label, label, label] = 1
        assert np.max(np.abs(effects.sum(axis=0)-np.eye(D))) < 1e-14
        success = 0.0
        for k in range(M):
            state = np.zeros(D, complex)
            state[k % D] = 1
            success += float(np.vdot(state, effects[k]@state).real)/M
        assert abs(success-D/M) < 1e-14
        dimension_checks.append(dict(D=D, M=M, exact_key_success=success,
                                     dimension_upper=D/M, achieves_bound=True))
    dump_csv('dimension_bound_tight_examples.csv', dimension_checks)

    helstrom = []
    for s in [0.01, 0.1, 0.5, 0.9, 0.99]:
        for m in range(1, 5):
            u = np.array([1.0, 0.0], complex)
            v = np.array([math.sqrt(s), math.sqrt(1-s)], complex)
            uu, vv = np.array([1.0], complex), np.array([1.0], complex)
            for _ in range(m):
                uu, vv = np.kron(uu, u), np.kron(vv, v)
            difference = np.outer(uu, uu.conj())-np.outer(vv, vv.conj())
            actual = (1-0.5*float(np.sum(np.abs(np.linalg.eigvalsh(difference)))))/2
            expected = (1-math.sqrt(1-s**m))/2
            assert abs(actual-expected) < 1e-13
            helstrom.append(dict(squared_overlap=s, copies=m, numerical_error_probability=actual,
                                 formula_error_probability=expected, absolute_error=abs(actual-expected)))
    dump_csv('binary_state_verification_checks.csv', helstrom)

    results = dict(
        scope='Original abstract E P R operator; unspecified HCS/HG-Phi law is not substituted.',
        grover_probability_checks=len(grover), maximum_grover_probability_error=max_grover_error,
        original_operator_checks=len(algebra),
        maximum_block_formula_error=max(x['block_formula_error'] for x in algebra),
        maximum_layer_reparametrization_error=max(x['alternative_gate_choices_same_operator_error'] for x in algebra),
        qirbp_known_image_collision=qirbp,
        conditional_cost_rows=len(estimates), exact_key_dimension_bound_rows=len(dimension_rows),
        binary_state_verification_checks=len(helstrom),
        maximum_binary_probability_error=max(x['absolute_error'] for x in helstrom),
        dimension_bound_tight_examples=len(dimension_checks),
        NO_claim_of_2_to_192_global_QEF_lower_bound=True,
        NO_actual_384_bit_key_search_executed=True,
        NO_claim_reparameterized_angles_correspond_to_HCS_seeds=True,
        references=['https://arxiv.org/abs/quant-ph/9605034',
                    'https://arxiv.org/abs/quant-ph/9711070',
                    'https://eprint.iacr.org/2019/1146'])
    (OUT/'verification_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
