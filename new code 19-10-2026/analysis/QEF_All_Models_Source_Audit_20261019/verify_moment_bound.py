"""Original QEF rotation moments and a conditional sample-complexity theorem.

All numerical security values are analytical benchmarks for an explicitly
specified angle law, not certifications of the paper's HCS/HG-Phi sources.
"""
import csv
import itertools
import json
import math
from decimal import Decimal, localcontext, ROUND_CEILING
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)


def kron(ops):
    a = np.array([[1]], complex)
    for v in ops: a = np.kron(a, v)
    return a


def rotation(axis, theta):
    return math.cos(theta/2)*I - 1j*math.sin(theta/2)*axis


def binary_entropy(p):
    return -sum(v*math.log(v) for v in [p, 1-p] if v > 0)


def entropy(rho):
    ev = np.maximum(0.0, np.linalg.eigvalsh((rho+rho.conj().T)/2))
    return -sum(v*math.log(v) for v in ev if v > 1e-16)


def tdist(a, b):
    diff = a-b
    return float(np.sum(np.abs(np.linalg.eigvalsh((diff+diff.conj().T)/2))))/2


def pinching(rho, N, f):
    out = np.zeros_like(rho)
    for k in range(N):
        out[k*f:(k+1)*f, k*f:(k+1)*f] = rho[k*f:(k+1)*f, k*f:(k+1)*f]
    return out


def averaged_substitution(rho, N, r, mean):
    f = 1 << r
    out = np.zeros_like(rho)
    for k in range(N):
        for l in range(N):
            block = rho[k*f:(k+1)*f, l*f:(l+1)*f]
            out[k*f:(k+1)*f, l*f:(l+1)*f] = (
                np.eye(f)*np.trace(block)/f if k == l else mean@block@mean.conj().T)
    return out


def public_layers(N, r, rng):
    n, f = (N-1).bit_length(), 1 << r
    perm = rng.permutation(N)
    P = np.zeros((N*f, N*f), complex)
    for k in range(N): P[perm[k]*f:(perm[k]+1)*f, k*f:(k+1)*f] = np.eye(f)
    choices = rng.integers(0, 2, (n, r))
    E = np.zeros_like(P)
    for t in range(N):
        ops = [I.copy() for _ in range(r)]
        for p in range(n):
            if (t >> (n-1-p)) & 1:
                for j in range(r): ops[j] = (Z if choices[p, j] else X)@ops[j]
        E[t*f:(t+1)*f, t*f:(t+1)*f] = kron(ops)
    return E@P


def write_csv(name, rows):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def high_precision_cost(N, r, m, precision):
    with localcontext() as ctx:
        ctx.prec = precision
        one, two = Decimal(1), Decimal(2)
        a, b, t, p = one, one/two.sqrt(), one/4, one
        for _ in range(10):
            next_a = (a+b)/2
            b = (a*b).sqrt()
            t -= p*(a-next_a)**2
            a, p = next_a, 2*p
        pi = (a+b)**2/(4*t)
        if m is None:
            beta = (two/pi)**(6*r)
        else:
            x = pi/(2*m)
            term, sine = x, x
            for j in range(1, 300):
                term *= -x*x/((2*j)*(2*j+1))
                sine += term
                if abs(term) < Decimal(10)**(-precision+5): break
            beta = (one/(m*sine))**(6*r)
        target = two/3
        numerator = two.ln()+target*target.ln()+(one-target)*(one-target).ln()
        C = (one+Decimal(N*(1 << r))*beta*beta*(one-one/N)).ln()
        q = int((numerator/C).to_integral_value(rounding=ROUND_CEILING))
        return q, str(beta), str(C)


def main():
    rng = np.random.default_rng(2026101902)  # deterministic public verification cases
    moment_rows = []
    channel_rows = []
    sample_rows = []
    robustness = []
    for N, r, m in [(2, 1, 2), (3, 1, 2), (2, 1, 4), (2, 2, 2)]:
        f, D = 1 << r, N*(1 << r)
        angles = [2*math.pi*j/m for j in range(m)]
        lookup = []
        for indices in itertools.product(range(m), repeat=3*r):
            lookup.append(kron([rotation(Z, angles[indices[3*j+2]]) @
                                rotation(Y, angles[indices[3*j+1]]) @
                                rotation(X, angles[indices[3*j]]) for j in range(r)]))
        mean = sum(lookup)/len(lookup)
        c, s = 1/m, 1/(m*math.tan(math.pi/(2*m)))
        analytic_single = (c*I-1j*s*Z)@(c*I-1j*s*Y)@(c*I-1j*s*X)
        analytic_mean = kron([analytic_single]*r)
        beta = (1/(m*math.sin(math.pi/(2*m))))**(6*r)
        channel_error = 0.0
        for i in range(f):
            for j in range(f):
                basis = np.zeros((f, f), complex); basis[i, j] = 1
                avg = sum(t@basis@t.conj().T for t in lookup)/len(lookup)
                channel_error = max(channel_error, float(np.max(np.abs(avg-np.eye(f)*(i == j)/f))))
        mean_error = float(np.max(np.abs(mean-analytic_mean)))
        assert mean_error < 1e-12 and channel_error < 1e-12
        assert abs(np.linalg.norm(mean, 2)**2-beta) < 1e-12
        moment_rows.append(dict(N=N, r=r, angle_grid=m, per_pixel_tuples=len(lookup),
                                first_moment_error=mean_error, feature_twirl_error=channel_error,
                                beta=beta))
        F = public_layers(N, r, rng)
        superop = np.zeros((D*D, D*D), complex)
        count = len(lookup)**N
        for pixel_indices in itertools.product(range(len(lookup)), repeat=N):
            R = np.zeros((D, D), complex)
            for k, idx in enumerate(pixel_indices): R[k*f:(k+1)*f, k*f:(k+1)*f] = lookup[idx]
            U = F@R
            superop += np.kron(U.conj(), U)/count
        expected = np.zeros_like(superop)
        for col in range(D*D):
            basis = np.zeros(D*D, complex); basis[col] = 1
            avg = F@averaged_substitution(basis.reshape((D, D), order='F'), N, r, mean)@F.conj().T
            expected[:, col] = avg.reshape(-1, order='F')
        super_error = float(np.max(np.abs(superop-expected)))
        assert super_error < 2e-11
        tau = np.eye(D)/D
        hs_bound = beta*math.sqrt(1-1/N)
        eps_bound = beta*(1-1/N)
        info_bound = math.log1p(D*beta**2*(1-1/N))
        for trial in range(20):
            branches = rng.normal(size=(N, f))+1j*rng.normal(size=(N, f))
            branches /= np.linalg.norm(branches, axis=1)[:, None]
            image = branches.reshape(-1)/math.sqrt(N)
            rho = np.outer(image, image.conj())
            sigma = (superop@rho.reshape(-1, order='F')).reshape((D, D), order='F')
            td = tdist(sigma, tau)
            hs = float(np.linalg.norm(sigma-tau, 'fro'))
            rel = max(0.0, math.log(D)-entropy(sigma))
            assert td <= eps_bound+1e-11 and hs <= hs_bound+1e-11
            assert rel <= info_bound+1e-11
            channel_rows.append(dict(N=N, r=r, m=m, trial=trial, enumerated_keys=count,
                                     superoperator_error=super_error, trace_distance=td,
                                     trace_distance_upper=eps_bound, hilbert_schmidt_distance=hs,
                                     hilbert_schmidt_upper=hs_bound, relative_entropy=rel,
                                     relative_entropy_upper=info_bound))
            if trial == 0:
                deterministic = F@rho@F.conj().T
                for delta in [0.001, 0.1, 0.5]:
                    perturbed = (1-delta)*sigma+delta*deterministic
                    C = math.log1p(D*(hs_bound+2*delta)**2)
                    rel_actual = max(0.0, math.log(D)-entropy(perturbed))
                    assert rel_actual <= C+1e-11
                    robustness.append(dict(N=N, r=r, m=m, source_distance_upper=delta,
                                           actual_relative_entropy=rel_actual, relative_entropy_upper=C))
        # Independent check of the binary discrimination converse on small joint states.
        images = []
        for colour in [0, 1]:
            vec = np.zeros(D, complex)
            vec[np.arange(N)*f+colour] = 1/math.sqrt(N)
            images.append(np.outer(vec, vec.conj()))
        sigmas = [(superop@rho.reshape(-1, order='F')).reshape((D, D), order='F') for rho in images]
        for q in range(1, 4 if D <= 4 else 3):
            joint = []
            for sigma in sigmas:
                value = np.array([[1]], complex)
                for _ in range(q): value = np.kron(value, sigma)
                joint.append(value)
            p = 0.5+tdist(*joint)/2
            chi = entropy((joint[0]+joint[1])/2)-(entropy(joint[0])+entropy(joint[1]))/2
            required_information = math.log(2)-binary_entropy(p)
            assert required_information <= chi+1e-10
            assert chi <= q*info_bound+1e-10
            sample_rows.append(dict(N=N, r=r, m=m, independent_ciphertexts=q,
                                    optimal_binary_success=p, required_information=required_information,
                                    actual_Holevo_information=chi, information_upper=q*info_bound))
    write_csv('rotation_moments.csv', moment_rows)
    write_csv('enumerated_channel_checks.csv', channel_rows)
    write_csv('binary_discrimination_checks.csv', sample_rows)
    write_csv('source_perturbation_checks.csv', robustness)
    tables = []
    numerator = math.log(2)-binary_entropy(2/3)
    images = [('IM1', 65536), ('IM2', 50394), ('IM3', 65536), ('IM4', 786432)]
    for angle_label, m in [('continuous_limit', None), ('2_point', 2), ('4_point', 4), ('256_point', 256)]:
        for label, N in images:
            r = 11
            beta = (2/math.pi)**(6*r) if m is None else (1/(m*math.sin(math.pi/(2*m))))**(6*r)
            D = N*(1 << r)
            C = math.log1p(D*beta*beta*(1-1/N))
            q_lower, _, _ = high_precision_cost(N, r, m, 100)
            repeated_q, beta_precise, C_precise = high_precision_cost(N, r, m, 140)
            assert q_lower == repeated_q
            tables.append(dict(image=label, N=N, r=r, valid_dimension=D, angle_reference=angle_label,
                               beta=beta, one_instance_advantage_upper=beta*(1-1/N),
                               information_per_instance_upper_nats=C,
                               success_target=2/3, necessary_independent_ciphertexts=q_lower,
                               log2_necessary_ciphertexts=math.log2(q_lower),
                               high_precision_beta=beta_precise,
                               high_precision_information_upper=C_precise,
                               status='CONDITIONAL ANGLE-LAW BENCHMARK; not certified original-source resistance'))
    write_csv('conditional_sample_lower_bounds.csv', tables)
    qirbp_rows = []
    for N in [2, 3, 4]:
        r, f = 6, 64
        mean = ((2/math.pi)**(3*r))*np.eye(f)
        beta = (2/math.pi)**(6*r)
        F = public_layers(N, r, rng)
        for name, data_colour in [('black', 0), ('white', 7)]:
            feature = np.zeros(f, complex); feature[data_colour*8:data_colour*8+8] = 1/math.sqrt(8)
            image = np.kron(np.full(N, 1/math.sqrt(N)), feature)
            rho = np.outer(image, image.conj())
            sigma = F@averaged_substitution(rho, N, r, mean)@F.conj().T
            qirbp_rows.append(dict(N=N, r=r, image=name, norm=float(np.vdot(image, image).real),
                                   privacy_distance=tdist(sigma, np.eye(N*f)/(N*f)),
                                   exact_tight_value=beta*(1-1/N)))
            assert abs(qirbp_rows[-1]['privacy_distance']-qirbp_rows[-1]['exact_tight_value']) < 1e-12
    write_csv('normalized_qirbp_checks.csv', qirbp_rows)
    # A different angle RANGE gives a different security outcome. This is
    # an explicit analytical counterexample to choosing a numeric benchmark
    # without specifying the original angle law, not an asserted HCS attack.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    angles = (nodes+1)*math.pi/2
    weights = weights/2
    axis_channels = []
    for axis in [X, Y, Z]:
        axis_channels.append(sum(w*np.kron(rotation(axis, theta).conj(), rotation(axis, theta))
                                 for theta, w in zip(angles, weights)))
    channel_pi = axis_channels[2]@axis_channels[1]@axis_channels[0]
    zero = np.array([[1, 0], [0, 0]], complex)
    output = (channel_pi@zero.reshape(-1, order='F')).reshape((2, 2), order='F')
    signal = float(np.trace(X@output).real)
    assert abs(signal-(2/math.pi)**2) < 1e-13
    range_attack = []
    for label, N in images:
        range_attack.append(dict(image=label, N=N, angle_reference='independent uniform [0,pi)',
                                 attack='measure position until zero; then one data-qubit X measurement',
                                 expected_ciphertext_samples=N,
                                 binary_success=(1+(2/math.pi)**2)/2,
                                 status='CONDITIONAL COUNTERMODEL; actual HCS angle law is unspecified'))
    write_csv('pi_range_countermodel_attack.csv', range_attack)
    results = dict(moment_models=len(moment_rows), enumerated_image_checks=len(channel_rows),
                   binary_joint_measurement_checks=len(sample_rows), source_perturbation_checks=len(robustness),
                   normalized_qirbp_checks=len(qirbp_rows), analytical_table_rows=len(tables),
                   maximum_first_moment_error=max(t['first_moment_error'] for t in moment_rows),
                   maximum_twirl_error=max(t['feature_twirl_error'] for t in moment_rows),
                   maximum_enumerated_superoperator_error=max(t['superoperator_error'] for t in channel_rows),
                   continuous_reference_beta=(2/math.pi)**66,
                   pi_range_countermodel_signal_error=abs(signal-(2/math.pi)**2),
                   sample_lower_bound_integer_checked_at_decimal_precisions=[100, 140],
                   success_information_threshold_nats=numerator,
                   original_HCS_HG_Phi_source_law_certified=False,
                   no_new_encryption_layer_used=True,
                   costs_are_ciphertext_samples_not_elementary_gates=True)
    (OUT/'verification_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    print(json.dumps([v for v in tables if v['angle_reference']=='continuous_limit'], indent=2))


if __name__ == '__main__': main()
