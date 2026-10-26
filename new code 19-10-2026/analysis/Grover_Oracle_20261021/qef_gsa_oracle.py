"""Coherent QEF candidate verification and concrete CX+U gate counts.

The public finite bank has four substitution records, two permutation records,
and two entanglement records. It is a 4-bit, 16-candidate experiment, not an
implementation of a 384-bit attack or a reversible HCS/HG-Phi generator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, qpy, transpile
from qiskit.circuit.library import (
    PhaseGate, RXGate, RYGate, RZGate, StatePreparation, UnitaryGate, XGate, ZGate,
)
from qiskit.quantum_info import Operator, Statevector

BASIS = ["cx", "u"]
SOURCE_CODE = Path(__file__).resolve().parents[2]


def load_reference(project: Path):
    spec = importlib.util.spec_from_file_location("qef_reference_for_oracle", project / "qef_qirbp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@dataclass(frozen=True)
class Config:
    n: int = 1
    wavelength_qubits: int = 0
    angle_bits: int = 12
    sub_bits: int = 2
    perm_bits: int = 1
    ent_bits: int = 1
    model: str = "paper-three-layer"
    total_oracle_error: float = 1e-6

    @property
    def r(self): return 6 + self.wavelength_qubits
    @property
    def Q(self): return self.n + self.r
    @property
    def N(self): return 1 << self.n
    @property
    def L(self): return self.sub_bits + self.perm_bits + self.ent_bits
    @property
    def M(self): return 1 << self.L
    @property
    def privacy_core(self): return self.model == "existing-qirbp-v2"

    def validate(self):
        if self.n != 1:
            raise ValueError("The audited experiment uses two pixel addresses (n=1).")
        if not 0 <= self.wavelength_qubits <= 5:
            raise ValueError("Use 0..5 wavelength qubits for the small experiment.")
        if not 2 <= self.angle_bits <= 24:
            raise ValueError("angle_bits must be in 2..24.")
        if (self.sub_bits, self.perm_bits, self.ent_bits) != (2, 1, 1):
            raise ValueError("This reproducible public bank uses (2,1,1) key bits.")
        if self.model not in ("paper-three-layer", "existing-qirbp-v2"):
            raise ValueError("Unknown circuit family.")
        if not 0 < self.total_oracle_error < 0.1:
            raise ValueError("Choose total_oracle_error in (0,0.1).")


class Bank:
    def __init__(self, config: Config, reference):
        config.validate()
        self.c, self.reference = config, reference
        self.spec = reference.ImageSpec(1, config.N, config.wavelength_qubits)
        self.sub = [self.schedule(i, 0, 0) for i in range(1 << config.sub_bits)]
        self.perm = [self.schedule(0, i, 0) for i in range(1 << config.perm_bits)]
        self.ent = [self.schedule(0, 0, i) for i in range(1 << config.ent_bits)]
        self.angle_raw = [self.raw_angles(s) for s in self.sub]
        self.angle_codes = [
            np.floor(a * (1 << config.angle_bits) / (2 * math.pi)).astype(np.uint32)
            for a in self.angle_raw
        ]
        self.angles = [a * (2 * math.pi / (1 << config.angle_bits)) for a in self.angle_codes]

    @staticmethod
    def public_bytes(role: str, value: int):
        return hashlib.sha256(f"public-oracle-fixture:{role}:{value}".encode()).digest()[:16]

    def key_material(self, sub: int, perm: int, ent: int):
        # All records are public test fixtures. No saved research key is loaded.
        permutation = self.public_bytes("perm", perm)[:8] + perm.to_bytes(8, "big")
        return self.reference.KeyMaterial(
            self.public_bytes("sub", sub), permutation, self.public_bytes("ent", ent)
        )

    def schedule(self, sub: int, perm: int, ent: int):
        return self.reference.Schedule(self.spec, self.key_material(sub, perm, ent))

    def raw_angles(self, schedule):
        c = self.c
        raw = b"".join(
            hashlib.shake_256(
                schedule.context + schedule.mixing_seed + k.to_bytes(8, "big")
            ).digest(c.r * 3 * 8)
            for k in range(c.N)
        )
        return (
            (np.frombuffer(raw, dtype="<u8") >> 11).astype(np.float64)
            * (2 * math.pi / (1 << 53))
        ).reshape(c.N, c.r, 3)

    def decode(self, candidate: int):
        c = self.c
        sub = candidate & ((1 << c.sub_bits) - 1)
        perm = (candidate >> c.sub_bits) & ((1 << c.perm_bits) - 1)
        ent = candidate >> (c.sub_bits + c.perm_bits)
        return sub, perm, ent

    def fixed(self, candidate: int, quantized: bool = True):
        c = self.c
        sub, perm, ent = self.decode(candidate)
        qc = QuantumCircuit(c.Q, name=f"U_{candidate}")
        angles = self.angles[sub] if quantized else self.angle_raw[sub]
        self.append_substitution(qc, list(range(c.Q)), sub, angles)
        self.append_permutation(qc, list(range(c.Q)), perm)
        self.append_entanglement(qc, list(range(c.Q)), ent)
        return qc

    def append_substitution(self, qc, data, sub, angles, key=None):
        c = self.c
        key = list(key or [])
        for k in range(c.N):
            controls = key + data[:c.n]
            pattern = sub + (k << len(key)) if key else k
            for f in range(c.r):
                target = data[c.n + c.r - 1 - f]
                for axis, gate in enumerate((RXGate, RYGate, RZGate)):
                    qc.append(
                        gate(float(angles[k, f, axis])).control(
                            len(controls), ctrl_state=pattern
                        ),
                        controls + [target],
                    )
        if c.privacy_core:
            schedule = self.sub[sub]
            count = 0
            for f in range(c.r):
                target = data[c.n + c.r - 1 - f]
                for word, gate in ((schedule.feature_x, XGate), (schedule.feature_z, ZGate)):
                    if (word >> f) & 1:
                        append_selected(qc, gate(), key, sub, [target])
                        count += 1
            for p in range(c.n):
                if (schedule.phase_z >> p) & 1:
                    append_selected(qc, ZGate(), key, sub, [data[p]])
            phase = -math.pi * count / 2
            if key:
                phase_on_pattern(qc, key, sub, phase)
            else:
                qc.global_phase += phase

    def append_permutation(self, qc, data, perm, key=None):
        c = self.c
        mapping = np.asarray(self.perm[perm].permute(np.arange(c.N)), dtype=int)
        if np.array_equal(mapping, np.arange(c.N)):
            return
        if c.n == 1:
            append_selected(qc, XGate(), list(key or []), perm, [data[0]])
        else:
            matrix = np.zeros((c.N, c.N), dtype=complex)
            matrix[mapping, np.arange(c.N)] = 1
            append_selected(qc, UnitaryGate(matrix), list(key or []), perm, data[:c.n])

    def append_entanglement(self, qc, data, ent, key=None):
        c = self.c
        key = list(key or [])
        choices = self.ent[ent].entanglement_choices
        for p in range(c.n):
            position = data[c.n - 1 - p]
            for f in range(c.r):
                target = data[c.n + c.r - 1 - f]
                gate = ZGate() if choices[p, f] else XGate()
                controls = key + [position]
                pattern = ent + (1 << len(key)) if key else 1
                qc.append(
                    gate.control(len(controls), ctrl_state=pattern),
                    controls + [target],
                )

    def programmable(self):
        c = self.c
        ks = QuantumRegister(c.sub_bits, "K_sub")
        kp = QuantumRegister(c.perm_bits, "K_perm")
        ke = QuantumRegister(c.ent_bits, "K_ent")
        data = QuantumRegister(c.Q, "image")
        qc = QuantumCircuit(ks, kp, ke, data, name="U_program")
        stages = []
        for role, register, values, method in (
            ("Substitution", ks, self.sub, self.append_substitution),
            ("Permutation", kp, self.perm, self.append_permutation),
            ("Entanglement", ke, self.ent, self.append_entanglement),
        ):
            layer = QuantumCircuit(len(register) + c.Q, name=role)
            layer_key = list(range(len(register)))
            layer_data = list(range(len(register), len(register) + c.Q))
            for value in range(len(values)):
                if role == "Substitution":
                    method(layer, layer_data, value, self.angles[value], key=layer_key)
                else:
                    method(layer, layer_data, value, key=layer_key)
            qc.append(layer.to_gate(), list(register) + list(data))
            stages.append(layer)
        return qc, stages

    def block_data(self, candidate):
        c = self.c
        sub, perm, ent = self.decode(candidate)
        schedule = self.sub[sub]
        mapping = np.asarray(self.perm[perm].permute(np.arange(c.N)), dtype=int)
        blocks = []
        scalars = []
        pauli_x = np.array([[0, 1], [1, 0]], complex)
        pauli_z = np.diag([1, -1]).astype(complex)
        eye = np.eye(2, dtype=complex)
        for k, t in enumerate(mapping):
            local = []
            for f in range(c.r):
                x, y, z = self.angles[sub][k, f]
                rx = math.cos(x/2)*eye - 1j*math.sin(x/2)*pauli_x
                ry = np.array([[math.cos(y/2), -math.sin(y/2)],
                               [math.sin(y/2), math.cos(y/2)]], complex)
                rz = np.diag([np.exp(-.5j*z), np.exp(.5j*z)])
                operator = rz @ ry @ rx
                if c.privacy_core:
                    if (schedule.feature_x >> f) & 1: operator = pauli_x @ operator
                    if (schedule.feature_z >> f) & 1: operator = pauli_z @ operator
                for p in range(c.n):
                    if (t >> (c.n-1-p)) & 1:
                        operator = (
                            pauli_z if self.ent[ent].entanglement_choices[p, f] else pauli_x
                        ) @ operator
                local.append(operator)
            phase = 1.0 + 0j
            if c.privacy_core:
                count = schedule.feature_x.bit_count() + schedule.feature_z.bit_count()
                phase = (-1j)**count * (-1)**((k & schedule.phase_z).bit_count())
            blocks.append(local)
            scalars.append(phase)
        return mapping, blocks, scalars

    def operator_overlap(self, candidate, target):
        p, w, a = self.block_data(candidate)
        pt, wt, at = self.block_data(target)
        total = 0j
        for k in range(self.c.N):
            if p[k] == pt[k]:
                product = np.conj(at[k])*a[k]
                for f in range(self.c.r):
                    product *= np.trace(wt[k][f].conj().T @ w[k][f])/2
                total += product
        return abs(total/self.c.N)


def append_selected(qc, gate, key, value, targets):
    if key:
        qc.append(gate.control(len(key), ctrl_state=value), key + list(targets))
    else:
        qc.append(gate, targets)


def phase_on_pattern(qc, key, value, angle):
    if abs(angle) < 1e-15:
        return
    for j, q in enumerate(key):
        if not (value >> j) & 1: qc.x(q)
    if len(key) == 1:
        qc.p(angle, key[0])
    else:
        qc.mcp(angle, key[:-1], key[-1])
    for j, q in enumerate(key):
        if not (value >> j) & 1: qc.x(q)


def plaintext_circuit(rgb, config: Config):
    """Exact QIRBP loader for the two selected RGB pixels."""
    c = config
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(c.N, 3)
    qc = QuantumCircuit(c.Q, name="S_P")
    qc.h(list(range(c.n)))
    plane = list(range(c.n+c.wavelength_qubits, c.n+c.wavelength_qubits+3))
    qc.h(plane)
    for q in range(c.n, c.n+c.wavelength_qubits):
        qc.h(q)
    controls = list(range(c.n)) + plane
    for k in range(c.N):
        for bitplane in range(8):
            pattern = k + (bitplane << c.n)
            for channel in range(3):
                if (int(rgb[k, channel]) >> bitplane) & 1:
                    target = c.n + c.wavelength_qubits + 5 - channel
                    qc.append(XGate().control(len(controls), ctrl_state=pattern), controls+[target])
    return qc


def reference_state(reference, bank: Bank, rgb, candidate):
    sub, perm, ent = bank.decode(candidate)
    schedule = bank.schedule(sub, perm, ent)
    chunk = reference.encrypt_chunk(rgb, np.arange(bank.c.N), schedule)
    out = np.zeros(1 << bank.c.Q, dtype=complex)
    for row, position in enumerate(chunk.positions):
        branch = chunk.core[row]
        for j in range(bank.c.wavelength_qubits):
            branch = np.kron(branch, chunk.wavelength[row, j])
        for feature, amplitude in enumerate(branch):
            out[int(position) + (feature << bank.c.n)] = amplitude/math.sqrt(bank.c.N)
    return out


def compile_basis(circuit):
    return transpile(circuit, basis_gates=BASIS, optimization_level=0, seed_transpiler=17)


def counts(circuit):
    result = Counter(dict(circuit.count_ops()))
    unexpected = set(result) - set(BASIS)
    if unexpected:
        raise AssertionError(f"Unexpanded gates in CX+U circuit: {unexpected}")
    return {"cx": int(result["cx"]), "u": int(result["u"])}


def add_counts(*terms):
    return {g: sum(t.get(g, 0) for t in terms) for g in BASIS}


def scale_counts(term, factor):
    return {g: term[g]*int(factor) for g in BASIS}


def zero_phase(qc, data, work):
    """I-2|0><0| with a linear clean-ancilla Toffoli ladder."""
    size = len(data)
    qc.x(data)
    if size == 1:
        qc.z(data[0])
    elif size == 2:
        qc.cz(*data)
    else:
        if len(work) < size-2:
            raise ValueError("The clean-ancilla ladder needs size-2 ancillas.")
        qc.ccx(data[0], data[1], work[0])
        for j in range(2, size-1):
            qc.ccx(data[j], work[j-2], work[j-1])
        qc.cz(work[size-3], data[-1])
        for j in reversed(range(2, size-1)):
            qc.ccx(data[j], work[j-2], work[j-1])
        qc.ccx(data[0], data[1], work[0])
    qc.x(data)


def zero_phase_counts(size):
    if size == 1: return {"cx": 0, "u": 3}
    return {"cx": 12*(size-2)+1, "u": 20*size-34}


def test_block(prep, cipher_prep, program, config):
    c = config
    qc = QuantumCircuit(c.L+c.Q, name="A")
    data = list(range(c.L, c.L+c.Q))
    qc.compose(prep, data, inplace=True)
    qc.compose(program, list(range(c.L+c.Q)), inplace=True)
    qc.compose(cipher_prep.inverse(), data, inplace=True)
    return qc


def oracle_circuit(block, config, repetitions, compiled=False):
    c = config
    size = repetitions*c.Q
    ancillas = max(size-2, c.L-2, 0)
    key = QuantumRegister(c.L, "K")
    tests = QuantumRegister(size, "test")
    work = QuantumRegister(ancillas, "clean")
    qc = QuantumCircuit(key, tests, work, name=f"Oracle_s{repetitions}")
    gate = block.to_gate(label="A = S_C^dag U_K S_P")
    inverse = block.inverse().to_gate(label="A^dag")
    for j in range(repetitions):
        qc.append(gate, list(key)+list(tests[j*c.Q:(j+1)*c.Q]))
    if compiled:
        zero_phase(qc, list(tests), list(work))
    else:
        reflection = QuantumCircuit(size+ancillas, name="R_zero")
        zero_phase(reflection, list(range(size)), list(range(size, size+ancillas)))
        qc.append(reflection.to_gate(label="I - 2|0><0|"), list(tests)+list(work))
    for j in reversed(range(repetitions)):
        qc.append(inverse, list(key)+list(tests[j*c.Q:(j+1)*c.Q]))
    return qc


def diffusion_circuit(L):
    ancillas = max(L-2, 0)
    qc = QuantumCircuit(L+ancillas, name="Diffusion")
    qc.h(list(range(L)))
    zero_phase(qc, list(range(L)), list(range(L,L+ancillas)))
    qc.h(list(range(L)))
    qc.global_phase += math.pi
    return qc


def gram_work_vectors(base_vectors, repetitions, tolerance=1e-11):
    """Coordinates for span{|0>, v_k^tensor(s)} without a large statevector."""
    v = np.asarray(base_vectors, dtype=complex)
    M = len(v)
    gram = np.empty((M+1,M+1), complex)
    gram[0,0] = 1
    gram[0,1:] = v[:,0]**repetitions
    gram[1:,0] = np.conj(gram[0,1:])
    gram[1:,1:] = (v.conj() @ v.T)**repetitions
    gram = (gram+gram.conj().T)/2
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues.min() < -1e-8:
        raise AssertionError("Work-state Gram matrix is not positive semidefinite.")
    selected = eigenvalues > tolerance
    coordinates = np.sqrt(eigenvalues[selected])[:,None]*eigenvectors[:,selected].conj().T
    residual = float(np.max(np.abs(coordinates.conj().T @ coordinates-gram)))
    if residual > 1e-8:
        raise AssertionError(f"Gram reconstruction residual: {residual}")
    return coordinates, residual


def simulate_grover_gram(base_vectors, repetitions, marked, iterations):
    coordinates, residual = gram_work_vectors(base_vectors, repetitions)
    M = len(base_vectors)
    state = np.tile(coordinates[:,0], (M,1))/math.sqrt(M)
    norms = np.sum(np.abs(coordinates[:,1:])**2,axis=0)
    assert np.max(np.abs(norms-1)) < 1e-8
    work_directions = coordinates[:,1:].T
    theta = math.asin(math.sqrt(len(marked)/M))
    trace = []
    for step in range(iterations+1):
        probabilities = np.sum(np.abs(state)**2,axis=1).real
        trace.append({
            "oracle_queries":step,
            "success_probability":float(probabilities[marked].sum()),
            "ideal_probability":math.sin((2*step+1)*theta)**2,
            "total_probability":float(probabilities.sum()),
        })
        if step < iterations:
            overlaps = np.sum(work_directions.conj()*state,axis=1)
            state -= 2*overlaps[:,None]*work_directions
            state = 2*state.mean(axis=0,keepdims=True)-state
    return trace, residual, np.sum(np.abs(state)**2,axis=1).real


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ensure_pylatexenc(project):
    if importlib.util.find_spec("pylatexenc"):
        return
    package = (
        project.parent / "images encryption and decryption models" / "code"
        / "QEF" / "Lib" / "site-packages" / "pylatexenc"
    )
    if not (package/"__init__.py").is_file():
        raise RuntimeError("Install pylatexenc to draw the circuits with output='mpl'.")
    spec=importlib.util.spec_from_file_location(
        "pylatexenc",package/"__init__.py",submodule_search_locations=[str(package)]
    )
    module=importlib.util.module_from_spec(spec)
    sys.modules["pylatexenc"]=module
    spec.loader.exec_module(module)


def save_circuit(circuit, stem, title, fold=45):
    figure=circuit.draw(output="mpl",fold=fold,idle_wires=True,
                        style={"name":"bw","fontsize":9,"subfontsize":7})
    figure.axes[0].set_title(title,fontsize=12,pad=14)
    for ext in ("png","svg"):
        figure.savefig(str(stem)+"."+ext,dpi=190,bbox_inches="tight")
    plt.close(figure)


def full_oracle_overview(result, output):
    figure,ax=plt.subplots(figsize=(12.8,4.2))
    ax.set_xlim(0,12.8);ax.set_ylim(-.8,3.1);ax.axis("off")
    lanes=[(2.2,"Candidate key",f'{result["key_qubits"]} qubits'),
           (1.1,"QIRBP test copies",f'{result["test_qubits"]:,} qubits'),
           (0.0,"Clean work",f'{result["clean_ancillas"]:,} qubits')]
    for y,label,width in lanes:
        ax.plot([2.7,12.3],[y,y],color="#364152",lw=1.4)
        ax.text(.1,y+.12,label,fontsize=11)
        ax.text(.1,y-.18,width,fontsize=10,color="#566273")
        ax.plot([2.92,3.08],[y-.08,y+.08],color="#364152")
    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.07",
                                    facecolor=color,edgecolor="#364152",zorder=3))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=11,zorder=4)
    box(3.7,.75,2.05,1.8,"V\ns repeated tests","#e6effb")
    box(6.45,-.35,2.05,1.8,"Zero-state\nphase reflection","#fcead7")
    box(9.2,.75,2.05,1.8,"V inverse\nuncompute","#e6effb")
    ax.text(6.4,2.95,"Coherent QEF verification oracle",ha="center",fontsize=15,weight="bold")
    ax.text(6.4,-.65,
            f's = {result["repetitions"]:,};  '
            f'CX = {result["oracle_cx"]:,};  U = {result["oracle_u"]:,}',
            ha="center",fontsize=11)
    for ext in ("png","svg"):
        figure.savefig(output/f"oracle_full_mpl.{ext}",dpi=200,bbox_inches="tight")
    plt.close(figure)


def selector_circuit():
    qc=QuantumCircuit(3,name="CNOT_or_CZ")
    qc.x(0);qc.ccx(0,1,2);qc.x(0)
    qc.h(2);qc.ccx(0,1,2);qc.h(2)
    return qc


def arabic_summary(report, output):
    lines=[
        "تجربة بناء Oracle متماسك لخوارزمية QEF",
        "",
        "استخدمت التجربة بكسلين من بداية الصف الأول لكل صورة أصلية، دون تعديل الصور.",
        "عدد مرشحي المفتاح 16: بتان لاختيار الاستبدال، وبت للتبديل، وبت للتشابك.",
        "تم اختبار QIRBP بسجلي خصائص r=6 وr=11، مع n=1.",
        "دقة الزوايا: شبكة 12 بت بخطوة 2*pi/4096 راديان.",
        "معرفة المهاجم في هذه التجربة: وصف كلاسيكي كامل لحالتي النص الأصلي والمشفر،",
        "مما يتيح بناء دائرتي التحضير ومعكوسهما. تحضير الحالات محسوب ضمن البوابات.",
        "كل صورة تجربة مستقلة بمفتاح اختبار خاص بها؛ لا تجمع أزواج مفاتيح مختلفة.",
        "",
        "حساب الكلفة: بوابات CX وU أحادية الكيوبت، دون تحسين يلغي بوابات بين المكونات.",
        "اختبارات المرشحين المتكررة تشمل اختيار المفتاح داخل الدائرة وعكس الحساب.",
        "تم حساب Delta باستنفاد المرشحين الستة عشر، وليس بافتراض قيمة عشوائية.",
        "",
        "الصورة | r | Delta | التكرارات | بوابات Oracle واحد | بوابات هجوم Grover",
    ]
    for row in report["configurations"]:
        lines.append(
            f'{row["image_id"]} | {row["r"]} | {row["delta"]:.9f} | '
            f'{row["repetitions"]} | {row["oracle_total_gates"]:,} | '
            f'{row["attack_total_gates"]:,}'
        )
    lines += [
        "",
        "عدد دورات Grover في الحالات المعروضة: 3.",
        "احتمال المفتاح الصحيح: نحو 0.961318970.",
        "المحاكاة تحسب Oracle المقارن المبني فعلياً، مع بقايا عمل التحقق؛",
        "تستخدم فضاءً جزئياً ثابتاً اشتق رياضياً، وقورنت بمحاكاة الدائرة الكاملة.",
        "",
        "المجال الذي تثبته التجربة:",
        "تثبت اختبارات عددية بناء Oracle وصحة تشغيله وعد بواباته للإعداد المصغر المحدد.",
        "هذه الأعداد لا تمثل كلفة هجوم على مفاتيح الورقة الكاملة ذات 384 بت.",
        "سجلات المعلمات هنا سجلات اختبار معلنة من الجدولة الموجودة، وليست إعادة تنفيذ",
        "للمولدين الأصليين nD-HCS وHG-Phi أو حساباً انعكاسياً لجدولتهما الكاملة.",
        "النمط الافتراضي يستخدم الطبقات الثلاث فقط؛ يختبر نمط منفصل إضافات كود v2.",
        "تكلفة أجهزة حقيقية وتصحيح الأخطاء وتحويل U إلى Clifford+T ليست ضمن العد.",
        "",
        "لم يتم تعديل البحث أو برامج التشفير الأصلية أو مفاتيح التجارب المحفوظة.",
    ]
    (output/"SUMMARY_AR.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run_case(config, bank, program, rgb, image_id, target, output, detailed=False):
    c=config
    prep=plaintext_circuit(rgb,c)
    plaintext=Statevector.from_instruction(prep)
    ciphertext=plaintext.evolve(bank.fixed(target)).data
    cipher_prep=QuantumCircuit(c.Q,name="S_C")
    cipher_prep.append(StatePreparation(ciphertext),list(range(c.Q)))
    base_vectors=[];candidate_rows=[]
    for candidate in range(c.M):
        vector=Statevector(ciphertext).evolve(bank.fixed(candidate).inverse()).evolve(prep.inverse()).data
        fidelity=float(np.clip(abs(vector[0])**2,0,1))
        overlap=bank.operator_overlap(candidate,target)
        sub,perm,ent=bank.decode(candidate)
        candidate_rows.append({
            "candidate":candidate,"sub_index":sub,"perm_index":perm,"ent_index":ent,
            "fidelity":fidelity,"accepted":abs(1-fidelity)<1e-10,
            "operator_equivalent":abs(1-overlap)<1e-10,
            "normalized_operator_overlap":overlap,
        })
        base_vectors.append(vector)
    marked=[row["candidate"] for row in candidate_rows if row["accepted"]]
    equivalent=[row["candidate"] for row in candidate_rows if row["operator_equivalent"]]
    if marked != equivalent:
        raise AssertionError("Known pair does not distinguish this bank's encryption operators.")
    rejected=[row for row in candidate_rows if not row["accepted"]]
    delta=min(1-row["fidelity"] for row in rejected)
    if delta <= 1e-10:
        raise AssertionError("The finite bank's verification gap is numerically unresolved.")
    theta=math.asin(math.sqrt(len(marked)/c.M))
    optimal_queries=max(0,math.floor(math.pi/(4*theta)))
    budget=max(1,2*optimal_queries+1)
    s=max(1,math.ceil(2/delta*math.log(2*budget/c.total_oracle_error)))
    trace,residual,final_distribution=simulate_grover_gram(base_vectors,s,marked,budget)
    result_at_optimal=trace[optimal_queries]
    maximum_wrong_fidelity=max(row["fidelity"] for row in rejected)
    oracle_accumulated_bound=2*budget*maximum_wrong_fidelity**(s/2)
    assert oracle_accumulated_bound <= c.total_oracle_error*(1+1e-9)
    assert max(abs(row["total_probability"]-1) for row in trace)<1e-8
    assert abs(result_at_optimal["success_probability"]-result_at_optimal["ideal_probability"]) <= 2*c.total_oracle_error+1e-8

    block=test_block(prep,cipher_prep,program,c)
    compiled_block=compile_basis(block)
    compiled_inverse=compile_basis(block.inverse())
    forward_count=counts(compiled_block)
    inverse_count=counts(compiled_inverse)
    reflection=zero_phase_counts(s*c.Q)
    oracle_count=add_counts(scale_counts(forward_count,s),scale_counts(inverse_count,s),reflection)
    diffusion=compile_basis(diffusion_circuit(c.L))
    diffusion_count=counts(diffusion)
    search_count=scale_counts(add_counts(oracle_count,diffusion_count),optimal_queries)
    initialization={"cx":0,"u":c.L}
    total_count=add_counts(search_count,initialization)
    sub,perm,ent=bank.decode(target)
    sample_hash=hashlib.sha256(np.asarray(rgb,dtype=np.uint8).tobytes()).hexdigest()
    result={
        "image_id":image_id,"model":c.model,"sample_pixels":c.N,"sample_rgb_sha256":sample_hash,
        "n":c.n,"r":c.r,"Q":c.Q,"wavelength_qubits":c.wavelength_qubits,
        "angle_bits":c.angle_bits,"angle_step_radians":2*math.pi/(1<<c.angle_bits),
        "key_qubits":c.L,"candidates":c.M,"target_fixture":target,
        "target_sub_index":sub,"target_perm_index":perm,"target_ent_index":ent,
        "accepted_candidates":len(marked),"marked_keys":marked,
        "operator_equivalent_candidates":len(equivalent),"delta":delta,
        "maximum_wrong_fidelity":maximum_wrong_fidelity,"repetitions":s,
        "oracle_queries":optimal_queries,"oracle_error_query_budget":budget,
        "requested_total_oracle_error":c.total_oracle_error,
        "accumulated_oracle_error_bound":oracle_accumulated_bound,
        "ideal_grover_success":result_at_optimal["ideal_probability"],
        "actual_oracle_grover_success":result_at_optimal["success_probability"],
        "gram_reconstruction_error":residual,"gram_work_dimension":c.M+1,
        "test_qubits":s*c.Q,"clean_ancillas":max(s*c.Q-2,c.L-2),
        "total_circuit_qubits":c.L+s*c.Q+max(s*c.Q-2,c.L-2),
        "block_cx":forward_count["cx"],"block_u":forward_count["u"],
        "inverse_block_cx":inverse_count["cx"],"inverse_block_u":inverse_count["u"],
        "oracle_cx":oracle_count["cx"],"oracle_u":oracle_count["u"],
        "oracle_total_gates":sum(oracle_count.values()),
        "diffusion_cx":diffusion_count["cx"],"diffusion_u":diffusion_count["u"],
        "attack_cx":total_count["cx"],"attack_u":total_count["u"],
        "attack_total_gates":sum(total_count.values()),
        "plaintext_loader_cx":counts(compile_basis(prep))["cx"],
        "ciphertext_loader_cx":counts(compile_basis(cipher_prep))["cx"],
        "program_cx":counts(compile_basis(program))["cx"],
        "program_u":counts(compile_basis(program))["u"],
        "classical_data_access":"full state-description preparation benchmark",
        "gap_source":"exhaustive finite-bank evaluation",
        "gate_basis":"CX plus arbitrary single-qubit U; optimization_level=0",
        "ciphertext_description_complex_entries":1 << c.Q,
        "quantum_data_acquisition_cost_given_descriptions":0,
        "simulation_method":"exact invariant work-subspace Gram simulation",
    }
    case=output/f'{image_id}_w{c.wavelength_qubits}'
    case.mkdir(exist_ok=True,parents=True)
    write_csv(case/"candidate_verification.csv",candidate_rows)
    write_csv(case/"grover_trace.csv",trace)
    (case/"resources.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if detailed:
        one=oracle_circuit(block,c,1)
        full=oracle_circuit(block,c,s)
        assert full.num_qubits==result["total_circuit_qubits"]
        with (output/"oracle_full.qpy").open("wb") as stream:
            qpy.dump(full,stream)
        with (output/"oracle_single_block.qpy").open("wb") as stream:
            qpy.dump(one,stream)
        save_circuit(one,output/"oracle_single_block_mpl",
                     "QIRBP coherent verification: one test block (s = 1)",fold=-1)
        summary=QuantumCircuit(c.L+c.Q)
        data=list(range(c.L,c.L+c.Q))
        summary.append(prep.to_gate(label="S_P"),data)
        summary.compose(program,inplace=True)
        summary.append(cipher_prep.inverse().to_gate(label="S_C inverse"),data)
        save_circuit(summary,output/"verification_test_mpl",
                     "One QIRBP verification test: prepare, encrypt candidate, compare",fold=-1)
        full_oracle_overview(result,output)
        figure,ax=plt.subplots(figsize=(7.6,4.3))
        ax.plot([row["oracle_queries"] for row in trace],
                [row["ideal_probability"] for row in trace],
                "o-",label="Ideal marked-set Grover")
        ax.plot([row["oracle_queries"] for row in trace],
                [row["success_probability"] for row in trace],
                "x--",label="Constructed coherent verification oracle")
        ax.set(xlabel="Oracle calls",ylabel="Accepted-key probability",
               title=f'QIRBP: {c.M} candidates, {len(marked)} marked, s = {s}',
               ylim=(-.02,1.04))
        ax.grid(alpha=.25);ax.legend();figure.tight_layout()
        for ext in ("png","svg"):
            figure.savefig(output/f"grover_success.{ext}",dpi=200,bbox_inches="tight")
        plt.close(figure)
        component_rows=[]
        for name,circuit in (("programmable_QEF",program),("plaintext_preparation",prep),
                             ("ciphertext_preparation",cipher_prep),("test_block",block),
                             ("inverse_test_block",block.inverse()),("diffusion",diffusion_circuit(c.L))):
            compiled=compile_basis(circuit)
            count=counts(compiled)
            component_rows.append({"component":name,"qubits":compiled.num_qubits,
                                   "cx":count["cx"],"u":count["u"],
                                   "total_gates":sum(count.values()),"depth":compiled.depth()})
        write_csv(output/"component_gate_counts.csv",component_rows)
    return result, {"block":block,"base_vectors":base_vectors,"marked":marked,"prep":prep,
                    "cipher_prep":cipher_prep,"ciphertext":ciphertext}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=SOURCE_CODE)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"results")
    parser.add_argument("--wavelength-qubits",type=int,nargs="+",default=[0,5])
    parser.add_argument("--angle-bits",type=int,default=12)
    parser.add_argument("--model",choices=["paper-three-layer","existing-qirbp-v2"],default="paper-three-layer")
    parser.add_argument("--error",type=float,default=1e-6)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    ensure_pylatexenc(args.project)
    reference=load_reference(args.project)
    manifest=json.loads((args.project/"input_manifest.json").read_text(encoding="utf-8"))
    results=[];start=time.perf_counter()
    for wavelength in args.wavelength_qubits:
        config=Config(wavelength_qubits=wavelength,angle_bits=args.angle_bits,
                      model=args.model,total_oracle_error=args.error)
        bank=Bank(config,reference)
        program,stages=bank.programmable()
        if wavelength==args.wavelength_qubits[0]:
            save_circuit(program,args.output/"programmable_qef_mpl",
                         "Key-controlled QEF: substitution, permutation, entanglement",fold=-1)
            selector=selector_circuit()
            save_circuit(selector,args.output/"entanglement_selector_mpl",
                         "b = 0: CNOT(position, feature); b = 1: CZ(position, feature)",fold=-1)
            with (args.output/"programmable_qef.qpy").open("wb") as stream:qpy.dump(program,stream)
        for index,item in enumerate(manifest):
            image=Image.open(args.project/item["relative_input"]).convert("RGB")
            # Public two-pixel crop; the full-size photograph is not simulated.
            rgb=np.asarray(image)[0,:config.N,:].copy()
            target=(11+index)%config.M
            result,_=run_case(config,bank,program,rgb,item["image_id"],target,args.output,
                              detailed=wavelength==args.wavelength_qubits[0] and index==0)
            result["source_image_height"]=image.height
            result["source_image_width"]=image.width
            results.append(result)
            print(f'{item["image_id"]} w={wavelength}: Delta={result["delta"]:.6g}, '
                  f's={result["repetitions"]}, oracle gates={result["oracle_total_gates"]:,}, '
                  f'Grover success={result["actual_oracle_grover_success"]:.9f}',flush=True)
    fields=[key for key in results[0] if key!="marked_keys"]
    write_csv(args.output/"resource_summary.csv",[{key:row[key] for key in fields} for row in results])
    report={
        "experiment":"finite public-bank QEF coherent-oracle benchmark",
        "configurations":results,"runtime_seconds":time.perf_counter()-start,
        "qiskit_version":qiskit.__version__,"numpy_version":np.__version__,
        "scope":{
            "full_384_bit_search_executed":False,
            "full_images_simulated":False,
            "original_generators_reimplemented":False,
            "existing_cipher_modules_modified":False,
            "paper_modified":False,
            "parameter_source":"public fixtures from existing Schedule; quantized Euler angles",
            "paper_mode":"three layers only; omits the v2 privacy core",
            "existing_mode":"includes v2 privacy core and is separately checked against encrypt_chunk",
            "angle_precision":"finite angle grid; numerical CX+U synthesis uses floating point",
        "gate_counts_exclude":"classical acquisition of state descriptions, measurement, routing, error correction and Clifford+T synthesis",
            "state_access":"classical plaintext/ciphertext state descriptions and coherent preparation/inverse preparation",
        },
    }
    (args.output/"experiment_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    arabic_summary(report,args.output)
    print(f'Completed {len(results)} cases in {report["runtime_seconds"]:.1f}s',flush=True)


if __name__=="__main__":
    main()
