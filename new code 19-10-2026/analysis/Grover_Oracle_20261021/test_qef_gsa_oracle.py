"""Independent circuit and resource checks for the finite QEF oracle."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import qef_gsa_oracle as oracle


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project",type=Path,default=oracle.SOURCE_CODE)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"results")
    args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    reference=oracle.load_reference(args.project)
    checks={}
    rgb=np.array([[31,162,87],[209,14,115]],dtype=np.uint8)

    # The three-qubit selector must implement both branches on every basis state.
    selector=oracle.selector_circuit()
    error=0.0
    for index in range(8):
        b=(index>>0)&1;p=(index>>1)&1;f=(index>>2)&1
        expected_index=index^(4 if b==0 and p else 0)
        phase=-1 if b==1 and p and f else 1
        expected=np.zeros(8,complex);expected[expected_index]=phase
        initial=np.zeros(8,complex);initial[index]=1
        actual=Statevector(initial).evolve(selector).data
        error=max(error,float(np.linalg.norm(actual-expected)))
    assert error<1e-12
    checks["selector_max_state_error"]=error
    print("Selector branches verified.",flush=True)

    # Check zero-state phase, clean work, and exact CX+U composition counts.
    reflection_errors=[]
    for size in range(1,9):
        work=max(size-2,0)
        qc=QuantumCircuit(size+work)
        oracle.zero_phase(qc,list(range(size)),list(range(size,size+work)))
        compiled=oracle.compile_basis(qc)
        assert oracle.counts(compiled)==oracle.zero_phase_counts(size)
        rng=np.random.default_rng(1000+size)
        amplitudes=rng.normal(size=1<<size)+1j*rng.normal(size=1<<size)
        amplitudes/=np.linalg.norm(amplitudes)
        initial=np.zeros(1<<(size+work),complex);initial[:1<<size]=amplitudes
        expected=initial.copy();expected[0]*=-1
        actual=Statevector(initial).evolve(compiled).data
        reflection_errors.append(float(np.linalg.norm(actual-expected)))
    assert max(reflection_errors)<1e-11
    checks["zero_phase_max_error"]=max(reflection_errors)
    checks["zero_phase_count_cases"]=8
    print("Clean-ancilla reflections and counts verified.",flush=True)

    all_program_errors=[]
    reference_errors=[]
    quantization_errors=[]
    for model in ("paper-three-layer","existing-qirbp-v2"):
        config=oracle.Config(model=model)
        bank=oracle.Bank(config,reference)
        program,_=bank.programmable()
        prep=oracle.plaintext_circuit(rgb,config)
        plain=Statevector.from_instruction(prep)
        for candidate in range(config.M):
            initial=np.zeros(1<<(config.L+config.Q),complex)
            initial[candidate::config.M]=plain.data
            actual=Statevector(initial).evolve(program).data
            expected=np.zeros_like(initial)
            encrypted=plain.evolve(bank.fixed(candidate)).data
            expected[candidate::config.M]=encrypted
            error=float(np.linalg.norm(actual-expected))
            all_program_errors.append(error)
            assert error<1e-10,(model,candidate,error)
            overlap=bank.operator_overlap(candidate,candidate)
            assert abs(overlap-1)<1e-11
            if model=="existing-qirbp-v2":
                independent=oracle.reference_state(reference,bank,rgb,candidate)
                unquantized=plain.evolve(bank.fixed(candidate,quantized=False)).data
                ref_error=float(np.linalg.norm(independent-unquantized))
                reference_errors.append(ref_error)
                assert ref_error<1e-10,(candidate,ref_error)
                q_error=float(np.linalg.norm(encrypted-unquantized))
                quantization_errors.append(q_error)
                upper=3*config.r*math.pi/(1<<config.angle_bits)
                assert q_error<=upper+1e-10
        # A superposition of all key branches detects lost key-dependent phases.
        initial=np.kron(plain.data,np.full(config.M,1/math.sqrt(config.M),complex))
        actual=Statevector(initial).evolve(program).data
        expected=np.zeros_like(initial)
        for candidate in range(config.M):
            expected[candidate::config.M]=plain.evolve(bank.fixed(candidate)).data/math.sqrt(config.M)
        assert np.linalg.norm(actual-expected)<1e-10
    checks["key_controlled_QEF_max_error"]=max(all_program_errors)
    checks["key_controlled_QEF_basis_branches"]=len(all_program_errors)
    checks["existing_v2_reference_max_error"]=max(reference_errors)
    checks["angle_quantization_max_state_error"]=max(quantization_errors)
    print("All key branches, key superpositions, and the existing cipher reference verified.",flush=True)

    # Exercise the default eleven-feature-qubit QIRBP layout as well.
    wave_errors=[]
    for model in ("paper-three-layer","existing-qirbp-v2"):
        wave_config=oracle.Config(wavelength_qubits=5,model=model)
        wave_bank=oracle.Bank(wave_config,reference)
        wave_program,_=wave_bank.programmable()
        wave_plain=Statevector.from_instruction(oracle.plaintext_circuit(rgb,wave_config))
        for candidate in (0,11,15):
            initial=np.zeros(1<<(wave_config.L+wave_config.Q),complex)
            initial[candidate::wave_config.M]=wave_plain.data
            actual=Statevector(initial).evolve(wave_program).data
            expected=np.zeros_like(initial)
            expected[candidate::wave_config.M]=wave_plain.evolve(wave_bank.fixed(candidate)).data
            error=float(np.linalg.norm(actual-expected))
            wave_errors.append(error)
            assert error<1e-10
            if model=="existing-qirbp-v2":
                independent=oracle.reference_state(reference,wave_bank,rgb,candidate)
                unquantized=wave_plain.evolve(wave_bank.fixed(candidate,quantized=False)).data
                assert np.linalg.norm(independent-unquantized)<1e-10
    checks["eleven_feature_register_max_error"]=max(wave_errors)
    checks["eleven_feature_register_branches_checked"]=len(wave_errors)
    print("Eleven-feature-qubit QIRBP layouts verified.",flush=True)

    config=oracle.Config()
    bank=oracle.Bank(config,reference)
    program,_=bank.programmable()
    result,objects=oracle.run_case(config,bank,program,rgb,"independent_test",11,args.output)
    print("Finite-bank gap and sufficient oracle verified.",flush=True)
    block=objects["block"]
    single=oracle.oracle_circuit(block,config,1)
    compiled_single=oracle.compile_basis(single)
    expected_counts=oracle.add_counts(
        oracle.counts(oracle.compile_basis(block)),
        oracle.counts(oracle.compile_basis(block.inverse())),
        oracle.zero_phase_counts(config.Q),
    )
    assert oracle.counts(compiled_single)==expected_counts
    checks["single_oracle_gate_counts"]=expected_counts
    print("Single-block oracle compiled and counted.",flush=True)
    for s in (2,3):
        actual_counts=oracle.counts(oracle.compile_basis(oracle.oracle_circuit(block,config,s)))
        predicted=oracle.add_counts(
            oracle.scale_counts(oracle.counts(oracle.compile_basis(block)),s),
            oracle.scale_counts(oracle.counts(oracle.compile_basis(block.inverse())),s),
            oracle.zero_phase_counts(s*config.Q),
        )
        assert actual_counts==predicted,(s,actual_counts,predicted)
    checks["compositional_count_repetitions_checked"]=[1,2,3]
    print("Repeated-oracle gate counts verified.",flush=True)

    # Compare the Gram simulation with the physical CX+U circuit, including
    # coherent verification error and all clean ancillas, after one Grover step.
    physical=QuantumCircuit(single.num_qubits)
    physical.h(list(range(config.L)))
    physical.compose(single,inplace=True)
    diffusion=oracle.diffusion_circuit(config.L)
    work_start=config.L+config.Q
    physical.compose(
        diffusion,list(range(config.L))+list(range(work_start,work_start+config.L-2)),
        inplace=True,
    )
    executable=oracle.compile_basis(physical)
    assert set(executable.count_ops()) <= set(oracle.BASIS)
    executable.save_probabilities(list(range(config.L)))
    simulator=AerSimulator(method="statevector",max_parallel_threads=1,
                           fusion_enable=False)
    print(f"Running full {single.num_qubits}-qubit oracle simulation.",flush=True)
    observed=np.asarray(simulator.run(executable).result().data(0)["probabilities"])
    _,_,predicted=oracle.simulate_grover_gram(
        objects["base_vectors"],1,objects["marked"],1
    )
    simulation_error=float(np.max(np.abs(observed-predicted)))
    assert simulation_error<1e-9,simulation_error
    checks["gram_vs_full_circuit_probability_max_error"]=simulation_error
    checks["full_circuit_simulated_qubits"]=single.num_qubits
    checks["sufficient_oracle_actual_success"]=result["actual_oracle_grover_success"]
    checks["sufficient_oracle_error_bound"]=result["accumulated_oracle_error_bound"]
    checks["passed"]=True
    (args.output/"test_report.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2),flush=True)


if __name__=="__main__":
    main()
