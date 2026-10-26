# python Entanglement_Operation.py
# Requirements: Qiskit, Matplotlib
# This script builds a quantum circuit to illustrate the QEF entanglement layer.
# It entangles position and color qubits using controlled gates (CNOT or CZ).

from qiskit import QuantumCircuit, QuantumRegister

# Define two quantum registers: 2 position qubits and 2 color qubits (for 2 pixels example)
pos = QuantumRegister(2, 'pos')
color = QuantumRegister(2, 'color')
qc = QuantumCircuit(pos, color)

# (Optional) Prepare superposition or other state if needed to mimic image state.
# For demonstration, we start with all |0> state; entangling gates will be applied directly.

# Apply entangling gates:
qc.cx(pos[0], color[0])   # CNOT: control = pos0, target = color0
# This entangles pixel 0's position qubit with its color qubit (flips color0 if pos0 is |1>).
qc.cz(pos[1], color[1])   # CZ: control = pos1, target = color1
# This entangles pixel 1's position and color qubits with a phase flip (adds phase if pos1 and color1 are |1>).

# Draw and save the quantum circuit diagram
qc.draw(output='mpl', filename='entanglement_circuit.png')
print("Entanglement circuit drawn and saved as entanglement_circuit.png")
