# substitution.py
# Requirements: Qiskit, Matplotlib
# This script builds a quantum circuit to illustrate the QEF substitution layer.
# Each qubit (e.g., representing a pixel's color) is rotated about the Z, Y, and X axes by specified angles.

import numpy as np
from math import pi
from qiskit import QuantumCircuit

# Define number of qubits (e.g., 3 feature/color qubits for demonstration)
num_qubits = 3
qc = QuantumCircuit(num_qubits)

# Example rotation angles (in radians) for each qubit.
# In a real scenario, these would be derived from a chaotic key:contentReference[oaicite:5]{index=5}.
angles_z = [3*pi/8,  pi/4,  1*pi/8]   # angles for Rz on qubits 0,1,2
angles_y = [1*pi/4,  3*pi/8, 1*pi/8]   # angles for Ry on qubits 0,1,2
angles_x = [1*pi/8,  1*pi/8,  3*pi/8]   # angles for Rx on qubits 0,1,2

# Apply the substitution rotations on each qubit
for i in range(num_qubits):
    qc.rz(angles_z[i], i)   # Rotate qubit i around Z-axis by angles_z[i]
    qc.ry(angles_y[i], i)   # Rotate qubit i around Y-axis by angles_y[i]
    qc.rx(angles_x[i], i)   # Rotate qubit i around X-axis by angles_x[i]
    # These rotations alter the state of qubit i, analogous to changing a pixel's color value.

# Draw and save the quantum circuit diagram
qc.draw(output='mpl', filename='substitution_circuit.png')
print("Substitution circuit drawn and saved as substitution_circuit.png")
