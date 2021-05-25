import cirq
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction

def swap(qubit1, qubit2):
    circuit = cirq.Circuit()

    circuit.append([cirq.CNOT(qubit1, qubit2),
                    cirq.CNOT(qubit2, qubit1),
                    cirq.CNOT(qubit1, qubit2)])
    return circuit

def QFT(qubits):
    circuit = cirq.Circuit()
    n = len(qubits)
    m = n-1

    for qH in range(n):
        if qH > 0:
            i=0
            for qC in range(qH, n):
                i+=1
                circuit.append(cirq.CZ(qubits[qH-1], qubits[qC])**(1/(2**i)))
        circuit.append(cirq.H(qubits[qH]))

    for q1, q2 in zip(range(int(m/2)+1), range(m,int(m/2),-1)):
        circuit.append(swap(qubits[q1],qubits[q2]))


    return circuit

def Shor(qubits):
    circuit = cirq.Circuit()

    # Classical part: from factoring problem to period finding problem

    # Quantum part: Find priod with QFT
    QFT(qubits)

    return circuit

if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(linewidth = 180)
    qubits = [cirq.GridQubit(0,i) for i in range(3)]
    circuit = QFT(qubits)
    print_circuit = lambda circuit : "  " + (str(circuit).replace('\n','\n  ') if len(circuit) > 0 else "<<This circuit contains no gates.>>")
    print("We will be checking whether the correct 3-qubit quantum Fourier transform is constructed.")
    print("The circuit you constructed is:")
    print()
    print(print_circuit(circuit))
