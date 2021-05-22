
qubits 4
prep_z q[0:3]
H q[0:1]

.encode_new_data_point
#encode the new data point, this implements a cRy(omega)
CNOT q[1], q[2]
Ry q[2], 0.1105             # (Ry q[2], -omega/2)
CNOT q[1], q[2]
Ry q[2], -0.1105            # (Ry q[2], omega/2)
X q[1]

.encode_first_training_point
# encode the first data point, this implements a ccRy(theta)
toffoli q[0],q[1],q[2]
CNOT q[0],q[2]
Ry q[2], 0                 # (Ry q[2], theta/4)
CNOT q[0],q[2]
Ry q[2], 0                 # (Ry q[2], -theta/4)
toffoli q[0],q[1],q[2]
CNOT q[0],q[2]
Ry q[2], 0                 # (Ry q[2], -theta/4)
CNOT q[0],q[2]
Ry q[2], 0                 # (Ry q[2], theta/4)
X q[0]

.encode_second_training_point
# encode the second data point, this implements a ccRy(phi)
toffoli q[0],q[1],q[2]
CNOT q[0],q[2]
Ry q[2], -1.511125          # (Ry q[2], phi/4)
CNOT q[0],q[2]
Ry q[2], 1.511125           # (Ry q[2], -phi/4)
toffoli q[0],q[1],q[2]
CNOT q[0],q[2]
Ry q[2], 1.511125           # (Ry q[2], -phi/4)
CNOT q[0],q[2]
Ry q[2], -1.511125          # (Ry q[2], phi/4)

.labels
# encode the labels
CNOT q[0], q[3]

.algorithm
# The actual algorithm
H q[1]
measure_z q[1,3]