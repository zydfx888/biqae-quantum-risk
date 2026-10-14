import numpy as np
from qiskit import QuantumCircuit

def loss_lookup(levels):
    levels = np.asarray(levels)
    if levels.ndim != 1 or len(levels) < 2 or np.any(levels < 0) or np.any(levels != levels.astype(int)):
        raise ValueError("Loss levels must be nonnegative integers")
    n = int(np.ceil(np.log2(len(levels))))
    bits = max(1, int(max(levels)).bit_length())
    qc = QuantumCircuit(n+bits, name="loss_LUT")
    for j, level in enumerate(levels):
        zeros = [i for i in range(n) if not (j >> i)&1]
        for i in zeros:
            qc.x(i)
        for bit in range(bits):
            if (int(level) >> bit)&1:
                qc.mcx(list(range(n)), n+bit)
        for i in zeros:
            qc.x(i)
    return qc

