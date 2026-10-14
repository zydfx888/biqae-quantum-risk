import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from q3_biqae.arithmetic_oracle import add_constant,greater_equal

def test_controlled_integer_multiply_add_exhaustive():
    for value in range(32):
      for control in [0,1]:
        for constant in [1,2,3,6,-7]:
          q=QuantumCircuit(6)
          for bit in range(5):
            if (value>>bit)&1:q.x(bit)
          if control:q.x(5)
          add_constant(q,list(range(5)),constant,[5])
          expected=((value+control*constant)%32)+32*control
          assert abs(Statevector.from_instruction(q).data[expected])>1-1e-10

def test_binary_prefix_comparator_exhaustive():
    for threshold in range(34):
      for value in range(32):
        q=QuantumCircuit(6)
        for bit in range(5):
          if (value>>bit)&1:q.x(bit)
        greater_equal(q,list(range(5)),5,threshold)
        expected=value+32*(value>=threshold)
        assert abs(Statevector.from_instruction(q).data[expected])>1-1e-10
