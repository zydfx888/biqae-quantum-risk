"""Reversible integer weighted return arithmetic; never loads portfolio losses."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, grover_operator
from .state_preparation import probability_circuit

def toggle_pattern(qc,controls,target,value):
    zeros=[q for j,q in enumerate(controls) if not (value>>j)&1]
    for q in zeros:qc.x(q)
    qc.mcx(controls,target)
    for q in zeros:qc.x(q)

def add_power(qc,reg,shift,controls=()):
    # Carry computed from pre-increment lower bits, highest target first.
    for target in range(len(reg)-1,shift,-1):
        qc.mcx(list(controls)+reg[shift:target],reg[target])
    if controls:qc.mcx(list(controls),reg[shift])
    else:qc.x(reg[shift])

def add_constant(qc,reg,value,controls=()):
    value=int(value)%(2**len(reg))
    for bit in range(len(reg)):
        if (value>>bit)&1:add_power(qc,reg,bit,controls)

def greater_equal(qc,reg,target,threshold):
    # Disjoint binary-prefix decomposition of [threshold, 2**bits).
    if threshold<=0:qc.x(target);return
    if threshold>=2**len(reg):return
    toggle_pattern(qc,reg,target,threshold)
    for bit in range(len(reg)):
        if not (threshold>>bit)&1:
            controls=reg[bit:]
            pattern=((threshold>>(bit+1))<<1)|1
            toggle_pattern(qc,controls,target,pattern)

class ArithmeticOracle:
    def __init__(self,p,asset_codes,integer_weights,threshold,stage,loss_bits=5):
        self.p=np.asarray(p);self.codes=np.asarray(asset_codes,dtype=int)
        self.weights=np.asarray(integer_weights,dtype=int);self.threshold=int(threshold)
        self.n=int(np.ceil(np.log2(len(p))));self.flag=self.n+2+loss_bits+1
        count=self.flag+1;index=list(range(self.n));returns=list(range(self.n,self.n+2));loss=list(range(self.n+2,self.n+2+loss_bits));tail=self.flag-1
        self.components={}
        load=QuantumCircuit(count,name='probability_loading');load.compose(probability_circuit(p),index,inplace=True)
        self.components['probability_loading']=load
        arithmetic=QuantumCircuit(count,name='return_qROM_weighted_arithmetic')
        for asset,weight in enumerate(self.weights):
            qrom=QuantumCircuit(count,name=f'asset_{asset}_return_qROM')
            for state,value in enumerate(self.codes[:,asset]):
                for bit in range(2):
                    if (value>>bit)&1:toggle_pattern(qrom,index,returns[bit],state)
            arithmetic.compose(qrom,inplace=True)
            for bit,control in enumerate(returns):add_constant(arithmetic,loss,int(weight)*(2**bit),[control])
            arithmetic.compose(qrom.inverse(),inplace=True)
        self.components['return_qROM_weighted_arithmetic']=arithmetic
        encode=QuantumCircuit(count,name='compare_excess_encode_uncompute')
        if stage=='var':
            greater_equal(encode,loss,self.flag,self.threshold+1);encode.x(self.flag)
        else:
            greater_equal(encode,loss,tail,self.threshold+1)
            add_constant(encode,loss,-self.threshold)
            bound=int(np.ptp(self.codes@self.weights))
            if bound:
                for value in range(1,min(bound,2**loss_bits-1)+1):
                    zeros=[q for bit,q in enumerate(loss) if not (value>>bit)&1]
                    for q in zeros:encode.x(q)
                    encode.append(RYGate(2*np.arcsin(np.sqrt(value/bound))).control(loss_bits+1,annotated=True),loss+[tail,self.flag])
                    for q in zeros:encode.x(q)
            add_constant(encode,loss,self.threshold)
            greater_equal(encode,loss,tail,self.threshold+1)
        self.components['compare_excess_encode_uncompute']=encode
        self.components['arithmetic_uncompute']=arithmetic.inverse()
        self.circuit=QuantumCircuit(count,name='A_arithmetic')
        for qc in self.components.values():self.circuit.compose(qc,inplace=True)
        values=self.codes@self.weights
        self.payload=(values<=threshold).astype(float) if stage=='var' else np.maximum(values-threshold,0)/max(1,np.ptp(values))
    def amplified(self,k):
        qc=self.circuit.copy()
        if k:
            marker=QuantumCircuit(qc.num_qubits);marker.z(self.flag)
            grover=grover_operator(marker,state_preparation=self.circuit,reflection_qubits=list(range(self.n))+[self.flag])
            for _ in range(k):qc.compose(grover,inplace=True)
        return qc
