"""Run the authors' original FindNextK method unchanged, with an external cap."""
import ast
import numpy as np
from .config import ROOT

SOURCE=ROOT/'artifacts/validation_suite_v6/paper/author_biae.py'
tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='BayesianIQAE')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_find_next_k')
scope={'np':np,'AlgorithmError':ValueError}
# Execute only this deterministic method; no Qiskit imports or other code.
exec(compile(ast.Module(body=[method],type_ignores=[]),str(SOURCE),'exec'),scope)
AUTHOR=scope['_find_next_k']

def raw(theta,current):
    q=int(np.floor((2*current+1)*theta[0]/(np.pi/2)))
    return AUTHOR(None,current,q%2==0,[float(theta[0]/(2*np.pi)),float(theta[1]/(2*np.pi))],2.)

def next_power(theta,current,maximum):
    k,_=raw(theta,current)
    # Reject a proposed circuit beyond the external cap; keep sampling the
    # current stage. Do not substitute a different, non-author-selected power.
    if k>maximum or k==current:
        return current,None
    return k,int(np.floor((2*k+1)*theta[0]/(np.pi/2)))
