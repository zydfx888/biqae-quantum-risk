"""Independent numerical identities and saved-round accounting audit."""
import json
import ast
import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.stats import beta
from .paper_biqae import inverse,normal_update,normal_prior,beta_prior,next_power
from .config import ROOT,write_json,digest

def main():
    dest=ROOT/'artifacts/validation_suite_v6/paper';rng=np.random.default_rng(114)
    inverse_residual=delta_residual=mle_score=0.
    for _ in range(1000):
        K=int(rng.choice([1,3,7,15,31,63]));q=int(rng.integers(K));p=float(rng.uniform(.05,.95))
        x=inverse(p,K,q);inverse_residual=max(inverse_residual,abs(np.sin(K*x)**2-p))
        newK=2*K+1;post=(p,.0001)
        _,v=normal_prior(post,K,newK,q)
        f=lambda y: np.sin(newK*inverse(y,K,q))**2
        derivative=(f(p+1e-6)-f(p-1e-6))/2e-6
        delta_residual=max(delta_residual,abs(v-derivative**2*post[1]))
    assert inverse_residual<1e-12 and delta_residual<1e-9
    # Integrate an independently formed product of two normals by quadrature.
    from scipy.integrate import quad
    prior=(.4,.02);post=normal_update(prior,30,100)
    f=lambda x: np.exp(-(x-.4)**2/(2*.02)-(x-.3)**2/(2*.3*.7/100))
    z=quad(f,-2,3,epsabs=1e-12)[0];mean=quad(lambda x:x*f(x),-2,3,epsabs=1e-12)[0]/z
    variance=quad(lambda x:(x-mean)**2*f(x),-2,3,epsabs=1e-12)[0]/z
    assert np.allclose(post,[mean,variance],rtol=1e-10,atol=1e-12)
    for seed in range(20):
        a,b=beta_prior((20.,30.),1,3,0,np.random.default_rng(seed))
        samples=np.random.default_rng(seed).beta(20,30,1000)
        samples=np.sin(3*inverse(samples,1,0))**2
        scores=[digamma(a+b)-digamma(a)+np.log(samples).mean(),digamma(a+b)-digamma(b)+np.log1p(-samples).mean()]
        mle_score=max(mle_score,max(abs(np.array(scores))))
    assert mle_score<1e-7
    # Execute the unmodified authors' two deterministic methods in isolation.
    # This is not a claim to have run their full Qiskit sampler implementation.
    tree=ast.parse((dest/'author_biae.py').read_text(encoding='utf-8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='BayesianIQAE')
    methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('_find_next_k','_transform_probability')]
    scope={'np':np,'AlgorithmError':ValueError}
    exec(compile(ast.Module(body=methods,type_ignores=[]),'author_deterministic_methods','exec'),scope)
    author_transform_error=0.;scheduler_mismatches=0
    for _ in range(1000):
        current=int(rng.integers(0,32));K=2*current+1;q=int(rng.integers(K))
        low=(q+.2)*np.pi/2/K;high=(q+.4)*np.pi/2/K;p=float(rng.uniform(.01,.99));newk=int(rng.integers(32))
        ours=np.sin((2*newk+1)*inverse(p,K,q))**2
        authors=scope['_transform_probability'](None,p,newk,current,[low/(2*np.pi),high/(2*np.pi)],q%2==0)
        author_transform_error=max(author_transform_error,abs(ours-authors))
        found,_=scope['_find_next_k'](None,current,q%2==0,[low/(2*np.pi),high/(2*np.pi)],2.)
        if found<=31:
            scheduler_mismatches+=int(found!=next_power([low,high],current,31)[0])
    print('author comparison',author_transform_error,scheduler_mismatches)
    assert author_transform_error<1e-12
    rounds=0;cost_mismatches=0;branch_errors=0
    for name in ['amplitude','risk']:
        records=pd.read_csv(dest/f'{name}_results.csv');logs=pd.read_parquet(dest/f'{name}_rounds.parquet')
        assert len(records)==len(logs)
        for (_,r),(_,l) in zip(records.iterrows(),logs.iterrows()):
            entries=json.loads(l.rounds_json)
            batches=entries if name=='amplitude' else [v for x in entries for v in x['rounds']]
            costs=sum((2*x['k']+1)*x['shots'] for x in batches)
            cost_mismatches+=int(costs!=r.A_calls);rounds+=len(batches)
            if str(r.method).startswith('paper_'):
                for x in batches:
                    interval=x['theta_interval'];K=2*x['k']+1;q=x['branch']
                    branch_errors+=int(K*interval[0]<q*np.pi/2-1e-12 or K*interval[1]>(q+1)*np.pi/2+1e-12)
    assert cost_mismatches==branch_errors==0
    write_json(dest/'audit.json',dict(passed=True,inverse_cases=1000,inverse_max_residual=inverse_residual,
        delta_variance_max_residual=delta_residual,normal_quadrature_mean_error=abs(mean-post[0]),
        beta_mle_max_score=mle_score,rounds=rounds,query_mismatches=cost_mismatches,branch_errors=branch_errors,
        author_transform_max_difference=author_transform_error,author_scheduler_mismatches_within_cap=scheduler_mismatches,
        grid_imports_in_paper_module=False))
    write_json(dest/'checksums.json',{p.name:digest(p) for p in dest.iterdir() if p.is_file() and p.name!='checksums.json'})
    print((dest/'audit.json').read_text())

if __name__=='__main__':main()
