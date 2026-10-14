"""Independent implementation of arXiv:2507.23074v2 Algorithms 1--6.

No posterior grid. Normal uses the paper's normal likelihood approximation;
Beta transports 1000 posterior draws and fits by maximum likelihood.
Budget and Grover caps are external experimental limits, not paper theorems.
"""
import numpy as np
from scipy.stats import beta, norm
from .amplitude import AmplitudeResult
from .statistics import clopper_pearson


def inverse(p, K, quadrant):
    angle = np.arcsin(np.sqrt(np.clip(p, 0, 1)))
    if quadrant % 2:
        angle = np.pi/2-angle
    return (angle+quadrant*np.pi/2)/K


def normal_update(prior, hits, shots):
    mean, variance = prior
    sample = hits/shots
    likelihood_variance = sample*(1-sample)/shots
    # Continuous zero-variance limit of Eq.20; deliberately no smoothing.
    if likelihood_variance == 0:
        return sample, 0.
    precision = 0. if np.isinf(variance) else 1/variance
    posterior_variance = 1/(precision+1/likelihood_variance)
    return posterior_variance*(mean*precision+sample/likelihood_variance), posterior_variance


def normal_prior(post, K, next_K, quadrant):
    mean, variance = post
    new_mean = float(np.sin(next_K*inverse(mean,K,quadrant))**2)
    if not 0 < mean < 1:
        raise ValueError('Normal delta-method requires an interior posterior mean')
    return new_mean, (next_K/K)**2*new_mean*(1-new_mean)/(mean*(1-mean))*variance


def beta_prior(post, K, next_K, quadrant, rng):
    draws = rng.beta(*post, size=1000)
    transported = np.sin(next_K*inverse(draws,K,quadrant))**2
    # Endpoint protection is floating point only, not pseudo counts.
    transported = np.clip(transported,np.nextafter(0.,1.),np.nextafter(1.,0.))
    a,b,_,_ = beta.fit(transported,floc=0,fscale=1)
    return (float(a),float(b))


def next_power(theta, current, maximum):
    """Largest legal standard-schedule odd K, minimum growth ratio two."""
    width = theta[1]-theta[0]
    top = min(maximum,int((np.pi/(2*width)-1)/2)) if width>0 else maximum
    for k in range(top,current,-1):
        K=2*k+1
        if K<2*(2*current+1):
            break
        q=int(np.floor(K*theta[0]/(np.pi/2)))
        if K*theta[1] <= (q+1)*np.pi/2+1e-14:
            return k,q
    return current,None


def estimate(measurement, config, budget, delta, epsilon, method, prior_seed=0):
    theta=np.array([0.,np.pi/2]);ci=np.array([0.,1.])
    k=0;quadrant=0;remaining=budget;hits=shots=stage=0
    is_normal=method=='paper_normal'
    prior=(.5,float('inf')) if is_normal else (.5,.5)
    post=prior; rng=np.random.default_rng(prior_seed)
    max_stages=max(1,int(np.log(2*np.pi/(8*epsilon))/np.log(2))+1)
    local_delta=delta/max_stages
    for iteration in range(config.max_rounds):
        new_k,new_q=next_power(theta,k,config.max_grover_power)
        if new_k!=k and remaining>=2*new_k+1:
            prior=(.5,.5) if method=='paper_iqae_cp' else (normal_prior(post,2*k+1,2*new_k+1,quadrant) if is_normal else beta_prior(post,2*k+1,2*new_k+1,quadrant,rng))
            k,quadrant=new_k,new_q;hits=shots=0;stage+=1
        n=min(config.shots_per_round,remaining//(2*k+1))
        if n<1: break
        hits+=measurement.sample(k,n);shots+=n;remaining-=n*(2*k+1)
        if method=='paper_iqae_cp':
            post=(prior[0]+hits,prior[1]+shots-hits)
            pci=clopper_pearson(hits,shots,local_delta)
        elif is_normal:
            post=normal_update(prior,hits,shots)
            pci=np.clip(post[0]+np.array([-1,1])*norm.ppf(1-local_delta/2)*np.sqrt(post[1]),0,1)
        else:
            post=(prior[0]+hits,prior[1]+shots-hits)
            pci=beta.ppf([local_delta/2,1-local_delta/2],*post)
        theta=np.sort(inverse(pci,2*k+1,quadrant));ci=np.sin(theta)**2
        measurement.rounds[-1].update(stage=stage,stage_hits=hits,stage_shots=shots,branch=quadrant,
            prior=[None if np.isinf(x) else float(x) for x in prior],posterior=list(map(float,post)),
            theta_interval=theta.tolist(),amplitude_interval=ci.tolist(),local_delta=local_delta,
            normal_zero_variance=bool(is_normal and post[1]==0))
        if (ci[1]-ci[0])/2<=epsilon: break
    measurement.finalize_resources()
    kind='paper_iqae_cp_stage_spending' if method=='paper_iqae_cp' else method+'_paper_credible'
    return AmplitudeResult(float(np.mean(ci)),tuple(map(float,ci)),kind,
        bool((ci[1]-ci[0])/2<=epsilon),measurement.ledger,measurement.rounds)
