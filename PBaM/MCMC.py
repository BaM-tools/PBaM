"""
This module implements MCMC samplers
"""
import numpy as np
import math
from typing import Callable as function_type
from random import gauss,uniform
from tqdm import tqdm
from scipy.stats import norm

def MCMC_options(nAdapt:int=50,nCycles:int=20,
                 minMoveRate:float=0.2,maxMoveRate:float=0.5,downMult:float=0.8,upMult:float=1.2,
                 burnCov:float=0.2,dofCovMin:int=5,nCovMax:int=10000,
                 C0_factor:float=0.02,C0_eps:float=0.1):
    """
    MCMC options.

    Parameters
    ----------        
    nAdapt: integer
        Number of iterations before adapting jump properties (covariance C and scaleFactor). 
    
    nCycles: integer
        Number of adaption cycles. Total number of iterations is hence equal to nAdapt*nCycles.
        nCycles=1 leads to the standard non-adaptive Metropolis sampler.

    minMoveRate: float in (0;1)
        Lower bound for the desired move rate interval.

    maxMoveRate: float in (0;1)
        Upper bound for the desired move rate interval.

    downMult: float  in (0;1)
        Multiplicative factor used to decrease scaleFactor when move rate is too low.

    upMult: float (>1, avoid 1/downMult)
        Multiplicative factor used to increase scaleFactor when move rate is too high.

    burnCov: float  in (0;1)
        Fraction of initial values to be discarded before computing the empirical covariance of
        sampled vectors, which is used to adapt the jump covariance.

    dofCovMin: integer
        Minimum number of degrees of freedom required to compute the empirical covariance of
        sampled vectors and hence to adapt the jump covariance. If D denotes the length of x0, at least
        dofCovMin*(D+0.5*(D-1)*(D-2)) iterations are required before adapting the jump covariance
        (i.e. dofCovMin times the number of unknown elements in the covariance matrix).

    nCovMax: integer
        Maximum number of iterations used to compute the empirical covariance.
        If the number of available iterations is larger than nCovMax, iterations are 'slimmed' to reach nCovMax.

    C0_factor: float
        When not provided, the initial covariance of the Gaussian jump distribution, C0, will be equal to:
        C0=np.diag((C0_factor*(abs(x0)+C0_eps))**2).

    C0_eps: float
        When not provided, the initial covariance of the Gaussian jump distribution, C0, will be equal to:
        C0=np.diag((C0_factor*(abs(x0)+C0_eps))**2).

    Returns
    -------
    A dictionary containing the MCMC options

    Examples
    --------
    MCMC_options(nAdapt=100,nCycles=100)
    """
    options={'nAdapt':nAdapt,'nCycles':nCycles,'minMoveRate':minMoveRate,'maxMoveRate':maxMoveRate,
             'downMult':downMult,'upMult':upMult,'burnCov':burnCov,'dofCovMin':dofCovMin,'nCovMax':nCovMax,
             'C0_factor':C0_factor,'C0_eps':C0_eps}
    return options

def MCMC_AM(logPdf:function_type,x0:np.ndarray,
            C0:np.ndarray | None=None,scaleFactor:float | None=None,
            options:dict=MCMC_options(),**kwargs):
    """
    An adaptive Metropolis sampler largely inspired by Haario et al. (2001, https://doi.org/10.2307/3318737).
    The jump covariance is adapted using the empirical covariance of previously-sampled values,
    and the scaling factor is adapted in order to comply with a specified move rate interval.

    Parameters
    ----------        
    logPdf: function
        Function evaluating the log-density of the distribution to sample from (up to a proportionality constant).
        logPdf can return either a single numeric value (interpreted as the target log-pdf),
        or a tuple of 3 values (interpreted as log-post, log-prior and log-likelihood), 
        or a tuple of 4 values (interpreted as log-post, log-prior, log-likelihood and simulated model outputs). 
    
    x0: 1D array
        Starting point.

    C0: 2D array
        Initial covariance matrix of the Gaussian jump distribution (up to a scale factor, see next).

    scaleFactor: float
        Value used to scale the jump covariance. The covariance of the jump distribution is equal to (scaleFactor^2)*C0.

    options: dictionary
        MCMC options, see function MCMC_options()

    **kwargs: dictionary
        Additional information passed to function logPdf.
        
    Returns
    -------
    samples: 2D array
        Simulated MCMC samples.
        
    comps: 2D array (3 columns)
        Corresponding values for logpost, logprior and loglikelihood.
          
    C: 2D array
        Adapted covariance matrix of the jump distribution.
        
    scaleFactor: float
        Adapted scaling factor for the jump covariance.

    Examples
    --------
    # Define a 2-dimensional target log-pdf
    def logPdf(x):
        p1=np.log(0.6*norm.pdf(x[0],0,1)+0.4*norm.pdf(x[0],2,0.5)) # mixture of 2 Gaussians
        p2=norm.logpdf(x[1],0,1) # standard Gaussian
        return p1+p2

    # Sample from it
    samples,comps,C,sd=MCMC_AM(logPdf,x0=np.array([1,1]))
    """
    # Set up
    D=len(x0)
    if C0 is None:
       C0=np.diag((options['C0_factor']*(abs(x0)+options['C0_eps']))**2)
    if scaleFactor is None:
       scaleFactor=2.4/math.sqrt(D)
    samples=np.nan*np.ones(shape=(options['nAdapt']*options['nCycles']+1,D)) # samples
    comps=np.nan*np.ones(shape=(options['nAdapt']*options['nCycles']+1,3)) # components: post, prior, lkh\
    # Starting value and posterior
    k=0
    samples[k,:]=x0.copy()
    fx=_getComponents(logPdf(x0,**kwargs))
    lpost=fx[0]
    if math.isinf(lpost) or math.isnan(lpost) or lpost is None: # unfeasible starting point, abandon
        raise ValueError('Unfeasible starting point x0')
    comps[k,:]=fx[0:3]
    # Jump covariance
    C=C0.copy()
    nelement=int(D+0.5*(D-1)*(D-2)) # number of elements in covariance matrix
    nmin=int(nelement*options['dofCovMin'])
    nmax=max(nmin,options['nCovMax'])
    # Start iterations
    for j in tqdm(range(options['nCycles']),position=0, desc='j', leave=False, colour='blue',):
        jumps=np.random.multivariate_normal(mean=np.zeros(D),cov=(scaleFactor**2)*C,size=options['nAdapt'])
        move=np.full(options['nAdapt'],False)
        for i in tqdm(range(options['nAdapt']),position=1, desc='i', leave=False, colour='cyan'):
            k=k+1
            candid=samples[k-1,:]+jumps[i,:]
            fcandid=_getComponents(logPdf(candid,**kwargs))
            # Apply Metropolis rule
            foo=_applyMetropolisRule(x=samples[k-1,],candid=candid,fx=fx,fcandid=fcandid)
            samples[k,]=foo[0].copy()
            fx=foo[1]
            comps[k,]=fx[0:3]
            move[i]=foo[2]
        # Adapt scale factor
        mr=np.mean(move)
        if mr<options['minMoveRate']:
            scaleFactor=scaleFactor*options['downMult']
        elif mr>options['maxMoveRate']:
            scaleFactor=scaleFactor*options['upMult']
        # Adapt covariance
        # 2DO: recursive formula ?
        n0=max([int(options['burnCov']*k),0])
        nval=k-n0
        if nval>nmin: # enough values to update Cov
            ix=np.unique(np.linspace(start=n0,stop=k,num=min(nval,nmax)).astype(int))
            varis=np.apply_along_axis(np.var,0,samples[ix,:])
            if min(varis)>0:
                C=np.cov(samples[ix,:],rowvar=False)
        elif nval>(D*options['dofCovMin']): # enough values to estimate a diagonal Cov
            varis=np.apply_along_axis(np.var,0,samples[n0:k,:])
            if min(varis)>0:
                C=np.diag(varis)
    return samples,comps,C,scaleFactor

def _getComponents(obj:tuple,returnSim=False):
    """
    Utility function to re-interpret the output of the target logPdf function in terms of 
    4 components: lpost, lprior, ll, Ysim.

    Parameters
    ----------        
    obj: tuple, returned by logPdf function.
    returnSim: boolean, return Ysim if available?
        
    Returns
    -------
    lpost: float, posterior log-pdf
    lprior:float, prior log-pdf
    ll:float, log-likelihood
    Ysim: object, simulated outputs 

    Examples
    --------
    _getComponents((-100,-10,-110))
    _getComponents(-100.)
    """
    if type(obj)==tuple:
        D=len(obj)
    else:
        D=1
    Ysim=np.nan
    if D==4: # function returns posterior, prior, ll, Ysim
        lpost=obj[0];lprior=obj[1];ll=obj[2]
        if returnSim:
            Ysim=obj[3]
    elif D==1: # function only return posterior
        lpost=obj;lprior=np.nan;ll=np.nan
    elif D==3: # function returns posterior, prior, ll
        lpost=obj[0];lprior=obj[1];ll=obj[2]
    else: # can't interpret 
        lpost=np.nan;lprior=np.nan;ll=np.nan;Ysim=np.nan
    return lpost,lprior,ll,Ysim

def _applyMetropolisRule(x:np.ndarray,candid:np.ndarray,fx:tuple,fcandid:tuple):
    """
    Apply Metropolis rule

    Parameters
    ----------        
    x: 1D array, current sample.
    candid: 1D array, candidate sample.
    fx: tuple, with fx[0] equal to the target log-pdf at x.
    fcandid: tuple, with fcandid[0] equal to the target log-pdf at candid.
        
    Returns
    -------
    x: a 1D array equal to the selected sample.
    fx: a tuple equal to the selected fx.
    move: a boolean equal to True if the selected sample is the candidate, False otherwise.

    Examples
    --------
    _applyMetropolisRule(x=np.array([0,1]),candid=np.array([1,1]),fx=(10,0,10),fcandid=(9.5,1,8.5))
    """
    move=False
    # if NA or -Inf, reject candid
    if math.isinf(fcandid[0]) or math.isnan(fcandid[0]) or (fcandid[0] is None): 
        return x,fx,move
    # Metropolis rule
    try:
        ratio=math.exp(fcandid[0]-fx[0])
    except OverflowError:
            ratio=math.inf
    u=uniform(0,1) # throw the dice
    if u<=ratio: # accept
        move=True
        x=candid.copy()
        fx=fcandid
    return x,fx,move

