"""
This module defines inference functions such as log-likelihoods, 
log-priors, log-posteriors etc.
"""

import model_abstract
import numpy as np
from scipy.stats import norm
import math
from typing import Callable as function_type

def llfunk_iid_Gaussian(Ysim:np.ndarray,Yobs:np.ndarray,Yu:np.ndarray,gamma:np.ndarray):
    """
    Computes the log-likelihood from model-simulated values, based on a Gaussian iid error model:
    * Yobs = Ysim + 𝛿 + 𝜀
    * Measurement errors: 𝛿~ N(0,sdev=Yu)
    * Structural errors: 𝜀~ N(0,sdev=𝛾)
    
    If Yobs/Ysim are multi-variate, this error model is applied independently to each component.

    Parameters
    ----------
    Ysim: 2-D array with dimensions (nOutputVariables,nValues)
        Simulated values. If the model has a single output, Ysim is an 'horizontal' vector
        with dimensions (1,nValues).

    Yobs: 2-D array with same dimensions as Ysim
        Corresponding observed values. np.nan values are skipped.

    Yu: 2-D array with same dimensions as Ysim and Yobs
        Measurement uncertainties (standard deviations).

    gamma: 1-D array
        Structural error parameters. length(gamma) = number of columns in Ysim.

    Returns
    -------
    A float equal to the log-likelihood.
        
    Examples
    --------
    Ysim=np.array([0.,1.,2.,3.],ndmin=2)
    Yobs=np.array([0.02,0.9,2.2,3.4],ndmin=2)
    Yu=0*Ysim
    gamma=np.array([0.1])
    llfunk_iid_Gaussian(Ysim,Yobs,Yu,gamma)
    """
    if any(gamma<=0):
        return -math.inf
    p=Ysim.shape[0]
    out=0
    for i in range(0,p):
        ps=norm.logpdf(x=Yobs[i,:],loc=Ysim[i,:],scale=np.sqrt(gamma[i]**2+Yu[i,:]**2))
        out=out+np.nansum(ps)    
    return out

def llfunk_iLinear_Gaussian(Ysim:np.ndarray,Yobs:np.ndarray,Yu:np.ndarray,gamma:np.ndarray):
    """
    Computes the log-likelihood from model-simulated values, based on a Gaussian error model
    with linearly-varying standard deviation:
    * Yobs = Ysim + 𝛿 + 𝜀
    * Measurement errors: 𝛿~ N(0,sdev=Yu)
    * Structural errors: 𝜀~ N(0,sdev=g1+g2*|Ysim|)
    
    If Yobs/Ysim are multi-variate, this error model is applied independently to each component.

    Parameters
    ----------
    Ysim: 2-D array with dimensions (nOutputVariables,nValues)
        Simulated values. If the model has a single output, Ysim is an 'horizontal' vector
        with dimensions (1,nValues).

    Yobs: 2-D array with same dimensions as Ysim
        Corresponding observed values. np.nan values are skipped.

    Yu: 2-D array with same dimensions as Ysim and Yobs
        Measurement uncertainties (standard deviations).

    gamma: 1-D array
        Structural error parameters, organized as: gamma=(g1,g2) for the 1st component of Ysim,
        (g1,g2) for the 2nd component of Ysim, etc. => length(gamma) = 2*(number of columns in Ysim).

    Returns
    -------
    A float equal to the log-likelihood.
        
    Examples
    --------
    Ysim=np.array([0.,1.,2.,3.],ndmin=2)
    Yobs=np.array([0.02,0.9,2.2,3.4],ndmin=2)
    Yu=0*Ysim
    gamma=np.array([0.01,0.1])
    llfunk_iLinear_Gaussian(Ysim,Yobs,Yu,gamma)
    """
    if any(gamma<0):
        return -math.inf
    p=Ysim.shape[0]
    out=0
    for i in range(0,p):
        g0=gamma[2*i]
        g1=gamma[2*i+1]
        m=Ysim[i,:]
        s=g0+g1*abs(m)
        if any(s<=0):
            return -math.inf
        ps=norm.logpdf(x=Yobs[i,:],loc=m,scale=np.sqrt(s**2+Yu[i,:]**2))
        out=out+np.nansum(ps)    
    return out

def llfunk_AR1Linear_Gaussian(Ysim:np.ndarray,Yobs:np.ndarray,Yu:np.ndarray,gamma:np.ndarray):
    """
    Computes the log-likelihood from model-simulated values, based on an AR1 Gaussian error model
    with linearly-varying standard deviation:
    * Yobs = Ysim + 𝛿 + 𝜀
    * Total errors: (𝛿 + 𝜀) ~ AR1(mu=0,sigma=sqrt(Yu^2+(g1+g2*|Ysim|)^2),rho)
    
    If Yobs/Ysim are multi-variate, this error model is applied independently to each component.

    Parameters
    ----------
    Ysim: 2-D array with dimensions (nOutputVariables,nValues)
        Simulated values. If the model has a single output, Ysim is an 'horizontal' vector
        with dimensions (1,nValues).

    Yobs: 2-D array with same dimensions as Ysim
        Corresponding observed values. np.nan values are skipped.

    Yu: 2-D array with same dimensions as Ysim and Yobs
        Measurement uncertainties (standard deviations).

    gamma: 1-D array
        Structural error parameters, organized as: gamma=(g1,g2,rho) for the 1st component of Ysim,
        (g1,g2,rho) for the 2nd component of Ysim, etc. => length(gamma) = 3*(number of columns in Ysim).

    Returns
    -------
    A float equal to the log-likelihood.
        
    Examples
    --------
    Ysim=np.array([0.,1.,2.,3.],ndmin=2)
    Yobs=np.array([0.02,0.9,2.2,3.4],ndmin=2)
    Yu=0*Ysim
    gamma=np.array([0.01,0.1,0.5])
    llfunk_AR1Linear_Gaussian(Ysim,Yobs,Yu,gamma)
    """
    if any(gamma<0):
        return -math.inf
    p,n=Ysim.shape
    out=0
    for i in range(0,p):
        g0=gamma[3*i]
        g1=gamma[3*i+1]
        rho=gamma[3*i+2]
        if abs(rho)>=1:
            return -math.inf
        s=g0+g1*abs(Ysim[i,:])
        if any(s<=0):
            return -math.inf
        sres=(Yobs[i,:]-Ysim[i,:])/np.sqrt(s**2+Yu[i,:]**2)
        ps=norm.logpdf(x=Yobs[i,1:n],loc=Ysim[i,1:n]+rho*sres[:(n-1)],scale=s[1:n])
        out=out+np.nansum(ps)    
    return out

def logLikelihood(parvector:np.ndarray,Yobs:np.ndarray,Yu:np.ndarray,
                  model:model_abstract.aModel,
                  llfunk:function_type=llfunk_iLinear_Gaussian,
                  Ysim:np.ndarray=None,**kwargs):
    """
    Log-likelihood engine for a model available as a subclass of abstract class aModel.
    Unlike functions llfunk_***, which compute the log-likelihood from already-simulated values Ysim, 
    this function runs the model internally.

    Parameters
    ----------
    parvector: 1-D array
        Parameter vector, including thetas (model parameters) and gammas (structural errors parameters).

    Yobs: 2-D array with dimensions (nOutputVariables,nValues)
        Observed values. np.nan values are skipped.

    Yu: 2-D array with same dimensions as Yobs
        Measurement uncertainties (standard deviations).

    model: Instance from model_abstract.aModel class
        Model to be calibrated.

    llfunk: function
        Function computing the log-likelihood given Ysim, typically one of llfunk_*** functions.
    
    Ysim: 2-D array with same dimensions as Yobs and Yu
        Model-simulated values. When NULL (default), the model is run internally to provide simulations.
        When a non-NULL array is provided, it is used as pre-computed simulations, and the model is
        hence not run within this function. This is useful to speed-up some MCMC strategies.

    **kwargs: dictionary
        Model-specific information passed to aModel.run() and aModel.npar()

    Returns
    -------
    ll: a float equal to the log-likelihood.
    Ysim: a 2D-array containing simulated values
        
    Examples
    --------
    TODO
    """
    # separate parvector into theta and gamma
    nTheta = model.npar(**kwargs)
    nPar=parvector.size
    # model parameters
    theta=parvector[0:nTheta] 
     # structural errors parameters
    if nTheta<nPar:
        gamma=parvector[nTheta:nPar]
    else:
        gamma=None
    if Ysim is None: # only run model if Ysim not provided as input argument
        # Run model
        Ysim=model.run(theta,**kwargs)
    if Ysim is None:
        return -math.inf,Ysim
    
    ll=llfunk(Ysim=Ysim,Yobs=Yobs,Yu=Yu,gamma=gamma)
    return ll,Ysim

def logPrior_Flat(parvector:np.ndarray):
    """
    Log-density for an improper flat distribution on all parameters.

    Parameters
    ----------
    parvector: 1-D array
        Parameter vector, including thetas (model parameters) and gammas (structural errors parameters).

    Returns
    -------
    A float equal to the prior log-density, always equal to zero here.
        
    Examples
    --------
    logPrior_Flat(np.array([0, 1]))
    """
    return 0.

def logPosterior(parvector:np.ndarray,Yobs:np.ndarray,Yu:np.ndarray,model:model_abstract.aModel,
                 llfunk:function_type=llfunk_iLinear_Gaussian,lpfunk:function_type=logPrior_Flat,
                 Ysim:np.ndarray=None,logLikelihood_engine:function_type=logLikelihood,**kwargs):
    """
    Log-density of the unnormalized posterior distribution for a model available as a subclass of abstract class aModel.

    Parameters
    ----------
    parvector: 1-D array
        Parameter vector, including thetas (model parameters) and gammas (structural errors parameters).

    Yobs: 2-D array with dimensions (nOutputVariables,nValues)
        Observed values. np.nan values are skipped.

    Yu: 2-D array with same dimensions as Yobs
        Measurement uncertainties (standard deviations).

    model: Instance from model_abstract.aModel class
        Model to be calibrated.

    llfunk: function
        Function computing the log-likelihood given Ysim, typically one of llfunk_*** functions.
    
    lpfunk: function
        Function computing the log-prior.

    Ysim: 2-D array with same dimensions as Yobs and Yu
        Model-simulated values. When NULL (default), the model is run internally to provide simulations.
        When a non-NULL array is provided, it is used as pre-computed simulations, and the model is
        hence not run within this function. This is useful to speed-up some MCMC strategies.

    logLikelihood_engine: function
        Function that runs the model and computes the log-likelihood (unlike functions llfunk_***, 
        which compute the log-likelihood from already-simulated values Ysim).

    **kwargs: dictionary
        Model-specific information passed to aModel.run() and aModel.npar()

    Returns
    -------
    lpost: a float equal to the unnormalized posterior log-density.
    lprior: a float equal to the prior log-density.
    ll: a float equal to the log-likelihood.
    Ysim: a 2D-array containing simulated values.
        
    Examples
    --------
    TODO
    """
    # start with the prior to exit early and cheaply if parvector is prior-incompatible
    lprior=lpfunk(parvector)
    if math.isinf(lprior):
        return -math.inf,-math.inf,None,None
    # finish
    ll,Ysim=logLikelihood_engine(parvector=parvector,Yobs=Yobs,Yu=Yu,model=model,llfunk=llfunk,Ysim=Ysim,**kwargs)
    lpost=lprior+ll
    return lpost,lprior,ll,Ysim

def minusLogPosterior(parvector:np.ndarray,Yobs:np.ndarray,Yu:np.ndarray,model:model_abstract.aModel,
                 llfunk:function_type=llfunk_iLinear_Gaussian,lpfunk:function_type=logPrior_Flat,
                 Ysim:np.ndarray=None,logLikelihood_engine:function_type=logLikelihood,**kwargs):
    """
    Minus one times the log-density of the unnormalized posterior distribution for a model available as a subclass of abstract class aModel.
    Used to be passed to minimizers.

    Parameters
    ----------
    parvector: 1-D array
        Parameter vector, including thetas (model parameters) and gammas (structural errors parameters).

    Yobs: 2-D array with dimensions (nOutputVariables,nValues)
        Observed values. np.nan values are skipped.

    Yu: 2-D array with same dimensions as Yobs
        Measurement uncertainties (standard deviations).

    model: Instance from model_abstract.aModel class
        Model to be calibrated.

    llfunk: function
        Function computing the log-likelihood given Ysim, typically one of llfunk_*** functions.
    
    lpfunk: function
        Function computing the log-prior.

    Ysim: 2-D array with same dimensions as Yobs and Yu
        Model-simulated values. When NULL (default), the model is run internally to provide simulations.
        When a non-NULL array is provided, it is used as pre-computed simulations, and the model is
        hence not run within this function. This is useful to speed-up some MCMC strategies.

    logLikelihood_engine: function
        Function that runs the model and computes the log-likelihood (unlike functions llfunk_***, 
        which compute the log-likelihood from already-simulated values Ysim).

    **kwargs: dictionary
        Model-specific information passed to aModel.run() and aModel.npar()

    Returns
    -------
    a float equal to the unnormalized posterior log-density.
        
    Examples
    --------
    TODO
    """
    lpost,lprior,ll,Ysim=logPosterior(parvector,Yobs,Yu,model,llfunk,lpfunk,
                 Ysim,logLikelihood_engine,**kwargs)
    return -1*lpost
