"""
This module contains various public utilities that are meant to be exported
"""

import numpy as np
from typing import Callable as function_type
from inferenceFunk import llfunk_iid_Gaussian, llfunk_iLinear_Gaussian, llfunk_AR1Linear_Gaussian
from scipy.optimize import minimize

def rescale(x:np.ndarray,bounds:tuple=(0,1)):
    """
    Rescale an array between bounds.

    Parameters
    ----------
    x: array
        Array to be rescaled.

    bounds: tuple 
        bounds.

    Returns
    -------
    An array equal to the rescaled x.
        
    Examples
    --------
    x=np.array([1.,4.,8.,10.]) 
    rescale(x,bounds=(100,120))
    """
    mini=np.min(x)
    maxi=np.max(x)
    return bounds[0]+(bounds[1]-bounds[0])*((x-mini)/(maxi-mini))

def sigmoid(x:np.ndarray,bounds:tuple=(0,1)):
    """
    Transform an array of floats by a sigmoid function rescaled between bounds.

    Parameters
    ----------
    x: array
        Array to be transformed.

    bounds: tuple 
        bounds.

    Returns
    -------
    An array equal to the transformed x.
        
    Examples
    --------
    x=np.array([-5,0,0.1,8,10,100]) 
    sigmoid(x,bounds=(100,120))
    """
    res=bounds[0]+(bounds[1]-bounds[0])/(1+np.exp(-1*x))
    return res

def sigmoidInv(x:np.ndarray,bounds:tuple=(0,1)):
    """
    Transform an array of numbers included within bounds by the inverse sigmoid function.

    Parameters
    ----------
    x: array
        Array to be transformed.

    bounds: tuple 
        bounds.

    Returns
    -------
    An array equal to the transformed x.
        
    Examples
    --------
    x=np.array([100,110,119,119.99,120,121]) 
    sigmoidInv(x,bounds=(100,120))
    """
    res=-1*np.log(((bounds[1]-bounds[0])/(x-bounds[0]))-1.)
    return res

def decompose_covariance(covar:np.ndarray):
    """
    Decompose a covariance matrix into a vector of standard deviations and a correlation matrix.

    Parameters
    ----------
    covar: 2D array, the covariance matrix to decompose.

    Returns
    -------
    sdev: 1D array, standard deviations
    correl: 2D array, correlation matrix
        
    Examples
    --------
    x=np.array([[10,4],[4,8]])
    sdev,correl=decompose_covariance(x)
    """
    sdev=(np.sqrt(np.diag(covar))) 
    dmat=np.diag(sdev) 
    dinv=np.linalg.inv(dmat)
    correl=np.matmul(np.matmul(dinv,covar),dinv)
    return sdev,correl

def estimate_gamma(Ysim:np.ndarray,Yobs:np.ndarray,Yu:np.ndarray,
                  llfunk:function_type=llfunk_iLinear_Gaussian,
                  gamma0:np.ndarray=None,
                  **kwargs):
    """
    Estimate the structural errors parameters.

    Parameters
    ----------
    Ysim: 2-D array with dimensions (nOutputVariables,nValues)
        Simulated values. If the model has a single output, Ysim is an 'horizontal' vector
        with dimensions (1,nValues).

    Yobs: 2-D array with same dimensions as Ysim
        Corresponding observed values.

    Yu: 2-D array with same dimensions as Ysim and Yobs
        Measurement uncertainties (standard deviations).
    
    llfunk: function
        Function computing the log-likelihood given Ysim, typically one of llfunk_*** functions.

    gamma0: 1-D array
        Initial guess for the structural error parameters. length(gamma) depends on the selected llfunk.
        If None, a default initial guess is computed internally.
    
    **kwargs: dictionary
        Other arguments passed to function scipy.optimize.minimize


    Returns
    -------
    The OptimizeResult object returned by scipy.optimize.minimize
        
    Examples
    --------
    rng = np.random.default_rng()
    n=100
    X=rng.normal(size=n)
    Yobs=np.reshape(2*X+rng.normal(scale=0.1,size=n),(1,n))
    Yu=0.1+0*Yobs
    Ysim=np.reshape(2*X+rng.normal(scale=0.2+abs(0.2*X),size=n),(1,n))
    foo=estimate_gamma(Ysim,Yobs,Yu)
    foo.success
    foo.x
    """   
    if gamma0 is None:
        if llfunk==llfunk_iid_Gaussian:
            gamma0=np.reshape(np.nanstd(Yobs,axis=1),Yobs.shape[0])
        elif llfunk==llfunk_iLinear_Gaussian:
            gamma0=np.reshape(np.dstack((np.nanstd(Yobs,axis=1),
                                         np.zeros(shape=Yobs.shape[0]))),
                                         2*Yobs.shape[0])
        elif llfunk==llfunk_AR1Linear_Gaussian:
            gamma0=np.reshape(np.dstack((np.nanstd(Yobs,axis=1),
                                         np.zeros(shape=Yobs.shape[0]),
                                         np.zeros(shape=Yobs.shape[0]))),
                                         3*Yobs.shape[0])
        else:
            raise ValueError('No default gamma0 is implemented for this llfunk. Please provide your own gamma0')
        
    def minusFunk(x,llfunk,Ysim,Yobs,Yu):
        out=llfunk(gamma=x,Ysim=Ysim,Yobs=Yobs,Yu=Yu)
        return -1*out
    
    out=minimize(fun=minusFunk,x0=gamma0,args=(llfunk,Ysim,Yobs,Yu),method='Nelder-Mead')
    return out