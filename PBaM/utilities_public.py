"""
This module contains various public utilities that are meant to be exported
"""

import numpy as np


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