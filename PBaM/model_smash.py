"""
This module defines smash model classes
"""

from model_abstract import aModel
import numpy as np
import smash
from utilities_public import sigmoid

class smash_linearMapping(aModel):
    """
    Smash model with a linear descriptors-to-parameter mapping.
    Note that a sigmoid function is applied after the linear mapping to
    produce a parameter map included within specified bounds.
    Also note that there is no intercept in the linear mapping: if you want one, 
    you need to include a descriptor identically equal to 1.

    """
    
    def __init__(self,model:smash.Model,D:dict,bounds:dict):
        """
        Constructor of a smash_linearMapping model

        Parameters
        ----------        
        model: smash.Model object
            The model created using smash.
        D: dictionary
            The descriptors used to derive the map for each rr_parameter.            
            The keys should correspond to names of rr_parameters in the model.
            The values should be 3-D arrays, with dimensions (nX,nY,nDescriptors). 
            For instance `D['cp'][:,:,0]` is the map for the first descriptor used to compute rr_parameter 'cp'.
        bounds: Dictionary
            Lower and upper bounds for each rr_parameter.
            For instance: `bounds={'cp':(1e-06,1000.0),'ct':(1e-06, 1000.0),'kexc':(-50, 50),'llr':(1e-06, 1000.0)}`
        
        Returns
        -------
        A smash_linearMapping object 

        Examples
        --------
        TODO
        """
        self.model = model
        self.D = D
        self.bounds = bounds

    def npar(self):
        """
        Get the number of parameters of the model.

        Returns
        -------
        An integer.
        
        Examples
        --------
        TODO        
        """
        out = sum([v.shape[2] for v in self.D.values()])
        return out
    
    def run(self,theta:np.ndarray):
        """
        Run a smash_linearMapping model.

        Parameters
        ----------        
        theta: numeric 1-D array
            Vector of parameters of the model, corresponding to the coefficients of linear mappings, 
            ordered following the order used in descriptors dictionary D. 
            Note that there is no intercept in the linear mapping: if you want one, 
            you need to include a descriptor identically equal to 1.
        
        Returns
        -------
        A 2-D array with dimensions (nGauges,nT), containing the model-simulated discharge.

        Examples
        --------
        TODO
        """
        k=0
        for par,des in self.D.items():
            nd=des.shape[2] # number of descriptors
            coeff=theta[(k):(k+nd)] # coefficients
            foo=np.sum(np.multiply(coeff,des),axis=2) # linear combination
            res=sigmoid(foo,self.bounds[par]) # rescale to natural units
            self.model.set_rr_parameters(par,res) # set smash parameter map
            k=k+nd
            
        self.model.forward_run(common_options={'verbose':False}) # run smash
        out=self.model.response.q
        return out
    