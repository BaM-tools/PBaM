"""
This module defines the abstract aModel class
"""

from abc import ABC, abstractmethod
import numpy as np

class aModel(ABC):
    """
    Abstract class representing a model - any model.
    Any subclass should implement the following methods: 
    * run(theta,**kwargs) to run the model
    * npar(**kwargs) to retrieve the number of parameters of the model
    """
    
    @abstractmethod
    def npar(self,**kwargs):
        """
        Get the number of parameters of the model.

        Parameters
        ----------        
        **kwargs: Dictionary
            Model-specific information.
        
        Returns
        -------
        An integer.
        """
        pass
    
    @abstractmethod
    def run(self,theta:np.ndarray,**kwargs):
        """
        Run the model.

        Parameters
        ----------        
        theta: numeric 1-D array
            Vector of parameters of the model.
        **kwargs: Dictionary
            Model-specific information beyond its parameter vector.
        
        Returns
        -------
        A model-specific object 
        """
        pass