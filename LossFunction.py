"""
Loss Function Module
====================
This module contains the loss function class for the FSP optimization problem.
"""
#Libraries
import pandas as pd
class Evaluator:
    """
    Class for evaluating the subset of features.
    """
    def __init__(self, data, loss_function: str = 'r2'):
        """
        Constructor for the loss function
        
        Parameters
        ----------
        data : pandas.DataFrame
            The data for the loss function.
        """
        # Load the data
        self.data = data
        
        # Set the loss function
        if loss_function == 'r2':
            self.loss_function = self.r2
        elif loss_function == 'rmse':
            self.loss_function = self.rmse
        else:
            raise ValueError('Invalid loss function.')
        
    def subset(self, features) -> pd.DataFrame:
        """
        Get a subset of the data based on the features.
        """ 
        # Return the subset of the data
        return self.data[features]
    
    def evaluate(self, features)
        """
        Evaluate the loss function for a subset of features.
        """
        # Get the subset of the data
        subset = self.subset(features)
        y_true = subset['y']
        y_pred = subset['y_pred']
        # Evaluate the loss function
        return self.loss_function(subset)
    
    
    # Loss Functions
    def r2(y_true, y_pred):
        """
        R2 Loss Function
        """
        return 1 - ((y_true - y_pred)**2).sum() / ((y_true - y_true.mean())**2).sum()
    
    def rmse(y_true, y_pred):
        """
        RMSE Loss Function
        """
        return ((y_true - y_pred)**2).mean()**0.5