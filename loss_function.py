"""
Loss Function Module
====================
This module contains the Evaluator class for the FSP optimization problem.
    >Note: The warnings are disabled because the ridge model throws a warning on every fit.
    >Note: The regression models settings are taken from the paper "P. Cunningham, B. Kathirgamaranathan, and S. J. Delany, ''Feature Selection Tutorial with Python Examples'', arXiv:2106.06437 [cs.LG], Jun. 2021."
"""
#Libraries
import pandas as pd

#Regression models
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

#Disable warnings
import warnings
warnings.filterwarnings("ignore")

#Data split
from sklearn.model_selection import train_test_split

class Evaluator:
    """
    Class for evaluating the subset of features.
    """
    def __init__(self, data_path: str, categorical_encoding: str, regression_model:str , loss_function: str) -> None:
        """
        Constructor for the loss function
        
        Parameters
        ----------
        data : str
            Path to the csv file containing the data.
        categorical_encoding : str
            The type of encoding to use for the categorical columns.
        regression_model : str
            The regression model to fit.
        loss_function : str
            The loss function to use.
        """        
        # Set the loss function
        if loss_function == 'r2':
            self.loss_function = self.r2
        elif loss_function == 'rmse':
            self.loss_function = self.rmse
        elif loss_function == 'combined':
            self.loss_function = self.combined
        else:
            raise ValueError('Invalid loss function.')
        
        # Set the regression model
        if regression_model == 'ridge':
            self.regression_model = Ridge (alpha=1.0, copy_X=True, fit_intercept=True, 
                                           max_iter=None, random_state=None, 
                                           solver='auto', tol=0.001)
            
        elif regression_model == 'lasso':
            self.regression_model = Lasso(alpha=1.0, copy_X=True, fit_intercept=True, 
                                          max_iter=1000, positive=False, 
                                          random_state=None,selection='cyclic', 
                                          tol=0.0001, warm_start=False)
        
        elif regression_model == 'linear':
            self.regression_model = LinearRegression(copy_X=True, fit_intercept=True, 
                                                     n_jobs=1)
        
        else:
            raise ValueError('Invalid regression model.')
        
        #Load the data
        self.load_data(data_path, categorical_encoding)
        
    def load_data(self, data_path: str, categorical_encoding: str = None):
        """
        Load the data from the csv file.
        """
        #Load the csv
        df = pd.read_csv(data_path, index_col='id')
        
        #Find the numeric and categorical columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        if categorical_encoding == 'target':
            #Apply target encoding to the categorical columns
            for column in categorical_columns:
                df[column] = df.groupby(column)['price'].transform('mean')
        elif categorical_encoding == 'label':
            #Apply label encoding to the categorical columns
            for column in categorical_columns:
                df[column] = df.groupby(column).ngroup()   
        else:
            #Drop the non-numeric columns
            df = df[numeric_columns]
            
        #Fill the missing values with 0
        for column in df.columns:
            df[column] = df[column].fillna(0)
            
        #Split the data
        Y = df['price']
        X = df.drop(columns=['price'])
                
        self.features = X.columns
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
    def evaluate(self, features: list) -> float:
        """
        Evaluate the loss function for a subset of features.
        
        Parameters
        ----------
        features : list
            The subset of features to evaluate coded as a binary array.
        """
        #Turn features into a boolean arrat umbralized at 0.5
        features = [feature > 0.5 for feature in features]
        
        #If no features are selected, return None
        if not any(features):
            return None
        
        #Get the selected features
        features = [feature for feature, include in zip(self.features, features) if include]
        X_train = self.x_train[features]

        #Fit the model
        self.regression_model.fit(X_train, self.y_train)
        
        #Evaluate the model
        X_test = self.x_test[features]
        y_pred = self.regression_model.predict(X_test)
        
        return self.loss_function(self.y_test, y_pred)  
    
    # Loss Functions
    def r2(sefl, y_true, y_pred) -> float:
        """
        R2 Loss Function
        """
        return 1 - ((y_true - y_pred)**2).sum() / ((y_true - y_true.mean())**2).sum()
    
    def rmse(self, y_true, y_pred) -> float:
        """
        RMSE Loss Function
        """
        return ((y_true - y_pred)**2).mean()**0.5
    
    def combined(self, y_true, y_pred) -> float:
        """
        Linear combination of R2 and RMSE
        """
        return self.r2(y_true, y_pred) + 1/self.rmse(y_true, y_pred)