#%%
"""
Test Script
================
This script tests the search algorithms on the FSP problem.
"""
#Libraries
from loss_function import Evaluator, ROOT_DIR
from difev_real import RealDifferentialEvolution
from difev_comb import ComibnatorialDifferentialEvolution
from pso import PSO
import numpy as np
from colorama import Fore

def test_realDE():
    encoders = ['label', 'target', 'drop']
    loss_functions = ['r2', 'rmse', 'combined']
    regression_models = ['ridge', 'lasso', 'linear']
    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  
    
    data_path = ROOT_DIR + '/data/listings.csv'
    
    for encoder in encoders:
        for regression_model in regression_models:
            for loss_function in loss_functions:
                for seed in seeds:
                    np.random.seed(seed)
                    print(Fore.YELLOW + f'\n\nExperiment: Encoder: {encoder} - Regression Model: {regression_model} - Loss Function: {loss_function} - Seed: {seed}')
                    #Set the evaluator
                    evaluator = Evaluator(data_path, encoder, regression_model, loss_function)
                    
                    print(Fore.CYAN + 'All features Loss:', evaluator.evaluate([1]*len(evaluator.features)))
                    
                    #Set the differential evolution
                    de = RealDifferentialEvolution(evaluator)
                    if loss_function == 'rmse':
                        action = 'minimize'
                    else:
                        action = 'maximize'
                        
                    #Optimize
                    de.optimize(5, 50, 0.1, 0.1, 0.4, action)

def test_combDE():
    encoders = ['label', 'target', 'drop']
    loss_functions = ['r2', 'rmse', 'combined']
    regression_models = ['ridge', 'lasso', 'linear']
    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  
    
    data_path = ROOT_DIR + '/data/listings.csv'
    
    for encoder in encoders:
        for regression_model in regression_models:
            for loss_function in loss_functions:
                for seed in seeds:
                    np.random.seed(seed)
                    print(Fore.YELLOW + f'\n\nExperiment: Encoder: {encoder} - Regression Model: {regression_model} - Loss Function: {loss_function} - Seed: {seed}')
                    #Set the evaluator
                    evaluator = Evaluator(data_path, encoder, regression_model, loss_function)
                    
                    print(Fore.CYAN + 'All features Loss:', evaluator.evaluate([1]*len(evaluator.features)))
                    
                    #Set the differential evolution
                    de = ComibnatorialDifferentialEvolution(evaluator)
                    if loss_function == 'rmse':
                        action = 'minimize'
                    else:
                        action = 'maximize'
                        
                    #Optimize
                    de.optimize(5, 50, 0.1, 0.1, action)

def test_PSO():
    encoders = ['label', 'target', 'drop']
    loss_functions = ['r2', 'rmse', 'combined']
    regression_models = ['ridge', 'lasso', 'linear']
    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  
    
    data_path = ROOT_DIR + '/data/listings.csv'
    
    for encoder in encoders:
        for regression_model in regression_models:
            for loss_function in loss_functions:
                for seed in seeds:
                    np.random.seed(seed)
                    print(Fore.YELLOW + f'\n\nExperiment: Encoder: {encoder} - Regression Model: {regression_model} - Loss Function: {loss_function} - Seed: {seed}')
                    #Set the evaluator
                    evaluator = Evaluator(data_path, encoder, regression_model, loss_function)
                    
                    print(Fore.CYAN + 'All features Loss:', evaluator.evaluate([1]*len(evaluator.features)))
                    
                    #Set the PSO
                    pso = PSO(evaluator)
                    
                    if loss_function == 'rmse':
                        pso.minimize(50, 15, 0, 0.9, 0.1, -np.inf)
                    else:
                        pso.maximize(50, 15, 0, 0.9, 0.1, np.inf)


# %%
test_combDE()

# %%
