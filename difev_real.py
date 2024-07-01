#%%
"""
difev.py
Differential Evolution Applied to a combinatorial problem (FSP)
"""
from LossFunction import Evaluator
import numpy as np
import random
from colorama import Fore
import sys


class Individual:
    def __init__(self, dimension) -> None:
        """
        Constructor for the Individual class.
        """
        self.features = np.random.uniform(0, 1, dimension)
        self.value = None
    
    def evaluate(self, evaluator: Evaluator) -> None:
        self.value = evaluator.evaluate(self.features)
        
    def getFeatures(self, evaluator: Evaluator):
        all_features = evaluator.features
        features = [i>0.5 for i in self.features]
        selected_features = [feature for feature, include in zip(all_features, features) if include]
        return selected_features

class DifferentialEvolution:
    def __init__(self, evaluator: Evaluator) -> None:
        """
        Constructor for the Differential Evolution class.
        
        Parameters
        ----------
        evaluator : Evaluator
            The evaluator object for the FSP problem.
        """
        
        # Set for the FSP problem
        self.evaluator = evaluator
        self.dimension = len(evaluator.features)
    
    def optimize(self, generations, population_size, cr, mr, f, action='minimize'):

        #Initialize the population
        self.population = [Individual(self.dimension) for _ in range(population_size)]

        #Initial evaluation
        for individual in self.population:
           
            individual.evaluate(self.evaluator)

        #Main loop
        for i in range(generations):
            
            print(Fore.WHITE + f'\nGeneration {i+1}/{generations}')
            replaces = 0
            for i, target in enumerate(self.population):
                progress = int((i / population_size) * 25)

                fitness = [individual.value for individual in self.population]
                
                #Select r1 id
                r1 = np.argmin(fitness) if action == 'minimize' else np.argmax(fitness)
                
                #Select random r2, r3 ids
                r2, r3 = random.sample([num for num in range(population_size)], k=2)
                
                #Create trial vector
                trial = self.population[r1].features + f*(self.population[r2].features - self.population[r3].features)
                
                #Mutate the trial vector
                for j, feature in enumerate(trial):
                    if random.uniform(0, 1) < mr:
                        trial[j] += random.uniform(-1, 1)
                
                #Fix the trial vector
                trial = np.clip(trial, 0, 1)
                
                #Cross-over
                fixed = random.randint(0, self.dimension)
                
                for j, feature in enumerate(trial):
                    if random.uniform(0, 1) < cr or feature == fixed:
                        pass
                    else:
                        feature = target.features[j]
                
                #Evaluate the trial
                trial_fitness = self.evaluator.evaluate(trial)
                
                if action == 'minimize':
                    if trial_fitness < target.value:
                        replaces += 1
                        target.features = trial
                        target.value = trial_fitness
                else:
                    if trial_fitness > target.value:
                        replaces += 1
                        target.features = trial
                        target.value = trial_fitness
                        
                best = np.argmin(fitness) if action == 'minimize' else np.argmax(fitness)
                
                sys.stdout.write("\r{}/{} [{}{}{}] - Selected Features: {} - Loss: {} - Target Replaces: {}".format(i+1, population_size, "=" * progress, ">" , " " * (25 - progress), self.population[best].getFeatures(self.evaluator), self.population[best].value, replaces))
                sys.stdout.flush()
       
        print('\n')
        print('-'*150)        
        print(f'\nSelected features: {self.population[best].getFeatures(self.evaluator)} \nLoss: {self.population[best].value}',)

# %%
evaluator = Evaluator('data/listings.csv', 'target', 'linear', 'r2')
print(f'All features selected loss: {evaluator.evaluate([1]*9)}')
difev = DifferentialEvolution(evaluator)
difev.optimize(3, 50, 0.2, 0.1, 0.6, 'maximize')


# %%
