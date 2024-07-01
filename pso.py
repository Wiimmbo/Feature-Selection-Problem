#%%
"""
Particle Swarm Optimization Module
====================
This module contains the particle swarm optimization algorithm classes for the FSP optimization problem.
"""

import numpy as np
import random
from loss_function import Evaluator
from colorama import Fore
import sys

class Particle:
    """
    Particle Class
    """
    def __init__(self, dimension, position_min, position_max):
        """
        Constructor for the Particle class. Initilizes a particle with a random position and zero velocity.
        
        Parameters
        ----------
        dimension : int
            The dimension of the particle (lenght).
        position_min : float
            The minimum value for the position.
        position_max : float
            The maximum value for the position.
        """
        # Randomly initialize the position and velocity
        self.position = np.array([random.uniform(position_min, position_max) for _ in range(dimension)])
        self.velocity = np.zeros(dimension)
        
        # Initialize the best position and fitness as the initial position and fitness
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def update_velocity(self, global_best_position, w, c1, c2):
        """
        Update the velocity of the particle.
        
        Parameters
        ----------
        global_best_position : numpy.ndarray
            The best position of the swarm.
        w : float
            Inertia weight.
        c1 : float
            Cognitive weight.
        c2 : float
            Social weight.
        """
        # Random factors
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        
        # Update the velocity
        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.best_position - self.position)
            + c2 * r2 * (global_best_position - self.position)
        )

    def update_position(self, position_min, position_max):
        """
        Update the position of the particle. Clip the position to the position_min and position_max values.
        
        Parameters
        ----------
        position_min : float
            The minimum value for the position.
        position_max : float
            The maximum value for the position.
        """
        self.position += self.velocity
        self.position = np.clip(self.position, position_min, position_max)


class PSO:
    """
    PSO Class
    """
    def __init__(self, evaluator: Evaluator):
        """
        Constructor for the PSO class.
        
        Parameters
        ----------
        evaluator : Evaluator
            The evaluator object for the FSP problem    
        """
        # Set for the FSP problem
        self.dimension = 9
        self.position_min = 0
        self.position_max = 1
        self.evaluator = evaluator
    
    def minimize(self, generations, population_size, w, c1, c2, fitness_criterion):
        """
        Function to minimize the loss function using the PSO algorithm.
        
        Parameters
        ----------
        generations : int
            The number of generations to run the algorithm.
        population_size : int
            The number of particles in the swarm.
        w : float
            Inertia weight.
        c1 : float
            Cognitive weight.
        c2 : float
            Social weight.
        fitness_criterion : float
            The fitness value to reach to stop the algorithm.
        """
        # Hyperparameters
        self.population_size = population_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Initialize the particles
        self.particles = [Particle(self.dimension, self.position_min, self.position_max) for _ in range(population_size)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.fitness_history = []
        
        #Iterate over the generations
        for t in range(generations):
            progress = int((t / generations) * 25)
            
            # Iterate over the particles
            for particle in self.particles:
                
                fitness = self.evaluator.evaluate(particle.position)
                if fitness is None:
                    fitness = np.inf
                    
                if fitness < particle.best_fitness:
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = fitness

                if fitness < self.global_best_fitness:
                    self.global_best_position = particle.position.copy()
                    self.global_best_fitness = fitness

            if np.mean([particle.best_fitness for particle in self.particles]) <= fitness_criterion:
                break

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position(self.position_min, self.position_max)

            self.fitness_history.append(self.global_best_fitness)
            
            sys.stdout.write("\r{}/{} [{}{}{}]  - Loss: {} ".format(t+1, generations, "=" * progress, ">" , " " * (25 - progress), self.global_best_fitness))
            sys.stdout.flush()
       
        features = self.global_best_position
        features = [i>0.5 for i in features]
        all_features = self.evaluator.features
        selected_features = [feature for feature, include in zip(all_features, features) if include]
        
        print(Fore.GREEN + f'\nSelected features: {selected_features} \nLoss: {self.global_best_fitness}')
    
    def maximize(self, generations, population_size, w, c1, c2, fitness_criterion):
        """
        Function to minimize the loss function using the PSO algorithm.
        
        Parameters
        ----------
        generations : int
            The number of generations to run the algorithm.
        population_size : int
            The number of particles in the swarm.
        w : float
            Inertia weight.
        c1 : float
            Cognitive weight.
        c2 : float
            Social weight.
        fitness_criterion : float
            The fitness value to reach to stop the algorithm.
        """
        # Hyperparameters
        self.population_size = population_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Initialize the particles
        self.particles = [Particle(self.dimension, self.position_min, self.position_max) for _ in range(population_size)]
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        self.fitness_history = []
        
        for t in range(generations):
            progress = int((t / generations) * 25)
            for particle in self.particles:
                fitness = self.evaluator.evaluate(particle.position)
                
                if fitness is None:
                    fitness = -np.inf
                
                if fitness > particle.best_fitness:
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = fitness

                if fitness > self.global_best_fitness:
                    self.global_best_position = particle.position.copy()
                    self.global_best_fitness = fitness

            if np.mean([particle.best_fitness for particle in self.particles]) >= fitness_criterion:
                break

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position(self.position_min, self.position_max)

            self.fitness_history.append(self.global_best_fitness)
            
            # Print the progress
            sys.stdout.write("\r{}/{} [{}{}{}]  - Loss: {} ".format(t+1, generations, "=" * progress, ">" , " " * (25 - progress), self.global_best_fitness))
            sys.stdout.flush()
            
        # Print the selected features   
        features = self.global_best_position
        features = [i>0.5 for i in features]
        all_features = self.evaluator.features
        selected_features = [feature for feature, include in zip(all_features, features) if include]
        
        #Print results
        print(Fore.GREEN + f'\nSelected features: {selected_features} \nLoss: {self.global_best_fitness}')