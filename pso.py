#%%
"""
Particle Swarm Optimization Module
====================
This module contains the particle swarm optimization algorithm classes for the FSP optimization problem.
"""

import numpy as np
import random
from LossFunction import Evaluator

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
        
        for t in range(generations):
            for particle in self.particles:
                fitness = self.evaluator.evaluate(particle.position)
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

        self.print_results(t)

        return self.global_best_fitness, self.fitness_history
    
    def maximize(self, generations, population_size, w, c1, c2, fitness_criterion):
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
            for particle in self.particles:
                fitness = self.evaluator.evaluate(particle.position)
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

        self.print_results(t)

        return self.global_best_fitness, self.fitness_history

    def print_results(self, generations):
        print(f'Initial setup: w={self.w}, c1={self.c1}, c2={self.c2}')
        print('Global Best Position: ', self.global_best_position)
        print('Best Fitness Value: ', self.global_best_fitness)
        print('Average Particle Best Fitness Value: ', np.mean([particle.best_fitness for particle in self.particles]))
        print('Number of Generations: ', generations + 1)
        print('-' * 150)

# %%
evaluator = Evaluator('data/listings.csv', 'linear', 'rmse')
evaluator.evaluate([1,1,1,1,1,1,1,1,1])

#%%
pso = PSO(evaluator)
features, _ = pso.minimize(100, 100, 0.5, 1, 1, 0)
features = pso.global_best_position
features = [i>0.5 for i in features]
all_features = evaluator.features
selected_features = [feature for feature, include in zip(all_features, features) if include]
print(f'Selected features: {selected_features}')

# %%
# %%
