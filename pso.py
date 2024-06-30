"""
Particle Swarm Optimization Module
====================
This module contains the particle swarm optimization algorithm classes for the FSP optimization problem.
"""

import numpy as np
import random

class Particle:
    def __init__(self, dimension, position_min, position_max):
        self.position = np.array([random.uniform(position_min, position_max) for _ in range(dimension)])
        self.velocity = np.zeros(dimension)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def update_velocity(self, global_best_position, w, c1, c2):
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.best_position - self.position)
            + c2 * r2 * (global_best_position - self.position)
        )

    def update_position(self, position_min, position_max):
        self.position += self.velocity
        self.position = np.clip(self.position, position_min, position_max)


class PSO:
    def __init__(self, population_size, dimension, position_min, position_max, w, c1, c2, fitness_function):
        self.population_size = population_size
        self.dimension = dimension
        self.position_min = position_min
        self.position_max = position_max
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.fitness_function = fitness_function
        self.particles = [Particle(dimension, position_min, position_max) for _ in range(population_size)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.fitness_history = []

    def optimize(self, generations, fitness_criterion):
        for t in range(generations):
            for particle in self.particles:
                fitness = self.fitness_function(particle.position)
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

    def print_results(self, generations):
        print(f'Initial setup: w={self.w}, c1={self.c1}, c2={self.c2}')
        print('Global Best Position: ', self.global_best_position)
        print('Best Fitness Value: ', self.global_best_fitness)
        print('Average Particle Best Fitness Value: ', np.mean([particle.best_fitness for particle in self.particles]))
        print('Number of Generations: ', generations + 1)
        print('-' * 150)
