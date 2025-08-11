#mawada ebrahim ali 
#20210415

from typing import List
import random

class GeneticAlgorithm:
    def __init__(self, pop_size: int, R_max: float, R_min: float, crossover_prob: float, mutation_prob: float):
        self.pop_size = pop_size
        self.R_max = R_max
        self.R_min = R_min
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = self.initialize_population()

    def initialize_population(self) -> List[List[float]]:
        return [[random.uniform(self.R_min, self.R_max), random.uniform(2, self.R_max)] for _ in range(self.pop_size)]

    def evaluate_fitness(self, individual: List[float]) -> float:
        x1, x2 = individual
        return 8 - (x1 + 0.0317)**2 + x2**2

    def tournament_selection(self, k: int) -> List[List[float]]:
        selected_individuals = []
        for _ in range(2):
            tournament_pool = random.sample(self.population, k)
            selected_individuals.append(max(tournament_pool, key=self.evaluate_fitness))
        return selected_individuals

    def arithmetic_crossover(self, parents: List[List[float]]) -> List[List[float]]:
        parent1, parent2 = parents
        if random.random() < self.crossover_prob:
            offspring1 = [min(max((parent1[i] + parent2[i]) / 2, self.R_min), self.R_max) for i in range(len(parent1))]
            offspring2 = [min(max((parent1[i] - parent2[i]) / 2, self.R_min), self.R_max) for i in range(len(parent1))]
            return offspring1, offspring2
        else:
            return parent1, parent2

    def gaussian_mutation(self, individual: List[float]) -> List[float]:
        mutated_individual = individual.copy()
        for i in range(len(mutated_individual)):
            if random.random() < self.mutation_prob:
                mutated_individual[i] += random.gauss(0, 0.5)
                mutated_individual[i] = min(max(mutated_individual[i], self.R_min), self.R_max)
        return mutated_individual

    def elitism(self, offspring: List[List[float]]) -> List[List[float]]:
        combined_population = self.population + offspring
        sorted_population = sorted(combined_population, key=self.evaluate_fitness, reverse=True)
        return sorted_population[:self.pop_size]

    def run(self, max_generations: int, tournament_k_small: int, tournament_k_large: int):
        for generation in range(max_generations):
            offspring = []
            for _ in range(self.pop_size // 2):
                selected_parents_small = self.tournament_selection(tournament_k_small)
                selected_parents_large = self.tournament_selection(tournament_k_large)
                
                offspring_small = self.arithmetic_crossover(selected_parents_small)
                offspring_large = self.arithmetic_crossover(selected_parents_large)
                
                mutated_offspring_small = [self.gaussian_mutation(child) for child in offspring_small]
                mutated_offspring_large = [self.gaussian_mutation(child) for child in offspring_large]
                
                offspring.extend(mutated_offspring_small)
                offspring.extend(mutated_offspring_large)
            
            self.population = self.elitism(offspring)
        
        best_solution = max(self.population, key=self.evaluate_fitness)
        return best_solution


class OneMaxGA(GeneticAlgorithm):
    def __init__(self, pop_size: int, R_max: float, R_min: float, crossover_prob: float, mutation_prob: float, max_generations: int, tournament_k_small: int, tournament_k_large: int):
        super().__init__(pop_size, R_max, R_min, crossover_prob, mutation_prob)
        self.max_generations = max_generations
        self.tournament_k_small = tournament_k_small
        self.tournament_k_large = tournament_k_large

    def solve(self):
        return self.run(self.max_generations, self.tournament_k_small, self.tournament_k_large)


# Example usage:
if __name__ == "__main__":
    pop_size = 100
    R_max = 2
    R_min = -2
    crossover_prob = 0.6
    mutation_prob = 0.05
    max_generations = 100
    tournament_k_small = 3
    tournament_k_large = 10

    onemax_ga = OneMaxGA(pop_size, R_max, R_min, crossover_prob, mutation_prob, max_generations, tournament_k_small, tournament_k_large)
    best_solution = onemax_ga.solve()

    print("Best solution:", best_solution)
    print("Fitness:", onemax_ga.evaluate_fitness(best_solution))
