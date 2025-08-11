import numpy as np

# Define the objective function to be optimized
def objective_function(x):
    return sum(x**2)

# Genetic Algorithm
def genetic_algorithm(objective_function, n_gen, n_pop, n_genome, n_parents, mutation_rate):
    # Initialize population randomly
    population = np.random.randint(0, 2, size=(n_pop, n_genome))
    
    # Main loop
    for generation in range(n_gen):
        # Evaluate fitness of population
        fitness = np.array([objective_function(individual) for individual in population])
        
        # Select parents based on tournament selection
        parents = []
        for _ in range(n_parents):
            tournament_indices = np.random.choice(range(n_pop), size=2, replace=False)
            tournament_fitness = fitness[tournament_indices]
            parent_index = tournament_indices[np.argmin(tournament_fitness)]
            parents.append(population[parent_index])
        
        # Crossover to create offspring
        offspring = []
        for _ in range(n_pop - n_parents):
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            crossover_point = np.random.randint(1, n_genome)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring.append(child)
        
        # Mutation
        for i in range(len(offspring)):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(n_genome)
                offspring[i][mutation_point] = 1 - offspring[i][mutation_point]
        
        # Replace old population with new population
        population = np.vstack([parents, offspring])
        
    # Select the best individual as the solution
    best_individual = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    
    return best_individual, best_fitness

# Simulated Annealing
def simulated_annealing(objective_function, n_gen, n_params, initial_temperature, cooling_schedule):
    # Initialize the current solution randomly
    current_solution = np.random.randint(0, 2, size=n_params)
    current_fitness = objective_function(current_solution)
    
    # Main loop
    for t in range(n_gen):
        # Update temperature
        temperature = cooling_schedule(t, initial_temperature)
        
        # Generate a candidate solution by flipping a random bit
        candidate_solution = current_solution.copy()
        flip_index = np.random.randint(n_params)
        candidate_solution[flip_index] = 1 - candidate_solution[flip_index]
        candidate_fitness = objective_function(candidate_solution)
        
        # Accept or reject the candidate solution
        delta_fitness = candidate_fitness - current_fitness
        if delta_fitness < 0 or np.random.rand() < np.exp(-delta_fitness / temperature):
            current_solution = candidate_solution
            current_fitness = candidate_fitness
    
    return current_solution, current_fitness

# Cooling schedule for Simulated Annealing (exponential decay)
def exponential_decay(t, initial_temperature, decay_rate=0.01):
    return initial_temperature * np.exp(-decay_rate * t)

# Evolutionary Algorithm
def evolutionary_algorithm(objective_function, n_gen, n_pop, n_genome, mutation_rate):
    # Initialize population randomly
    population = np.random.randint(0, 2, size=(n_pop, n_genome))
    
    # Main loop
    for generation in range(n_gen):
        # Evaluate fitness of population
        fitness = np.array([objective_function(individual) for individual in population])
        
        # Mutation
        for i in range(len(population)):
            for j in range(len(population[i])):
                if np.random.rand() < mutation_rate:
                    population[i][j] = 1 - population[i][j]
        
    # Select the best individual as the solution
    best_individual = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    
    return best_individual, best_fitness

# Example usage
if __name__ == "__main__":
    n_gen = 50       # Number of generations
    n_pop = 100      # Population size
    n_genome = 10    # Number of genes in an individual
    mutation_rate = 0.1  # Probability of mutation
    n_parents = 5    # Number of parents for crossover
    initial_temperature = 10  # Initial temperature for simulated annealing
    n_params = 10             # Number of parameters for simulated annealing
    decay_rate = 0.01         # Decay rate for the cooling schedule

    # Genetic Algorithm
    best_individual_ga, best_fitness_ga = genetic_algorithm(objective_function, n_gen, n_pop, n_genome, n_parents, mutation_rate)
    print("Genetic Algorithm:")
    print("Best individual:", best_individual_ga)
    print("Best fitness:", best_fitness_ga)

    # Simulated Annealing
    best_solution_sa, best_fitness_sa = simulated_annealing(objective_function, n_gen, n_params, initial_temperature, lambda t, T: exponential_decay(t, T, decay_rate))
    print("\nSimulated Annealing:")
    print("Best solution:", best_solution_sa)
    print("Best fitness:", best_fitness_sa)

    # Evolutionary Algorithm
    best_individual_ea, best_fitness_ea = evolutionary_algorithm(objective_function, n_gen, n_pop, n_genome, mutation_rate)
    print("\nEvolutionary Algorithm:")
    print("Best individual:", best_individual_ea)
    print("Best fitness:", best_fitness_ea)
