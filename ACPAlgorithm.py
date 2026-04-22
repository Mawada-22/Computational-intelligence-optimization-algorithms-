import random
import numpy as np

# Read city coordinates from file
with open("TSPDATA.txt", "r") as file:
    city_data = [list(map(float, line.split())) for line in file]

# Number of cities
num_cities = len(city_data)

# Calculate distances between cities
distances = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        x1, y1 = city_data[i][1], city_data[i][2]
        x2, y2 = city_data[j][1], city_data[j][2]
        distances[i][j] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# ACS parameters
num_ants = 10
num_iterations = 20
alpha = 1.0
beta = 2.0
evaporation_rate = 0.5
Q = 1.0  # Pheromone update quantity

# Initialize pheromone levels
tau_0 = 1 / (num_cities * np.mean(distances))
pheromone = np.ones((num_cities, num_cities)) * tau_0

# Function to choose the next node based on probabilities
def choose_next_node(current_city, visited, pheromone, distances):
    probabilities = []
    for i in range(num_cities):
        if i not in visited:
            prob = (pheromone[current_city][i] ** alpha) * ((1 / distances[current_city][i]) ** beta)
            probabilities.append(prob)
        else:
            probabilities.append(0)

    # Normalize probabilities
    probabilities = probabilities / np.sum(probabilities)

    # Choose next node based on probabilities
    next_city = np.random.choice(range(num_cities), p=probabilities)
    return next_city

# Main loop for the ant colony optimization
best_tour_length = float('inf')
best_tour = None
for iteration in range(num_iterations):
    for ant in range(num_ants):
        # Start the tour at a random city
        current_city = random.randint(0, num_cities - 1)
        tour = [current_city]
        visited = set([current_city])
        tour_distance = 0.0

        # Construct the tour
        while len(visited) < num_cities:
            # Choose the next city to visit
            next_city = choose_next_node(current_city, visited, pheromone, distances)
            tour.append(next_city)
            visited.add(next_city)
            tour_distance += distances[current_city][next_city]
            current_city = next_city

        # Complete the tour by returning to the start city
        tour_distance += distances[tour[-1]][tour[0]]

        # Update pheromone levels
        for i in range(num_cities - 1):
            pheromone[tour[i]][tour[i + 1]] += Q / tour_distance
        pheromone[tour[-1]][tour[0]] += Q / tour_distance

        # Update best tour if the current tour is better
        if tour_distance < best_tour_length:
            best_tour_length = tour_distance
            best_tour = tour.copy()

    # Evaporate some pheromone
    pheromone *= (1 - evaporation_rate)

# Print the best tour and its length
print("Best tour:", best_tour)
print("Best tour length:", best_tour_length)
