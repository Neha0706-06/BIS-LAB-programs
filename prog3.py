
import random
import math
import matplotlib.pyplot as plt

# Coordinates of six cities
cities = [(0, 0), (2, 6), (5, 5),
          (7, 1), (4, -2), (1, -3)]
names = ["A", "B", "C", "D", "E", "F"]

random.seed(1)
n = len(cities)
dist = [[math.dist(cities[i], cities[j])
         for j in range(n)] for i in range(n)]

pheromone = [[1.0] * n for _ in range(n)]
alpha, beta = 1, 2
rho, q0 = 0.1, 0.4
ants, iterations = 4, 50

best_route = None
best_distance = float("inf")
history = []

for _ in range(iterations):
    for _ in range(ants):
        route = [random.randrange(n)]

        while len(route) < n:
            i = route[-1]
            choices = [j for j in range(n) if j not in route]
            weights = [
                pheromone[i][j]**alpha *
                (1 / dist[i][j])**beta for j in choices
            ]

            if random.random() < q0:
                nxt = choices[weights.index(max(weights))]
            else:
                nxt = random.choices(choices, weights=weights)[0]

            route.append(nxt)

            # Local pheromone update
            pheromone[i][nxt] = (
                (1-rho) * pheromone[i][nxt] + rho
            )
            pheromone[nxt][i] = pheromone[i][nxt]

        route.append(route[0])
        length = sum(
            dist[route[k]][route[k+1]] for k in range(n)
        )

        if length < best_distance:
            best_route = route[:]
            best_distance = length

    # Global pheromone evaporation
    for i in range(n):
        for j in range(n):
            pheromone[i][j] *= (1-rho)

    # Reinforce best-known route
    for k in range(n):
        a, b = best_route[k], best_route[k+1]
        pheromone[a][b] += rho / best_distance
        pheromone[b][a] = pheromone[a][b]

    history.append(best_distance)

print("Best route:", " -> ".join(names[i] for i in best_route))
print("Best distance:", round(best_distance, 2))

# Graph 1: Best route
x = [cities[i][0] for i in best_route]
y = [cities[i][1] for i in best_route]

plt.figure()
plt.plot(x, y, marker="o")
for i, (cx, cy) in enumerate(cities):
    plt.annotate(names[i], (cx, cy))
plt.title("Best TSP Route using ACS")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid()
plt.show()

# Graph 2: Convergence
plt.figure()
plt.plot(range(1, iterations+1), history, marker=".")
plt.title("ACS Convergence Graph")
plt.xlabel("Iteration")
plt.ylabel("Best Tour Distance")
plt.grid()
plt.show()
