import random
import numpy as np
import matplotlib.pyplot as plt

# VM requirements: [CPU, Memory]
vms = [
    [4, 8], [2, 4], [6, 8], [3, 6],
    [2, 2], [4, 6], [1, 2], [3, 4],
    [2, 4], [5, 8]
]

# Physical machine capacities
NUM_HOSTS = 5
CPU_CAPACITY = 16
RAM_CAPACITY = 32

# Power consumption (Watts)
P_MAX = 215
P_IDLE = 162

# Cuckoo Search parameters
NUM_NESTS = 20
MAX_ITERATIONS = 100
PA = 0.25

random.seed(10)
np.random.seed(10)


# Calculate CPU and memory usage of each host
def get_usage(solution):
    cpu = [0] * NUM_HOSTS
    ram = [0] * NUM_HOSTS

    for i, host in enumerate(solution):
        cpu[host] += vms[i][0]
        ram[host] += vms[i][1]

    return cpu, ram


# Check whether the placement is valid
def is_valid(solution):
    cpu, ram = get_usage(solution)

    for h in range(NUM_HOSTS):
        if cpu[h] > CPU_CAPACITY or ram[h] > RAM_CAPACITY:
            return False

    return True


# Calculate energy consumption
def fitness(solution):
    if not is_valid(solution):
        return float("inf")

    cpu, ram = get_usage(solution)
    energy = 0

    for h in range(NUM_HOSTS):
        if cpu[h] > 0:
            utilization = cpu[h] / CPU_CAPACITY

            power = P_IDLE + (
                P_MAX - P_IDLE
            ) * utilization

            energy += power

    return energy


# Generate a random valid solution
def create_nest():
    solution = [-1] * len(vms)

    order = list(range(len(vms)))
    random.shuffle(order)

    cpu = [0] * NUM_HOSTS
    ram = [0] * NUM_HOSTS

    for i in order:
        possible_hosts = []

        for h in range(NUM_HOSTS):
            if (cpu[h] + vms[i][0] <= CPU_CAPACITY and
                    ram[h] + vms[i][1] <= RAM_CAPACITY):
                possible_hosts.append(h)

        if not possible_hosts:
            return create_nest()

        h = random.choice(possible_hosts)
        solution[i] = h

        cpu[h] += vms[i][0]
        ram[h] += vms[i][1]

    return solution


# Generate a new solution by moving some VMs
def generate_new_nest(solution):
    new_solution = solution.copy()

    number = random.randint(1, max(1, len(vms) // 3))
    selected = random.sample(range(len(vms)), number)

    for i in selected:
        old_host = new_solution[i]
        new_host = random.randrange(NUM_HOSTS)

        if new_host == old_host:
            continue

        new_solution[i] = new_host

        if not is_valid(new_solution):
            new_solution[i] = old_host

    return new_solution


# Initialize nests
nests = [create_nest() for _ in range(NUM_NESTS)]

best_solution = min(nests, key=fitness).copy()
best_fitness = fitness(best_solution)

history = []

# Cuckoo Search optimization
for iteration in range(MAX_ITERATIONS):

    # Generate new solutions
    for i in range(NUM_NESTS):
        new_solution = generate_new_nest(nests[i])

        # Compare with a randomly selected nest
        j = random.randrange(NUM_NESTS)

        if fitness(new_solution) < fitness(nests[j]):
            nests[j] = new_solution

    # Abandon some poor nests
    nests.sort(key=fitness)

    abandon_count = int(PA * NUM_NESTS)

    for i in range(NUM_NESTS - abandon_count, NUM_NESTS):
        nests[i] = create_nest()

    # Update the best solution
    current_best = min(nests, key=fitness)

    if fitness(current_best) < best_fitness:
        best_solution = current_best.copy()
        best_fitness = fitness(best_solution)

    history.append(best_fitness)


# Display final allocation
cpu, ram = get_usage(best_solution)

print("CUCKOO SEARCH - VM PLACEMENT")
print("--------------------------------")

for i, host in enumerate(best_solution):
    print(f"VM {i+1} -> Host {host+1}")

print("\nHOST UTILIZATION")
for h in range(NUM_HOSTS):
    if cpu[h] > 0:
        print(
            f"Host {h+1}: CPU = {cpu[h]}/{CPU_CAPACITY}, "
            f"RAM = {ram[h]}/{RAM_CAPACITY}"
        )
    else:
        print(f"Host {h+1}: Inactive")

active_hosts = sum(1 for x in cpu if x > 0)

print("\nRESULT")
print("Active Hosts:", active_hosts)
print("Total Energy Consumption:", round(best_fitness, 2), "Watts")


# Plot convergence graph
plt.plot(history)
plt.xlabel("Iterations")
plt.ylabel("Energy Consumption (Watts)")
plt.title("Cuckoo Search - VM Placement")
plt.grid(True)
plt.savefig("cuckoo_search_graph.png")
plt.show()
