
import random
import matplotlib.pyplot as plt

random.seed(10)

# Courses and students
courses = {
    "AI": ["S1", "S2", "S3", "S4"],
    "DBMS": ["S1", "S2", "S5"],
    "OS": ["S3", "S4", "S6"],
    "CN": ["S1", "S5", "S6"],
    "ML": ["S2", "S3", "S5"],
    "Java": ["S4", "S5", "S6"],
    "Python": ["S1", "S3", "S6"],
    "Maths": ["S2", "S4", "S5"]
}

rooms = {"R1": 30, "R2": 40, "R3": 60}

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
periods = ["9-10", "10-11", "11-12", "12-1"]

NUM_SLOTS = len(days) * len(periods)
POPULATION = 30
GENERATIONS = 150

course_names = list(courses.keys())

# Generate random timetable
def create_timetable():
    return [random.randrange(NUM_SLOTS) for _ in course_names]

# Assign rooms to courses
def assign_rooms(chromosome):
    schedule = {}
    hard = 0

    for slot in range(NUM_SLOTS):
        events = [
            course_names[i]
            for i in range(len(course_names))
            if chromosome[i] == slot
        ]

        events.sort(
            key=lambda c: len(courses[c]),
            reverse=True
        )

        available = list(rooms.items())

        for event in events:
            required = len(courses[event]) * 10

            room_found = False

            for room, capacity in available:
                if capacity >= required:
                    schedule[event] = (slot, room)
                    available.remove((room, capacity))
                    room_found = True
                    break

            if not room_found:
                hard += 1

    return schedule, hard

# Calculate fitness
def fitness(chromosome):
    hard = 0
    soft = 0

    # Check student clashes
    for i in range(len(course_names)):
        for j in range(i + 1, len(course_names)):
            if chromosome[i] == chromosome[j]:
                common = set(courses[course_names[i]]) & set(
                    courses[course_names[j]]
                )
                if common:
                    hard += len(common)

    schedule, room_errors = assign_rooms(chromosome)
    hard += room_errors

    # Soft constraint: avoid last period
    for i, slot in enumerate(chromosome):
        if slot % 4 == 3:
            soft += 1

    # Soft constraints for student schedules
    for student in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        student_slots = []

        for i, course in enumerate(course_names):
            if student in courses[course]:
                student_slots.append(chromosome[i])

        # Check single class per day
        daily_count = {}

        for slot in student_slots:
            day = slot // 4
            period = slot % 4
            daily_count.setdefault(day, []).append(period)

        for periods_list in daily_count.values():
            if len(periods_list) == 1:
                soft += 1

            periods_list.sort()

            # More than two consecutive classes
            for k in range(len(periods_list) - 2):
                if (periods_list[k + 1] == periods_list[k] + 1
                    and periods_list[k + 2] == periods_list[k] + 2):
                    soft += 1

    return hard * 1000 + soft

# Selection
def selection(population):
    candidates = random.sample(population, 3)
    return min(candidates, key=fitness)

# Crossover
def crossover(parent1, parent2):
    child = []

    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])

    return child

# Mutation
def mutation(chromosome):
    chromosome = chromosome[:]

    if random.random() < 0.5:
        position = random.randrange(len(chromosome))
        chromosome[position] = random.randrange(NUM_SLOTS)

    return chromosome

# Initialize population
population = [create_timetable() for _ in range(POPULATION)]

best_fitness_history = []

# Genetic Algorithm
for generation in range(GENERATIONS):

    population.sort(key=fitness)

    best = population[0]
    best_fitness_history.append(fitness(best))

    new_population = [best[:]]

    while len(new_population) < POPULATION:
        parent1 = selection(population)
        parent2 = selection(population)

        child = crossover(parent1, parent2)
        child = mutation(child)

        new_population.append(child)

    population = new_population

# Best solution
best = min(population, key=fitness)
final_schedule, room_errors = assign_rooms(best)

hard_violations = fitness(best) // 1000

print("\nFINAL UNIVERSITY TIMETABLE")
print("-" * 55)

for day in range(5):
    print("\n", days[day])

    for period in range(4):
        slot = day * 4 + period

        events = [
            course for course, (s, room) in final_schedule.items()
            if s == slot
        ]

        if events:
            for course in events:
                print(
                    periods[period], "->",
                    course, "(", final_schedule[course][1], ")"
                )
        else:
            print(periods[period], "-> Free")

print("\nFinal Fitness:", fitness(best))
print("Hard Violations:", hard_violations)

# GRAPH 1: Fitness improvement
plt.figure(figsize=(9, 5))
plt.plot(best_fitness_history)
plt.title("Genetic Algorithm - Fitness Improvement")
plt.xlabel("Generation")
plt.ylabel("Fitness Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# GRAPH 2: Timetable visualization
fig, ax = plt.subplots(figsize=(12, 6))

ax.set_xlim(0, 5)
ax.set_ylim(0, 4)
ax.set_xticks(range(5))
ax.set_xticklabels(days)
ax.set_yticks(range(4))
ax.set_yticklabels(periods)
ax.invert_yaxis()

ax.set_title("University Timetable Generated by GA")
ax.set_xlabel("Days")
ax.set_ylabel("Time Slots")
ax.grid(True)

for course, (slot, room) in final_schedule.items():
    day = slot // 4
    period = slot % 4

    ax.text(
        day + 0.5,
        period + 0.5,
        course + "\n" + room,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.3)
    )

plt.tight_layout()
plt.show()
