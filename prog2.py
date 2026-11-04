
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# ------------------------------------------------
# STEP 1: LOAD DIABETIC RETINOPATHY DATASET
# ------------------------------------------------

dataset = fetch_ucirepo(id=329)

X = dataset.data.features.to_numpy(dtype=float)
y = dataset.data.targets.iloc[:, 0].to_numpy(dtype=int)

feature_names = np.array(dataset.data.features.columns)

print("Dataset loaded successfully")
print("Total records:", len(X))
print("Total features:", X.shape[1])

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ------------------------------------------------
# STEP 2: DEFINE NEURAL NETWORK
# ------------------------------------------------

def create_model():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(16, 8),
            max_iter=250,
            random_state=42,
            early_stopping=True
        )
    )

# ------------------------------------------------
# STEP 3: DEFINE PSO PARAMETERS
# ------------------------------------------------

particles = 8
iterations = 10

w = 0.7
c1 = 1.5
c2 = 1.5

n_features = X_train.shape[1]

# Initialize particle positions and velocities
positions = np.random.randint(0, 2, (particles, n_features))
velocities = np.random.uniform(-1, 1, (particles, n_features))

pbest = positions.copy()
pbest_fitness = np.full(particles, float("inf"))

gbest = None
gbest_fitness = float("inf")

fitness_history = []

# ------------------------------------------------
# STEP 4: FITNESS FUNCTION
# ------------------------------------------------

def fitness_function(position):

    selected = np.where(position == 1)[0]

    if len(selected) == 0:
        return 1.0

    model = create_model()

    cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=42
    )

    scores = cross_val_score(
        model,
        X_train[:, selected],
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    accuracy = scores.mean()

    # Prefer good accuracy with fewer features
    fitness = 1 - accuracy + 0.01 * (
        len(selected) / n_features
    )

    return fitness

# ------------------------------------------------
# STEP 5: PSO OPTIMIZATION
# ------------------------------------------------

for iteration in range(iterations):

    for i in range(particles):

        fitness = fitness_function(positions[i])

        # Update Personal Best
        if fitness < pbest_fitness[i]:
            pbest_fitness[i] = fitness
            pbest[i] = positions[i].copy()

        # Update Global Best
        if fitness < gbest_fitness:
            gbest_fitness = fitness
            gbest = positions[i].copy()

    # Update velocity and position
    for i in range(particles):

        r1 = np.random.rand(n_features)
        r2 = np.random.rand(n_features)

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest[i] - positions[i])
            + c2 * r2 * (gbest - positions[i])
        )

        # Convert velocity into probability
        probability = 1 / (1 + np.exp(-np.clip(
            velocities[i], -10, 10
        )))

        # Update binary position
        positions[i] = (
            np.random.rand(n_features) < probability
        ).astype(int)

    fitness_history.append(gbest_fitness)

    print(
        "Iteration:", iteration + 1,
        "Best Fitness:", round(gbest_fitness, 4)
    )

# ------------------------------------------------
# STEP 6: SELECT BEST FEATURES
# ------------------------------------------------

selected_features = np.where(gbest == 1)[0]

print("\nSelected Features:")

for index in selected_features:
    print(feature_names[index])

print("Number of selected features:", len(selected_features))

# ------------------------------------------------
# STEP 7: TRAIN NEURAL NETWORK
# ------------------------------------------------

X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

model = create_model()
model.fit(X_train_selected, y_train)

# Predict diabetic retinopathy classes
y_pred = model.predict(X_test_selected)

# ------------------------------------------------
# STEP 8: EVALUATE PERFORMANCE
# ------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nPSO + NEURAL NETWORK RESULTS")
print("--------------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["No DR", "DR"]
))

# ------------------------------------------------
# STEP 9: COMPARE WITH NORMAL NEURAL NETWORK
# ------------------------------------------------

normal_model = create_model()
normal_model.fit(X_train, y_train)

normal_pred = normal_model.predict(X_test)

normal_accuracy = accuracy_score(y_test, normal_pred)

print("\nACCURACY COMPARISON")
print("--------------------------------")
print("Normal Neural Network:",
      round(normal_accuracy * 100, 2), "%")

print("PSO + Neural Network:",
      round(accuracy * 100, 2), "%")

# ------------------------------------------------
# STEP 10: DISPLAY GRAPHS
# ------------------------------------------------

plt.figure(figsize=(7, 4))
plt.plot(range(1, iterations + 1), fitness_history, marker="o")
plt.xlabel("Iterations")
plt.ylabel("Best Fitness")
plt.title("PSO Optimization Progress")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(
    ["Normal NN", "PSO + NN"],
    [normal_accuracy * 100, accuracy * 100]
)
plt.ylabel("Accuracy (%)")
plt.title("Diabetic Retinopathy Classification")
plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No DR", "DR"],
    cmap="Blues"
)
plt.title("PSO + Neural Network Confusion Matrix")
plt.show()
