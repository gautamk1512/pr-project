# Comparison of Classification Accuracy of SVM(Support Vector Machine) for given dataset
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset (you can replace this with your own CSV using pd.read_csv)
iris = load_iris()
X = iris.data
y = iris.target 

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compare different SVM kernels
svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_results = {}

for kernel in svm_kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    svm_results[f"SVM ({kernel})"] = acc

# Compare with other classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

other_models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

other_results = {}

for name, clf in other_models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    other_results[name] = acc

# Combine all results
all_results = {**svm_results, **other_results}

# Print accuracies
print("Classification Accuracy Results:\n")
for model_name, accuracy in all_results.items():
    print(f"{model_name}: {accuracy:.4f}")

# Plot the comparison
plt.figure(figsize=(12, 6))
plt.bar(all_results.keys(), all_results.values(), color='skyblue')
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Accuracy (SVM vs Other Classifiers)')
plt.xticks(rotation=45)
plt.ylim(0.0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
