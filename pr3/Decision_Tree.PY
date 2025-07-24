import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 🌤 Sample dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                   'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# 🔢 Encode categorical variables
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

X = df_encoded.drop('PlayTennis', axis=1)
y = df_encoded['PlayTennis']

# ✂️ Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🌳 Train Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf.fit(X_train, y_train)

# 📝 Print decision tree rules
print(export_text(clf, feature_names=list(X.columns)))

# 🌲 Plot Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["No", "Yes"])
plt.title("🌳 Decision Tree by Gautam")
plt.show()

# 🔮 Predict on test data
y_pred = clf.predict(X_test)

# 📊 Accuracy & Confusion Matrix
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])

# 🔷 Confusion Matrix Plot
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix (Accuracy: {acc:.2f})")
plt.show()

# 📈 Bar Chart of Predictions
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.value_counts().plot(kind='bar', color=['red', 'green'])
plt.title("📊 Prediction vs Actual Counts")
plt.xlabel("Outcome (Actual, Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
