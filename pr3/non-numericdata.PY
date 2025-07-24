import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import graphviz

# Step 1: Create dataset with nominal + numeric data
data = pd.DataFrame({
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild'],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95],
    'Windy': [False, True, False, False, False, True, True, False],  # Use bool type
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No']
})

# Step 2: Encode categorical data
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    elif data[column].dtype == 'bool':
        # Convert bool to int 0/1 explicitly
        data[column] = data[column].astype(int)

# Step 3: Split features and target
X = data.drop('Play', axis=1)
y = data['Play']

# Step 4: Split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build decision tree with pruning (max_depth)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf.fit(X_train, y_train)

# Step 6: Predict
predicted = clf.predict(X_test)
print("Predictions on Test Data:", predicted)

# Step 7: Visualize the decision tree
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=X.columns,
    class_names=label_encoders['Play'].classes_,
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)

# Render to PNG file
graph.render("decision_tree_output", format='png', cleanup=False)

# Open the output file viewer (this will open the PNG or PDF depending on your system)
graph.view()
