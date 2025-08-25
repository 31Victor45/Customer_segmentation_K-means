# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
df = pd.read_csv("segmented_customers.csv")

# Identify features (X) and the target variable (y)
# The 'Cluster' column is the target, all others are features.
X = df.drop('Cluster', axis=1)
y = df['Cluster']

# Separate categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns

# Create the preprocessing pipelines
# Pipeline for numerical features: standard scaling
numerical_transformer = StandardScaler()

# Pipeline for categorical features: one-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine both pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline with preprocessing and the model
# Using RandomForestClassifier as it's robust and performs well
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model (optional, but good practice)
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy:.2%}")

# Save the trained pipeline (preprocessor + model) to a file
# This will be loaded by the Streamlit app
dump(model_pipeline, 'customer_cluster_model.joblib')
print("Model pipeline saved as 'customer_cluster_model.joblib'")
