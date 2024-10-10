import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Load the dataset
df = pd.read_csv('/kaggle/input/customer/customer.csv')

# Separate features and target
X = df.iloc[:, 0:2]  # Assuming the first two columns are features
y = df.iloc[:, -1]   # Assuming the last column is the target

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the OrdinalEncoder (ensure categories match your actual data)
oe = OrdinalEncoder()

# Fit and transform the training data
x_train_encoded = oe.fit_transform(x_train)

# Fit and transform y_train using LabelEncoder (if y_train is categorical)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Output the transformed x_train and y_train
print("Encoded x_train:\n", x_train_encoded)
print("Encoded y_train:\n", y_train_encoded)
