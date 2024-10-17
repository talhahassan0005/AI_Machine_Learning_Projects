import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# dataset 
df = pd.read_csv("/kaggle/input/cars-data/cars.csv")

#check that the values of Fuel columns
df['fuel'].value_counts()


# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    df.iloc[:, 0:4],  
    df.iloc[:, -1],   
    test_size=0.2,    
    random_state=0    
)

# Initialize OneHotEncoder with 'drop='first'' to avoid the dummy variable trap
# This means the first category in each feature will be dropped to prevent multicollinearity.
ohe = OneHotEncoder(drop='first')

# Apply One-Hot Encoding and convert to array to avoid sparse matrix issues
# Fit and transform on the training set 'fuel' and 'owner' columns
x_train_new = ohe.fit_transform(x_train[['fuel', 'owner']]).toarray().astype(int)

# Apply the fitted encoder on the test set
# Note: We use 'transform' instead of 'fit_transform' to keep the same encoding as training.
x_test_new = ohe.transform(x_test[['fuel', 'owner']]).toarray().astype(int)

# ------------------------------------------------
# Defining a threshold for rare categories:
# ------------------------------------------------
# This is used to group low-frequency categories into an 'uncommon' category.
# This helps in reducing the number of columns created during One-Hot Encoding,

df['brand'].nunique()  # Get the number of unique brands to assess category diversity

threshold = 100  # Set the threshold for categorizing a brand as 'uncommon'
counts = df['brand'].value_counts()  # Get counts of each brand
repl = counts[counts <= threshold].index  # Identify brands that have occurrences below the threshold

# Replace rare brands with 'uncommon' for easier encoding
df['brand'] = df['brand'].replace(repl, 'uncommon')

# ------------------------------------------------
# Combining One-Hot Encoded columns with other features:
# ------------------------------------------------
# Use np.hstack() to combine the numerical columns ('brand' and 'km_driven')
# with the newly one-hot encoded features for 'fuel' and 'owner'.
np.hstack((x_train[['brand', 'km_driven']].values, x_train_new))

# ------------------------------------------------
# Applying One-Hot Encoding to the 'brand' column with replaced values:
# ------------------------------------------------
# Use get_dummies() to automatically one-hot encode the modified 'brand' column,
# where infrequent brands have been replaced with 'uncommon'.
# Display a random sample of 5 rows to verify the encoding.
pd.get_dummies(df['brand'].replace(repl, 'uncommon')).sample(5)
