import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

# Load data
data = pd.read_csv(r"C:\TrafficTelligence\traffic volume.csv")

# Feature engineering: Split date and Time columns
data[["day", "month", "year"]] = data["date"].str.split("-", expand=True)
data[["hours", "minutes", "seconds"]] = data["Time"].str.split(":", expand=True)
data.drop(columns=['date', 'Time'], axis=1, inplace=True)

# Prepare features and target
y = data['traffic_volume']
x = data.drop(columns=['traffic_volume'], axis=1)
names = x.columns

# Identify categorical and numerical columns
categorical_features = ['holiday', 'weather']
numerical_features = x.columns.difference(categorical_features).tolist()

# Inspect and clean categorical columns
print("\nChecking unique values and types in categorical columns:")
for col in categorical_features:
    print(f"Column '{col}':")
    unique_values = x[col].unique()
    print(f"  Unique values: {unique_values[:10]}...")  # Print a sample of unique values
    print(f"  Value types: {[type(val) for val in unique_values[:10]]}...")  # Print types of sample values

# Convert categorical columns to string
for col in categorical_features:
    x[col] = x[col].astype(str)

# Create and fit the imputer for numerical features
imputer = SimpleImputer(strategy='mean')
imputer.fit(x[numerical_features])

# Save the imputer
joblib.dump(imputer, 'Flask/imputer.pkl')

# Apply imputation to numerical columns
x[numerical_features] = imputer.transform(x[numerical_features])

# Create a column transformer with passthrough for numerical features and encoding for categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),  # Numerical data is already imputed
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Fit and transform the data
x_processed = preprocessor.fit_transform(x)

# Save the preprocessor (includes encoder)
joblib.dump(preprocessor, 'Flask/encoder.pkl')

# Scale the processed data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_processed)

# Save the scaler
joblib.dump(scaler, 'Flask/scale.pkl')

print("Preprocessing objects saved successfully!")