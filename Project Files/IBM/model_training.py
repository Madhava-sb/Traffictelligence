import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, ensemble, svm
import xgboost
import joblib
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Load preprocessed data
data = pd.read_csv(r"C:\TrafficTelligence\traffic volume.csv")
data[["day", "month", "year"]] = data["date"].str.split("-", expand=True)
data[["hours", "minutes", "seconds"]] = data["Time"].str.split(":", expand=True)
data.drop(columns=['date', 'Time'], axis=1, inplace=True)

y = data['traffic_volume']
x = data.drop(columns=['traffic_volume'], axis=1)

# Ensure categorical columns match the preprocessing step
categorical_features = ['holiday', 'weather']
for col in categorical_features:
    x[col] = x[col].astype(str)  # Convert to string to match preprocessing

# Load imputer and apply to numerical columns
imputer = joblib.load('Flask/imputer.pkl')
numerical_features = x.columns.difference(categorical_features).tolist()
x[numerical_features] = imputer.transform(x[numerical_features])

# Load preprocessor and transform data
preprocessor = joblib.load('Flask/encoder.pkl')
x_processed = preprocessor.transform(x)

# Scale the processed data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_processed)

# Save the scaler
joblib.dump(scaler, 'Flask/scale.pkl')

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)

# Initialize models
lin_reg = linear_model.LinearRegression()
Dtree = tree.DecisionTreeRegressor()
Rand = ensemble.RandomForestRegressor()
svr = svm.SVR()
XGB = xgboost.XGBRegressor()

# Fit models
lin_reg.fit(x_train, y_train)
Dtree.fit(x_train, y_train)
Rand.fit(x_train, y_train)
svr.fit(x_train, y_train)
XGB.fit(x_train, y_train)

# Predict on training data
p1 = lin_reg.predict(x_train)
p2 = Dtree.predict(x_train)
p3 = Rand.predict(x_train)
p4 = svr.predict(x_train)
p5 = XGB.predict(x_train)

# Evaluate on training data
print("Training R2 Scores:")
print(metrics.r2_score(p1, y_train))
print(metrics.r2_score(p2, y_train))
print(metrics.r2_score(p3, y_train))
print(metrics.r2_score(p4, y_train))
print(metrics.r2_score(p5, y_train))

# Predict on test data
p1 = lin_reg.predict(x_test)
p2 = Dtree.predict(x_test)
p3 = Rand.predict(x_test)
p4 = svr.predict(x_test)
p5 = XGB.predict(x_test)

# Evaluate on test data
print("\nTest R2 Scores:")
print(metrics.r2_score(p1, y_test))
print(metrics.r2_score(p2, y_test))
print(metrics.r2_score(p3, y_test))
print(metrics.r2_score(p4, y_test))
print(metrics.r2_score(p5, y_test))

# Calculate RMSE for RandomForest
MSE = metrics.mean_squared_error(p3, y_test)
print("\nRMSE for RandomForest:", np.sqrt(MSE))

# Save the best model (RandomForest)
joblib.dump(Rand, 'Flask/model.pkl')

print("Model and preprocessing objects saved successfully!")