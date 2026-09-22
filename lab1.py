import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("CarPrice_Assignment.csv")
print("First 5 records:")
print(df.head())
print("\nLast 5 records:")
print(df.tail())
print("\nDataset shape:", df.shape)
print("\nNumerical attributes:")
print(df.select_dtypes(include=np.number).columns.tolist())
print("\nCategorical attributes:")
print(df.select_dtypes(include="object").columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate records:", df.duplicated().sum())
print("\nDescriptive statistics:")
print(df.describe())

df = df.drop_duplicates()
df = df.drop(columns=["car_ID"])
X = df.drop(columns=["price"])
y = df["price"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True)
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

correlation = df.select_dtypes(include=np.number).corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="horsepower", y="price")
plt.title("Horsepower vs Price")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="enginesize", y="price")
plt.title("Enginesize vs Price")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="curbweight", y="price")
plt.title("Curbweight vs Price")
plt.show()

print("\nCorrelation with price:")
print(correlation["price"].sort_values(ascending=False))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()

new_car = pd.DataFrame([{
    "symboling": 0,
    "CarName": "toyota camry",
    "fueltype": "gas",
    "aspiration": "std",
    "doornumber": "four",
    "carbody": "sedan",
    "drivewheel": "fwd",
    "enginelocation": "front",
    "wheelbase": 100.0,
    "carlength": 190.0,
    "carwidth": 70.0,
    "carheight": 55.0,
    "curbweight": 3000,
    "enginetype": "ohc",
    "cylindernumber": "four",
    "enginesize": 150,
    "fuelsystem": "mpfi",
    "boreratio": 3.2,
    "stroke": 3.2,
    "compressionratio": 9.0,
    "horsepower": 110,
    "peakrpm": 5500,
    "citympg": 25,
    "highwaympg": 31
}])
predicted_price = model.predict(new_car)
print("\nPredicted price for new car:", predicted_price[0])