import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

df = pd.read_csv("./data.csv")
print(df.columns.tolist(), sep="\n")

X = df[["street", "city", "country", "yr_built", "yr_renovated"]]
y = df["price"]

categorial_columns = ["street", "city", "country"]
numerical_column = ["yr_built", "yr_renovated"]

preporcessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorial_columns),
        ("num", "passthrough", numerical_column),
    ]
)

model = Pipeline(
    steps=[("preprocessor", preporcessor), ("regressor", LinearRegression())]
)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
)
# model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(Y_test, y_pred))
print("MSE:", mean_squared_error(Y_test, y_pred))
print("R2 Score:", r2_score(Y_test, y_pred))

strr = input("Enter Street Name: ").strip()
ci = input("Enter City Name: ").strip()
country = input("Enter Country Name: ").strip()
built = float(input("Enter Build year: "))
renovate = float(input("Enter Renovate Year: "))
new_house = pd.DataFrame({
    "street": [strr],
    "city": [ci],
    "country": [country],
    "yr_built": [built],
    "yr_renovated": [renovate]
})

predicted_price = model.predict(new_house)

def format(number):
    s = str(int(number))
    if len(s) <= 3:
        return s
    last_three = s[-3:]
    remaining = s[:-3]
    remaining = ",".join(
        [remaining[max(i-2, 0):i] for i in range(len(remaining), 0, -2)][::-1]
    )
    return remaining + "," + last_three

print("Predicted Price: ₹" + format(predicted_price[0]))
