
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("./Airline Dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

df = df.drop_duplicates()
df = df.dropna(subset=["Flight Status"]).copy()

df["Departure Date"] = pd.to_datetime(
    df["Departure Date"],
    errors="coerce"
)

df["Departure Month"] = df["Departure Date"].dt.month
df["Departure Day"] = df["Departure Date"].dt.day
df["Departure Weekday"] = df["Departure Date"].dt.day_of_week



categorical_features = [
    "Airport Name",
    "Airport Country Code",
    "Airport Continent",
    "Arrival Airport"
]

numerical_features = [
    "Departure Month",
    "Departure Day",
    "Departure Weekday"
]

features = categorical_features + numerical_features

X = df[features]
y = df["Flight Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_pipeline, categorical_features),
    ("num", numerical_pipeline, numerical_features)
])


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

print("\nModel Training Completed!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

labels = model.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")
plt.title("Flight Status Prediction")
plt.show()

joblib.dump(model, "flight_status_model.pkl")

print("\nModel Saved Successfully!")




print("\n========== FLIGHT STATUS PREDICTION ==========")

airport_name = input("Enter Airport Name: ")
country_code = input("Enter Airport Country Code: ")
continent = input("Enter Airport Continent (NAM, AS, EU, etc.): ")
arrival_airport = input("Enter Arrival Airport Code: ")

month = int(input("Enter Departure Month (1-12): "))
day = int(input("Enter Departure Day (1-31): "))

departure_date = pd.to_datetime(
    f"2026-{month:02d}-{day:02d}",
    errors="coerce"
)

if pd.isna(departure_date):
    print("Invalid date entered!")
else:
    weekday = departure_date.day_of_week

    new_flight = pd.DataFrame([{
        "Airport Name": airport_name,
        "Airport Country Code": country_code,
        "Airport Continent": continent,
        "Arrival Airport": arrival_airport,
        "Departure Month": month,
        "Departure Day": day,
        "Departure Weekday": weekday
    }])

    prediction = model.predict(new_flight)

    print("\n========== PREDICTION RESULT ==========")
    print("Predicted Flight Status:", prediction[0])

    probabilities = model.predict_proba(new_flight)

    print("\nPrediction Probabilities:")

    for status, probability in zip(model.classes_, probabilities[0]):
        print(status, ":", round(probability * 100, 2), "%")

