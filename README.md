# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts house prices using **Supervised Machine Learning** and **Linear Regression**.

The model learns from historical house data and predicts the price of a new house based on its location and construction details.

## 🎯 Project Objectives

- Predict house prices using Linear Regression.
- Understand supervised learning and regression.
- Perform data splitting into training and testing sets.
- Convert categorical data into numerical values using One-Hot Encoding.
- Evaluate model performance using regression metrics.
- Predict house prices using user input from the terminal.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning |
| Linear Regression | Price prediction |

## 📂 Features Used

The model uses the following input features:

- `street` – Street name
- `city` – City name
- `country` – Country name
- `yr_built` – Year the house was built
- `yr_renovated` – Year the house was renovated

**Target variable:** `price`

## ⚙️ Machine Learning Workflow

1. Load the dataset using Pandas.
2. Select independent variables (`X`) and dependent variable (`y`).
3. Separate categorical and numerical features.
4. Apply One-Hot Encoding to categorical columns.
5. Split the dataset into 80% training and 20% testing data.
6. Train the Linear Regression model.
7. Predict house prices.
8. Evaluate model performance.
9. Accept new house details through terminal input.

## 📊 Model Evaluation

The model is evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
- **MSE (Mean Squared Error):** Measures the average squared prediction error.
- **R² Score:** Measures how much variation in house prices is explained by the model.

The actual metric values are displayed when the program runs.

## 📈 Data Visualizations

### 1. Actual vs Predicted Prices

Compares actual house prices with the prices predicted by the model.

### 2. Residual Plot

Displays prediction errors (residuals) to help examine model performance.

### 3. House Price Distribution

Shows the distribution of house prices in the dataset.


## 💻 How to Run the Project

**1. Clone the repository**

```bash
git clone https://github.com/Rainunstabble/House-prediction
```

**2. Open the project directory**

```bash
cd House-prediction
```

**3. Install required libraries**

```bash
pip install pandas numpy matplotlib scikit-learn
```

**4. Run the Python file**

```bash
python house.py
```

**5. Enter house details**

The program asks for:
- Street name
- City name
- Country name
- Year built
- Year renovated

It then displays the predicted house price.

## 📌 Future Improvements

- Experiment with Random Forest Regression.
- Compare multiple regression algorithms.
- Improve preprocessing and feature selection.
- Use a larger and more representative dataset.
- Improve prediction accuracy through model tuning.

## 👨‍💻 Project Type

Machine Learning | Supervised Learning | Regression | Python
