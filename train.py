import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor  # Fixed import
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv("D:/DEMO COVID/source/pneumonia_covid_diagnosis_dataset (1).csv")

# Columns to label encode
columns = ["Gender", "fever", "cough", "Fatigue", "Breathlessness", "comorbidty", "type"]

# Apply Label Encoding
le = LabelEncoder()
for col in columns:
    df[col] = le.fit_transform(df[col])

# Drop the 'Is_curable' column
df = df.drop("Is_curable", axis=1)

# Features and target variable
x = df.drop(columns=["Survival_Rate"], axis=1)
y = df["Survival_Rate"]

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()  # Fixed class name
model.fit(x_train, y_train)

# Predict and save the model
prd = model.predict(x_test)
joblib.dump(model, "covid_diag.pkl")

# Display sample data
print(df.head())
