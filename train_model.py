import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data from SQLite
conn = sqlite3.connect("database.db")
df = pd.read_sql_query("SELECT * FROM healthcare", conn)
conn.close()

# Drop non-useful columns
drop_cols = [
    "name", "doctor", "hospital",
    "date_of_admission", "discharge_date"
]
df = df.drop(columns=drop_cols)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features & target
X = df.drop("test_results", axis=1)
y = df["test_results"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model & encoders
joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "encoders.pkl")

print("✅ Model and encoders saved")
