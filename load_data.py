import pandas as pd
import sqlite3

# Load CSV
df = pd.read_csv("healthcare_dataset.csv")

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Connect to SQLite
conn = sqlite3.connect("database.db")

# Save to SQL table
df.to_sql("healthcare", conn, if_exists="replace", index=False)

conn.close()

print("✅ Dataset loaded into SQLite database successfully")
