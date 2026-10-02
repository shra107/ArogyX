import sqlite3
import pandas as pd

conn = sqlite3.connect("fraud_detection.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS mrp_reference")
cursor.execute("CREATE TABLE mrp_reference (item TEXT PRIMARY KEY, mrp_price REAL)")

try:
    df = pd.read_csv("mrp_master.csv")
    df.to_sql("mrp_reference", conn, if_exists="append", index=False)
    print("✅ Database loaded with MRP dataset!")
except Exception as e:
    print("❌ Error loading mrp_master.csv:", e)

conn.commit()
conn.close()