import sqlite3
from pathlib import Path

db_path = Path(r"C:\Users\aravi\OneDrive\Desktop\proj\sih26\anpr-trajectory-dashboard\anpr_demo.db")


if not db_path.exists():
    raise FileNotFoundError(f"Database file not found:\n{db_path}")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name;
""")

print("Tables:", cur.fetchall())

cur.execute("SELECT COUNT(*) FROM events;")

for row in cur.fetchall():
    print(row)

conn.close()