import sqlite3
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()
cursor.execute("SELECT person_id, name, role FROM persons")
for row in cursor.fetchall():
    print(row)
conn.close()
