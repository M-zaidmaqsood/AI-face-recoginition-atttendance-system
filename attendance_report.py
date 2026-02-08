import os
import pandas as pd

ATTENDANCE_FILE = "attendance/attendance.csv"

# Create file with header if not exists
if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
    with open(ATTENDANCE_FILE, "w") as f:
        f.write("Name,Date,Time\n")

df = pd.read_csv(ATTENDANCE_FILE)

print("\n📊 Attendance Summary\n")
print(df["Name"].value_counts())

print("\n🗓 Daily Attendance\n")
print(df.groupby("Date")["Name"].count())
