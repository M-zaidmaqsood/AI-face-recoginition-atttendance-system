import cv2
import numpy as np
import mediapipe as mp
import csv
import os
from datetime import datetime

# Load model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")
label_map = np.load("labels.npy", allow_pickle=True).item()

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(0.5)

ATTENDANCE_FILE = "attendance/attendance.csv"

# Create file with header if not exists
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

marked_today = set()
today = datetime.now().strftime("%Y-%m-%d")

cap = cv2.VideoCapture(0)

print("📋 Attendance system started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(rgb)

    if result.detections:
        h, w, _ = frame.shape

        for det in result.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            face = frame[y:y+bh, x:x+bw]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))

            label, confidence = recognizer.predict(gray)
            name = label_map.get(label, "Unknown")

            if confidence < 65:
                if name not in marked_today:
                    time_now = datetime.now().strftime("%H:%M:%S")

                    with open(ATTENDANCE_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, today, time_now])

                    marked_today.add(name)
                    print(f"✅ Attendance marked for {name}")

                display = name
                color = (0, 255, 0)
            else:
                display = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(
                frame,
                display,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
