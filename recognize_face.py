import cv2
import numpy as np
import mediapipe as mp

# Load trained model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")
label_map = np.load("labels.npy", allow_pickle=True).item()

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("🎥 Camera started. Press Q to quit.")

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

            # LBPH: lower confidence = better match
            if confidence < 70:
                text = f"{name} ({int(confidence)})"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(
                frame, text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv2.imshow("AI Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
