import cv2
import os
import numpy as np
import mediapipe as mp

DATASET_PATH = "dataset"

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

faces = []
labels = []
label_map = {}
current_label = 0

print("📌 Starting training process...")

for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name
    print(f"🔹 Label {current_label} → {person_name}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_detection.process(rgb)

        if not result.detections:
            continue

        h, w, _ = image.shape

        for det in result.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            face = image[y:y+bh, x:x+bw]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))

            faces.append(gray)
            labels.append(current_label)
            break

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

recognizer.save("face_model.yml")
np.save("labels.npy", label_map)

print("✅ Training complete!")
print("💾 Model saved as face_model.yml")
print("💾 Labels saved as labels.npy")
