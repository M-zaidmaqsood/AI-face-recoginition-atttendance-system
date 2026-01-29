import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

print("✅ Camera opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            mp_draw.draw_detection(frame, detection)

    # Show output
    cv2.imshow("Face Detection Test", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
