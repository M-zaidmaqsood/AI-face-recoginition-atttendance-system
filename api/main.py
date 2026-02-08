from fastapi import FastAPI
import cv2
import mediapipe as mp
import threading

app = FastAPI()

cap = None
running = False
thread = None

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(0.5)

def recognition_loop():
    global cap, running

    cap = cv2.VideoCapture(0)

    while running:
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
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)

        cv2.imshow("Recognition (API)", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            stop_recognition()
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

@app.post("/start")
def start_recognition():
    global running, thread

    if running:
        return {"message": "Recognition already running"}

    running = True
    thread = threading.Thread(target=recognition_loop)
    thread.start()

    return {"message": "Recognition started"}

@app.post("/stop")
def stop_recognition():
    global running
    running = False
    return {"message": "Recognition stopped"}
