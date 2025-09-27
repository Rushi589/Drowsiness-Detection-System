import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp

# ---------------- Load trained Yawn model ----------------
model = load_model("yawn_model.h5")
class_labels = {0: "No Yawn", 1: "Yawn"}
IMG_SIZE = (32, 32)

# ---------------- Mediapipe Face Mesh ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# EAR threshold and frame counter
EAR_THRESH = 0.25
EAR_CONSEC_FRAMES = 15
closed_frames = 0

# ---------------- Video Capture ----------------
cap = cv2.VideoCapture(0)  # 0 = webcam

def calculate_EAR(landmarks, eye_indices, w, h):
    """Calculate Eye Aspect Ratio (EAR)"""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    EAR = (A + B) / (2.0 * C)
    return EAR

# Eye indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    h, w = frame.shape[:2]

    # ---------------- Preprocess frame for Yawn ----------------
    face = cv2.resize(frame, IMG_SIZE)
    face = face.astype("float32") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)  # batch
    face = np.expand_dims(face, axis=1)  # time step -> (1, 1, 32, 32, 3)

    preds = model.predict(face, verbose=0)
    yawn_label = np.argmax(preds, axis=1)[0]

    # ---------------- Eye Blink Detection ----------------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    EAR = 0
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            left_EAR = calculate_EAR(landmarks.landmark, LEFT_EYE_IDX, w, h)
            right_EAR = calculate_EAR(landmarks.landmark, RIGHT_EYE_IDX, w, h)
            EAR = (left_EAR + right_EAR) / 2.0

            if EAR < EAR_THRESH:
                closed_frames += 1
            else:
                closed_frames = 0

    # ---------------- Drowsiness Alert ----------------
    if closed_frames >= EAR_CONSEC_FRAMES or yawn_label == 1:
        cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Display labels
    text = class_labels[yawn_label]
    cv2.putText(frame, f"Yawn: {text}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"EAR: {EAR:.2f}", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()







