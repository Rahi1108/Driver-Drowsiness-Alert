import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

# --- CONSTANTS & THRESHOLDS ---
# Eye Aspect Ratio thresholds
EAR_THRESHOLD = 0.21  
CONSECUTIVE_FRAMES = 20 # Number of frames to confirm eyes are closed

# Mouth Aspect Ratio thresholds (Yawn detection)
MAR_THRESHOLD = 0.5
YAWN_TIME_LIMIT = 1.0 

# Colors (BGR)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# --- MEDIAPIPE ASSETS ---
mp_face_mesh = mp.solutions.face_mesh
# Landmark indices for EAR and MAR
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308] # Upper lip, lower lip, left corner, right corner

def calculate_ear(landmarks, eye_indices):
    # Vertical distances
    v1 = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) - 
                        np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
    v2 = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) - 
                        np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
    # Horizontal distance
    h = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) - 
                       np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
    return (v1 + v2) / (2.0 * h)

def calculate_mar(landmarks, mouth_indices):
    # Vertical distance (center of lips)
    v = np.linalg.norm(np.array([landmarks[mouth_indices[0]].x, landmarks[mouth_indices[0]].y]) - 
                       np.array([landmarks[mouth_indices[1]].x, landmarks[mouth_indices[1]].y]))
    # Horizontal distance (corners of mouth)
    h = np.linalg.norm(np.array([landmarks[mouth_indices[2]].x, landmarks[mouth_indices[2]].y]) - 
                       np.array([landmarks[mouth_indices[3]].x, landmarks[mouth_indices[3]].y]))
    return v / h

# --- INITIALIZATION ---
cap = cv2.VideoCapture(0)
frame_counter = 0
yawn_start_time = 0
is_yawning = False

with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # 1. EAR Calculation
                left_ear = calculate_ear(landmarks, LEFT_EYE)
                right_ear = calculate_ear(landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0

                # 2. MAR Calculation (Yawn)
                mar = calculate_mar(landmarks, MOUTH)

                # --- DROWSINESS LOGIC ---
                if avg_ear < EAR_THRESHOLD:
                    frame_counter += 1
                    if frame_counter >= CONSECUTIVE_FRAMES:
                        cv2.putText(frame, "DROWSINESS DETECTED!", (50, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 3)
                        winsound.Beep(1500, 200)
                else:
                    frame_counter = 0

                # --- YAWN LOGIC ---
                if mar > MAR_THRESHOLD:
                    if not is_yawning:
                        yawn_start_time = time.time()
                        is_yawning = True
                    
                    if (time.time() - yawn_start_time) > YAWN_TIME_LIMIT:
                        cv2.putText(frame, "YAWN ALERT!", (50, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, YELLOW, 2)
                else:
                    is_yawning = False

                # --- ADVANCED HUD (No Mesh) ---
                # Draw subtle indicators only for eyes/mouth
                for idx in LEFT_EYE + RIGHT_EYE:
                    pt = landmarks[idx]
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, GREEN, -1)
                
                # Display Stats
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (w-150, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, CYAN, 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (w-150, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, CYAN, 2)

        cv2.imshow('Advanced Driver Fatigue Monitor', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()