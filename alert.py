import cv2
import mediapipe as mp
import math
import time
import winsound  # Built-in Windows library for sound

# --- INITIALIZATION ---
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

# --- CONFIGURATION ---
EYE_AR_THRESHOLD = 0.012  # Threshold for closed eyes
DROWSY_TIME_LIMIT = 1.5   # Seconds before alarm sounds
NEON_CYAN = (255, 255, 0)
ALARM_RED = (0, 0, 255)

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Timer variables
eye_closed_start_time = 0
is_eye_closed = False

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. Calculate Eye Distance
                p_top = face_landmarks.landmark[RIGHT_EYE_TOP]
                p_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM]
                dist = get_distance(p_top, p_bottom)

                # 2. Drowsiness Logic
                if dist < EYE_AR_THRESHOLD:
                    if not is_eye_closed:
                        # Eye just closed, start the timer
                        eye_closed_start_time = time.time()
                        is_eye_closed = True
                    
                    # Calculate how long eyes have been closed
                    elapsed_time = time.time() - eye_closed_start_time
                    
                    if elapsed_time >= DROWSY_TIME_LIMIT:
                        # TRIGGER ALARM
                        cv2.putText(image, "!!! WAKE UP !!!", (w//2 - 150, h//2), 
                                    cv2.FONT_HERSHEY_TRIPLEX, 1.5, ALARM_RED, 3)
                        # winsound.Beep(Frequency, Duration_in_ms)
                        winsound.Beep(1000, 100) 
                else:
                    # Eye is open, reset timer logic
                    is_eye_closed = False
                    eye_closed_start_time = 0

                # 3. Aesthetics (Mesh)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=NEON_CYAN, thickness=1, circle_radius=0))

        cv2.imshow('Drowsiness Alert System', image)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()