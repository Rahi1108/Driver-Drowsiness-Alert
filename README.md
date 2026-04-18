# 🛡️ Advanced Driver Fatigue Monitor

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00a0a0?style=for-the-badge&logo=google&logoColor=white)


An advanced, real-time computer vision system designed to detect driver drowsiness and fatigue. Unlike basic distance-based trackers, this system utilizes **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** to provide a robust, non-intrusive safety solution.

-----

## 🚀 Key Features

  * **EAR-Based Drowsiness Detection:** Uses a 6-point landmark system for each eye to calculate precise closure duration, distinguishing between natural blinks and fatigue-induced micro-sleeps.
  * **MAR-Based Yawn Detection:** Monitors mouth geometry to identify frequent yawning, an early indicator of driver exhaustion.
  * **Minimalist Professional HUD:** No distracting face meshes—only subtle tracking indicators for a clean, production-ready interface.
  * **Real-time Audio Alerts:** Integrated with `winsound` to provide immediate feedback when danger is detected.

-----

## 📐 How It Works

The system identifies 468 3D face landmarks using **MediaPipe Face Mesh**. It then isolates specific coordinates to calculate ratios:

### 1\. Eye Aspect Ratio (EAR)

The EAR formula determines if the eye is open or closed based on the distance between vertical and horizontal landmarks:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2||p_1 - p_4||}$$

### 2\. Mouth Aspect Ratio (MAR)

Detects the vertical expansion of the lips relative to the horizontal width to identify a yawn state.

-----

## 🛠️ Installation & Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Rahi1108/Driver-Drowsiness-Alert.git
    cd Driver-Drowsiness-Alert
    ```

2.  **Install dependencies:**

    ```bash
    pip install opencv-python mediapipe numpy
    ```

3.  **Run the application:**

    ```bash
    python alert.py
    ```

    *Press `ESC` to exit the application.*

-----

## 🧠 Technical Highlights

  * **Smoothing:** Uses frame-consecutive logic (`CONSECUTIVE_FRAMES`) to eliminate false positives from rapid blinking.
  * **Performance:** Optimized for CPU-only environments by disabling heavy mesh rendering.
  * **Scalability:** The logic is modular, allowing for easy integration into larger vehicle-telematics systems.

