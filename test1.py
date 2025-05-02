import cv2
import numpy as np
import onnxruntime as ort
import time
import winsound  # For Windows audio alerts
import threading  # For non-blocking audio playback
import sys

# Config
MODEL_PATH = "C:\\Users\\jason\\temp1\\posture-detection\\src\\resnet18.onnx"
LABELS_PATH = "C:\\Users\\jason\\temp1\\posture-detection\\src\\labels.txt"
CAMERA_INDEX = 0  # adjust if needed

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [l.strip().lower() for l in f.readlines()]

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}")

# Set desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Posture tracking started. Press 'q' to stop.\n")

# Tracking vars
current_posture = None
session_start_time = time.time()
start_time = session_start_time
good_time = 0.0
bad_time = 0.0

# Audio alert vars
last_audio_alert_time = 0.0
audio_alert_interval = 10.0  # seconds
bad_posture_duration_for_alert = 5.0  # seconds

# Helper functions
def play_alert():
    winsound.Beep(1000, 500)

def format_time(seconds):
    minutes = int(seconds) // 60
    seconds = int(seconds) % 60
    return f"{minutes:02d}:{seconds:02d}"

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))[None, ...]

        # Run inference
        preds = session.run([out_name], {inp_name: img})[0]
        class_id = int(preds.argmax())
        score = float(preds[0, class_id])
        label = labels[class_id]

        now = time.time()
        elapsed_session = now - session_start_time

        # Initialize current posture
        if current_posture is None:
            current_posture = label
            start_time = now

        # Handle posture switch
        elif label != current_posture:
            duration = now - start_time
            if current_posture == "good-posture":
                good_time += duration
            else:
                bad_time += duration

            # Update posture and overlay message
            switch_message = f"Switched to {label.upper()}"
            last_switch_time = now
            current_posture = label
            start_time = now

            # If switched to bad posture: beep and freeze frame
            if label == "bad-posture":
                # Immediate audio alert
                for _ in range(4):
                    play_alert()

                # Freeze the last frame for 5 seconds
                freeze_start = time.time()
                while time.time() - freeze_start < 5.0:
                    # Display the frozen frame with overlay
                    overlay = frame.copy()
                    ui_h = 180
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], ui_h), (40, 40, 40), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    cv2.putText(frame, "BAD POSTURE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    cv2.putText(frame, "Fix your sitting position", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                    cv2.imshow("Posture Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt()

                # After freeze, continue loop

        # Ongoing alert for sustained bad posture
        current_duration = now - start_time
        if label == "bad-posture" and \
           current_duration > bad_posture_duration_for_alert and \
           now - last_audio_alert_time > audio_alert_interval:
            threading.Thread(target=play_alert).start()
            last_audio_alert_time = now

        # Draw UI overlay
        overlay = frame.copy()
        ui_h = 180
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], ui_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.line(frame, (0, ui_h), (frame.shape[1], ui_h), (200, 200, 200), 2)

        # Status text
        status_color = (0, 255, 0) if label == "good-posture" else (0, 0, 255)
        status_text = label.replace('-', ' ').upper()
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2)
        cv2.putText(frame, f"Confidence: {score*100:.1f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Time stats
        cv2.putText(frame, f"Session: {format_time(elapsed_session)}", (frame.shape[1]-220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        cv2.putText(frame, f"Good Posture: {format_time(good_time + (current_duration if label=='good-posture' else 0))}",
                    (frame.shape[1]-320, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        cv2.putText(frame, f"Bad  Posture: {format_time(bad_time + (current_duration if label=='bad-posture' else 0))}",
                    (frame.shape[1]-320, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        # Footer
        cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Posture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Final duration tally
    final_duration = time.time() - start_time
    if current_posture == "good-posture":
        good_time += final_duration
    else:
        bad_time += final_duration

except KeyboardInterrupt:
    pass

finally:
    total_time = time.time() - session_start_time
    print("\nTracking stopped.")
    print(f"Total session time: {format_time(total_time)}")
    print(f"Good posture time: {format_time(good_time)} ({good_time/total_time*100:.1f}%)")
    print(f"Bad posture time:  {format_time(bad_time)} ({bad_time/total_time*100:.1f}%)")
    cap.release()
    cv2.destroyAllWindows()
