import cv2 
import numpy as np 
import onnxruntime as ort 
import time 
import winsound  # For Windows audio alerts
import threading  # For non-blocking audio playback
 
# Load model & labels 
session = ort.InferenceSession("C:\\Users\\jason\\temp1\\posture-detection\\src\\resnet18.onnx", providers=["CPUExecutionProvider"]) 
inp_name = session.get_inputs()[0].name 
out_name = session.get_outputs()[0].name 
 
with open("C:\\Users\\jason\\temp1\\posture-detection\\src\\labels.txt", "r") as f: 
    labels = [l.strip().lower() for l in f.readlines()] 
 
# Initialize camera 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
if not cap.isOpened(): 
    raise RuntimeError("Could not open webcam") 

# Set desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
print("Posture tracking started. Press 'q' to stop.\n") 
 
# Posture tracking vars 
current_posture = None 
start_time = time.time() 
session_start_time = time.time()
good_time = 0 
bad_time = 0 
switch_message = "" 
last_switch_time = 0 
font = cv2.FONT_HERSHEY_SIMPLEX

# Audio alert vars
last_audio_alert_time = 0
audio_alert_interval = 10  # Seconds between audio alerts
bad_posture_duration_for_alert = 5  # Seconds in bad posture before alerting

# Audio alert function
def play_alert():
    # Play a beep sound (frequency=1000, duration=500ms)
    winsound.Beep(1000, 500)

# Helper function to format time as minutes:seconds
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
 
        # Run model 
        preds = session.run([out_name], {inp_name: img})[0] 
        class_id = int(preds.argmax()) 
        score = float(preds[0, class_id]) 
        label = labels[class_id] 
 
        now = time.time() 
        elapsed_session_time = now - session_start_time
 
        if current_posture is None: 
            current_posture = label 
            start_time = now 
 
        elif label != current_posture: 
            # Update time spent in previous posture 
            duration = now - start_time 
            if current_posture == "good-posture": 
                good_time += duration 
            elif current_posture == "bad-posture":
                bad_time += duration 

            # Update posture + switch overlay 
            switch_message = f"Switched to {label.upper()}" 
            last_switch_time = now 
            current_posture = label 
            start_time = now 

            # Play audio alert 4 times immediately if switched to bad-posture
            if label == "bad-posture":
                for _ in range(4):
                    play_alert()

        
        # Calculate current duration for display
        current_duration = now - start_time
        
        # Check if we need to play an audio alert for bad posture
        if label == "bad-posture":
            if current_duration > bad_posture_duration_for_alert and now - last_audio_alert_time > audio_alert_interval:
                # Use a thread to play audio to avoid blocking the main loop
                threading.Thread(target=play_alert).start()
                last_audio_alert_time = now
        
        # Create a semi-transparent overlay for the UI
        overlay = frame.copy()
        ui_height = 180
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], ui_height), (40, 40, 40), -1)
        
        # Apply the overlay with proper transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add divider line
        cv2.line(frame, (0, ui_height), (frame.shape[1], ui_height), (200, 200, 200), 2)
        
        # Current posture status with color-coded indicator
        if label == "good-posture":
            status_color = (0, 255, 0)  # Green for good posture
            status_text = "GOOD POSTURE"
        else:
            status_color = (0, 0, 255)  # Red for bad posture
            status_text = "BAD POSTURE"
            
        # Draw current posture status
        cv2.putText(frame, status_text, (20, 40), font, 1.2, status_color, 2)
        
        # Draw confidence score
        conf_text = f"Confidence: {score * 100:.1f}%"
        cv2.putText(frame, conf_text, (20, 80), font, 0.7, (200, 200, 200), 1)
        
        # Time stats
        session_time_text = f"Session: {format_time(elapsed_session_time)}"
        cv2.putText(frame, session_time_text, (frame.shape[1] - 210, 40), font, 0.8, (200, 200, 200), 1)
        
        good_time_text = f"Good Posture: {format_time(good_time + (current_duration if label == 'good-posture' else 0))}"
        cv2.putText(frame, good_time_text, (frame.shape[1] - 310, 80), font, 0.8, (0, 255, 0), 1)
        
        bad_time_text = f"Bad Posture: {format_time(bad_time + (current_duration if label == 'bad-posture' else 0))}"
        cv2.putText(frame, bad_time_text, (frame.shape[1] - 310, 120), font, 0.8, (0, 0, 255), 1)
        
        # Current streak
        streak_text = f"Current streak: {format_time(current_duration)}"
        cv2.putText(frame, streak_text, (20, 120), font, 0.7, (200, 200, 200), 1)
        
        # Show switch message (for 2 sec) with fade effect
        if time.time() - last_switch_time < 2 and switch_message:
            fade_factor = 1.0 - (time.time() - last_switch_time) / 2.0
            text_color = (0, 255, 255)
            
            # Calculate text size for centering
            text_size = cv2.getTextSize(switch_message, font, 1.5, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = ui_height + 70
            
            # Add background for better visibility
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - 40), 
                         (text_x + text_size[0] + 10, text_y + 10), 
                         (0, 0, 0), 
                         -1)
            
            # Fixed color calculation for fade effect
            fade_color = tuple(int(c * fade_factor) for c in text_color)
            cv2.putText(frame, 
                       switch_message,
                       (text_x, text_y), 
                       font, 
                       1.5, 
                       fade_color, 
                       2)
 
        # Instructions at bottom
        cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0] - 20), font, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Posture Detection", frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
 
    # Final posture duration 
    final_duration = time.time() - start_time 
    if current_posture == "good-posture": 
        good_time += final_duration 
    elif current_posture == "bad-posture":
        bad_time += final_duration 
 
except KeyboardInterrupt: 
    final_duration = time.time() - start_time 
    if current_posture == "good-posture": 
        good_time += final_duration 
    elif current_posture == "bad-posture":
        bad_time += final_duration 

except Exception as e:
    print(f"Error: {e}")
    
finally: 
    total_session_time = time.time() - session_start_time
    print("\nTracking stopped.") 
    print(f"Total session time: {format_time(total_session_time)}")
    print(f"Good posture time: {format_time(good_time)} ({good_time/total_session_time*100:.1f}%)")
    print(f"Bad posture time: {format_time(bad_time)} ({bad_time/total_session_time*100:.1f}%)")
 
    cap.release() 
    cv2.destroyAllWindows()