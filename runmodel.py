import cv2
import time
# from picamera2 import Picamera2 # Removed: Not needed for standard webcams
# from libcamera import controls  # Removed: Not needed for standard webcams
from ultralytics import YOLO

MODEL_PATH = './seperate-v8s-test.pt'
model = YOLO(MODEL_PATH) 

# --- Webcam Setup Changes ---
# Use cv2.VideoCapture(0) for the default (usually built-in) webcam.
# Change '0' to '1', '2', etc., if you have multiple cameras and need a different one.
cap = cv2.VideoCapture(0) 

# Check if the webcam was opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open video stream or file (Webcam not found/available).")
    exit()

# Set desired resolution for the webcam (optional, but good practice)
CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
# Note: Autofocus control is typically managed by the operating system/driver
# and is not usually set via a simple line like in the Picamera2 library.

print("Starting object detection on laptop webcam...")

prev_time = time.time()

try:
    while True:
        # 1. Read a frame from the webcam
        ret, frame = cap.read() 
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # 2. Perform YOLO prediction
        results = model.predict(
            frame, 
            conf=0.5,
            iou=0.7,
            imgsz=640,
            verbose=False,
            device='cpu'
        )
        
        # 3. Annotate the frame
        annotated_frame = results[0].plot()

        # 4. Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        cv2.putText(
            annotated_frame, 
            f"FPS: {fps:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2
        )
        
        # 5. Display the result
        cv2.imshow("YOLOv8 Detection (Laptop Webcam)", annotated_frame)
        
        # 6. Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # --- Cleanup Changes ---
    cap.release() # Release the webcam object
    cv2.destroyAllWindows()
    print("Operation stopped and resources released.")
