import cv2
import time
import pytesseract
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Path model YOLOv8s
MODEL_PATH = './seperate-v8s.pt'

# Define Class ID : 0 for char, 1 for province
CHAR_CLASS_ID = 0
PROVINCE_CLASS_ID = 1

TESSERACT_CHAR_CONFIG = '--psm 6 -l tha'
TESSERACT_PROVINCE_CONFIG = '--psm 6 -l tha'

# --- FUNCTIONS ---

def process_frame_for_ocr(model, frame):
    """run YOLO and Tesseract on frame that detected"""
    print("\n--- start scan OCR... ---")
    
    # Run YOLO Prediction
    # use frame directly replace using path
    results = model.predict(
        frame,
        conf=0.5,
        iou=0.7,
        imgsz=640,
        verbose=False,
        device='0' # select device for run (cpu = 'cpu', gpu = '0')
    )
    
    # store data that model detected
    detections = results[0].boxes.data.cpu().numpy()
    
    # variable Bounding Box
    char_boxes = []
    province_box = None
    
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det.astype(int)
        
        # seperate by class_id
        if class_id == CHAR_CLASS_ID:
            # (x1, y1, x2, y2) = rectangle that model 
            # x1 = position of rectangle
            char_boxes.append(((x1, y1, x2, y2), x1)) 
        elif class_id == PROVINCE_CLASS_ID:
            province_box = (x1, y1, x2, y2)

    # LETTER
    plate_chars = ""
    
    if char_boxes:
        # sort left to right letter
        char_boxes.sort(key=lambda x: x[1]) 
        
        # create new box for tesseract
        all_x1 = [box[0] for box, _ in char_boxes]
        all_y1 = [box[1] for box, _ in char_boxes]
        all_x2 = [box[2] for box, _ in char_boxes]
        all_y2 = [box[3] for box, _ in char_boxes]
        
        # define position for box
        top_line_bbox = (min(all_x1), min(all_y1), max(all_x2), max(all_y2))
        
        # crop letter
        x_min, y_min, x_max, y_max = top_line_bbox
        cropped_char_img = frame[y_min:y_max, x_min:x_max]
        
        # change color to gray scale
        gray_char_img = cv2.cvtColor(cropped_char_img, cv2.COLOR_BGR2GRAY)
        
        # run tesseract for scan letter
        try:
            # change image to array
            plate_chars = pytesseract.image_to_string(
                Image.fromarray(gray_char_img), 
                config=TESSERACT_CHAR_CONFIG
            ).strip().replace(' ', '') # for delete space that tesseract create
        except Exception as e:
            print(f"❌ Error during Tesseract OCR (letter): {e}")

    # PROVINCE
    plate_province = ""
    if province_box:
        x1, y1, x2, y2 = province_box
        cropped_province_img = frame[y1:y2, x1:x2]
        
        # change color to grayscale
        gray_province_img = cv2.cvtColor(cropped_province_img, cv2.COLOR_BGR2GRAY)
        
        # run tesseract for scan province
        try:
            plate_province = pytesseract.image_to_string(
                Image.fromarray(gray_province_img), 
                config=TESSERACT_PROVINCE_CONFIG
            ).strip()
        except Exception as e:
            print(f"❌ Error during Tesseract OCR (Province): {e}")

    # show result
    if plate_chars or plate_province:
        print("--------------------------------------------------")
        print(f"   Letter: {plate_chars}")
        print(f"   Province: {plate_province}")
        print("--------------------------------------------------")
    else:
        print("letter and province not found")
    
    return results[0].plot() # return frame that annotated

# --- MAIN EXECUTION ---

# Load model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Error: Cannot load model from {MODEL_PATH}, Error detail: {e}")
    exit()

# connect camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot connect to camera")
    exit()

print("Connected camera")
print(f"model (Path: {MODEL_PATH}) loaded")
print("---------------------------------------------------------")
print("ℹ️  press 's' for scan OCR")
print("ℹ️  press 'q' for quit program")
print("---------------------------------------------------------")

prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        annotated_frame = frame # using origin frame
        key = cv2.waitKey(1) & 0xFF
        
        # --- checking for 's' button ---
        if key == ord('s'):
            # run yolo and ocr
            annotated_frame = process_frame_for_ocr(model, frame)
        else:
            # run yolo realtime if no 's' button input
            results = model.predict(frame, conf=0.5, iou=0.7, imgsz=640, verbose=False, device='cpu')
            annotated_frame = results[0].plot()
        
        # show FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        cv2.putText(annotated_frame, f"FPS: {fps:.2f} | Press 's' to Scan", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv8 LPR (Press s to scan)", annotated_frame)
        
        # --- checking for 'q' button ---
        if key == ord('q'):
            break

except Exception as e:
    print(f"An unexpected error occurred in main loop: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped program")
