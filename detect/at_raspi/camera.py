# camera.py
#  from picamera2 import Picamera2

# picam2 = None

# def init_camera():
#     global picam2
#     picam2 = Picamera2()
#     config = picam2.create_preview_configuration(
#         main={"format": "RGB888", "size": (1600, 1080)}
#     )
#     picam2.configure(config)
#     picam2.start()
#     picam2.set_controls({"AfMode": 2})
#     print("📷 Camera ready")

# def capture_frame():
#     return picam2.capture_array()

import cv2
import time

# ตัวแปรเก็บการเชื่อมต่อกล้อง
cap = None

def init_camera():
    global cap
    # เปิดใช้งาน Webcam (0 คือกล้องตัวแรกของเครื่อง)
    cap = cv2.VideoCapture(0)
    
    # เช็คว่าเปิดกล้องได้ไหม
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    # รอให้กล้องปรับแสงเล็กน้อย
    time.sleep(2)
    print("📷 Camera ready (Webcam Simulation)")

def capture_frame():
    global cap
    if cap is None or not cap.isOpened():
        print("⚠️ Camera not initialized")
        return None
        
    # อ่านภาพจาก Webcam
    ret, frame = cap.read()
    
    if ret:
        # แปลงสีจาก BGR (มาตรฐาน OpenCV) เป็น RGB (เพื่อให้เหมือน picamera2 เดิม)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    else:
        return None