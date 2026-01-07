# camera.py
import cv2
import uvicorn
from picamera2 import Picamera2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

picam2 = None
app = FastAPI()

def init_camera():
    global picam2
    picam2 = Picamera2()
    # ตั้งค่า format เป็น RGB888 เพื่อคุณภาพสีที่ถูกต้อง
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1600, 1080)}
    )
    picam2.configure(config)
    picam2.start()
    picam2.set_controls({"AfMode": 2}) # Auto Focus
    print("📷 Camera ready")

def capture_frame():
    if picam2:
        return picam2.capture_array()
    return None

# --- ส่วนที่เพิ่มเข้ามา ---

def generate_frames():
    """ฟังก์ชัน Generator สำหรับแปลงภาพเป็น MJPEG Stream"""
    while True:
        frame = capture_frame()
        if frame is None:
            continue
        
        # แปลงภาพเป็น JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        # ส่งข้อมูลภาพกลับไปในรูปแบบ Multipart Stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
def video_feed():
    """Route สำหรับดู Preview"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == "__main__":
    # เริ่มต้นกล้อง
    init_camera()
    
    # เริ่มต้น Web Server บน Port 8020
    print("🚀 Starting Preview Server at http://0.0.0.0:8020")
    uvicorn.run(app, host="0.0.0.0", port=8020)