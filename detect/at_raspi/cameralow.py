# camera.py
import io
import time
import threading
import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# -----------------------------
# Global State
# -----------------------------
app = FastAPI()
picam2 = None

# Stream vars
latest_jpeg = None
frame_cond = threading.Condition()

# Logic vars (สำหรับ main.py)
is_frozen = False        # สถานะ Freeze
last_raw_frame = None    # เก็บภาพ Raw ตอน Freeze ไว้ทำ OCR

# -----------------------------
# Output Writer Class
# -----------------------------
class FrameOutputWriter(io.BufferedIOBase):
    def write(self, buf: bytes) -> int:
        global latest_jpeg
        # ถ้า Freeze อยู่ ไม่ต้องอัปเดตภาพใหม่เข้าสู่ระบบ Stream
        if is_frozen:
            return len(buf)

        with frame_cond:
            latest_jpeg = bytes(buf)
            frame_cond.notify_all()
        return len(buf)

# -----------------------------
# Camera Logic
# -----------------------------
def init_camera():
    global picam2
    if picam2 is not None:
        return

    print("Initializing Camera...")
    picam2 = Picamera2()

    # Config: RGB888 เพื่อให้ capture_frame ได้สีที่ถูกต้องสำหรับ AI
    config = picam2.create_video_configuration(
        main={"size": (1280,720), "format": "RGB888"}
    )
    picam2.configure(config)

    # Start MJPEG Stream (Low CPU)
    picam2.start_recording(MJPEGEncoder(), FileOutput(FrameOutputWriter()))
    
    try:
        picam2.set_controls({"AfMode": 2})
    except Exception as e:
        print(f"AF Warning: {e}")

    print("Camera Started (High Performance + OCR Ready)")

def capture_frame():
    """ดึงภาพ Raw (Numpy) จากกล้องทันที (สำหรับ OCR/Detection)"""
    if picam2:
        # capture_array ทำงานได้แม้กำลัง record stream อยู่
        return picam2.capture_array()
    return None

def toggle_freeze():
    """สลับสถานะ Freeze/Unfreeze และเก็บภาพ Raw ไว้"""
    global is_frozen, last_raw_frame
    
    is_frozen = not is_frozen
    
    if is_frozen:
        # จังหวะที่กด Freeze ให้ถ่ายภาพ Raw เก็บไว้เลย เพื่อความคมชัดสูงสุดตอน Scan
        print("Freezing frame...")
        last_raw_frame = capture_frame()
    else:
        print("Resuming stream...")
        last_raw_frame = None
        
    return is_frozen

# -----------------------------
# FastAPI Generator
# -----------------------------
def generate_frames():
    while True:
        if is_frozen:
            time.sleep(0.1)
            # ถ้ามีภาพค้างเก่า ส่งภาพเดิมไปให้ Browser
            if latest_jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_jpeg + b'\r\n')
            continue

        # โหมดปกติ: รอภาพใหม่จาก Camera
        with frame_cond:
            frame_cond.wait()
            frame = latest_jpeg
        
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -----------------------------
# Local Test Routes
# -----------------------------
@app.get("/")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == "__main__":
    # ทดสอบกล้อง Raspberry Pi ได้ที่ Port 8020
    init_camera()
    print("Starting Preview Server at http://0.0.0.0:8020")
    uvicorn.run(app, host="0.0.0.0", port=8020)
