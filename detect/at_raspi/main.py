# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
import cv2
import os
from datetime import datetime
from config import LOG_PATH
import cameralow
from detector import detect
from ocr import run_ocr

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent

LOG_DIR = Path(LOG_PATH)
LOG_DIR.mkdir(parents=True, exist_ok=True) # สร้างโฟลเดอร์ถ้ายังไม่มี

if (BASE_DIR / "web").exists():
    app.mount("/static", StaticFiles(directory=BASE_DIR / "web"), name="static")

@app.on_event("startup")
def startup():
    cameralow.init_camera()

@app.get("/")
async def index():
    try:
        html = (BASE_DIR / "web" / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: index.html not found</h1>")

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        cameralow.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/toggle_freeze")
async def toggle_freeze_api():
    frozen_state = cameralow.toggle_freeze()
    status_text = "frozen" if frozen_state else "streaming"
    return {"status": status_text}

@app.get("/debug_yolo")
async def debug_yolo():
    if cameralow.is_frozen and cameralow.last_raw_frame is not None:
        frame = cameralow.last_raw_frame.copy()
    else:
        frame = cameralow.capture_frame()
    if frame is None: return Response(content=b"", media_type="image/jpeg")
    results = detect(frame, conf=0.5)
    annotated_frame = results[0].plot() if results else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", annotated_frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

# --- ฟังก์ชันช่วยบันทึก Log ---
def save_log(chars, province):
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M")
    
    # เก็บโดย Path เต็ม โดยใช้ LOG_DIR ที่ประกาศไว้ข้างบน
    filepath = LOG_DIR / f"record-LPR-of-{date_str}.log"
    
    header = f"Record of LPR for {date_str}\n"
    
    mode = 'a'
    # เช็คว่ามีไฟล์นี้ใน Directory หรือไม่
    if not filepath.exists():
        mode = 'w'
        next_num = 1
    else:
        # อ่านไฟล์จาก path ที่กำหนด
        with filepath.open('r', encoding='utf-8') as f:
            lines = f.readlines()
            next_num = len(lines)
            if next_num < 1: next_num = 1
            
    # เขียนไฟล์ลงใน path ที่กำหนด
    with filepath.open(mode, encoding='utf-8') as f:
        if mode == 'w':
            f.write(header)
        
        log_line = f"{next_num}. {chars} - {province} - เวลา {time_str}\n"
        f.write(log_line)

@app.get("/scan")
async def scan():
    if cameralow.is_frozen and cameralow.last_raw_frame is not None:
        frame = cameralow.last_raw_frame.copy()
    else:
        frame = cameralow.capture_frame()

    if frame is None:
        return {"error": "Could not capture frame"}

    results = detect(frame, conf=0.5)
    
    if results and len(results) > 0 and results[0].boxes:
        detections = results[0].boxes.data.cpu().numpy()
        data = run_ocr(frame, detections) # ได้ผลลัพธ์เป็น dict {chars, province}
        
        # ตรวจสอบความถูกต้องก่อนบันทึก Log
        c = data["chars"]
        p = data["province"]
        
        # กรองสิ่งที่บัคๆ ไม่บันทึกถ้าไม่พบข้อมูล
        is_valid = (c != "ไม่พบอักษร") and (p != "ไม่พบจังหวัด") and c and p
        
        if is_valid:
            save_log(c, p)
            
        return data # ส่งค่ากลับไป Frontend (ให้ Frontend ตัดสินใจเรื่องการแสดงผลเองอีกที หรือจะใช้ข้อมูลนี้ก็ได้)
    else:
        return {"chars": "ไม่พบอักษร", "province": "ไม่พบจังหวัด"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)