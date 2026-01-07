# frompy/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

# Import ทุกอย่างจาก camera module ที่เราเพิ่งแก้
import cameralow
from detector import detect
from ocr import run_ocr

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

# (Optional) ถ้ามี folder web ให้ mount ไว้เหมือนเดิม
if (BASE_DIR / "web").exists():
    app.mount("/static", StaticFiles(directory=BASE_DIR / "web"), name="static")

@app.on_event("startup")
def startup():
    # เริ่มต้นกล้อง
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
    # ใช้ generator จาก camera.py
    return StreamingResponse(
        cameralow.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/toggle_freeze")
async def toggle_freeze_api():
    # เรียกใช้ฟังก์ชัน toggle ใน camera.py
    frozen_state = cameralow.toggle_freeze()
    
    status_text = "frozen" if frozen_state else "streaming"
    return {"status": status_text}

@app.get("/scan")
async def scan():
    # Logic: ถ้า Freeze อยู่ ให้เอาภาพที่แช่ไว้ (last_raw_frame) มาใช้
    # ถ้าไม่ Freeze ให้ถ่ายภาพใหม่เดี๋ยวนั้นเลย
    if cameralow.is_frozen and cameralow.last_raw_frame is not None:
        print("🔍 Scanning FROZEN frame")
        frame = cameralow.last_raw_frame.copy()
    else:
        print("📷 Scanning LIVE frame")
        frame = cameralow.capture_frame()

    if frame is None:
        return {"error": "Could not capture frame"}

    # ส่งเข้า process detect และ ocr
    results = detect(frame, conf=0.5)
    
    # ตรวจสอบว่ามีผลลัพธ์ไหมป้องกัน error
    if results and len(results) > 0 and results[0].boxes:
        detections = results[0].boxes.data.cpu().numpy()
        return run_ocr(frame, detections)
    else:
        return {"message": "No object detected", "data": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)