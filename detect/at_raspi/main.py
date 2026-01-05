# frompy/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

from camera import init_camera, capture_frame
from detector import detect
from ocr import run_ocr
# แก้ไข 1: import stream ทั้ง module เพื่อให้อ้างอิงตัวแปร global ล่าสุดได้เสมอ
import stream 

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "web"),
    name="static"
)

@app.on_event("startup")
def startup():
    init_camera()

@app.get("/")
async def index():
    # อ่านไฟล์ html ส่งกลับไป
    html = (BASE_DIR / "web" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

@app.get("/video_feed")
async def video_feed():
    # เรียกใช้ function จาก module stream
    return StreamingResponse(
        stream.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/toggle_freeze")
async def toggle_freeze_api():
    # เรียกใช้ฟังก์ชัน toggle จาก stream
    frozen_state = stream.toggle_freeze()
    
    # แก้ไข 2: แปลง Boolean (True/False) เป็น String ("frozen"/"streaming")
    # เพื่อให้ตรงกับที่ Javascript ใน index.html (บรรทัด 122) รอตรวจสอบ
    status_text = "frozen" if frozen_state else "streaming"
    
    return {"status": status_text}

@app.get("/scan")
async def scan():
    # แก้ไข 3: เช็คค่าจาก stream.is_frozen และดึงภาพจาก stream.last_raw_frame
    # เพื่อให้ได้ภาพเดียวกับที่กำลัง Freeze อยู่บนหน้าจอ
    if stream.is_frozen and stream.last_raw_frame is not None:
        frame = stream.last_raw_frame.copy() # ควร copy เพื่อความปลอดภัย
    else:
        frame = capture_frame()

    # ส่วนการ detect และ ocr ทำเหมือนเดิม
    results = detect(frame, conf=0.5)
    detections = results[0].boxes.data.cpu().numpy()
    return run_ocr(frame, detections)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)