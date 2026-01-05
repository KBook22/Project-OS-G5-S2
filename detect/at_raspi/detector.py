# detector.py
from ultralytics import YOLO
from config import MODEL_PATH

print("🧠 Loading YOLO model...")
model = YOLO(MODEL_PATH)

def detect(frame, conf=0.4):
    return model.predict(
        frame,
        conf=conf,
        iou=0.7,
        imgsz=640,
        verbose=False,
        device="cpu"
    )
