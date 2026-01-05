# stream.py
import cv2
import time
from camera import capture_frame
from detector import detect

last_raw_frame = None
last_annotated_frame = None
is_frozen = False

def toggle_freeze():
    global is_frozen
    is_frozen = not is_frozen
    return is_frozen

def generate_frames():
    global last_raw_frame, last_annotated_frame

    while True:
        if is_frozen and last_annotated_frame is not None:
            frame = last_annotated_frame
            time.sleep(0.1)
        else:
            raw = capture_frame()
            last_raw_frame = raw.copy()
            results = detect(raw)
            annotated = results[0].plot()
            last_annotated_frame = annotated.copy()
            frame = annotated

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )
