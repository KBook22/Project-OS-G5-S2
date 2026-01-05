# camera.py
from picamera2 import Picamera2

picam2 = None

def init_camera():
    global picam2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1600, 1080)}
    )
    picam2.configure(config)
    picam2.start()
    picam2.set_controls({"AfMode": 2})
    print("📷 Camera ready")

def capture_frame():
    return picam2.capture_array()
