import cv2
import pytesseract
import re
from PIL import Image
from config import (
    CHAR_CLASS_ID,
    PROVINCE_CLASS_ID,
    TESSERACT_CHAR_CONFIG,
    TESSERACT_PROVINCE_CONFIG
)
import difflib
# รายชื่อจังหวัดในประเทศไทย สำหรับการตรวจสอบและแก้ไขข้อความ
THAI_PROVINCES = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท",
    "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม", "นครราชสีมา",
    "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
    "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
    "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี",
    "ลพบุรี", "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร",
    "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย", "หนองบัวลำภู",
    "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี"
]

def fix_province(ocr_text):
    if not ocr_text: return ""
    clean_text = ocr_text.replace(" ", "").replace(".", "").replace("-", "")
    matches = difflib.get_close_matches(clean_text, THAI_PROVINCES, n=1, cutoff=0.4)
    return matches[0] if matches else ocr_text

def run_ocr(frame, detections):
    char_boxes = []
    province_box = None

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det.astype(int)
        if class_id == CHAR_CLASS_ID:
            char_boxes.append(((x1, y1, x2, y2), x1))
        elif class_id == PROVINCE_CLASS_ID:
            province_box = (x1, y1, x2, y2)

    plate_chars = ""
    plate_province = ""

    if char_boxes:
        char_boxes.sort(key=lambda x: x[1])
        xs = [b[0][0] for b in char_boxes]
        ys = [b[0][1] for b in char_boxes]
        xe = [b[0][2] for b in char_boxes]
        ye = [b[0][3] for b in char_boxes]

        crop = frame[min(ys):max(ye), min(xs):max(xe)]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        txt = pytesseract.image_to_string(
            Image.fromarray(gray),
            config=TESSERACT_CHAR_CONFIG
        )
        
        # 2. แก้ไข Logic กรองตัวอักษร: เอาเฉพาะ ก-ฮ และ 0-9
        # Regex [^0-9ก-ฮ] หมายถึง อะไรที่ไม่ใช่เลขและไทย ให้แทนที่ด้วยค่าว่าง
        plate_chars = re.sub(r'[^0-9ก-ฮ]', '', txt.strip())

    if province_box:
        x1, y1, x2, y2 = province_box
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        txt = pytesseract.image_to_string(
            Image.fromarray(gray),
            config=TESSERACT_PROVINCE_CONFIG
        )
        plate_province = txt.strip()
        plate_province = fix_province(plate_province)

    return {
        "chars": plate_chars or "ไม่พบอักษร",
        "province": plate_province or "ไม่พบจังหวัด"
    }