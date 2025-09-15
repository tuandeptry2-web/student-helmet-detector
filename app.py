# app.py (fixed, full)
import os
import io
import re
import math
import tempfile
import traceback
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import cloudinary
import cloudinary.uploader
import cv2
import numpy as np

# ----------------- CONFIG / ENV -----------------
load_dotenv(".env")

CLOUD_NAME = os.getenv('CLOUD_NAME')
API_KEY    = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
if CLOUD_NAME and API_KEY and API_SECRET:
    cloudinary.config(cloud_name=CLOUD_NAME, api_key=API_KEY, api_secret=API_SECRET, secure=True)

DETECT_MODEL_PATH = os.getenv("DETECT_MODEL_PATH", "best_No_Helmet_Detection.pt")
PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", "best_License_Plate_Recognition.pt")

DETECT_CONF = float(os.getenv("DETECT_CONF", 0.25))
CHAR_CONF = float(os.getenv("CHAR_CONF", 0.3))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")

SKIP_IF_TRUNCATED = os.getenv("SKIP_IF_TRUNCATED", "1") == "1"
SKIP_IF_MISSING_PLATE = os.getenv("SKIP_IF_MISSING_PLATE", "1") == "1"
SKIP_IF_MISSING_HEAD = os.getenv("SKIP_IF_MISSING_HEAD", "1") == "1"
TRUNCATE_AREA_RATIO = float(os.getenv("TRUNCATE_AREA_RATIO", 0.85))

HELMET_IN_MOTOR_RATIO = float(os.getenv("HELMET_IN_MOTOR_RATIO", 0.30))
PLATE_IN_MOTOR_RATIO = float(os.getenv("PLATE_IN_MOTOR_RATIO", 0.35))
NEAREST_DIAG_FACTOR = float(os.getenv("NEAREST_DIAG_FACTOR", 2.5))
CENTER_INSIDE_ACCEPT = os.getenv("CENTER_INSIDE_ACCEPT", "1") == "1"

# ----------------- HELPERS -----------------
def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1_cl = max(0, min(int(round(x1)), w-1))
    y1_cl = max(0, min(int(round(y1)), h-1))
    x2_cl = max(0, min(int(round(x2)), w-1))
    y2_cl = max(0, min(int(round(y2)), h-1))
    if x2_cl <= x1_cl: x2_cl = min(w-1, x1_cl + 1)
    if y2_cl <= y1_cl: y2_cl = min(h-1, y1_cl + 1)
    return [x1_cl, y1_cl, x2_cl, y2_cl]

def center_point(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    return interW * interH

def iou(boxA, boxB):
    inter = intersection_area(boxA, boxB)
    if inter == 0: return 0.0
    aA = box_area(boxA); aB = box_area(boxB)
    return inter / float(aA + aB - inter + 1e-9)

def box_contains_point(box, pt):
    x, y = pt
    return (x >= box[0] and x <= box[2] and y >= box[1] and y <= box[3])

def is_truncated_box(box, img_w, img_h, area_ratio_threshold=TRUNCATE_AREA_RATIO):
    x1,y1,x2,y2 = box
    orig_area = max(0, x2 - x1) * max(0, y2 - y1)
    if orig_area <= 0:
        return True
    clamped = clamp_box(box, img_w, img_h)
    clamp_area = box_area(clamped)
    ratio = clamp_area / (orig_area + 1e-9)
    return ratio < area_ratio_threshold

def normalize_plate(txt):
    if not txt:
        return ""
    s = str(txt).upper()
    s = s.replace('O', '0').replace('Q', '0').replace('I', '1').replace('L', '1')
    s = re.sub(r'[^A-Z0-9]', '', s)
    m = re.match(r'^(\d{2})([A-Z]{1,2})(\d)(\d{2,5})$', s)
    if m:
        prov, letters, digit, nums = m.groups()
    else:
        m2 = re.search(r'(\d{2})\D*([A-Z]{1,2})\D*(\d)(\d{2,5})', s)
        if m2:
            prov, letters, digit, nums = m2.groups()
        else:
            return s
    group = letters + digit
    if len(nums) == 5:
        return f"{prov}-{group} {nums[:-2]}.{nums[-2:]}"
    else:
        return f"{prov}-{group} {nums}"

def preprocess_plate_smooth_binary(pil_img):
    w, h = pil_img.size
    scale = max(640 / w, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    up = pil_img.resize((new_w, new_h), Image.LANCZOS)
    gray = np.array(up.convert('L'))
    _, bin0 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    g = cv2.GaussianBlur(bin0, (3, 3), 0)
    m = cv2.medianBlur(g, 3)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    c = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    _, final = cv2.threshold(c, 127, 255, cv2.THRESH_BINARY)
    if np.sum(final == 255) < np.sum(final == 0):
        final = 255 - final
    return final

def ocr_plate_yolo_your_model(pil_crop):
    try:
        bin_img = preprocess_plate_smooth_binary(pil_crop)
    except Exception:
        return ""
    img_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    results = plate_char_model.predict(source=img_bgr, conf=CHAR_CONF, device=YOLO_DEVICE, verbose=False)
    chars = []
    for r in results:
        for b in r.boxes:
            cls_idx = int(b.cls)
            label = plate_char_model.names[cls_idx]
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            chars.append({'char': label, 'x': (x1+x2)/2.0, 'y': (y1+y2)/2.0})
    if not chars: return ""
    ys = [c['y'] for c in chars]
    y_mean = np.mean(ys)
    line1 = [c for c in chars if c['y'] < y_mean]
    line2 = [c for c in chars if c['y'] >= y_mean]
    text = ''.join([c['char'] for c in sorted(line1, key=lambda c: c['x'])]) + \
           ''.join([c['char'] for c in sorted(line2, key=lambda c: c['x'])])
    return normalize_plate(text)

def has_associated_plate(motor_box, plate_boxes, img_diag):
    if not plate_boxes: return None
    best_candidate, best_score = None, -1.0
    for pb in plate_boxes:
        inter = intersection_area(motor_box, pb)
        pa = box_area(pb)
        ratio_plate_in_motor = inter / (pa + 1e-9)
        if ratio_plate_in_motor >= PLATE_IN_MOTOR_RATIO:
            return pb
        if CENTER_INSIDE_ACCEPT and box_contains_point(motor_box, center_point(pb)):
            return pb
        score = iou(motor_box, pb) * 2.0 + ratio_plate_in_motor
        if score > best_score:
            best_score, best_candidate = score, pb
    if best_candidate is not None:
        if iou(motor_box, best_candidate) >= 0.05:
            return best_candidate
        tcx, tcy = center_point(motor_box)
        pcx, pcy = center_point(best_candidate)
        d2 = (pcx - tcx)**2 + (pcy - tcy)**2
        if math.sqrt(d2) <= img_diag * NEAREST_DIAG_FACTOR:
            return best_candidate
    return None

def has_associated_head(motor_box, viol_boxes, img_diag):
    if not viol_boxes: return None
    best_candidate, best_score = None, -1.0
    for vb in viol_boxes:
        inter = intersection_area(motor_box, vb)
        va = box_area(vb)
        ratio = inter / (va + 1e-9)
        if ratio >= HELMET_IN_MOTOR_RATIO:
            return vb
        if CENTER_INSIDE_ACCEPT and box_contains_point(motor_box, center_point(vb)):
            return vb
        score = iou(motor_box, vb) + ratio * 2.0
        if score > best_score:
            best_score, best_candidate = score, vb
    if best_candidate is not None:
        if iou(motor_box, best_candidate) >= 0.05:
            return best_candidate
        tcx, tcy = center_point(motor_box)
        pcx, pcy = center_point(best_candidate)
        d2 = (pcx - tcx)**2 + (pcy - tcy)**2
        if math.sqrt(d2) <= img_diag * NEAREST_DIAG_FACTOR:
            return best_candidate
    return None

# ----------------- LOAD MODELS -----------------
print("Loading detect model:", DETECT_MODEL_PATH)
detect_model = YOLO(DETECT_MODEL_PATH)
print("Loading plate-char model:", PLATE_MODEL_PATH)
plate_char_model = YOLO(PLATE_MODEL_PATH)
print("Models loaded.")

# ----------------- FASTAPI APP -----------------
app = FastAPI(title="Helmet / Motorcyclist Violation API")
cors_env = os.getenv("CORS_ORIGINS", "*")
origins = ["*"] if cors_env.strip() in ("", "*") else [o.strip() for o in cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Cannot decode image bytes")
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def extract_frame_from_video_bytes(data: bytes) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(data); tmp.flush()
        vid = cv2.VideoCapture(tmp.name)
        if not vid.isOpened():
            raise RuntimeError("Cannot open video")
        length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_no = 0 if length <= 0 else max(0, length // 2)
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = vid.read()
        vid.release()
        if not ret or frame is None:
            raise RuntimeError("Cannot read frame from video")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

@app.get("/")
def root():
    return {"status": "ok", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        content = await file.read()
        kind = (file.content_type or "").lower()
        pil_img = extract_frame_from_video_bytes(content) if kind.startswith("video") else read_image_from_bytes(content)
    except Exception as e:
        print("Read file error:", e, traceback.format_exc())
        return JSONResponse(status_code=400, content={"error": f"Cannot read file: {str(e)}"})

    try:
        results = detect_model.predict(source=pil_img, conf=DETECT_CONF, device=YOLO_DEVICE, verbose=False)
    except Exception as e:
        print("Detection error:", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"Detection failed: {str(e)}"})

    w_img, h_img = pil_img.size
    diag_img = math.hypot(w_img, h_img)
    violations = []

    for res in results:
        motor_front_boxes, motor_back_boxes, helmet_boxes, plate_boxes, violation_boxes = [], [], [], [], []
        for b in res.boxes:
            cls_name = detect_model.names[int(b.cls)]
            box = b.xyxy[0].cpu().numpy().astype(int).tolist()
            if cls_name == 'motorcyclist_front':
                motor_front_boxes.append(box)
            elif cls_name == 'motorcyclist_back':
                motor_back_boxes.append(box)
            elif cls_name == 'helmet':
                helmet_boxes.append(box)
            elif cls_name == 'license_plate':
                plate_boxes.append(box)
            elif cls_name in ('no_helmet_front','no_helmet_back','wrong_helmet_front','wrong_helmet_back'):
                violation_boxes.append(box)

        for motor in motor_front_boxes + motor_back_boxes:
            if SKIP_IF_TRUNCATED and is_truncated_box(motor, w_img, h_img):
                print("Skipping motor (truncated):", motor); continue
            best_plate = has_associated_plate(motor, plate_boxes, diag_img)
            best_head = has_associated_head(motor, violation_boxes, diag_img)
            if SKIP_IF_MISSING_PLATE and best_plate is None:
                print("Skipping motor (no plate):", motor); continue
            if SKIP_IF_MISSING_HEAD and best_head is None:
                print("Skipping motor (no head):", motor); continue
            motor_clamped = clamp_box(motor, w_img, h_img)
            mot_crop = pil_img.crop((motor_clamped[0], motor_clamped[1], motor_clamped[2], motor_clamped[3]))
            plate_text = ""
            if best_plate is not None:
                pb = clamp_box(best_plate, w_img, h_img)
                try:
                    plate_crop = pil_img.crop((pb[0], pb[1], pb[2], pb[3]))
                    plate_text = ocr_plate_yolo_your_model(plate_crop)
                except Exception as e:
                    print("Plate OCR failed:", e)
            cropped_url = ""
            try:
                buf = io.BytesIO()
                mot_crop.save(buf, format='JPEG'); buf.seek(0)
                if CLOUD_NAME and API_KEY and API_SECRET:
                    upl = cloudinary.uploader.upload(buf, folder='violations')
                    cropped_url = upl.get('secure_url','')
            except Exception as e:
                print("Cloudinary upload error:", e)
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            violations.append({
                "time": ts,
                "license_plate": normalize_plate(plate_text),
                "cropped_image_url": cropped_url,
                "violation_type": "no_helmet_back" if motor in motor_back_boxes else "no_helmet_front",
                "motor_box": motor_clamped
            })
    return JSONResponse(content=violations)
