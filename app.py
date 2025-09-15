# app.py
import os
import io
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

# ----------------- CONFIG -----------------
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

# skip rules
SKIP_IF_TRUNCATED = os.getenv("SKIP_IF_TRUNCATED", "1") == "1"
SKIP_IF_MISSING_PLATE = os.getenv("SKIP_IF_MISSING_PLATE", "1") == "1"
SKIP_IF_MISSING_HEAD = os.getenv("SKIP_IF_MISSING_HEAD", "1") == "1"
TRUNCATE_AREA_RATIO = float(os.getenv("TRUNCATE_AREA_RATIO", 0.85))

# ----------------- Load models -----------------
print("Loading detect model:", DETECT_MODEL_PATH)
detect_model = YOLO(DETECT_MODEL_PATH)
print("Loading plate-char model:", PLATE_MODEL_PATH)
plate_char_model = YOLO(PLATE_MODEL_PATH)
print("Models loaded.")

app = FastAPI(title="Helmet / Motorcyclist Violation API")

# CORS config: có thể set CORS_ORIGINS="https://your-netlify.app,https://other"
cors_env = os.getenv("CORS_ORIGINS", "*")
if cors_env.strip() == "" or cors_env.strip() == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Utilities -----------------
def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1_cl = max(0, min(int(round(x1)), w-1))
    y1_cl = max(0, min(int(round(y1)), h-1))
    x2_cl = max(0, min(int(round(x2)), w-1))
    y2_cl = max(0, min(int(round(y2)), h-1))
    if x2_cl <= x1_cl: x2_cl = min(w-1, x1_cl + 1)
    if y2_cl <= y1_cl: y2_cl = min(h-1, y1_cl + 1)
    return [x1_cl, y1_cl, x2_cl, y2_cl]

def center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def area(box):
    x1,y1,x2,y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-9)

def normalize_plate(txt):
    if not txt:
        return ""
    s = str(txt).upper()
    s = s.replace('O', '0').replace('Q', '0').replace('I', '1').replace('L', '1')
    s = ''.join([c for c in s if c.isalnum()])
    # keep as simple normalization (you can reuse your regex formatting if needed)
    return s

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
    results = plate_char_model.predict(source=img_bgr, conf=CHAR_CONF, verbose=False)
    chars = []
    for r in results:
        for b in r.boxes:
            cls_idx = int(b.cls)
            label = plate_char_model.names[cls_idx]
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            chars.append({'char': label, 'x': (x1+x2)/2.0, 'y': (y1+y2)/2.0})
    if not chars:
        return ""
    ys = [c['y'] for c in chars]
    y_mean = np.mean(ys)
    line1 = [c for c in chars if c['y'] < y_mean]
    line2 = [c for c in chars if c['y'] >= y_mean]
    line1_sorted = sorted(line1, key=lambda c: c['x'])
    line2_sorted = sorted(line2, key=lambda c: c['x'])
    text = ''.join([c['char'] for c in line1_sorted]) + ''.join([c['char'] for c in line2_sorted])
    return normalize_plate(text)

def find_box_by_iou_or_nearest(target_box, candidates, img_diag, require_overlap=False):
    if not candidates:
        return None, 0.0
    best_iou = 0.0; best_box = None
    for c in candidates:
        score = iou(target_box, c)
        if score > best_iou:
            best_iou = score; best_box = c
    if best_iou > 0.0:
        return best_box, best_iou
    if require_overlap:
        return None, 0.0
    tcx, tcy = center(target_box)
    dmin = float('inf'); nearest = None
    for c in candidates:
        cx, cy = center(c)
        d = (cx-tcx)**2 + (cy-tcy)**2
        if d < dmin:
            dmin = d; nearest = c
    tw = target_box[2] - target_box[0]; th = target_box[3] - target_box[1]
    diag = math.hypot(tw, th)
    if math.sqrt(dmin) <= diag * 2.5:
        return nearest, 0.0
    return None, 0.0

def is_truncated_box(box, img_w, img_h, area_ratio_threshold=TRUNCATE_AREA_RATIO):
    x1,y1,x2,y2 = box
    orig_area = max(0, x2 - x1) * max(0, y2 - y1)
    if orig_area <= 0:
        return True
    clamped = clamp_box(box, img_w, img_h)
    clamp_area = area(clamped)
    ratio = clamp_area / (orig_area + 1e-9)
    return ratio < area_ratio_threshold

# ----------------- FastAPI endpoints -----------------
@app.get("/")
def root():
    return {"status": "ok", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accept image or video file.
    Logic: detect -> find motor_front/back boxes -> apply skip rules -> crop motor box -> find associated plate -> ocr -> upload -> return results
    """
    try:
        content = await file.read()
        kind = (file.content_type or "").lower()
        if kind.startswith("video"):
            # write temp and extract frame
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
                tmp.write(content); tmp.flush()
                vid = cv2.VideoCapture(tmp.name)
                if not vid.isOpened():
                    return JSONResponse(status_code=400, content={"error": "Cannot open video file"})
                length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                frame_no = 0 if length <= 0 else max(0, length // 2)
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = vid.read()
                vid.release()
                if not ret or frame is None:
                    return JSONResponse(status_code=400, content={"error": "Cannot read frame from video"})
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
        else:
            # try PIL then fallback to cv2
            try:
                pil_img = Image.open(io.BytesIO(content)).convert("RGB")
            except Exception:
                arr = np.frombuffer(content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    return JSONResponse(status_code=400, content={"error": "Cannot decode image bytes"})
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
    except Exception as e:
        tb = traceback.format_exc()
        print("read error:", e, tb)
        return JSONResponse(status_code=400, content={"error": f"Read file failed: {str(e)}"})

    # run detection
    try:
        results = detect_model(pil_img, conf=DETECT_CONF)
    except Exception as e:
        tb = traceback.format_exc()
        print("detection error:", e, tb)
        return JSONResponse(status_code=500, content={"error": f"Detection failed: {str(e)}"})

    w_img, h_img = pil_img.size
    diag_img = math.hypot(w_img, h_img)

    violations = []

    for res in results:
        # convert boxes into lists for easier processing
        motor_front_boxes, motor_back_boxes = [], []
        helmet_boxes, plate_boxes, violation_boxes = [], [], []

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

        # process front motors
        for motor in motor_front_boxes:
            # skip if truncated
            if SKIP_IF_TRUNCATED and is_truncated_box(motor, w_img, h_img, TRUNCATE_AREA_RATIO):
                continue

            plate_ok = has_associated_plate(motor, plate_boxes, diag_img)
            head_ok = has_associated_head(motor, violation_boxes, diag_img)

            if SKIP_IF_MISSING_PLATE and not plate_ok:
                continue
            if SKIP_IF_MISSING_HEAD and not head_ok:
                continue

            motor_clamped = clamp_box(motor, w_img, h_img)
            try:
                mot_crop = pil_img.crop((motor_clamped[0], motor_clamped[1], motor_clamped[2], motor_clamped[3]))
            except Exception as e:
                print("motor crop failed:", e)
                continue

            # find plate associated
            best_plate, _ = find_box_by_iou_or_nearest(motor, plate_boxes, diag_img, require_overlap=False)
            plate_text = ""
            if best_plate:
                pb = clamp_box(best_plate, w_img, h_img)
                try:
                    plate_crop = pil_img.crop((pb[0], pb[1], pb[2], pb[3]))
                    plate_text = ocr_plate_yolo_your_model(plate_crop)
                except Exception as e:
                    print("plate ocr failed:", e)
                    plate_text = ""

            # upload motor crop
            cropped_url = ""
            try:
                buf = io.BytesIO()
                mot_crop.save(buf, format='JPEG')
                buf.seek(0)
                if CLOUD_NAME and API_KEY and API_SECRET:
                    upl = cloudinary.uploader.upload(buf, folder='violations')
                    cropped_url = upl.get('secure_url','')
            except Exception as e:
                print("upload motor crop error:", e)

            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            violations.append({
                "time": ts,
                "license_plate": plate_text,
                "cropped_image_url": cropped_url,
                "violation_type": "no_helmet_front" if has_associated_head(motor, violation_boxes, diag_img) else "motorcyclist_front"
            })

        # process back motors
        for motor in motor_back_boxes:
            if SKIP_IF_TRUNCATED and is_truncated_box(motor, w_img, h_img, TRUNCATE_AREA_RATIO):
                continue

            plate_ok = has_associated_plate(motor, plate_boxes, diag_img)
            head_ok = has_associated_head(motor, violation_boxes, diag_img)

            if SKIP_IF_MISSING_PLATE and not plate_ok:
                continue
            if SKIP_IF_MISSING_HEAD and not head_ok:
                continue

            motor_clamped = clamp_box(motor, w_img, h_img)
            try:
                mot_crop = pil_img.crop((motor_clamped[0], motor_clamped[1], motor_clamped[2], motor_clamped[3]))
            except Exception as e:
                print("motor crop failed:", e)
                continue

            best_plate, _ = find_box_by_iou_or_nearest(motor, plate_boxes, diag_img, require_overlap=False)
            plate_text = ""
            if best_plate:
                pb = clamp_box(best_plate, w_img, h_img)
                try:
                    plate_crop = pil_img.crop((pb[0], pb[1], pb[2], pb[3]))
                    plate_text = ocr_plate_yolo_your_model(plate_crop)
                except Exception as e:
                    print("plate ocr failed:", e)
                    plate_text = ""

            cropped_url = ""
            try:
                buf = io.BytesIO()
                mot_crop.save(buf, format='JPEG')
                buf.seek(0)
                if CLOUD_NAME and API_KEY and API_SECRET:
                    upl = cloudinary.uploader.upload(buf, folder='violations')
                    cropped_url = upl.get('secure_url','')
            except Exception as e:
                print("upload motor crop error:", e)

            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            violations.append({
                "time": ts,
                "license_plate": plate_text,
                "cropped_image_url": cropped_url,
                "violation_type": "no_helmet_back" if has_associated_head(motor, violation_boxes, diag_img) else "motorcyclist_back"
            })

    # trả về mảng violations (frontend hiện tại mong mảng)
    return JSONResponse(content=violations)
