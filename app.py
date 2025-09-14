import os
import io
import tempfile
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
import traceback

# Load .env (nếu có)
load_dotenv(".env")

# Cloudinary config
CLOUD_NAME = os.getenv("CLOUD_NAME")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
if CLOUD_NAME and API_KEY and API_SECRET:
    cloudinary.config(cloud_name=CLOUD_NAME, api_key=API_KEY, api_secret=API_SECRET, secure=True)

# Model paths
DETECT_MODEL_PATH = os.getenv("DETECT_MODEL_PATH", "best_No_Helmet_Detection.pt")
PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", "best_License_Plate_Recognition.pt")
DETECT_CONF = float(os.getenv("DETECT_CONF", 0.25))
CHAR_CONF = float(os.getenv("CHAR_CONF", 0.3))

print("Loading detect model:", DETECT_MODEL_PATH)
detect_model = YOLO(DETECT_MODEL_PATH)
print("Loading plate-char model:", PLATE_MODEL_PATH)
plate_char_model = YOLO(PLATE_MODEL_PATH)
print("Models loaded.")

app = FastAPI(title="Helmet Violation API")

# ✅ CORS FIX - hardcode domain của bạn
origins = [
    "http://localhost:3000",
    "https://clever-travesseiro-77cca4.netlify.app",  # Netlify mới
    "https://student-helmet-detector.netlify.app"     # Netlify cũ (nếu bạn rename lại)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image_from_bytes(data: bytes) -> Image.Image:
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return pil
    except Exception:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Cannot decode image bytes")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

def extract_frame_from_video_bytes(data: bytes) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

def ocr_plate_from_pil(pil_crop: Image.Image) -> str:
    try:
        arr = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
        results = plate_char_model.predict(source=arr, conf=CHAR_CONF, verbose=False)
        chars = []
        for r in results:
            for b in r.boxes:
                cls_idx = int(b.cls)
                label = plate_char_model.names[cls_idx]
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                chars.append({'char': label, 'x': float((x1 + x2) / 2.0)})
        if not chars:
            return ""
        chars_sorted = sorted(chars, key=lambda c: c['x'])
        text = "".join([c['char'] for c in chars_sorted])
        text = ''.join([c for c in text.upper() if c.isalnum()])
        return text
    except Exception:
        return ""

@app.get("/")
def root():
    return {"status": "ok", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        content = await file.read()
        kind = (file.content_type or "").lower()
        if kind.startswith("video"):
            pil_img = extract_frame_from_video_bytes(content)
        else:
            pil_img = read_image_from_bytes(content)
    except Exception as e:
        tb = traceback.format_exc()
        print("Read file error:", e, tb)
        return JSONResponse(status_code=400, content={"error": f"Cannot read file: {str(e)}"})

    try:
        results = detect_model(pil_img, conf=DETECT_CONF)
    except Exception as e:
        tb = traceback.format_exc()
        print("Detection error:", e, tb)
        return JSONResponse(status_code=500, content={"error": f"Detection failed: {str(e)}"})

    violations = []
    for res in results:
        plate_boxes = [b for b in res.boxes if detect_model.names[int(b.cls)] == "license_plate"]
        for b in res.boxes:
            cls_name = detect_model.names[int(b.cls)]
            if cls_name in ("no_helmet_front", "no_helmet_back", "wrong_helmet_front", "wrong_helmet_back"):
                xy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = xy
                w, h = pil_img.size
                x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                viol_crop = pil_img.crop((x1, y1, x2, y2))

                plate_text = ""
                if plate_boxes:
                    def center(box):
                        a, b2, c, d = box.xyxy[0].cpu().numpy()
                        return ((a + c) / 2.0, (b2 + d) / 2.0)
                    tx = (x1 + x2) / 2.0; ty = (y1 + y2) / 2.0
                    best = None; best_d = None
                    for pb in plate_boxes:
                        cx, cy = center(pb)
                        d = (cx - tx) ** 2 + (cy - ty) ** 2
                        if best is None or d < best_d:
                            best = pb; best_d = d
                    if best is not None:
                        pb_xy = best.xyxy[0].cpu().numpy().astype(int).tolist()
                        px1, py1, px2, py2 = pb_xy
                        px1 = max(0, min(px1, w - 1)); px2 = max(0, min(px2, w - 1))
                        py1 = max(0, min(py1, h - 1)); py2 = max(0, min(py2, h - 1))
                        if px2 > px1 and py2 > py1:
                            plate_crop = pil_img.crop((px1, py1, px2, py2))
                            plate_text = ocr_plate_from_pil(plate_crop)

                cropped_url = ""
                try:
                    buf = io.BytesIO()
                    viol_crop.save(buf, format="JPEG")
                    buf.seek(0)
                    if CLOUD_NAME and API_KEY and API_SECRET:
                        upl = cloudinary.uploader.upload(buf, folder="violations")
                        cropped_url = upl.get("secure_url", "")
                except Exception as e:
                    print("Cloudinary upload error:", e)

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                violations.append({
                    "time": ts,
                    "license_plate": plate_text,
                    "cropped_image_url": cropped_url,
                    "violation_type": cls_name
                })

    return JSONResponse(content=violations)
