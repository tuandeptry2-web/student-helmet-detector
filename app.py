import os, io, csv
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import cloudinary
import cloudinary.uploader

# Load biến môi trường (Cloudinary)
load_dotenv(".env")
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET"),
    secure=True
)

# Load models
detect_model = YOLO("best_No_Helmet_Detection.pt")           # model phát hiện vi phạm
plate_char_model = YOLO("best_License_Plate_Recognition.pt") # model OCR biển số

app = FastAPI()

# OCR biển số đơn giản (dựa theo model ký tự của bạn)
def ocr_plate(pil_crop):
    try:
        results = plate_char_model.predict(pil_crop, conf=0.3, verbose=False)
        chars = []
        for r in results:
            for b in r.boxes:
                label = plate_char_model.names[int(b.cls)]
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                chars.append((label, (x1 + x2) / 2))
        if not chars:
            return ""
        # sắp xếp ký tự theo trục X (trái → phải)
        chars_sorted = sorted(chars, key=lambda c: c[1])
        return "".join([c[0] for c in chars_sorted])
    except:
        return ""

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Đọc ảnh upload
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Chạy YOLO detect
    results = detect_model(pil_img, conf=0.25)

    violations = []

    for res in results:
        for b in res.boxes:
            cls = detect_model.names[int(b.cls)]
            if cls in ("no_helmet_front","no_helmet_back","wrong_helmet_front","wrong_helmet_back"):
                # Crop vi phạm
                box = b.xyxy[0].cpu().numpy().astype(int).tolist()
                viol_crop = pil_img.crop((box[0], box[1], box[2], box[3]))

                # Tìm biển số gần nhất
                plate_text = ""
                plate_boxes = [p for p in res.boxes if detect_model.names[int(p.cls)] == "license_plate"]
                if plate_boxes:
                    pb = plate_boxes[0].xyxy[0].cpu().numpy().astype(int).tolist()
                    plate_crop = pil_img.crop((pb[0], pb[1], pb[2], pb[3]))
                    plate_text = ocr_plate(plate_crop)

                # Upload Cloudinary
                buf = io.BytesIO()
                viol_crop.save(buf, "JPEG")
                buf.seek(0)
                upl = cloudinary.uploader.upload(buf, folder="violations")
                url = upl.get("secure_url", "")

                # Thời gian
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Trả kết quả
                violations.append({
                    "time": ts,
                    "plate": plate_text,
                    "image_url": url
                })

    return JSONResponse(content={"violations": violations})
