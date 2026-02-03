import threading, queue, uuid, io, base64, uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from contextlib import asynccontextmanager

class ModelManager:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.loaded_models = {}  # Модели сохраняются здесь после первой загрузки
        self.results = {}
        self.lock = threading.Lock()
        self.model_paths = {
            "fast": "yolo11n_custom.pt", 
            "accurate": "yolo11l_custom.pt"
        }

    def get_model(self, model_id: str):
        with self.lock:
            if model_id not in self.loaded_models:
                path = self.model_paths.get(model_id, "yolo11n.pt")
                print(f"--- [MEMORY] Загрузка модели {model_id} в оперативную память ---")
                self.loaded_models[model_id] = YOLO(path)
            return self.loaded_models[model_id]

    def worker(self):
        while True:
            task = self.task_queue.get()
            if task is None: break
            task_id, model_id, image_bytes, imgsz, conf = task
            try:
                model = self.get_model(model_id)
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                results = model.predict(source=img, imgsz=imgsz, conf=conf, verbose=True)
                count = len(results[0].boxes)
                
                annotated_frame = results[0].plot() 
                img_out = Image.fromarray(annotated_frame[..., ::-1])
                
                buffered = io.BytesIO()
                img_out.save(buffered, format="JPEG", quality=90)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

                with self.lock:
                    self.results[task_id] = {
                        "status": "completed",
                        "image": img_str,
                        "count": int(count)
                    }
            except Exception as e:
                print(f"[ERROR] {e}")
                with self.lock: self.results[task_id] = {"status": "error", "message": str(e)}
            finally:
                self.task_queue.task_done()

manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=manager.worker, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
async def predict(model_type: str = Form(...), imgsz_w: int = Form(...), imgsz_h: int = Form(...), conf: float = Form(...), file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    img_bytes = await file.read()
    with manager.lock: manager.results[task_id] = {"status": "pending"}
    manager.task_queue.put((task_id, model_type, img_bytes, (imgsz_w, imgsz_h), conf))
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    with manager.lock: return manager.results.get(task_id, {"status": "not_found"})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)