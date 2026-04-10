from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
import cv2
import numpy as np
import os

app = FastAPI()
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html") as f:
        return f.read()

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    operation: str = Form(...),
    parameter: int = Form(...)
):
    
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if operation == "grayscale":
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    elif operation == "blur":
        
        k_size = parameter if parameter % 2 != 0 else parameter + 1
        processed = cv2.GaussianBlur(img, (k_size, k_size), 0)
    
    elif operation == "edges":
        processed = cv2.Canny(img, 100, parameter)
    
    elif operation == "invert":
        processed = cv2.bitwise_not(img)
    
    else:
        processed = img 
    
    save_path = os.path.join(OUT_DIR, f"dynamic_{file.filename}")
    cv2.imwrite(save_path, processed)
    
    return FileResponse(save_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)