from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
import cv2
import numpy as np
import os

app = FastAPI()


SAVE_DIR = "processed_images"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def main():
    # Serves the HTML file we created in Step 1
    with open("templates/index.html", "r") as f:
        return f.read()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    file_path = os.path.join(SAVE_DIR, f"gray_{file.filename}")
    cv2.imwrite(file_path, gray_img)

    
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)