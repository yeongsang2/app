import json
import uuid
from PIL import Image
from typing import Union
from click import File
from fastapi import FastAPI, File
import io
import torch
import os
from queue import Queue
import sys
sys.path.append('/user/app/color_classification')
from color_classification import color


app = FastAPI()
UPLOAD_DIR = "/user/app/image/"  # 이미지를 저장할 서버 경로

@app.get("/")
def read_root():
    return {"ping"}

@app.post("/clothes-classify")
async def detect_clothes_return_json_result(file: bytes = File(...)):

    color = get_color(file)
    logo = get_logo(file)

    result = {'logo': logo, 'color': color}
    return {"result": result}

def get_yolov5():
    model = torch.hub.load('/user/app/yolov5', 'custom', path='/user/app/yolov5/north.pt', source='local')
    model.conf = 0.5
    return model

def get_image_from_bytes(binary_image, max_size=1024):
    print('test')
    input_image =Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize((
        int(input_image.width * resize_factor),
        int(input_image.height * resize_factor)
    ))
    return resized_image

def get_logo(file):

    input_image = get_image_from_bytes(file)
    model = get_yolov5()
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    print(detect_res)
    detect_res = json.loads(detect_res)
    if(detect_res):
        return detect_res[0].get("name")
    return
            

def get_color(file):

    filename = f"{str(uuid.uuid4())}.jpg" 
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(file)
    
    que = Queue()
    que = color.execute(UPLOAD_DIR + filename, que)
    return que.get()