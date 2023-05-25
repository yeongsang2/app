import json
import uuid
from PIL import Image
from click import File
from fastapi import FastAPI, File
import io
import torch
from torch import device
import os
from queue import Queue
from color_classification import color
from torchvision import models, transforms
import torch.nn as nn
import asyncio
import time

app = FastAPI()
UPLOAD_DIR = "/Users/kim-yeongsang/Desktop/app/image/"  # 이미지를 저장할 서버 경로
device = torch.device('cpu')

@app.get("/")
def read_root():
    return {"ping"}

@app.post("/clothes-classify")
async def detect_clothes_return_json_result(file: bytes = File(...)):
    
    # print(f"async started at {time.strftime('%X')}")
    start_time = time.time()


    color_time_s = time.time()
    color = asyncio.create_task(get_color(file))
    color = await color
    color_time_f = time.time()

    print(f"color time: {color_time_f - color_time_s:.3f} seconds")


    logo_time_s = time.time()
    logo = asyncio.create_task(get_logo(file))
    logo = await logo
    logo_time_f = time.time()
    print(f"logo time: {logo_time_f - logo_time_s:.3f} seconds")

    pattern_time_s = time.time()
    pattern = asyncio.create_task(get_pattern(file))
    pattern = await pattern
    pattern_time_f = time.time()
    print(f"pattern time: {pattern_time_f - pattern_time_s:.3f} seconds")
    
    # color = await color
    # logo = await logo
    # pattern = await pattern
    # print(f"async finished at {time.strftime('%X')}")
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.3f} seconds")
    result = {'logo': logo, 'color': color, 'pattern': pattern}
    return result


# @app.post("/clothes-classify/v2")
# async def detect_clothes_return_json_resultv2(file: bytes = File(...)):
    
#     print(f"started at {time.strftime('%X')}")
#     color = get_color(file)
#     logo = get_logo(file)
#     pattern = get_pattern(file)

#     print(f"finished at {time.strftime('%X')}")
#     result = {'logo': logo, 'color': color, 'pattern': pattern}
#     return result


def get_yolov5():
    model = torch.hub.load('/Users/kim-yeongsang/Desktop/app/yolov5', 'custom', path='/Users/kim-yeongsang/Desktop/app/resource/north.pt', source='local')
    model.conf = 0.5
    return model

def get_image_from_bytes(binary_image, max_size=1024):
    input_image =Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize((
        int(input_image.width * resize_factor),
        int(input_image.height * resize_factor)
    ))
    return resized_image

async def get_logo(file):

    input_image = get_image_from_bytes(file)
    model = get_yolov5()
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    print(detect_res)
    detect_res = json.loads(detect_res)
    if(detect_res):
        return detect_res[0].get("name")
    return
            

async def get_color(file):

    filename = f"{str(uuid.uuid4())}.jpg" 
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(file)
    
    que = Queue()
    que = color.execute(UPLOAD_DIR + filename, que)
    return que.get()

async def get_pattern(file):

    with open("/Users/kim-yeongsang/Desktop/app/resource/pattern.txt", 'r') as f:
        class_names = [line.strip() for line in f]
    # get model
    model = get_resnet18()
    # get image 
    image = preprocess(file)

    with torch.no_grad():
        outputs = model(image.to(device))
        _, preds = torch.max(outputs, 1)

    pred_class = class_names[preds[0]]
    return pred_class


def get_resnet18():

    model = models.resnet18(pretrained=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    
    model.load_state_dict(torch.load('/Users/kim-yeongsang/Desktop/app/resource/pattern.pt', map_location='cpu'))

    model.eval()
    model.to(device)
    
    return model

def preprocess(file):

    image = get_image_from_bytes(file)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)