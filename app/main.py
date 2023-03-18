import json
from PIL import Image
from typing import Union
from click import File
from fastapi import FastAPI, File
import io
import torch

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):

    input_image = get_image_from_bytes(file)
    print('----')
    model = get_yolov5()
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    return {"result": detect_res}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


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