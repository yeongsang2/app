## Build the image torchserve locally before running this, cf github torchserve:
## https://github.com/pytorch/serve/tree/master/docker
FROM python:3.9

RUN mkdir /user

COPY . /user/yolov5
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y python3-distutils
# RUN pip install --upgrade pip
RUN pip install -r /user/yolov5/requirements.txt
# RUN pip install -r /app/yolov5/requirements.txt

WORKDIR /user/yolov5

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

# CMD ["python", "detect.py", "--weights","/app/yolov5/north.pt","--source","/app/yolov5/adidas_check1.png"]
