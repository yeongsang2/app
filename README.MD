# Getting start

## Prerequisites
* Docker

## Installation
```
git clone https://github.com/yeongsang2/app.git
cd app
```
* /app/yolov5에 xxxx.pt 저장

## Usage
* 도커 이미지 빌드
```
$ docker build . -t "server:0.1"
```
* 컨테이너 실행 
```
$ docker run -p 8080:8080 server:0.1
```
* test
```
$ curl get http://localhost:8080
ping
```

### API Endpoint
* /clothes-classify
    * form-data(multipart/form-data)
    * key: file | value: image


### reference
* https://github.com/ultralytics/yolov5
