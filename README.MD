# Getting start

## Prerequisites
* Docker

## Installation
```
git clone https://github.com/yeongsang2/app.git
cd app
```
* /app/resource 에 .pt file 저장

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
    * request
        * form-data(multipart/form-data)
        * key: file | value: image
    * respones<br>

            {
                "logo": "nike",
                "color": "탁한 초록색 48퍼, 검정색 45퍼,",
                "pattern": "스트라이프무늬"
            }
                

### reference
* https://github.com/ultralytics/yolov5
* https://github.com/pytorch/tutorials
