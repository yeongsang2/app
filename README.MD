# Getting start

## Prerequisites
* Docker

## Installation
```
git clone https://github.com/yeongsang2/app.git
cd app
```
## Download
* https://drive.google.com/file/d/1a8cCUyyHKn7MHTWRxdc5fUP5UzBLETcM/view?usp=share_link
* https://drive.google.com/file/d/1W69rCGWimN4cQs_PuQqoak1UFmWCXEz7/view?usp=share_link

/resource 에 .pt file 저장

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
