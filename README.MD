# Getting start

## Prerequisites
* Docker

## Installation
```
git clone https://github.com/yeongsang2/app.git
```

## Usage
* 도커 이미지 빌드
```
$ docker build . -t "server:0.1"
```
* 컨테이너 실행 
```
$ docker run -p 8080:8080 server:0.1
```
