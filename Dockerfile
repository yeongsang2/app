## Build the image torchserve locally before running this, cf github torchserve:
## https://github.com/pytorch/serve/tree/master/docker
FROM python:3.9

RUN mkdir /user

COPY . /user/app

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y python3-distutils
RUN pip install -r /user/app/requirements.txt
RUN sed -i 's/numpy.asscalar(delta_e)/delta_e.item()/g' /usr/local/lib/python3.9/site-packages/colormath/color_diff.py

WORKDIR /user/app

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
