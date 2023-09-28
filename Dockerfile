FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG AWS_ACCESS
ARG AWS_BUCKET
ARG AWS_REGION
ARG AWS_SECRET

WORKDIR /

RUN apt-get update && \
    apt-get -y install git cmake g++ libgl1-mesa-glx libglib2.0-0 wget

RUN conda install -y -c conda-forge cudatoolkit-dev

RUN pip install --upgrade pip
RUN pip install potassium

RUN git clone https://github.com/luca-medeiros/lang-segment-anything

ENV AWS_ACCESS=${AWS_ACCESS}
ENV AWS_BUCKET=${AWS_BUCKET}
ENV AWS_REGION=${AWS_REGION}
ENV AWS_SECRET=${AWS_SECRET}

ADD download.py .
RUN python3 download.py

ADD . .

EXPOSE 8000

CMD ["sh", "-c", "python check_gpu.py && cd lang-segment-anything && pip install -v torch torchvision && pip install -v -e . && cd .. && python -u app.py"]
