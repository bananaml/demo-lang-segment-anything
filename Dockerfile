FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG AWS_ACCESS
ARG AWS_BUCKET
ARG AWS_REGION
ARG AWS_SECRET

WORKDIR /

# Update and install dependencies
RUN apt-get update && \
    apt-get -y install git cmake g++ libgl1-mesa-glx libglib2.0-0 wget

# Install cudatoolkit-dev
#RUN conda install -y -c conda-forge cudatoolkit-dev
RUN conda install -y -c conda-forge cudatoolkit=11.3

RUN pip3 install --upgrade pip
# Clone and install lang-segment-anything
RUN git clone https://github.com/luca-medeiros/lang-segment-anything && \
    cd lang-segment-anything && \
    pip3 install torch torchvision && \
    pip3 install -e .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENV AWS_ACCESS=${AWS_ACCESS}
ENV AWS_BUCKET=${AWS_BUCKET}
ENV AWS_REGION=${AWS_REGION}
ENV AWS_SECRET=${AWS_SECRET}

# Add your model weight files 
ADD download.py .
RUN python3 download.py

ADD . .

EXPOSE 8000

CMD python3 -u app.py