FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG AWS_ID
ARG AWS_KEY 
ARG AWS_REGION
ARG AWS_BUCKET_NAME 

WORKDIR /

# Update and install dependencies
RUN apt-get update && \
    apt-get -y install git cmake g++ libgl1-mesa-glx libglib2.0-0 wget

# Install cudatoolkit-dev
RUN conda install -y -c conda-forge cudatoolkit-dev

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

ENV AWS_ID=${AWS_ID}
ARG AWS_KEY=${AWS_KEY} 
ARG AWS_REGION=${AWS_REGION}
ARG AWS_BUCKET_NAME=${AWS_BUCKET_NAME}

# Add your model weight files 
ADD download.py .
RUN python3 download.py

ADD . .

EXPOSE 8000

CMD python3 -u app.py