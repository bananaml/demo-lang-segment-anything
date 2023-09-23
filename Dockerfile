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
RUN conda install -y -c conda-forge cudatoolkit-dev

RUN pip install --upgrade pip
RUN pip install potassium

RUN git clone https://github.com/luca-medeiros/lang-segment-anything && \
    cd lang-segment-anything && \
    pip install torch torchvision && \
    pip install -e .
# Clone lang-segment-anything
#RUN git clone https://github.com/luca-medeiros/lang-segment-anything
#pip install torch torchvision
#pip install -e .
# Install Python dependencies
#ADD requirements.txt requirements.txt
#RUN pip3 install -r requirements.txt

#RUN pip install --force-reinstall torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html

ENV AWS_ACCESS=${AWS_ACCESS}
ENV AWS_BUCKET=${AWS_BUCKET}
ENV AWS_REGION=${AWS_REGION}
ENV AWS_SECRET=${AWS_SECRET}

COPY install.sh /install.sh
RUN chmod +x /install.sh
# Add your model weight files 
ADD download.py .
RUN python3 download.py

ADD . .

EXPOSE 8000

CMD ["sh", "-c", "bash /install.sh && python -u app.py"]