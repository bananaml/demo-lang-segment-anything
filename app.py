from potassium import Potassium, Request, Response
import uuid
from PIL import Image
import numpy as np
import base64
from PIL import Image
import io
import os
import boto3
import torch 
import logging
import subprocess
logging.basicConfig(level=logging.DEBUG)

app = Potassium("lang-segment-anything")

AWS_ACCESS = os.getenv('AWS_ACCESS')
AWS_BUCKET = os.getenv('AWS_BUCKET')
AWS_REGION = os.getenv('AWS_REGION')
AWS_SECRET = os.getenv('AWS_SECRET')

@app.init
def init():
    context = {
    }
    return context

@app.handler()
def handler(context: dict, request: Request) -> Response:
    logger = logging.getLogger(__name__)
    logger.info("start handler:")
    """logger.info("run check_gpu.py:")
    subprocess.run(["python3", "check_gpu.py"], check=True)
    os.chdir("lang-segment-anything")
    logger.info("run torch:")
    subprocess.run(["pip", "install", "-v", "torch", "torchvision"], check=True)
    logger.info("run -e:")
    subprocess.run(["pip", "install", "-v", "-e", "."], check=True)
    os.chdir("..")"""
    """installed_packages = subprocess.check_output(["pip", "freeze"])
    installed_packages = installed_packages.decode("utf-8")
    logger.info(f"Installed packages: {installed_packages}")
    current_directory = os.getcwd()"""
    """logger.info(f"Current directory: {current_directory}")
    for file in os.listdir(current_directory):
        logger.info(file)"""
    from lang_sam import LangSAM
    from lang_sam import LangSAM
    from lang_sam import SAM_MODELS
    from lang_sam.utils import draw_image
    from lang_sam.utils import load_image
    model = LangSAM()
    logger = logging.getLogger(__name__)
    logger.info("new iteration:")
    current_directory = os.getcwd()
    logger.info(f"Current directory: {current_directory}")
    logger.info("Directory contents:")
    image_base64 = request.json.get("image")
    box_threshold = request.json.get("box_threshold")
    text_threshold = request.json.get("text_threshold")
    text_prompt = request.json.get("prompt")
    #model = context.get("model")
    image_pil = Image.open("banana.png").convert("RGB")
    image_array = np.asarray(image_pil)
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold, text_threshold)
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    filename = str(uuid.uuid4()) + '.png'
    image.save(filename)
    #image.save('output2.png')
    #filename='output2.png'
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION
    )
    s3 = session.client('s3')
    bucket_name = AWS_BUCKET
    with open(filename, 'rb') as data:
        s3.upload_fileobj(data, bucket_name, filename)
    url = f"https://{bucket_name}.s3.amazonaws.com/{filename}"
    return Response(json={"output": url}, status=200)
    """image_data = base64.b64decode(image_base64)
    image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold, text_threshold)
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return Response(json={"output": img_str}, status=200)"""
    
if __name__ == "__main__":
    app.serve()