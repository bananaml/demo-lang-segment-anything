from potassium import Potassium, Request, Response
from PIL import Image
from lang_sam import LangSAM
import numpy as np
from lang_sam import LangSAM
from lang_sam import SAM_MODELS
from lang_sam.utils import draw_image
from lang_sam.utils import load_image
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv
import boto3

load_dotenv()

app = Potassium("lang-segment-anything")

AWS_ID = os.getenv("AWS_ID")
AWS_KEY = os.getenv("AWS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

@app.init
def init():
    model = LangSAM()
    context = {
        "model":model
    }
    return context

@app.handler()
def handler(context: dict, request: Request) -> Response:
    image_base64 = request.json.get("image")
    box_threshold = request.json.get("box_threshold")
    text_threshold = request.json.get("text_threshold")
    text_prompt = request.json.get("prompt")
    model = context.get("model")

    image_pil = Image.open("banana.png").convert("RGB")
    image_array = np.asarray(image_pil)
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold, text_threshold)
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    image.save('output.png')
    #upload to s3
    session = boto3.Session(
        aws_access_key_id=AWS_ID,
        aws_secret_access_key=AWS_KEY,
        region_name=AWS_REGION
    )
    s3 = session.client('s3')
    bucket_name = AWS_BUCKET_NAME
    with open("output.png", 'rb') as data:
        s3.upload_fileobj(data, bucket_name, "output.png")
    url = f"https://{bucket_name}.s3.amazonaws.com/djasjdmaklmflln"

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