from transformers import ViTImageProcessor, ViTModel
from PIL import Image


class Classifier:
    def __init__(self) -> None:
        model_name = "NightingaleCen/plant-vit"

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

    def infer(self, image_fp):
        img = Image.open(image_fp)
        inputs = self.processor(images=img, return_tensors="pt")
        outputs = self.model(**inputs)
