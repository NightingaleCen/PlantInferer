import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image


class Classifier:
    """植物分类模型"""

    def __init__(self, model_name="NightingaleCen/plant-vit") -> None:
        """
        从Hugging Face初始化植物分类器
        使用private模型可能需要先登陆Hugging Face账号，在终端执行`huggingface-cli login`以登陆

        Args:
            model_name (str, optional): Hugging Face模型名称. Defaults to "NightingaleCen/plant-vit".

        Returns:
            int: 初始化成功则返回值为0
        """

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model = ViTForImageClassification.from_pretrained(model_name).to(
                self.device
            )
        except:
            return -1

        return 0

    def infer(self, image_fp):
        """
        根据输入的图像输出得到分类置信度
        Args:
            image_fp (`String`, `pathlib.Path object`, file object.): 图像的路径或是（类）文件指针，参见Image.open

        Returns:
            `Dict`: 以{学名：置信度}的格式返回的分类置信度数据
        """
        img = Image.open(image_fp)
        id2label = self.model.config.id2label

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        output_logits = self.model(**inputs).logits
        probs = torch.softmax(output_logits, -1).detach().numpy()[0]

        return {id2label[i]: prob for i, prob in enumerate(probs)}
