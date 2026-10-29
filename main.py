import base64
import binascii
import io
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

# 模型路径基于本文件所在目录，避免依赖启动时的工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "resnet18.pt")

# 设备自适应：有 GPU 用 GPU，没有就退回 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, map_location):
    """加载整个 nn.Module。

    torch>=2.6 的 torch.load 默认 weights_only=True，加载完整模型会报错，
    这里显式传 weights_only=False；旧版本没有该参数，会抛 TypeError 后回退。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# 加载预训练的ResNet模型
model = load_model(MODEL_PATH, device)
model.eval()

# 图像预处理：只构建一次，所有请求复用
transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])


def invert_gray(img):
    """灰度图反色：255 - 像素值。

    用 numpy 向量化替代原来的双层 for 循环，uint8 下 255-p 不会回绕，
    结果与原逐像素实现逐字节一致。
    """
    return Image.fromarray(255 - np.asarray(img, dtype=np.uint8))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    img_data = request.form.get('image')  # 获取前端传回的图像数据
    if not img_data:
        return "错误：未收到图像数据，请重新绘制后再试", 400

    img_data = img_data.split(',')[-1].strip()  # 去除Base64编码前缀
    if not img_data:
        return "错误：图像数据为空", 400

    # 解码Base64字符串并将其转换为图像对象
    try:
        img_bytes = base64.b64decode(img_data)
    except (binascii.Error, ValueError):
        return "错误：图像数据不是合法的 Base64 编码", 400

    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.load()  # 强制解码，损坏的图片会在这里抛错
    except (UnidentifiedImageError, OSError, ValueError):
        return "错误：无法解析上传的图片", 400

    if img.mode == 'RGBA':
        # 把透明通道合成到白底上（与原逻辑一致）
        rgb_image = Image.new("RGB", img.size, (255, 255, 255))
        rgb_image.paste(img, mask=img.split()[3])
        img = rgb_image
    else:
        # 无 alpha 通道的图（RGB/L/P 等）原先会在 img.split()[3] 处崩，这里直接转 RGB
        img = img.convert('RGB')

    img = invert_gray(img.convert('L'))

    # 使用预训练的模型进行预测
    with torch.no_grad():
        img = transform(img).unsqueeze(0).to(device)
        out = model(img)
        a = torch.softmax(out, 1)
        prb, predict = torch.max(a, 1)

    # 返回识别结果给前端（概率保留 4 位小数）
    return "预测结果为" + str(predict.item()) + "概率为" + "%.4f" % prb.item()


if __name__ == '__main__':
    app.run(debug=True)
