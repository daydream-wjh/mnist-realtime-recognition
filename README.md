# minist

实时手写数字识别：浏览器里用鼠标写一个 0–9 的数字，Flask 后端把画布图像转成 MNIST 风格的输入，交给微调过的 **ResNet18** 分类，弹窗返回预测结果与置信度。

## 项目简介

整个推理服务只有 `main.py` 一个文件 + 一个前端页面 `templates/index.html`：

- 前端是一块 300×300 的 `<canvas>`，鼠标按下拖动即画线（线宽 10px、圆头笔刷），"识别"按钮把画布 `toDataURL('image/png')` 后以 `image=<base64>` 表单 POST 到 `/recognize`；
- 后端解码 Base64 → 以白色为底把 RGBA 画布合成到 RGB（去掉透明通道）→ 转灰度 → **逐像素反色**（前端是黑笔白底，MNIST 是白字黑底）→ `Resize(224)` → `ToTensor` → `Normalize(0.5, 0.5)` → 送入模型；
- 输出 `torch.softmax(out, 1)` 的 top-1，直接把 `"预测结果为X概率为Y"` 以纯文本返回，前端用 `alert` 显示；
- 模型是 `torch.save(model)` 保存的**整个 nn.Module**，不是 state_dict，因此 `torch.load("resnet18.pt")` 即可得到可调用的模型。

## 效果展示

浏览器中手写数字 "5"，弹窗返回 `预测结果为5概率为0.99999…`：

![网页手写识别示例](./data/show.png)

## 数据集

[MNIST](http://yann.lecun.com/exdb/mnist/) 手写数字，10 类（0–9）。

- 仓库内 `data/MNIST/raw/` 已包含原始 idx 文件（`train-images-idx3-ubyte`、`train-labels-idx1-ubyte`、`t10k-*`）及其 `.gz` 压缩包，约 54 MB，可离线训练，无需 `download=True` 联网；
- 训练脚本（notebook）中的切分与预处理：
  - 训练集 `datasets.MNIST(root='data', train=True)`，按索引 `np.random.shuffle` 后取 **90% 训练 / 10% 验证**（`SubsetRandomSampler`），batch size 128；
  - 测试集 `train=False`，10000 张，batch size 128；
  - 变换：`Resize((224, 224))` → `ToTensor()` → `Normalize((0.5,), (0.5,))`。
- `main.py` 推理端的预处理与训练一致（同样 224×224、同样 0.5/0.5 归一化），差异见"已知问题"。

## 模型 / 算法

### 网络结构

`torchvision.models.resnet18(pretrained=True)` 改造两处（见 notebook 中 `pre_trained_model()`）：

| 位置 | 改造 |
| --- | --- |
| `conv1` | `nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)`，把 ImageNet 的 3 通道首层换成 **1 通道**（灰度输入） |
| `fc` | 换成 `Sequential(Linear(512,256), ReLU, Dropout(0.2), Linear(256,64), ReLU, Linear(64,10), LogSoftmax(dim=1))` |

`resnet18.pt` 反序列化后确实是上述结构：`conv1 … bn1 … relu … maxpool … layer1..4 … avgpool … fc`，其中 `fc` 为 7 个子模块的 `Sequential`，`conv1.weight` 形状对应单通道输入。

### 训练策略

- 损失：`nn.NLLLoss()`（与 `fc` 末尾的 `LogSoftmax` 配对）；
- 优化器：`Adam(lr=0.001)`，`num_of_iterations = 10` 轮；
- 每轮在训练 sampler 与验证 sampler 上各跑一遍 `check_accuracy`（`argmax` 比对），并记录 `running_loss/steps`；
- 训练结束用 matplotlib 画 训练/验证准确率 + 训练损失曲线（仅 `plt.show()`，不落盘）；
- 最后 `torch.save(resnet18_model, './resnet18.pt')` 保存整模型。

### 评估指标

notebook 里保留的唯一一次运行输出（10 轮、batch 128、单卡 GPU）：

```
Training Accuracy:  0.9963
Validation Accuracy: 0.9912
Training Time:  59.47  minutes
```

即**训练集 99.63% / 验证集 99.12%**。notebook 开头提到"上一个 notebook 用若干预训练模型做到约 99.30%"，本 notebook 的设想是把这些模型做加权集成再涨一点——但**仓库里没有集成代码，也没有任何输出记录 99.60% 的测试集准确率**，因此 README 标题里的 99.60% 标注为：**待复核**。

## 技术栈与依赖

按代码 import 与 notebook 元数据推断：

- Python 3.10（notebook `language_info.version = 3.10.12`）
- `torch`、`torchvision`（`models.resnet18`、`datasets.MNIST`、`transforms`）
- `flask`（`Flask`、`render_template`、`request`）
- `pillow`（`Image`、`Image.new/paste/split`）
- `numpy`、`matplotlib`、`tqdm`

没有 `requirements.txt`，需自行安装：

```bash
pip install torch torchvision flask pillow numpy matplotlib tqdm
```

## 目录结构

```
minist/
├── main.py                                    # Flask 入口：/ 与 /recognize，含推理预处理
├── resnet18.pt                                # 整模型 pickle（torch.save(model)，约 43 MB）
├── mnist-weighted-ensemble-99-60-accuracy.ipynb  # 训练 + 保存模型的 notebook（含一次运行输出）
├── mnist-weighted-ensemble-99-60-accuracy.py  # 空文件（0 字节）
├── templates/
│   └── index.html                             # canvas 画板 + /recognize 的 XHR 调用
├── data/
│   ├── show.png                               # 效果截图
│   └── MNIST/raw/                             # MNIST 原始 idx 文件（含 .gz 副本）
└── .idea/                                     # PyCharm 工程文件
```

## 快速开始

### 1. 直接启动 Web 服务（用仓库自带权重）

```bash
cd minist
pip install torch torchvision flask pillow numpy
python main.py
# Flask 默认监听 http://127.0.0.1:5000
```

浏览器打开页面 → 在白色画板里写一个数字 → 点"识别" → 弹窗显示 `预测结果为X概率为Y`。"清除"按钮清空画布。

> 必须在 `minist/` 目录下启动：`torch.load("resnet18.pt")`、`render_template('index.html')` 都是相对当前工作目录的路径。

### 2. 重新训练

打开 `mnist-weighted-ensemble-99-60-accuracy.ipynb`，从上到下执行即可（数据默认读 `./data`，`download=True` 时若本地已有 idx 文件不会重复下载）。10 轮约需 1 小时（单卡 GPU）。训练完成后最后一段 `torch.save(resnet18_model, './resnet18.pt')` 会覆盖权重文件。

配套的 `.py` 子文件是 0 字节占位，**不能用 `python mnist-weighted-ensemble-99-60-accuracy.py` 训练**，请用 notebook。

### 3. 只用权重做单图推理

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

model = torch.load("resnet18.pt")          # 整模型；torch>=2.6 需加 weights_only=False
model.eval()

tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])
img = Image.open("digit.png").convert("L")
with torch.no_grad():
    prob = torch.softmax(model(tf(img).unsqueeze(0)), 1)
    p, idx = prob.max(1)
print(idx.item(), p.item())
```

## 已知问题

- **99.60% 待复核**：仓库内 notebook 只有单模型 10 轮训练（训练 99.63% / 验证 99.12%），没有加权集成的代码，也没有 99.60% 的实测输出；"加权集成"目前只是 notebook 开头的设想。
- **`mnist-weighted-ensemble-99-60-accuracy.py` 是 0 字节空文件**，无法直接跑脚本训练。
- **GPU 硬编码**：`main.py` 中 `img = img.cuda().unsqueeze(0)`，没有 `torch.cuda.is_available()` 判断，无 GPU 环境会直接抛错；也无法用 `--device cpu` 切换。
- **整模型 pickle**：`resnet18.pt` 是 `torch.save(model)` 的整个模块，加载依赖 `torchvision.models.resnet.ResNet` 的类型/字段；PyTorch ≥ 2.6 默认 `weights_only=True` 会失败，需显式 `torch.load(..., weights_only=False)`。仅支持 CPU 推理时还需对反序列化出的模型做 `.cpu()`（或先在有 GPU 的机器上重存 state_dict）。
- **预处理与 MNIST 分布有差距**：前端画布 300×300 直接 `Resize(224)`，没做"找数字外接框 → 裁剪 → 等比缩放 → 居中/补边"的规范化，笔画粗细（10px）和数字位置随用户手写习惯变化，实际使用中误判比 notebook 的验证准确率差，效果图里那种 0.99999 的置信度并非普遍情况。
- **无输入校验**：`/recognize` 直接 `request.form.get('image')` 后 `split(',')[-1]`、`base64.b64decode`，参数缺失时抛异常返回 500。
- **`app.run(debug=True)`**：调试模式直接暴露在端口上，不适合任何公开部署。
- **大文件入库**：`resnet18.pt`（约 43 MB）+ `data/MNIST/raw`（约 54 MB）都提交进了仓库，未使用 Git LFS / Release 附件。
- **无 `requirements.txt`**，`.idea/` 工程文件与 `.DS_Store` 一并入库。
- 训练/评估只 `plt.show()`，损失与准确率曲线没有落到文件，也没有保存 test 集指标。
