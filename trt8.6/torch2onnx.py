import torch
import torchvision.models as models

# 加载预训练的resnet18模型
model = models.resnet18(pretrained=True)
model.eval()

# 为模型输入创建一个虚拟输入。这里使用1x3x224x224，因为ResNet期望的输入是224x224的RGB图像。
dummy_input = torch.randn(1, 3, 224, 224)

# 导出模型到ONNX
torch.onnx.export(model, dummy_input, "resnet18.onnx")