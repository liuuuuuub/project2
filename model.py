import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes=101, pretrained=True, dropout_rate=0.5):
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(weights=weights)

     #冻结前额部层参数
    for name, param in model.named_parameters():
         if "fc" not in name:
            param.requires_grad = False

    # 替换输出层
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.fc.in_features, num_classes))

    return model