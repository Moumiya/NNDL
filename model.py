import torch.nn as nn
import torchvision.models as models

class MedicalNet(nn.Module):
    def _init_(self, num_classes=2):
        super(MedicalNet, self)._init_()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
