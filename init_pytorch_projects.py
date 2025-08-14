# 这是一个 PyTorch 项目集模板仓库初始化脚本
# 保存为 init_pytorch_projects.py 并运行
import os

# 定义项目结构
structure = {
    'stage1_basic': ['mnist', 'linear_regression'],
    'stage2_advanced': ['cifar10', 'imdb_sentiment', 'style_transfer'],
    'stage3_complex': ['unet_segmentation', 'yolov5_mini', 'clip_text_image'],
    'stage4_deployment': ['distributed_training', 'torchscript_export', 'onnx_runtime']
}

# 创建目录结构并生成模板文件
for stage, projects in structure.items():
    if not os.path.exists(stage):
        os.mkdir(stage)
    for project in projects:
        path = os.path.join(stage, project)
        os.makedirs(path, exist_ok=True)
        # 创建 train.py
        train_file = os.path.join(path, 'train.py')
        if not os.path.exists(train_file):
            with open(train_file, 'w') as f:
                f.write("""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# TODO: import your dataset and model

# 配置
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
batch_size = 64
lr = 1e-3
epochs = 10

# TODO: prepare your dataset and dataloader

# TODO: define your model
model = nn.Module()  # replace with your model
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# TODO: training loop
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
"""
                )
        # 创建 model.py
        model_file = os.path.join(path, 'model.py')
        if not os.path.exists(model_file):
            with open(model_file, 'w') as f:
                f.write("""import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # TODO: define layers

    def forward(self, x):
        # TODO: define forward pass
        return x
"""
                )
        # 创建 dataset.py
        dataset_file = os.path.join(path, 'dataset.py')
        if not os.path.exists(dataset_file):
            with open(dataset_file, 'w') as f:
                f.write("""from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, train=True):
        # TODO: load data
        pass

    def __len__(self):
        return 0  # replace with dataset length

    def __getitem__(self, idx):
        # TODO: return sample
        return None
"""
                )
print('PyTorch 项目集模板生成完成！')
