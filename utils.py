import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from config import Config
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
class MyDataset(Dataset):
    def __init__(self, ir_folder, vi_folder, color_mode=Config.color, transform=None):
        self.ir_folder = ir_folder
        self.vi_folder = vi_folder
        self.transform = transform

        # 假设每个文件夹中的图像数量相同，且图像是按顺序对应的
        self.image_filenames = sorted(os.listdir(ir_folder))

        # 根据color_mode参数决定是否应用Grayscale转换
        if color_mode == 'grayscale':
            self.ir_transform = Compose([
                Resize((Config.image_size, Config.image_size)),  # 将所有图像调整为相同的大小
                Grayscale(num_output_channels=1),
                ToTensor()
            ])
            self.vi_transform = Compose([
                Resize((Config.image_size, Config.image_size)),  # 将所有图像调整为相同的大小
                Grayscale(num_output_channels=1),
                ToTensor()
            ])
        elif color_mode == 'color':
            self.ir_transform = Compose([
                Resize((Config.image_size, Config.image_size)),  # 将所有图像调整为相同的大小
                ToTensor()
            ])
            self.vi_transform = Compose([
                Resize((Config.image_size, Config.image_size)),  # 将所有图像调整为相同的大小
                ToTensor()
            ])
        else:
            raise ValueError(f'Unknown color_mode: {color_mode}')

    def __getitem__(self, index):
        # 读取一对图像
        ir_image = Image.open(os.path.join(self.ir_folder, self.image_filenames[index]))
        vi_image = Image.open(os.path.join(self.vi_folder, self.image_filenames[index]))

        # 如果有定义转换，则应用转换
        ir_image = self.ir_transform(ir_image)
        vi_image = self.vi_transform(vi_image)
        #print(type(ir_image))  # 应该打印出 <class 'torch.Tensor'>
        #print(type(vi_image))  # 应该打印出 <class 'torch.Tensor'>
        return ir_image, vi_image

    def __len__(self):
        return len(self.image_filenames)
    
'''from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

ir_folder = "./ir"
vi_folder = "./vi"
dataset = MyDataset(ir_folder, vi_folder, transform=ToTensor())
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(train_loader)
dataiter = iter(train_loader)
images1, images2 = dataiter.next()

# 打印图像的形状和标签
print(images1.shape)
print(images2.shape)'''