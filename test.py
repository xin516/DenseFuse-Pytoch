import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from net import DenseFuse
from tqdm import tqdm
import torch
from config import Config
def test(model_path, ir_path, vi_path):
    # 加载模型

    model = DenseFuse(Config.in_channel, Config.in_channel, fusion_strategy=Config.fusion_strategy)  # 这里需要你自行定义网络的参数
    model.load_state_dict(torch.load(model_path))

    if Config.cuda and torch.cuda.is_available():
        model = model.cuda()

    to_tensor = transforms.ToTensor()
    ir_images = sorted(os.listdir(ir_path), key=lambda x: int(x.split('.')[0]))
    vi_images = sorted(os.listdir(vi_path), key=lambda x: int(x.split('.')[0]))
    # 遍历测试数据
    for i, (ir_image_file, vi_image_file) in tqdm(enumerate(zip(ir_images, vi_images)), total=len(ir_images)):
        #print('Processing image pair {}/{}'.format(i+1, len(ir_images)))
        # 加载并转换图像
        ir_image = Image.open(os.path.join(ir_path, ir_image_file)).convert('L')
        vi_image = Image.open(os.path.join(vi_path, vi_image_file)).convert('L')
        ir_tensor = to_tensor(ir_image).unsqueeze(0)
        vi_tensor = to_tensor(vi_image).unsqueeze(0)

        if torch.cuda.is_available():
            ir_tensor = ir_tensor.cuda()
            vi_tensor = vi_tensor.cuda()

        # 前向传播
        outputs = model(ir_tensor, vi_tensor)

        # 将输出转换为图像并保存
        output_image = transforms.ToPILImage()(outputs.cpu().data[0])
        output_image.save(os.path.join(Config.result, 'output_{}.png'.format(i)))

# 使用模型
test(Config.model_path_gray, Config.test_Inf_img, Config.test_Vis_img)