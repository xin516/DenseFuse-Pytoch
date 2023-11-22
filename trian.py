import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from utils import MyDataset
import os
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pytorch_msssim import ssim
from net import DenseFuse
from tqdm import tqdm
from config import Config

dataset = MyDataset(Config.Vis_img, Config.Inf_img, transform=ToTensor(), color_mode=Config.color)
train_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
#print(train_loader)
# 创建模型、优化器和损失函数
model = DenseFuse(Config.in_channel, Config.in_channel, fusion_strategy=Config.fusion_strategy)
optimizer = Adam(model.parameters(), lr=Config.lr)
loss_fn = MSELoss()
# 如果有可用的GPU，将模型移动到GPU
if Config.cuda and torch.cuda.is_available():
    model = model.cuda()

# 训练模型
for epoch in range(Config.epoch):  # 例如，我们训练100个epoch
    print('Epoch {}/{}'.format(epoch+1, Config.epoch))
    progress = tqdm(total=len(train_loader), ncols=80, leave=False)
    ssim_weight = Config.ssim_weights[epoch % len(Config.ssim_weights)]  # 根据当前的epoch编号选择权重值
    total_loss = 0.0  # 用于累积一轮的总损失
    for x1, x2 in train_loader:
        # 如果有可用的GPU，将数据移动到GPU
        if Config.cuda and torch.cuda.is_available():
            x1 = x1.cuda()
            x2 = x2.cuda()
       
        # 前向传播
        outputs = model(x1, x2)

        # 计算损失
        pixel_loss_total = 0
        ssim_loss_total = 0
        for i in range(x1.shape[0]):
            # 获取模型的对应输出
            output = outputs[i].unsqueeze(0)  # 确保维度匹配
            # 计算像素损失和SSIM损失
            pixel_loss_total += torch.norm(output - x1[i].unsqueeze(0))  # 像素损失
            ssim_loss_total += 1 - ssim(output, x1[i].unsqueeze(0))  # SSIM 损失
        pixel_loss = pixel_loss_total / len(outputs)
        ssim_loss = ssim_loss_total / len(outputs)
        loss = pixel_loss + ssim_weight * ssim_loss  # 总损失，这里使用选择的权重值
        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_description('Loss: {:.4f}'.format(loss.item()))
        progress.update(1)
    print('Total Loss for Epoch {}/{}: {:.4f}'.format(epoch+1, Config.epoch, total_loss / len(train_loader)))  # 输出每个epoch的平均损失
    if Config.in_channel == 1:
        torch.save(model.state_dict(), os.path.join(Config.save_model_dir, 'DenseFuse_epoch_{}_gray.pth'.format(epoch+1)))
    elif Config.in_channel == 3:
        torch.save(model.state_dict(), os.path.join(Config.save_model_dir, 'DenseFuse_epoch_{}_rgb.pth'.format(epoch+1)))
    else:
        print("Invalid color value: ", Config.color)
    progress.close()