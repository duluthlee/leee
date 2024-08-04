import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from skimage import io

# 自定义数据集类
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# 读取所有图像和掩码路径
image_dir = r'C:\Users\dulut\Desktop\hackathon\org_img'
mask_dir = r"C:\Users\dulut\Desktop\hackathon\new_mask"
images = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
masks = sorted([os.path.join(mask_dir, msk) for msk in os.listdir(mask_dir)])

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
dataset = SegmentationDataset(images, masks, transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 使用SMP库加载预训练的Unet模型
model = smp.Unet(
    encoder_name="resnet34",        # 选择编码器，例如resnet34
    encoder_weights="imagenet",     # 使用在ImageNet上预训练的权重
    in_channels=3,                  # 输入通道数（RGB图像为3，灰度图像为1）
    classes=1,                      # 输出通道数（类别数）
)

# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# 计算 Dice 系数的函数
def dice_coefficient(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union.item() == 0:
        return 1.0
    else:
        return (2.0 * intersection) / union

# 保存结果的函数
def save_visualization(save_path, images, masks, predictions):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, (image, mask, prediction) in enumerate(zip(images, masks, predictions)):
        prediction = prediction.squeeze()  # 去除多余的维度
        prediction = (prediction > 0.5).astype(np.uint8)  # 二值化
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
        ax[0].set_title("Input Image")
        ax[1].imshow(mask.squeeze().cpu().numpy(), cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[2].imshow(prediction, cmap="gray")
        ax[2].set_title("Prediction")
        plt.savefig(os.path.join(save_path, f"{i}_comparison.png"))
        plt.close()

# 记录训练损失及指标
train_losses = []
train_dices = []

# 训练
num_epochs = 150  # 增加epoch数量
best_train_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_dice = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_dice += dice_coefficient(outputs, masks).item()
    train_losses.append(train_loss / len(train_loader))
    train_dices.append(train_dice / len(train_loader))

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Dice: {train_dices[-1]:.4f}")

    # 学习率调度
    scheduler.step(train_loss)

    # 保存最好的模型
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), 'best_model.pth')

# 保存最终模型
torch.save(model.state_dict(), 'final_model.pth')

# 可视化训练集上的效果
def visualize_training_set(model, train_loader, device, num_images=10):
    model.eval()
    images_shown = 0
    with torch.no_grad():
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()

            save_visualization("data/train_visualizations", images, masks, outputs)
            
            images_shown += len(images)
            if images_shown >= num_images:
                break

visualize_training_set(model, train_loader, device)

# 可视化训练损失及Dice系数
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'r', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_dices, 'r', label='Training Dice')
plt.title('Training Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()

plt.tight_layout()
plt.savefig("loss_and_dice.png")
plt.show()
