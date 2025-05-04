import os
import random
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

# ============================
# 1. 配置参数
# ============================
data_root = '/root/autodl-tmp/CPS4801'
class_dirs = {
    'no_sign': os.path.join(data_root, 'COVID-19_no_infection_sign'),
    'limited_sign': os.path.join(data_root, 'COVID-19_very_limited_sign'),
}
img_size    = 224
batch_size  = 32
num_epochs  = 20
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================
# 2. 构建 ImageFolder 结构
# ============================
# 将两个类别的文件夹临时映射到一个统一 root 下
# ImageFolder 要求 data_root/类名/图片.jpg
tmp_root = '/root/autodl-tmp/CPS4801/dataset_for_train'
if os.path.exists(tmp_root):
    # 如果之前跑过，先清空
    import shutil
    shutil.rmtree(tmp_root)

for cls_name, src_dir in class_dirs.items():
    dst_dir = os.path.join(tmp_root, cls_name)
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            os.symlink(src_path, dst_path)  # 建立软链接，节省空间

# ============================
# 3. 数据增广与加载
# ============================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_ds = datasets.ImageFolder(tmp_root, transform=transform)
class_names = full_ds.classes  # ['limited_sign', 'no_sign']
print("Classes:", class_names)

# 按 70/15/15 划分
total = len(full_ds)
n_train = int(0.7 * total)
n_val   = int(0.15 * total)
n_test  = total - n_train - n_val
train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test],
                                         generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ============================
# 4. 构建模型
# ============================
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ============================
# 5. 训练与验证
# ============================
for epoch in range(1, num_epochs+1):
    model.train()
    running_correct, running_total = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)
    train_acc = running_correct / running_total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    print(f"[Epoch {epoch:02d}/{num_epochs}] Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

# ============================
# 6. 测试评估
# ============================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

print("\n=== Test Set Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=class_names))

# === Test Set Classification Report ===
#               precision    recall  f1-score   support
#
# limited_sign       0.54      0.62      0.57        47
#      no_sign       0.83      0.78      0.81       114
#
#     accuracy                           0.73       161
#    macro avg       0.68      0.70      0.69       161
# weighted avg       0.75      0.73      0.74       161

