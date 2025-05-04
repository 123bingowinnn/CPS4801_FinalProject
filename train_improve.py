import os
import shutil
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import OneCycleLR
import seaborn as sns
import matplotlib.pyplot as plt

# ============ 配置 ============
data_root = '/root/autodl-tmp/CPS4801'
class_dirs = {
    'no_sign':      os.path.join(data_root, 'COVID-19_no_infection_sign'),
    'limited_sign': os.path.join(data_root, 'COVID-19_very_limited_sign'),
}
base_dir    = os.path.join(data_root, 'binary_covid')
k_folds     = 5
epochs      = 30
batch_size  = 32
img_size    = 224
patience    = 5
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ 准备数据 ============
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
for lbl, src in class_dirs.items():
    dst = os.path.join(base_dir, lbl)
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        if fn.lower().endswith(('.png','jpg','jpeg')):
            shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))

# ============ 变换定义 ============
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])
val_tf = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])

# ============ Focal Loss ============
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        if self.alpha is not None:
            at = self.alpha[targets]
            loss = at * (1 - pt)**self.gamma * ce
        else:
            loss = (1 - pt)**self.gamma * ce
        return loss.mean() if self.reduction=='mean' else loss.sum()

# ============ 模型定义 ============
class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feats, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ============ 加载全数据集 ============
full_ds = datasets.ImageFolder(base_dir, transform=train_tf)
labels  = [s[1] for s in full_ds.samples]
classes = full_ds.classes
print("Classes:", classes)

# ============ 5折交叉验证 ============
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
best_fold, best_acc = None, 0.0

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
    print(f"\n— Fold {fold}/{k_folds} —")
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(datasets.ImageFolder(base_dir, transform=val_tf), val_idx)

    lbls = [labels[i] for i in train_idx]
    cnts = Counter(lbls)
    weights = [1.0/cnts[l] for l in lbls]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model     = CustomResNet50(num_classes=len(classes)).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3,
                           steps_per_epoch=len(train_loader),
                           epochs=epochs, pct_start=0.3, anneal_strategy='cos')

    best_val, no_imp = 0.0, 0
    for ep in range(1, epochs+1):
        model.train()
        tcorr, ttot = 0, 0
        for imgs, lbl in train_loader:
            imgs, lbl = imgs.to(device), lbl.to(device)
            outs = model(imgs)
            loss = criterion(outs, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            preds = outs.argmax(1)
            tcorr += (preds==lbl).sum().item()
            ttot  += lbl.size(0)
        train_acc = tcorr/ttot

        model.eval()
        vcorr, vtot = 0, 0
        with torch.no_grad():
            for imgs, lbl in val_loader:
                imgs, lbl = imgs.to(device), lbl.to(device)
                preds = model(imgs).argmax(1)
                vcorr += (preds==lbl).sum().item()
                vtot  += lbl.size(0)
        val_acc = vcorr/vtot
        print(f" Epoch {ep:02d} — Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val, no_imp = val_acc, 0
            torch.save(model.state_dict(), f'best_fold{fold}.pth')
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f" Early stopping at epoch {ep}")
                break

    print(f" Fold {fold} best_val: {best_val:.4f}")
    if best_val > best_acc:
        best_acc, best_fold = best_val, fold

print(f"\n>> Best fold: {best_fold} with Val Acc: {best_acc:.4f}")

# ============ 全数据重新训练 ============
print("\n=== Final training on full dataset with best hyperparameters ===")
full_train_ds = datasets.ImageFolder(base_dir, transform=train_tf)
full_val_ds   = datasets.ImageFolder(base_dir, transform=val_tf)

# 打印类别统计信息
full_labels = [s[1] for s in full_train_ds.samples]
print("Full dataset label count:", Counter(full_labels))

# 构建权重 & 采样器
cnts = Counter(full_labels)
weights = [1.0 / cnts[l] for l in full_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
class_weight = [sum(cnts.values()) / cnts[i] for i in range(len(classes))]
class_weight = torch.tensor(class_weight).to(device)

# Loader
full_train_loader = DataLoader(full_train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
full_val_loader = DataLoader(full_val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# 模型 & 损失 & 优化器
final_model = CustomResNet50(num_classes=len(classes)).to(device)
final_model.load_state_dict(torch.load(f'best_fold{best_fold}.pth'))
final_criterion = FocalLoss(gamma=2.0, alpha=class_weight)
final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-2)
final_scheduler = OneCycleLR(final_optimizer, max_lr=1e-3,
                             steps_per_epoch=len(full_train_loader),
                             epochs=10, pct_start=0.3, anneal_strategy='cos')

# Train final model
for ep in range(1, 11):
    final_model.train()
    for imgs, lbl in full_train_loader:
        imgs, lbl = imgs.to(device), lbl.to(device)
        outs = final_model(imgs)
        loss = final_criterion(outs, lbl)
        final_optimizer.zero_grad()
        loss.backward()
        final_optimizer.step()
        final_scheduler.step()

# ============ Final Evaluation ============
final_model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for imgs, lbl in full_val_loader:
        imgs = imgs.to(device)
        preds = final_model(imgs).argmax(1).cpu().tolist()
        all_preds += preds
        all_true += lbl.tolist()

print("\n=== Final Classification Report on Entire Dataset ===")
print(classification_report(all_true, all_preds, target_names=classes))

# 混淆矩阵
cm = confusion_matrix(all_true, all_preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# === Final Classification Report on Entire Dataset ===
#               precision    recall  f1-score   support

# limited_sign       0.61      0.94      0.74       264
#      no_sign       0.98      0.80      0.88       801

#     accuracy                           0.84      1065
#    macro avg       0.80      0.87      0.81      1065
# weighted avg       0.89      0.84      0.85      1065
