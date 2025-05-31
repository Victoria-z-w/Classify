import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import DenseNet161_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=r'F:\train_path', transform=transform)
print(f"Total data: {len(dataset)}")

# 正确分割数据集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
print(f"Train data: {len(train_data)}, Val data: {len(val_data)}")

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, pin_memory=True)

model = models.densenet161(weights=DenseNet161_Weights.DEFAULT)
num_classes = 12
model.classifier = nn.Linear(model.classifier.in_features, num_classes)  # 重置分类头
model = model.to(device)

# 冻结除指定层外的所有参数
for param in model.parameters():
    param.requires_grad = False

for param in model.features.denseblock3.parameters():
    param.requires_grad = True
for param in model.features.denseblock4.parameters():
    param.requires_grad = True
model.classifier.requires_grad_(True)

# 正确配置优化器参数组
param_groups = [
    {'params': model.features.denseblock3.parameters(), 'lr': 5e-6},
    {'params': model.features.denseblock4.parameters(), 'lr': 2e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
]

# optimizer = optim.Adam(param_groups, lr=0.0001)
optimizer = optim.Adam(param_groups)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_data)
    train_acc = correct / total

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images_val, labels_val in val_loader:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = model(images_val)
            loss_val = criterion(outputs_val, labels_val)
            val_loss += loss_val.item() * images_val.size(0)
            _, predicted_val = torch.max(outputs_val.data, 1)
            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()

    val_loss /= len(val_data)
    val_acc = correct_val / total_val

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    scheduler.step()

# model.cpu()
example_input = torch.rand(1, 3, 224, 224)
script_model = torch.jit.trace(model, example_input)
script_model.save("ClassifyModel.pt")