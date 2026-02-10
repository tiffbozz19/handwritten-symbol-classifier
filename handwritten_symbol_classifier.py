import numpy as np 
import os
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import confusion_matrix, classification_report
import math
from torchinfo import summary
import onnx
from onnx import helper
from torchview import draw_graph

DATA_DIR = r"handwritten math symbols"
BATCH_SIZE = 64
IMAGE_SIZE = 64
EPOCHS = 35
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "math_symbol_cnn.pt"
HISTORY_CSV = "training_history.csv"
PLOT_LOSS_PATH = "training_curves.png"
PLOT_ACC_PATH = "training_accuracy.png"
CONF_MATRIX_PATH = "confusion_matrix.png"


# ---- TRANSFORMS ----
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---- DATASET AND SPLITS ----
base_dataset = datasets.ImageFolder(DATA_DIR)  
class_names = base_dataset.classes
num_classes = len(class_names)
print("Classes:", class_names)

total = len(base_dataset)
train_size = int(0.8 * total)
test_size = total - train_size

train_subset, test_subset = random_split(base_dataset, [train_size, test_size])

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = self.subset.dataset.loader(path)
        image = self.transform(image) 
        return image, label

train_ds = TransformSubset(train_subset, train_transform)
test_ds = TransformSubset(test_subset, eval_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

data_iterator = iter(train_loader)
X, Y = next(data_iterator)

image_grid = torchvision.utils.make_grid(X)
plt.imshow(np.transpose(image_grid.numpy(), (1, 2, 0)))
plt.show()

# ---- MODEL ----
class MathSymbolCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MathSymbolCNN(num_classes).to(DEVICE)

# ---------- CLEAN ARCHITECTURE DIAGRAM (TORCHVIEW) ----------
try:
    graph = draw_graph(
        model,
        input_size=(1, 1, IMAGE_SIZE, IMAGE_SIZE) 
    )

    graph.visual_graph.render("model_hierarchy", format="png", cleanup=True)
    print("Saved cleaner model diagram to model_hierarchy.png")
except Exception as e:
    print("Could not create torchview model diagram:", e)

summary(
    model,
    input_size=(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE),  # (batch, channels, H, W)
    col_names=("input_size", "output_size", "num_params", "kernel_size"),
    col_width=16,
    row_settings=("var_names",)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# ---- TRAINING LOOP ----
train_losses = []
train_accs = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"- train_loss: {train_loss:.4f} "
          f"- train_acc: {train_acc:.4f}")

print("Training done.")

# ---- Save final model checkpoint ----
torch.save(
    {
        "model": model.state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
    },
    MODEL_PATH,
)
print(f"Saved model to {MODEL_PATH}")

# ---- Save training history to CSV ----
with open(HISTORY_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc"])
    for i in range(EPOCHS):
        writer.writerow([i + 1, train_losses[i], train_accs[i]])

print(f"Saved training history to {HISTORY_CSV}")

# ---- PLOT TRAINING CURVES ----
epochs_range = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(PLOT_LOSS_PATH, dpi=150)

plt.figure()
plt.plot(epochs_range, train_accs, label="Train Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(PLOT_ACC_PATH, dpi=150)

print("Saved plots:", PLOT_LOSS_PATH, "and", PLOT_ACC_PATH)

# ---- TEST EVALUATION ----
best_checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(best_checkpoint["model"])
model.to(DEVICE)
model.eval()

# ---------- EXPORT MODEL TO ONNX FOR NETRON ----------
onnx_path = "math_symbol_cnn.onnx"
m = onnx.load(onnx_path)
print([(p.key, p.value) for p in m.metadata_props])

dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=16,
)

onnx_model = onnx.load(onnx_path)

def add_meta(key, value):
    meta = onnx_model.metadata_props.add()
    meta.key = key
    meta.value = value

add_meta("description", "Handwritten math symbol classifier (Pytorch -> ONNX")
add_meta("image_size", str(IMAGE_SIZE))
add_meta("num_classes", str(num_classes))
add_meta("author", "Tiffany Boswell")

onnx.save(onnx_model, onnx_path)
print("Added metadata to ONNX file.")

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

test_acc = (all_labels == all_preds).mean()
print(f"\nFinal Test Accuracy: {test_acc:.4f}\n")

model.eval()

mis_images = []
mis_true = []
mis_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        mismatch = preds != labels
        if mismatch.any():
            # Select only the misclassified ones
            wrong_images = images[mismatch]
            wrong_true = labels[mismatch]
            wrong_pred = preds[mismatch]

            mis_images.extend(wrong_images.cpu())
            mis_true.extend(wrong_true.cpu().tolist())
            mis_pred.extend(wrong_pred.cpu().tolist())

print(f"Total misclassified examples: {len(mis_images)}")

if len(mis_images) > 0:
    n_show = min(25, len(mis_images))
    cols = 5
    rows = math.ceil(n_show / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    axes = axes.flatten()

    for i in range(n_show):
        img = mis_images[i]
        img = img.squeeze(0)                 
        img = img * 0.5 + 0.5                

        axes[i].imshow(img.numpy(), cmap="gray")
        true_label = class_names[mis_true[i]]
        pred_label = class_names[mis_pred[i]]
        axes[i].set_title(f"T: {true_label}\nP: {pred_label}", fontsize=9)
        axes[i].axis("off")

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No misclassified examples to show!")

# ---- Confusion Matrix Plot ----
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=90)
plt.yticks = (tick_marks, class_names)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            f"{cm[i, j]}",
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=8
        )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH, dpi=150)

print("Saved confusion matrix plot to", CONF_MATRIX_PATH)