import torch
import torch.nn as nn
import os
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import argparse
import shutil
import matplotlib.pyplot as plt
from Dataset_CamXuc import CamXuc_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="C:/Hoc_May/All_Project/Predict_CamXuc/dataset_classification")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=260)  # chuẩn của EfficientNet-B2
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_path", type=str, default="C:/Hoc_May/All_Project/Predict_CamXuc/EfNet_tensorboard/EfficientNet_B2")
    parser.add_argument("--checkpoint_path", type=str, default="C:/Hoc_May/All_Project/Predict_CamXuc/EfNet_checkpoint/efficientnet_b2")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    return parser.parse_args([])

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CamXuc_dataset(root=args.data_path, is_train=True, transforms=transform)
    val_dataset = CamXuc_dataset(root=args.data_path, is_train=False, transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)

    start_epoch = args.start_epoch
    if args.resume and os.path.isfile(args.resume):
        print(f"🔁 Resume from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    if os.path.exists(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    writer = SummaryWriter(args.log_path)

    best_accuracy = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        train_acc = train_correct / total
        writer.add_scalar("Train/Accuracy", train_acc, epoch)

        # === VALIDATION ===
        model.eval()
        val_loss = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_loss.append(loss.item())

        val_acc = accuracy_score(val_labels, val_preds)
        val_loss_mean = np.mean(val_loss)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        writer.add_scalar("Val/Loss", val_loss_mean, epoch)

        scheduler.step(val_loss_mean)

        print(f"[Epoch {epoch}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss_mean:.4f}")

        cm = confusion_matrix(val_labels, val_preds)
        plot_confusion_matrix(writer, cm, train_dataset.categories, epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(checkpoint, os.path.join(args.checkpoint_path, "best.pt"))

if __name__ == "__main__":
    args = get_args()
    train(args)


# Epoch 0/100: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:15<00:00,  4.18it/s, loss=1.21]
# [Epoch 0] Train Acc: 0.5133 | Val Acc: 0.6487 | Val Loss: 1.1896
# Epoch 1/100: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:22<00:00,  4.07it/s, loss=0.954]
# [Epoch 1] Train Acc: 0.6919 | Val Acc: 0.6933 | Val Loss: 1.0968
# Epoch 2/100: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:24<00:00,  4.04it/s, loss=1.37]
# [Epoch 2] Train Acc: 0.7865 | Val Acc: 0.7027 | Val Loss: 1.1176
# Epoch 3/100: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:25<00:00,  4.02it/s, loss=1.27]
# [Epoch 3] Train Acc: 0.8557 | Val Acc: 0.7011 | Val Loss: 1.1433
# Epoch 4/100: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:26<00:00,  4.01it/s, loss=0.785]
# [Epoch 4] Train Acc: 0.9064 | Val Acc: 0.7063 | Val Loss: 1.1656
# Epoch 5/100: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:25<00:00,  4.03it/s, loss=0.887]
# [Epoch 5] Train Acc: 0.9359 | Val Acc: 0.7046 | Val Loss: 1.1749
# Epoch 6/100: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:25<00:00,  4.02it/s, loss=0.683] 
# [Epoch 6] Train Acc: 0.9535 | Val Acc: 0.7007 | Val Loss: 1.2169
# Epoch 7/100: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:25<00:00,  4.03it/s, loss=0.617] 
# [Epoch 7] Train Acc: 0.9609 | Val Acc: 0.7048 | Val Loss: 1.2035
# Epoch 8/100: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:25<00:00,  4.03it/s, loss=0.664] 
# [Epoch 8] Train Acc: 0.9802 | Val Acc: 0.7138 | Val Loss: 1.1874
# Epoch 9/100: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:25<00:00,  4.03it/s, loss=0.484] 
# [Epoch 9] Train Acc: 0.9847 | Val Acc: 0.7051 | Val Loss: 1.2409
# Epoch 10/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:28<00:00,  3.99it/s, loss=0.482] 
# [Epoch 10] Train Acc: 0.9888 | Val Acc: 0.7175 | Val Loss: 1.2083
# Epoch 11/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:29<00:00,  3.97it/s, loss=0.487] 
# [Epoch 11] Train Acc: 0.9894 | Val Acc: 0.7116 | Val Loss: 1.2465
# Epoch 12/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:26<00:00,  4.01it/s, loss=0.525] 
# [Epoch 12] Train Acc: 0.9911 | Val Acc: 0.7077 | Val Loss: 1.2497
# [Epoch 10] Train Acc: 0.9888 | Val Acc: 0.7175 | Val Loss: 1.2083
# Epoch 11/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:29<00:00,  3.97it/s, loss=0.487] 
# [Epoch 11] Train Acc: 0.9894 | Val Acc: 0.7116 | Val Loss: 1.2465
# Epoch 12/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:26<00:00,  4.01it/s, loss=0.525] 
# [Epoch 12] Train Acc: 0.9911 | Val Acc: 0.7077 | Val Loss: 1.2497
# Epoch 11/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:29<00:00,  3.97it/s, loss=0.487] 
# [Epoch 11] Train Acc: 0.9894 | Val Acc: 0.7116 | Val Loss: 1.2465
# Epoch 12/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:26<00:00,  4.01it/s, loss=0.525] 
# [Epoch 12] Train Acc: 0.9911 | Val Acc: 0.7077 | Val Loss: 1.2497
# [Epoch 11] Train Acc: 0.9894 | Val Acc: 0.7116 | Val Loss: 1.2465
# Epoch 12/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:26<00:00,  4.01it/s, loss=0.525] 
# [Epoch 12] Train Acc: 0.9911 | Val Acc: 0.7077 | Val Loss: 1.2497
# Epoch 12/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:26<00:00,  4.01it/s, loss=0.525] 
# [Epoch 12] Train Acc: 0.9911 | Val Acc: 0.7077 | Val Loss: 1.2497
# [Epoch 12] Train Acc: 0.9911 | Val Acc: 0.7077 | Val Loss: 1.2497
# Epoch 13/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:27<00:00,  4.00it/s, loss=0.496] 
# [Epoch 13] Train Acc: 0.9903 | Val Acc: 0.7151 | Val Loss: 1.2490
# Epoch 14/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:27<00:00,  4.00it/s, loss=0.483] 
# [Epoch 14] Train Acc: 0.9926 | Val Acc: 0.7168 | Val Loss: 1.2312
# Epoch 15/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:27<00:00,  4.00it/s, loss=0.514] 
# [Epoch 15] Train Acc: 0.9954 | Val Acc: 0.7153 | Val Loss: 1.2347
# Epoch 16/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:28<00:00,  3.98it/s, loss=0.492] 
# [Epoch 16] Train Acc: 0.9950 | Val Acc: 0.7137 | Val Loss: 1.2279
# Epoch 17/100: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:29<00:00,  3.97it/s, loss=0.48] 
# [Epoch 17] Train Acc: 0.9961 | Val Acc: 0.7175 | Val Loss: 1.2265
# [Epoch 17] Train Acc: 0.9961 | Val Acc: 0.7175 | Val Loss: 1.2265
# Epoch 18/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:32<00:00,  3.92it/s, loss=0.472] 
# [Epoch 18] Train Acc: 0.9965 | Val Acc: 0.7164 | Val Loss: 1.2240
# Epoch 19/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:36<00:00,  3.86it/s, loss=0.472] 
# [Epoch 19] Train Acc: 0.9963 | Val Acc: 0.7166 | Val Loss: 1.2332
# Epoch 20/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:47<00:00,  3.71it/s, loss=0.474] 
# [Epoch 20] Train Acc: 0.9979 | Val Acc: 0.7190 | Val Loss: 1.2277
# Epoch 21/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:45<00:00,  3.74it/s, loss=0.537] 
# [Epoch 21] Train Acc: 0.9972 | Val Acc: 0.7209 | Val Loss: 1.2372
# Epoch 22/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:48<00:00,  3.70it/s, loss=0.473] 
# [Epoch 22] Train Acc: 0.9974 | Val Acc: 0.7238 | Val Loss: 1.2263
# Epoch 23/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:43<00:00,  3.77it/s, loss=0.472] 
# [Epoch 23] Train Acc: 0.9981 | Val Acc: 0.7238 | Val Loss: 1.2244
# Epoch 24/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:26<00:00,  4.01it/s, loss=0.508] 
# [Epoch 24] Train Acc: 0.9971 | Val Acc: 0.7199 | Val Loss: 1.2222
# Epoch 25/100: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:25<00:00,  4.03it/s, loss=0.47]
# [Epoch 25] Train Acc: 0.9980 | Val Acc: 0.7222 | Val Loss: 1.2298
# Epoch 26/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:26<00:00,  4.01it/s, loss=0.479]
# [Epoch 26] Train Acc: 0.9978 | Val Acc: 0.7203 | Val Loss: 1.2320
# Epoch 27/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:27<00:00,  4.00it/s, loss=0.471]
# [Epoch 27] Train Acc: 0.9983 | Val Acc: 0.7199 | Val Loss: 1.2322
# Epoch 28/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:27<00:00,  4.00it/s, loss=0.477]
# [Epoch 28] Train Acc: 0.9983 | Val Acc: 0.7211 | Val Loss: 1.2276
# Epoch 29/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:25<00:00,  4.03it/s, loss=0.476]
# [Epoch 29] Train Acc: 0.9984 | Val Acc: 0.7209 | Val Loss: 1.2265
# Epoch 30/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:29<00:00,  3.97it/s, loss=0.492]
# [Epoch 30] Train Acc: 0.9985 | Val Acc: 0.7190 | Val Loss: 1.2297
# Epoch 31/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:30<00:00,  3.95it/s, loss=0.497]
# [Epoch 31] Train Acc: 0.9982 | Val Acc: 0.7207 | Val Loss: 1.2234
# Epoch 32/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:26<00:00,  4.02it/s, loss=0.481]
# [Epoch 32] Train Acc: 0.9988 | Val Acc: 0.7192 | Val Loss: 1.2265
# Epoch 33/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:32<00:00,  3.92it/s, loss=0.472]
# [Epoch 33] Train Acc: 0.9992 | Val Acc: 0.7194 | Val Loss: 1.2251
# Epoch 34/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:30<00:00,  3.96it/s, loss=0.476]
# [Epoch 34] Train Acc: 0.9984 | Val Acc: 0.7212 | Val Loss: 1.2264
# Epoch 35/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:30<00:00,  3.96it/s, loss=0.473]
# [Epoch 35] Train Acc: 0.9986 | Val Acc: 0.7257 | Val Loss: 1.2231
# Epoch 36/100: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:32<00:00,  3.92it/s, loss=0.47]
# [Epoch 36] Train Acc: 0.9986 | Val Acc: 0.7211 | Val Loss: 1.2140
# Epoch 37/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:29<00:00,  3.97it/s, loss=0.472]
# [Epoch 37] Train Acc: 0.9984 | Val Acc: 0.7248 | Val Loss: 1.2196
# Epoch 38/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:29<00:00,  3.97it/s, loss=0.498]
# [Epoch 38] Train Acc: 0.9985 | Val Acc: 0.7238 | Val Loss: 1.2167
# Epoch 39/100: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:27<00:00,  3.99it/s, loss=0.47]
# [Epoch 39] Train Acc: 0.9992 | Val Acc: 0.7166 | Val Loss: 1.2232
# Epoch 40/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:29<00:00,  3.96it/s, loss=0.472]
# [Epoch 40] Train Acc: 0.9985 | Val Acc: 0.7196 | Val Loss: 1.2210
# Epoch 41/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:34<00:00,  3.90it/s, loss=0.472]
# [Epoch 41] Train Acc: 0.9989 | Val Acc: 0.7183 | Val Loss: 1.2226
# Epoch 42/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:33<00:00,  3.91it/s, loss=0.474]
# [Epoch 42] Train Acc: 0.9992 | Val Acc: 0.7211 | Val Loss: 1.2207
# Epoch 43/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:31<00:00,  3.94it/s, loss=0.469]
# [Epoch 43] Train Acc: 0.9988 | Val Acc: 0.7183 | Val Loss: 1.2225
# Epoch 44/100: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1069/1069 [04:34<00:00,  3.90it/s, loss=0.471]
# [Epoch 44] Train Acc: 0.9990 | Val Acc: 0.7222 | Val Loss: 1.2235