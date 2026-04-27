import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

SEQUENCE_LENGTH = 60
BATCH_SIZE      = 32
NUM_CLASSES     = 12
FINE_CLASSES    = [0, 3, 7, 10]
EPOCHS          = 40

h5_files = []
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f.endswith('.h5'):
            h5_files.append(os.path.join(root, f))
print(f"Found {len(h5_files)} H5 files")

subject_files = {s: [] for s in [2, 3, 5, 6, 8, 9, 10, 11, 12, 13]}
for fp in h5_files:
    fname = os.path.basename(fp)
    parts = fname.replace('.h5', '').split('_')
    if len(parts) == 3:
        sid = int(parts[1])
        if sid in subject_files:
            subject_files[sid].append(fp)

FOLDS = [
    {'train': [2, 3, 5, 6, 8],     'test': [9, 10, 11, 12, 13]},
    {'train': [9, 10, 11, 12, 13], 'test': [2, 3, 5, 6, 8]},
]


class SoliDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_list[idx], 'r') as f:
            data  = f['ch0'][()]
            label = f['label'][()]

        g_label = int(label[0][0])
        nf      = data.shape[0]
        data    = data.reshape(nf, 32, 32).astype(np.float32)[:, np.newaxis, :, :]

        if nf < SEQUENCE_LENGTH:
            pad  = np.zeros((SEQUENCE_LENGTH - nf, 1, 32, 32), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
        else:
            data = data[:SEQUENCE_LENGTH]

        if data.max() > 0:
            data = data / data.max()

        return torch.FloatTensor(data), torch.LongTensor([g_label])[0]

class BaselineCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.lstm    = nn.LSTM(64, 128, batch_first=True,
                               bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x).view(B, T, 64)
        x, _ = self.lstm(x)
        return self.fc(self.dropout(x[:, -1, :]))


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    corr_g  = torch.zeros(NUM_CLASSES)
    tot_g   = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            preds = model(data).argmax(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
            for i in range(len(labels)):
                tl = labels[i].item()
                corr_g[tl] += int(preds[i].item() == tl)
                tot_g[tl]  += 1
    model.train()
    return 100. * correct / total, corr_g, tot_g

def get_class_weights():
    w = torch.ones(NUM_CLASSES)
    for c in FINE_CLASSES: w[c] = 2.5
    w[11] = 0.5
    return w.to(device)

results = []

for fi, fold in enumerate(FOLDS):
    print(f"\n{'='*60}")
    print(f"FOLD {fi+1}  |  Train: {fold['train']}")
    print(f"{'='*60}")

    train_files, test_files = [], []
    for s in fold['train']: train_files.extend(subject_files[s])
    for s in fold['test']:  test_files.extend(subject_files[s])
    print(f"Train: {len(train_files)}  |  Test: {len(test_files)}")

    train_loader = DataLoader(SoliDataset(train_files),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(SoliDataset(test_files),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = BaselineCNNLSTM().to(device)
    ce_loss   = nn.CrossEntropyLoss(weight=get_class_weights())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_acc   = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        correct = total = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = ce_loss(model(data), labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                preds    = model(data).argmax(1)
                correct += preds.eq(labels).sum().item()
                total   += labels.size(0)

        scheduler.step()
        train_acc = 100. * correct / total
        test_acc, _, _ = evaluate(model, test_loader)

        if test_acc > best_acc:
            best_acc   = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'baseline_fold{fi+1}.pth')

        print(f"Ep {epoch+1:2d}/{EPOCHS} | "
              f"Train: {train_acc:.2f}%  Test: {test_acc:.2f}%  Best: {best_acc:.2f}%")

    # Load best and get per-gesture breakdown
    model.load_state_dict(torch.load(f'baseline_fold{fi+1}.pth'))
    final_acc, corr_g, tot_g = evaluate(model, test_loader)

    print(f"\n{'='*60}")
    print(f"FOLD {fi+1} BASELINE BEST: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"{'='*60}")
    print("PER-GESTURE ACCURACY:")
    for g in range(NUM_CLASSES):
        if tot_g[g] > 0:
            acc = corr_g[g] / tot_g[g] * 100
            tag = "🔴 FINE-GRAINED" if g in FINE_CLASSES else "          "
            print(f"  Gesture {g:2d} {tag}: {acc:.2f}%")

    fine_corr = sum(corr_g[g] for g in FINE_CLASSES)
    fine_tot  = sum(tot_g[g]  for g in FINE_CLASSES)
    fine_acc  = fine_corr / max(fine_tot, 1) * 100
    print(f"\n  🎯 Fine-grained avg : {fine_acc:.2f}%")
    print(f"  📊 Overall accuracy : {final_acc:.2f}%")
    results.append({'overall': best_acc, 'fine': fine_acc})

print("\n" + "=" * 60)
print("BASELINE SUMMARY")
print("=" * 60)
for i, r in enumerate(results):
    print(f"  Fold {i+1}: Overall={r['overall']:.2f}%  Fine-grained={r['fine']:.2f}%")
print(f"\n  Mean Overall      : {np.mean([r['overall'] for r in results]):.2f}%")
print(f"  Mean Fine-grained : {np.mean([r['fine']     for r in results]):.2f}%")
print("=" * 60)