import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("UNIFIED ACGAN + DANN ADVERSARIAL TRAINING")
print("=" * 60)


SEQUENCE_LENGTH = 60
BATCH_SIZE      = 32
NUM_CLASSES     = 12
NUM_SUBJECTS    = 10
LATENT_DIM      = 128        
FEATURE_DIM     = 256        
EPOCHS_PHASE1   = 15         
EPOCHS_PHASE2   = 10         
EPOCHS_PHASE3   = 30         
GAN_LAMBDA      = 0.3        
DOMAIN_LAMBDA   = 0.5        
FINE_CLASSES    = [0, 3, 7, 10]   
GAN_REAL_FAKE_RATIO = 3      

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

h5_files = []
for root, dirs, files in os.walk('/kaggle/input'):
    for file in files:
        if file.endswith('.h5'):
            h5_files.append(os.path.join(root, file))
print(f"Found {len(h5_files)} H5 files")

subject_files = {s: [] for s in [2, 3, 5, 6, 8, 9, 10, 11, 12, 13]}
for fp in h5_files:
    fname = os.path.basename(fp)
    parts = fname.replace('.h5', '').split('_')
    if len(parts) == 3:
        sid = int(parts[1])
        if sid in subject_files:
            subject_files[sid].append(fp)

# Two-fold cross-validation splits
FOLDS = [
    {'train': [2, 3, 5, 6, 8],  'test': [9, 10, 11, 12, 13]},
    {'train': [9, 10, 11, 12, 13], 'test': [2, 3, 5, 6, 8]},
]

class SoliDataset(Dataset):
    def __init__(self, file_list, sequence_length=SEQUENCE_LENGTH):
        self.file_list = file_list
        self.sequence_length = sequence_length
        self.subject_to_idx = {2:0, 3:1, 5:2, 6:3, 8:4, 9:5, 10:6, 11:7, 12:8, 13:9}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_list[idx], 'r') as f:
            data  = f['ch0'][()]
            label = f['label'][()]

        fname      = os.path.basename(self.file_list[idx])
        subject_id = int(fname.replace('.h5', '').split('_')[1])
        subj_idx   = self.subject_to_idx.get(subject_id, 0)
        g_label    = int(label[0][0])

        nf   = data.shape[0]
        data = data.reshape(nf, 32, 32)[:, np.newaxis, :, :]

        if nf < self.sequence_length:
            pad  = np.zeros((self.sequence_length - nf, 1, 32, 32), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
        else:
            data = data[:self.sequence_length]

        if data.max() > 0:
            data = data / data.max()

        return (torch.FloatTensor(data),
                torch.LongTensor([g_label])[0],
                torch.LongTensor([subj_idx])[0])


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class SoliEncoder(nn.Module):
    """
    Processes a sequence of range-Doppler frames.
    Input : (B, T, 1, 32, 32)
    Output: (B, FEATURE_DIM)   where FEATURE_DIM = 256
    """
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                              # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                              # 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                      # 1x1
        )
        self.lstm = nn.LSTM(64, 128, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)                     # (B*T, 64, 1, 1)
        x = x.view(B, T, 64)               # (B, T, 64)
        x, _ = self.lstm(x)                 # (B, T, 256)
        feat = x[:, -1, :]                  # last timestep → (B, 256)
        return self.dropout(feat)


class GestureClassifier(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, feat):
        return self.fc(feat)

class DomainClassifier(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, num_subjects=NUM_SUBJECTS):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_subjects)
        )

    def forward(self, feat, alpha=1.0):
        reversed_feat = grad_reverse(feat, alpha)
        return self.fc(reversed_feat)


class ACGANGenerator(nn.Module):
    """
    Generates a fake flattened feature vector that mimics
    the encoder output for a given class.
    We generate in FEATURE space (256-d) rather than image space
    to keep training stable with sparse radar data.
    This is a 'feature-space GAN' — the discriminator also
    operates on features, not raw pixels.
    """
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES,
                 out_dim=FEATURE_DIM):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, out_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)          # (B, num_classes)
        x = torch.cat([z, label_input], dim=1)        # (B, latent+num_classes)
        return self.net(x)                             # (B, FEATURE_DIM)


class ACGANDiscriminator(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )
        self.real_fake = nn.Linear(128, 1)          # real/fake
        self.aux_class = nn.Linear(128, num_classes) # class prediction

    def forward(self, feat):
        h = self.shared(feat)
        return self.real_fake(h), self.aux_class(h)

def get_class_weights():
    """Higher weights for fine-grained (hard) classes."""
    w = torch.ones(NUM_CLASSES)
    for c in FINE_CLASSES:
        w[c] = 2.5
    w[11] = 0.5   # background — easy, downweight
    return w.to(device)

def evaluate(encoder, classifier, loader):
    encoder.eval(); classifier.eval()
    correct = total = 0
    correct_by_g = torch.zeros(NUM_CLASSES)
    total_by_g   = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for data, g_labels, _ in loader:
            data, g_labels = data.to(device), g_labels.to(device)
            feat  = encoder(data)
            logits = classifier(feat)
            preds  = logits.argmax(1)
            correct += preds.eq(g_labels).sum().item()
            total   += g_labels.size(0)
            for i in range(len(g_labels)):
                tl = g_labels[i].item()
                correct_by_g[tl] += int(preds[i].item() == tl)
                total_by_g[tl]   += 1
    acc = 100. * correct / total
    return acc, correct_by_g, total_by_g


def train_fold(fold_idx, train_subjects, test_subjects):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx+1}  |  Train subjects: {train_subjects}")
    print(f"{'='*60}")

    # Build file lists
    train_files = []
    test_files  = []
    for s in train_subjects: train_files.extend(subject_files[s])
    for s in test_subjects:  test_files.extend(subject_files[s])
    print(f"Train files: {len(train_files)} | Test files: {len(test_files)}")

    train_ds = SoliDataset(train_files)
    test_ds  = SoliDataset(test_files)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Models ──────────────────────────
    encoder    = SoliEncoder().to(device)
    classifier = GestureClassifier().to(device)
    domain_cls = DomainClassifier().to(device)
    generator  = ACGANGenerator().to(device)
    discriminator = ACGANDiscriminator().to(device)

    # ── Losses ──────────────────────────
    class_weights   = get_class_weights()
    ce_loss         = nn.CrossEntropyLoss(weight=class_weights)
    ce_loss_unw     = nn.CrossEntropyLoss()   # unweighted for domain / GAN aux
    bce_loss        = nn.BCEWithLogitsLoss()

    # ── Optimizers ──────────────────────
    opt_enc  = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3)
    opt_dom  = torch.optim.Adam(domain_cls.parameters(), lr=1e-3)
    opt_gen  = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    sch_enc  = torch.optim.lr_scheduler.StepLR(opt_enc,  step_size=20, gamma=0.5)
    sch_dom  = torch.optim.lr_scheduler.StepLR(opt_dom,  step_size=20, gamma=0.5)

    best_acc   = 0.0
    best_epoch = 0
    total_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2 + EPOCHS_PHASE3

    for epoch in range(total_epochs):
        phase = (1 if epoch < EPOCHS_PHASE1
                 else 2 if epoch < EPOCHS_PHASE1 + EPOCHS_PHASE2
                 else 3)

        # GRL alpha — ramp up slowly in phase 2/3
        p     = max(0, epoch - EPOCHS_PHASE1) / (EPOCHS_PHASE2 + EPOCHS_PHASE3)
        alpha = float(2. / (1. + np.exp(-10. * p)) - 1)

        encoder.train(); classifier.train()
        domain_cls.train(); generator.train(); discriminator.train()

        ep_loss_cls = ep_loss_dom = ep_loss_g = ep_loss_d = 0.0
        correct = total = 0

        for batch_idx, (data, g_labels, d_labels) in enumerate(train_loader):
            data, g_labels, d_labels = (data.to(device),
                                         g_labels.to(device),
                                         d_labels.to(device))
            B = data.size(0)

            # ── (A) Get real features from encoder ──────────────
            features = encoder(data)

            # ════════════════════════════════════════════════════
            # PHASE 1 & 2 & 3 : Gesture classification loss
            # ════════════════════════════════════════════════════
            logits = classifier(features)
            loss_cls = ce_loss(logits, g_labels)

            # ════════════════════════════════════════════════════
            # PHASE 2 & 3 : Domain adversarial loss (DANN)
            # ════════════════════════════════════════════════════
            loss_dom = torch.tensor(0.0, device=device)
            if phase >= 2:
                dom_logits = domain_cls(features.detach(), alpha)
                loss_dom   = ce_loss_unw(dom_logits, d_labels)
                opt_dom.zero_grad()
                loss_dom.backward()
                opt_dom.step()

                # Also encoder sees reversed gradient from domain head
                dom_enc = domain_cls(features, alpha)
                loss_dom_enc = ce_loss_unw(dom_enc, d_labels)
            else:
                loss_dom_enc = torch.tensor(0.0, device=device)

            # ════════════════════════════════════════════════════
            # PHASE 3 : ACGAN
            # ════════════════════════════════════════════════════
            loss_g_total = torch.tensor(0.0, device=device)
            loss_d_total = torch.tensor(0.0, device=device)

            if phase == 3:
                # Only augment fine-grained classes
                fine_mask = torch.zeros(B, dtype=torch.bool, device=device)
                for fc in FINE_CLASSES:
                    fine_mask |= (g_labels == fc)

                # Number of fakes to generate (1 per fine-grained sample)
                n_fake = max(fine_mask.sum().item(), 1)
                fine_labels_sample = g_labels[fine_mask]
                if fine_labels_sample.numel() == 0:
                    fine_labels_sample = torch.randint(0, len(FINE_CLASSES),
                                                       (4,), device=device)
                    fine_labels_sample = torch.tensor(
                        [FINE_CLASSES[i] for i in fine_labels_sample], device=device)

                z = torch.randn(n_fake, LATENT_DIM, device=device)
                fake_feats = generator(z, fine_labels_sample[:n_fake])

                # ── Train Discriminator ──────────────────────────
                # Real features from fine-grained samples only
                real_feats_fine = features[fine_mask].detach()
                if real_feats_fine.size(0) == 0:
                    real_feats_fine = features[:1].detach()

                real_labels_fine = g_labels[fine_mask]
                if real_labels_fine.size(0) == 0:
                    real_labels_fine = g_labels[:1]

                # Discriminator on real
                d_real_rf, d_real_cls = discriminator(real_feats_fine)
                d_fake_rf, d_fake_cls = discriminator(fake_feats.detach())

                real_tgt = torch.ones_like(d_real_rf)
                fake_tgt = torch.zeros_like(d_fake_rf)

                loss_d_rf  = bce_loss(d_real_rf, real_tgt) + \
                             bce_loss(d_fake_rf, fake_tgt)
                loss_d_cls = ce_loss_unw(d_real_cls,
                                         real_labels_fine[:d_real_cls.size(0)])
                loss_d_total = loss_d_rf + loss_d_cls

                opt_disc.zero_grad()
                loss_d_total.backward()
                opt_disc.step()

                # ── Train Generator ──────────────────────────────
                fake_feats2     = generator(z, fine_labels_sample[:n_fake])
                d_fake_rf2, d_fake_cls2 = discriminator(fake_feats2)
                loss_g_rf  = bce_loss(d_fake_rf2, torch.ones_like(d_fake_rf2))
                loss_g_cls = ce_loss_unw(d_fake_cls2,
                                          fine_labels_sample[:n_fake])
                loss_g_total = loss_g_rf + loss_g_cls

                # ── Augment batch with fake features for classifier ─
                # Detach generator output so no gradient flows back through G
                aug_feats  = torch.cat([features, fake_feats2.detach()], dim=0)
                aug_labels = torch.cat([g_labels,
                                         fine_labels_sample[:n_fake]], dim=0)
                logits_aug = classifier(aug_feats)
                loss_cls   = ce_loss(logits_aug, aug_labels)

            # ── Total encoder + classifier loss ─────────────────
            total_loss = (loss_cls
                          + DOMAIN_LAMBDA * loss_dom_enc
                          + GAN_LAMBDA    * loss_g_total)

            opt_enc.zero_grad()
            if phase == 3:
                opt_gen.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()), 1.0)
            opt_enc.step()
            if phase == 3:
                opt_gen.step()

            # Metrics
            ep_loss_cls += loss_cls.item()
            ep_loss_dom += loss_dom.item()
            ep_loss_g   += loss_g_total.item()
            ep_loss_d   += loss_d_total.item()

            with torch.no_grad():
                preds    = logits.argmax(1)
                correct += preds.eq(g_labels).sum().item()
                total   += B

        sch_enc.step(); sch_dom.step()

        train_acc = 100. * correct / total
        test_acc, _, _ = evaluate(encoder, classifier, test_loader)

        if test_acc > best_acc:
            best_acc   = test_acc
            best_epoch = epoch + 1
            torch.save({
                'encoder':    encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'generator':  generator.state_dict(),
            }, f'best_model_fold{fold_idx+1}.pth')

        n_batches = len(train_loader)
        print(f"Ep {epoch+1:3d}/{total_epochs} [Ph{phase}] "
              f"α={alpha:.2f} | "
              f"Cls: {ep_loss_cls/n_batches:.3f} "
              f"Dom: {ep_loss_dom/n_batches:.3f} "
              f"G: {ep_loss_g/n_batches:.3f} "
              f"D: {ep_loss_d/n_batches:.3f} | "
              f"Train: {train_acc:.2f}% Test: {test_acc:.2f}% "
              f"Best: {best_acc:.2f}%")

    # ── Per-gesture breakdown ────────────────────────────────
    ckpt = torch.load(f'best_model_fold{fold_idx+1}.pth')
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])

    final_acc, corr_g, tot_g = evaluate(encoder, classifier, test_loader)

    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx+1} — BEST TEST ACCURACY: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"{'='*60}")
    print("PER-GESTURE ACCURACY:")
    for g in range(NUM_CLASSES):
        if tot_g[g] > 0:
            acc = corr_g[g] / tot_g[g] * 100
            tag = "🔴 FINE-GRAINED" if g in FINE_CLASSES else "          "
            print(f"  Gesture {g:2d} {tag}: {acc:.2f}%")

    fine_corr = sum(corr_g[g] for g in FINE_CLASSES)
    fine_tot  = sum(tot_g[g]  for g in FINE_CLASSES)
    fine_acc  = fine_corr / fine_tot * 100 if fine_tot > 0 else 0
    print(f"\n  🎯 Fine-grained avg : {fine_acc:.2f}%")
    print(f"  📊 Overall accuracy : {final_acc:.2f}%")

    return best_acc, fine_acc


fold_results = []
for fi, fold in enumerate(FOLDS):
    acc, fg_acc = train_fold(fi, fold['train'], fold['test'])
    fold_results.append({'overall': acc, 'fine_grained': fg_acc})

print("\n" + "=" * 60)
print("TWO-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 60)
for i, r in enumerate(fold_results):
    print(f"  Fold {i+1}: Overall = {r['overall']:.2f}%  |  "
          f"Fine-grained = {r['fine_grained']:.2f}%")
mean_overall = np.mean([r['overall']     for r in fold_results])
mean_fine    = np.mean([r['fine_grained'] for r in fold_results])
print(f"\n  Mean Overall      : {mean_overall:.2f}%")
print(f"  Mean Fine-grained : {mean_fine:.2f}%")
print("=" * 60)