"""
Meteorite Identification - LoRA Pipeline v2
基于队友 train_dinov2_lora.py，修复 v1 过拟合问题

核心改进：
1. 深度分类头：MLP(512→256→1) + BatchNorm + Dropout
2. Label Smooth Loss：smoothing=0.05
3. Multi-seed Ensemble：3个seed训练取平均
4. 保守参数：lr=1e-4, weight_decay=1e-3, epochs=15, patience=3
"""

import argparse
import copy
import os
import random
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset import StoneDataset
from submission_utils import save_topk_submission, split_output_name


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_number_list(value, cast_type=float):
    if value is None or value == "":
        return []
    return [cast_type(item.strip()) for item in str(value).split(",") if item.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_resize_resample():
    try:
        return transforms.InterpolationMode.BICUBIC
    except AttributeError:
        return Image.BICUBIC


class PathImageDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = [Path(path) for path in paths]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is None:
            return image, index
        return image, int(self.labels[index])


def build_train_transform(image_size):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.72, 1.0),
                ratio=(0.90, 1.10),
                interpolation=pil_resize_resample(),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=12, fill=255),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(image_size, hflip=False, vflip=False, rotation=0):
    ops = [
        transforms.Resize(image_size, interpolation=pil_resize_resample()),
        transforms.CenterCrop(image_size),
    ]
    if hflip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    if vflip:
        ops.append(transforms.RandomVerticalFlip(p=1.0))
    if rotation:
        ops.append(transforms.RandomRotation((rotation, rotation), fill=255))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def tta_transforms(image_size, mode):
    transforms_for_tta = [build_eval_transform(image_size)]
    if mode in {"flip", "full"}:
        transforms_for_tta.extend(
            [
                build_eval_transform(image_size, hflip=True),
                build_eval_transform(image_size, vflip=True),
            ]
        )
    if mode == "full":
        transforms_for_tta.extend(
            [
                build_eval_transform(image_size, rotation=8),
                build_eval_transform(image_size, rotation=-8),
            ]
        )
    return transforms_for_tta


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))

    def forward(self, x):
        result = self.base(x)
        update = torch.nn.functional.linear(self.dropout(x), self.lora_a)
        update = torch.nn.functional.linear(update, self.lora_b)
        return result + self.scaling * update


def replace_module(root, module_name, new_module):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora(model, target_names, rank, alpha, dropout, last_n_blocks):
    for parameter in model.parameters():
        parameter.requires_grad = False

    target_names = tuple(item.strip() for item in target_names.split(",") if item.strip())
    block_indices = []
    for name, _ in model.named_modules():
        match = re.search(r"blocks\.(\d+)\.", name)
        if match:
            block_indices.append(int(match.group(1)))
    first_trainable_block = 0
    if block_indices and last_n_blocks > 0:
        first_trainable_block = max(block_indices) + 1 - int(last_n_blocks)

    injected = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if target_names and not name.endswith(target_names):
            continue
        match = re.search(r"blocks\.(\d+)\.", name)
        if match and int(match.group(1)) < first_trainable_block:
            continue
        if not match and last_n_blocks > 0:
            continue
        replace_module(model, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        injected.append(name)

    if not injected:
        raise RuntimeError(
            "No Linear modules received LoRA adapters. "
            "Try --lora-targets qkv,proj,fc1,fc2 or --lora-last-n-blocks 0."
        )
    return injected


class MLPHead(nn.Module):
    """深度分类头：MLP + BatchNorm + Dropout"""
    def __init__(self, feature_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(1)


class DinoV2LoRAClassifier(nn.Module):
    def __init__(self, backbone, pool="cls_mean", dropout=0.2, head_hidden_dim=256, head_dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.head = None
        self.head_hidden_dim = head_hidden_dim
        self.head_dropout = head_dropout

    def extract_features(self, images):
        if hasattr(self.backbone, "forward_features"):
            output = self.backbone.forward_features(images)
            if isinstance(output, dict):
                cls = output.get("x_norm_clstoken", None)
                patches = output.get("x_norm_patchtokens", None)
            else:
                cls = output
                patches = None
        else:
            output = self.backbone(images)
            cls = output[:, 0] if output.ndim == 3 else output
            patches = output[:, 1:] if output.ndim == 3 else None

        if self.pool == "cls":
            return cls
        if patches is None:
            return cls
        patch_mean = patches.mean(dim=1)
        if self.pool == "cls_mean":
            return torch.cat([cls, patch_mean], dim=1)
        if self.pool == "cls_mean_max":
            patch_max = patches.max(dim=1).values
            return torch.cat([cls, patch_mean, patch_max], dim=1)
        raise ValueError(f"Unknown pool: {self.pool}")

    def initialize_head(self, image_size):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size, device=DEVICE)
            feature_dim = self.extract_features(dummy).shape[1]
        self.head = MLPHead(feature_dim, self.head_hidden_dim, self.head_dropout).to(DEVICE)

    def forward(self, images):
        features = self.extract_features(images)
        return self.head(self.dropout(features))


def trainable_state_dict(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(model, state):
    named_parameters = dict(model.named_parameters())
    for name, value in state.items():
        named_parameters[name].data.copy_(value.to(named_parameters[name].device))


def find_best_threshold(labels, probs):
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.95, 0.005):
        f1 = f1_score(labels, (probs > threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold), float(best_f1)


def load_dinov2_model(args):
    model_name = args.model_name
    if model_name == "dinov2":
        model_name = "dinov2_vits14"
    try:
        backbone = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as exc:
        raise RuntimeError(
            "Could not load DINOv2 through torch.hub. "
            "Make sure the model is cached or the machine has internet for the first run."
        ) from exc
    backbone.to(DEVICE)
    injected = inject_lora(
        backbone,
        target_names=args.lora_targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        last_n_blocks=args.lora_last_n_blocks,
    )
    model = DinoV2LoRAClassifier(
        backbone, pool=args.pool, dropout=args.head_dropout,
        head_hidden_dim=args.head_hidden_dim, head_dropout=args.head_dropout
    ).to(DEVICE)
    model.initialize_head(args.image_size)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Injected LoRA modules: {len(injected)} | trainable_params={trainable:,}/{total:,}")
    return model


def make_loader(paths, labels, transform, batch_size, shuffle, num_workers):
    dataset = PathImageDataset(paths, labels=labels, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def predict_probs(model, paths, transform, batch_size, num_workers):
    loader = make_loader(paths, None, transform, batch_size, False, num_workers)
    probs = np.zeros(len(paths), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            probs[indices.numpy()] = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


class LabelSmoothingBCE(nn.Module):
    """BCEWithLogitsLoss with label smoothing for anti-overfitting."""
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        # Label smoothing: blend with 0.5
        smoothed = targets.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, smoothed)


def train_one_fold(args, train_paths, labels, train_idx, val_idx, fold_seed):
    set_seed(fold_seed)
    model = load_dinov2_model(args)
    train_loader = make_loader(
        [train_paths[index] for index in train_idx],
        labels[train_idx],
        build_train_transform(args.image_size),
        args.batch_size,
        True,
        args.num_workers,
    )
    val_paths = [train_paths[index] for index in val_idx]
    val_labels = labels[val_idx]

    positives = float(np.sum(labels[train_idx] == 1))
    negatives = float(np.sum(labels[train_idx] == 0))
    pos_weight = torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32, device=DEVICE)

    # v2: 使用 Label Smooth Loss
    criterion = LabelSmoothingBCE(smoothing=args.label_smoothing, pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available() and args.amp
        else nullcontext()
    )

    best_state = trainable_state_dict(model)
    best_f1 = -1.0
    best_threshold = 0.5
    stale_epochs = 0
    eval_transform = build_eval_transform(args.image_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, batch_labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.float().to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context:
                logits = model(images)
                loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu().item())
        scheduler.step()

        val_probs = predict_probs(model, val_paths, eval_transform, args.batch_size, args.num_workers)
        threshold, val_f1 = find_best_threshold(val_labels, val_probs)
        print(
            f"fold_seed={fold_seed} epoch={epoch}/{args.epochs} | "
            f"loss={running_loss / max(len(train_loader), 1):.4f} | "
            f"val_f1={val_f1:.4f} | threshold={threshold:.3f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_threshold = threshold
            best_state = trainable_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    load_trainable_state_dict(model, best_state)
    return model, best_f1, best_threshold


def load_data(root):
    train_dataset = StoneDataset(root, split="train", transforms=None)
    test_dataset = StoneDataset(root, split="test", transforms=None)
    missing = [image_id for image_id, path in zip(test_dataset.ids, test_dataset.samples) if path is None]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images, examples: {missing[:8]}")
    return (
        list(train_dataset.ids),
        np.asarray(train_dataset.labels, dtype=int),
        [Path(path) for path in train_dataset.samples],
        list(test_dataset.ids),
        [Path(path) for path in test_dataset.samples],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--sample-submission", default="", help="Defaults to <data-root>/sample_submission.csv.")
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--pool", default="cls_mean", choices=["cls", "cls_mean", "cls_mean_max"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)  # 保守：减少训练轮数
    parser.add_argument("--patience", type=int, default=3)  # 保守：更早停止
    parser.add_argument("--lr", type=float, default=1e-4)  # 保守：更低学习率
    parser.add_argument("--weight-decay", type=float, default=1e-3)  # 保守：更强正则
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)  # 同队友
    parser.add_argument("--lora-alpha", type=float, default=16.0)  # 同队友
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-last-n-blocks", type=int, default=4)  # 同队友
    parser.add_argument("--lora-targets", default="qkv,proj,fc1,fc2")
    parser.add_argument("--head-dropout", type=float, default=0.3)  # head dropout
    parser.add_argument("--head-hidden-dim", type=int, default=256)  # head 隐藏层维度
    parser.add_argument("--label-smoothing", type=float, default=0.05)  # v2 新增
    parser.add_argument("--tta", default="flip", choices=["none", "flip", "full"])
    parser.add_argument("--detail-output", default="dinov2_lora_v2_detail.csv")
    parser.add_argument("--fold-output", default="dinov2_lora_v2_folds.csv")
    parser.add_argument("--output", default="submission_dinov2_lora_v2.csv")
    parser.add_argument("--topk-list", default="85,90,95,100,105")
    args = parser.parse_args()

    train_ids, labels, train_paths, test_ids, test_paths = load_data(args.data_root)
    sample_submission = args.sample_submission or os.path.join(args.data_root, "sample_submission.csv")
    template = pd.read_csv(sample_submission)
    if template["id"].astype(str).tolist() != [str(item) for item in test_ids]:
        raise RuntimeError("Test ids do not match sample_submission.csv order.")

    indices = np.arange(len(labels))
    if args.folds <= 1:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_size,
            stratify=labels,
            random_state=args.seed,
        )
        splits = [(train_idx, val_idx)]
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(indices, labels))

    # v2: Multi-seed ensemble
    num_seeds = 3
    all_seed_oof_probs = []
    all_seed_test_probs = []

    for seed_idx in range(num_seeds):
        current_seed = args.seed + seed_idx
        set_seed(current_seed)
        print(f"\n{'='*60}")
        print(f"Training seed {seed_idx+1}/{num_seeds} (seed={current_seed})")
        print(f"{'='*60}")

        oof_probs = np.zeros(len(labels), dtype=np.float64)
        test_probs_by_fold = []
        fold_rows = []

        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            print(f"Training LoRA fold {fold}/{len(splits)}")
            model, val_f1, threshold = train_one_fold(
                args,
                train_paths,
                labels,
                np.asarray(train_idx),
                np.asarray(val_idx),
                fold_seed=current_seed + fold,
            )
            eval_transform = build_eval_transform(args.image_size)
            oof_probs[val_idx] = predict_probs(
                model,
                [train_paths[index] for index in val_idx],
                eval_transform,
                args.batch_size,
                args.num_workers,
            )

            fold_test_probs = []
            for transform in tta_transforms(args.image_size, args.tta):
                fold_test_probs.append(predict_probs(model, test_paths, transform, args.batch_size, args.num_workers))
            test_probs_by_fold.append(np.mean(fold_test_probs, axis=0))
            fold_rows.append(
                {
                    "fold": fold,
                    "val_f1": val_f1,
                    "threshold": threshold,
                    "val_positive_count": int((oof_probs[val_idx] > threshold).sum()),
                }
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        test_probs = np.mean(test_probs_by_fold, axis=0)
        all_seed_oof_probs.append(oof_probs)
        all_seed_test_probs.append(test_probs)

        oof_threshold, oof_f1 = find_best_threshold(labels, oof_probs)
        print(f"Seed {current_seed} OOF f1={oof_f1:.4f} | threshold={oof_threshold:.3f}")

    # Average across seeds
    final_oof_probs = np.mean(all_seed_oof_probs, axis=0)
    final_test_probs = np.mean(all_seed_test_probs, axis=0)

    oof_threshold, oof_f1 = find_best_threshold(labels, final_oof_probs)
    print(f"\n{'='*60}")
    print(f"Final ensemble ({num_seeds} seeds) OOF f1={oof_f1:.4f} | threshold={oof_threshold:.3f}")
    print(f"{'='*60}")

    detail = pd.DataFrame({"id": test_ids, "lora_prob": final_test_probs})
    detail["rank"] = detail["lora_prob"].rank(method="first", ascending=False).astype(int)
    detail.sort_values("rank").to_csv(args.detail_output, index=False)
    pd.DataFrame(fold_rows).to_csv(args.fold_output, index=False)
    print(f"Saved detail: {args.detail_output}")
    print(f"Saved fold summary: {args.fold_output}")

    stem, ext = split_output_name(args.output)
    for topk in parse_number_list(args.topk_list, int):
        save_topk_submission(template, test_ids, final_test_probs, topk, f"{stem}_top{topk}{ext}")


if __name__ == "__main__":
    main()