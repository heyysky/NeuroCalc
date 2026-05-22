"""
NeuroCalc CNN Training Script
Trains lightweight CNN on EMNIST digits + custom math operators.
Exports to ONNX for CPU inference.

Usage:
    python ml/train_cnn.py --epochs 20 --export

Requirements:
    pip install torch torchvision onnx
"""

import argparse
import os
import sys
import numpy as np

# ── Model definition using PyTorch ────────────────────────────────────────────

PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
    import torchvision
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.recognition.cnn_model import NUM_CLASSES, SYMBOL_CLASSES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── CNN Architecture ───────────────────────────────────────────────────────────

def build_model():
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for training. pip install torch torchvision")

    class NeuroCalcCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                   # 14x14

                # Block 2
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                   # 7x7

                # Block 3
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),           # 4x4
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, NUM_CLASSES),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return NeuroCalcCNN()


# ── Dataset ────────────────────────────────────────────────────────────────────

def get_emnist_loaders(batch_size=64, data_dir="./data"):
    """Download and prepare EMNIST digits dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = torchvision.datasets.EMNIST(
        root=data_dir, split="digits", train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.EMNIST(
        root=data_dir, split="digits", train=False, download=True, transform=transform
    )

    # EMNIST digits: classes 0-9 (first 10 of our NUM_CLASSES)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    return train_loader, test_loader


# ── Training ───────────────────────────────────────────────────────────────────

def train(epochs=20, batch_size=64, lr=0.001, export=True):
    if not PYTORCH_AVAILABLE:
        print("ERROR: PyTorch not installed. Run: pip install torch torchvision")
        return

    print("=" * 50)
    print("NeuroCalc CNN Training")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Classes: {NUM_CLASSES}")
    print("=" * 50)

    device = torch.device("cpu")  # CPU-only
    model = build_model().to(device)

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} (~{n_params*4/1024/1024:.1f}MB)")

    train_loader, test_loader = get_emnist_loaders(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # Clamp targets to NUM_CLASSES (EMNIST digits = 0-9)
            target = target.clamp(0, NUM_CLASSES - 1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch}/{epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                target = target.clamp(0, NUM_CLASSES - 1)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        acc = 100.0 * correct / total
        print(f"  → Epoch {epoch}: Acc={acc:.2f}%")
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "neurocalc_cnn_best.pth"))
            print(f"  ✓ Saved best model (acc={acc:.2f}%)")

    print(f"\n✓ Training complete. Best accuracy: {best_acc:.2f}%")

    if export:
        export_onnx(model)


# ── ONNX Export ────────────────────────────────────────────────────────────────

def export_onnx(model=None):
    """Export trained model to ONNX with optional quantization."""
    if not PYTORCH_AVAILABLE:
        print("ERROR: PyTorch required for ONNX export.")
        return

    try:
        import onnx
        from torch.quantization import quantize_dynamic
    except ImportError:
        print("ERROR: Install onnx: pip install onnx")
        return

    if model is None:
        model = build_model()
        weights_path = os.path.join(MODELS_DIR, "neurocalc_cnn_best.pth")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print(f"✓ Loaded weights from {weights_path}")
        else:
            print("WARNING: No trained weights found, exporting random weights")

    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    onnx_path = os.path.join(MODELS_DIR, "neurocalc_cnn.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Verify
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"✓ ONNX model exported: {onnx_path} ({size_mb:.1f}MB)")

    # Quantize for further size reduction
    try:
        quantized_path = os.path.join(MODELS_DIR, "neurocalc_cnn_quantized.onnx")
        from onnxruntime.quantization import quantize_dynamic as ort_quantize, QuantType
        ort_quantize(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
        q_size = os.path.getsize(quantized_path) / 1024 / 1024
        print(f"✓ Quantized model: {quantized_path} ({q_size:.1f}MB)")
    except Exception as e:
        print(f"  Quantization skipped: {e}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuroCalc CNN")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--export", action="store_true", default=True)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    if args.export_only:
        export_onnx()
    else:
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, export=args.export)
