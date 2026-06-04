import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (
    DeQuantStub,
    MinMaxObserver,
    QConfig,
    QuantStub,
    convert,
    fuse_modules,
    prepare,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
HOMEWORK2_MODELS_PATH = PROJECT_ROOT / "Homework 2" / "models.py"
DATA_DIR = THIS_DIR / "dataset"
ARTIFACTS_DIR = THIS_DIR / "artifacts"
FIGURES_DIR = THIS_DIR / "Figures"


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


DEFAULT_CONFIG = {
    "batch_size": _env_int("HW4_BATCH_SIZE", 256),
    "train_epochs": _env_int("HW4_EPOCHS", 8),
    "learning_rate": float(os.environ.get("HW4_LR", "0.001")),
    "train_subset_size": _env_int("HW4_TRAIN_SUBSET_SIZE", None),
    "test_subset_size": _env_int("HW4_TEST_SUBSET_SIZE", None),
    "calibration_batches": _env_int("HW4_CALIBRATION_BATCHES", 16),
    "error_batches": _env_int("HW4_ERROR_BATCHES", 4),
    "latency_warmup": _env_int("HW4_LATENCY_WARMUP", 30),
    "latency_repeats": _env_int("HW4_LATENCY_REPEATS", 200),
    "seed": _env_int("HW4_SEED", 42),
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_homework2_models_module():
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("homework2_models", HOMEWORK2_MODELS_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load {HOMEWORK2_MODELS_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def set_reproducible(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_quantize(x: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, float, int]:
    """Per-tensor asymmetric linear quantization implemented by hand."""
    if not torch.is_floating_point(x):
        x = x.float()

    qmin = 0
    qmax = 2**num_bits - 1
    x_min = float(x.min().item())
    x_max = float(x.max().item())

    if x_max == x_min:
        scale = 1.0
    else:
        scale = (x_max - x_min) / float(qmax - qmin)

    zero_point = int(round(qmin - x_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))

    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    dtype = torch.uint8 if num_bits <= 8 else torch.int32
    return q.to(dtype), float(scale), int(zero_point)


def linear_dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """Inverse operation of per-tensor asymmetric linear quantization."""
    return (q.float() - float(zero_point)) * float(scale)


class QuantizableSimpleCNN(nn.Module):
    """Homework 2 SimpleCNN with explicit ReLU modules and quant/dequant stubs."""

    def __init__(self, homework2_model: Optional[nn.Module] = None):
        super().__init__()
        self.model_name = "SimpleCNN_INT8_PTQ"
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.dequant = DeQuantStub()

        if homework2_model is not None:
            self.load_from_homework2(homework2_model)

    def load_from_homework2(self, homework2_model: nn.Module) -> None:
        self.load_state_dict(homework2_model.state_dict(), strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x


def build_svhn_loaders(config: Optional[Dict] = None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    ensure_dirs()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = datasets.SVHN(root=str(DATA_DIR), split="train", download=True, transform=transform)
    test_set = datasets.SVHN(root=str(DATA_DIR), split="test", download=True, transform=transform)

    if config["train_subset_size"]:
        train_set = Subset(train_set, range(min(config["train_subset_size"], len(train_set))))
    if config["test_subset_size"]:
        test_set = Subset(test_set, range(min(config["test_subset_size"], len(test_set))))

    batch_size = config["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    calib_count = min(len(train_set), batch_size * config["calibration_batches"])
    calibration_set = Subset(train_set, range(calib_count))
    calibration_loader = DataLoader(calibration_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, calibration_loader


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / total


def train_one_model(config: Optional[Dict] = None, force_train: bool = False):
    config = {**DEFAULT_CONFIG, **(config or {})}
    ensure_dirs()
    set_reproducible(config["seed"])

    checkpoint_path = ARTIFACTS_DIR / "hw2_simplecnn_fp32.pth"
    history_path = ARTIFACTS_DIR / "hw2_simplecnn_fp32_history.json"
    hw2_models = load_homework2_models_module()
    model = hw2_models.SimpleCNN()

    if checkpoint_path.exists() and not force_train:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        return model, history

    train_loader, test_loader, _ = build_svhn_loaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = []

    for epoch in range(1, config["train_epochs"] + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / total
        test_acc = evaluate_accuracy(model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{config['train_epochs']} "
            f"loss={train_loss:.4f} train_acc={train_acc:.2f}% test_acc={test_acc:.2f}%"
        )

    model.to("cpu")
    torch.save(model.state_dict(), checkpoint_path)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history


def make_fp32_quantizable_model(homework2_simplecnn: nn.Module) -> QuantizableSimpleCNN:
    model = QuantizableSimpleCNN(homework2_simplecnn)
    model.eval()
    model.to("cpu")
    return model


def fuse_quantizable_model(model: nn.Module) -> nn.Module:
    model.eval()
    fuse_modules(model, [["conv1", "relu1"], ["conv2", "relu2"], ["fc1", "relu3"]], inplace=True)
    return model


def choose_quant_engine() -> str:
    supported = list(torch.backends.quantized.supported_engines)
    for engine in ("fbgemm", "x86", "onednn", "qnnpack"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"No supported quantized engine found: {supported}")


def build_static_qconfig() -> QConfig:
    activation = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    # The local PyTorch build only exposes ONEDNN quantized kernels, which require
    # weight zero_point == 0 for Conv/Linear packing. Activations remain asymmetric.
    weight = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    return QConfig(activation=activation, weight=weight)


@torch.no_grad()
def calibrate(model: nn.Module, calibration_loader: DataLoader, max_batches: int) -> None:
    model.eval()
    for batch_idx, (images, _) in enumerate(calibration_loader):
        if batch_idx >= max_batches:
            break
        model(images.cpu())


def quantize_static_ptq(
    fp32_model: nn.Module, calibration_loader: DataLoader, config: Optional[Dict] = None
) -> Tuple[nn.Module, str]:
    config = {**DEFAULT_CONFIG, **(config or {})}
    engine = choose_quant_engine()
    model = copy.deepcopy(fp32_model).cpu().eval()
    fuse_quantizable_model(model)
    model.qconfig = build_static_qconfig()
    prepare(model, inplace=True)
    calibrate(model, calibration_loader, config["calibration_batches"])
    convert(model, inplace=True)
    model.eval()
    return model, engine


def model_state_size_mb(model: nn.Module, output_path: Path) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path.stat().st_size / (1024 * 1024)


@torch.no_grad()
def measure_single_image_latency_ms(
    model: nn.Module,
    sample: torch.Tensor,
    warmup: int = 30,
    repeats: int = 200,
) -> float:
    model.eval()
    sample = sample.cpu()
    for _ in range(warmup):
        model(sample)

    start = time.perf_counter()
    for _ in range(repeats):
        model(sample)
    end = time.perf_counter()
    return (end - start) * 1000.0 / repeats


def _tensor_to_float_cpu(output):
    if isinstance(output, tuple):
        output = output[0]
    if output.is_quantized:
        output = output.dequantize()
    return output.detach().float().cpu()


@torch.no_grad()
def _collect_outputs(model: nn.Module, images: torch.Tensor, layer_names: Iterable[str]) -> Dict[str, torch.Tensor]:
    outputs = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            outputs[name] = _tensor_to_float_cpu(output)

        return hook

    named_modules = dict(model.named_modules())
    for name in layer_names:
        handles.append(named_modules[name].register_forward_hook(make_hook(name)))

    try:
        final_output = model(images.cpu())
        outputs["output"] = _tensor_to_float_cpu(final_output)
    finally:
        for handle in handles:
            handle.remove()

    return outputs


@torch.no_grad()
def compute_layer_mse(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    loader: DataLoader,
    max_batches: int = 4,
    layer_names: Tuple[str, ...] = ("conv1", "conv2", "fc1", "fc2"),
) -> pd.DataFrame:
    fp32_compare = copy.deepcopy(fp32_model).cpu().eval()
    fuse_quantizable_model(fp32_compare)

    sums = {name: 0.0 for name in list(layer_names) + ["output"]}
    counts = {name: 0 for name in sums}

    for batch_idx, (images, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        fp32_outputs = _collect_outputs(fp32_compare, images, layer_names)
        int8_outputs = _collect_outputs(int8_model, images, layer_names)
        for name in sums:
            mse = F.mse_loss(int8_outputs[name], fp32_outputs[name]).item()
            sums[name] += float(mse)
            counts[name] += 1

    rows = [
        {"layer": name, "mse": sums[name] / max(1, counts[name])}
        for name in list(layer_names) + ["output"]
    ]
    return pd.DataFrame(rows)


def plot_metric_bars(metrics_df: pd.DataFrame) -> Tuple[Path, Path]:
    acc_path = FIGURES_DIR / "accuracy_comparison.png"
    latency_path = FIGURES_DIR / "latency_comparison.png"

    plt.figure(figsize=(5, 3.2))
    plt.bar(metrics_df["model"], metrics_df["accuracy_percent"], color=["#2f6f9f", "#c75b39"])
    plt.ylabel("Accuracy (%)")
    plt.title("FP32 vs INT8 Accuracy")
    plt.ylim(0, max(100, float(metrics_df["accuracy_percent"].max()) + 5))
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()

    plt.figure(figsize=(5, 3.2))
    plt.bar(metrics_df["model"], metrics_df["latency_ms"], color=["#2f6f9f", "#c75b39"])
    plt.ylabel("Latency per image (ms)")
    plt.title("CPU Inference Latency")
    plt.tight_layout()
    plt.savefig(latency_path, dpi=200)
    plt.close()

    return acc_path, latency_path


def plot_mse(error_df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "layer_mse.png"
    plt.figure(figsize=(6, 3.2))
    plt.bar(error_df["layer"], error_df["mse"], color="#5b7f55")
    plt.ylabel("MSE")
    plt.title("Layer Output Quantization Error")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def run_full_experiment(config: Optional[Dict] = None, force_train: bool = False):
    config = {**DEFAULT_CONFIG, **(config or {})}
    ensure_dirs()
    set_reproducible(config["seed"])
    train_loader, test_loader, calibration_loader = build_svhn_loaders(config)

    homework2_model, train_history = train_one_model(config, force_train=force_train)
    fp32_model = make_fp32_quantizable_model(homework2_model)

    cpu = torch.device("cpu")
    fp32_acc = evaluate_accuracy(fp32_model, test_loader, cpu)
    sample_images, _ = next(iter(test_loader))
    single_sample = sample_images[:1].cpu()
    fp32_latency = measure_single_image_latency_ms(
        fp32_model,
        single_sample,
        warmup=config["latency_warmup"],
        repeats=config["latency_repeats"],
    )
    fp32_size = model_state_size_mb(fp32_model, ARTIFACTS_DIR / "fp32_quantizable_simplecnn_state.pth")

    int8_model, engine = quantize_static_ptq(fp32_model, calibration_loader, config)
    int8_acc = evaluate_accuracy(int8_model, test_loader, cpu)
    int8_latency = measure_single_image_latency_ms(
        int8_model,
        single_sample,
        warmup=config["latency_warmup"],
        repeats=config["latency_repeats"],
    )
    int8_size = model_state_size_mb(int8_model, ARTIFACTS_DIR / "int8_static_simplecnn_state.pth")

    metrics_df = pd.DataFrame(
        [
            {
                "model": "FP32",
                "accuracy_percent": fp32_acc,
                "size_mb": fp32_size,
                "latency_ms": fp32_latency,
            },
            {
                "model": "INT8 PTQ",
                "accuracy_percent": int8_acc,
                "size_mb": int8_size,
                "latency_ms": int8_latency,
            },
        ]
    )
    accuracy_loss = fp32_acc - int8_acc
    compression_ratio = fp32_size / int8_size
    speedup = fp32_latency / int8_latency

    error_df = compute_layer_mse(
        fp32_model,
        int8_model,
        test_loader,
        max_batches=config["error_batches"],
    )

    acc_plot, latency_plot = plot_metric_bars(metrics_df)
    mse_plot = plot_mse(error_df)

    summary = {
        "quant_engine": engine,
        "accuracy_loss_percent": accuracy_loss,
        "compression_ratio": compression_ratio,
        "speedup": speedup,
        "config": config,
    }

    metrics_df.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False, encoding="utf-8")
    error_df.to_csv(ARTIFACTS_DIR / "layer_mse.csv", index=False, encoding="utf-8")
    (ARTIFACTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "config": config,
        "train_history": train_history,
        "metrics": metrics_df,
        "errors": error_df,
        "summary": summary,
        "figures": {
            "accuracy": acc_plot,
            "latency": latency_plot,
            "mse": mse_plot,
        },
        "fp32_model": fp32_model,
        "int8_model": int8_model,
    }
