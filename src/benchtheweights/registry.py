"""Pinned pretrained models and their evaluation datasets.

Every entry pins an exact Hugging Face revision, so a task that names a `model_id` is reproducible.
A problem registers its network here, and its loader in `models.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    hf_repo: str
    revision: str
    parquet_file: str
    modality: str  # "image" | "text" | "mnist" | "denoise" | "recon" | "lm"
    num_classes: int
    n_dev: int  # the split the backend serves to the solver
    n_hidden: int  # the split only the verifier sees
    split_seed: int = 0
    label_column: str = "label"
    input_column: str = "img"
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str  # "cnn" | "transformer" | "mlp" | "vit" | "unet" | "autoencoder" | "dit"
    architecture: str  # shown to the solver
    task: str  # shown to the solver
    hf_repo: str
    revision: str
    weights_file: str
    dataset_id: str
    loader: str
    extra: dict = field(default_factory=dict)
    objective: str = "classification"  # "classification" | "denoising" | "reconstruction" | "lm"


DATASETS: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        dataset_id="cifar10",
        hf_repo="uoft-cs/cifar10",
        revision="0b2714987fa478483af9968de7c934580d0bb9a2",
        parquet_file="plain_text/test-00000-of-00001.parquet",
        modality="image",
        num_classes=10,
        n_dev=1000,
        n_hidden=4000,
        label_column="label",
        input_column="img",
    ),
    "cifar100": DatasetSpec(
        dataset_id="cifar100",
        hf_repo="uoft-cs/cifar100",
        revision="aadb3af77e9048adbea6b47c21a81e47dd092ae5",
        parquet_file="cifar100/test-00000-of-00001.parquet",
        modality="image",
        num_classes=100,
        n_dev=1000,
        n_hidden=4000,
        label_column="fine_label",
        input_column="img",
    ),
    "sst2": DatasetSpec(
        dataset_id="sst2",
        hf_repo="stanfordnlp/sst2",
        revision="8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb",
        parquet_file="data/validation-00000-of-00001.parquet",
        modality="text",
        num_classes=2,
        n_dev=272,
        n_hidden=600,
        label_column="label",
        input_column="sentence",
    ),
    "mnist": DatasetSpec(
        dataset_id="mnist",
        hf_repo="ylecun/mnist",
        revision="77f3279092a1c1579b2250db8eafed0ad422088c",
        parquet_file="mnist/test-00000-of-00001.parquet",
        modality="mnist",
        num_classes=10,
        n_dev=1000,
        n_hidden=4000,
        label_column="label",
        input_column="image",
    ),
    # CIFAR-10 test images with a fixed (t, eps) per example, for noise prediction
    "cifar10-denoise": DatasetSpec(
        dataset_id="cifar10-denoise",
        hf_repo="uoft-cs/cifar10",
        revision="0b2714987fa478483af9968de7c934580d0bb9a2",
        parquet_file="plain_text/test-00000-of-00001.parquet",
        modality="denoise",
        num_classes=10,
        n_dev=256,
        n_hidden=512,
        split_seed=1,
        extra={"timesteps": 1000, "beta_start": 1e-4, "beta_end": 0.02, "noise_seed": 1234},
    ),
    # CIFAR-10 test images, bilinearly upsampled to 64x64 at evaluation time, for reconstruction
    "cifar10-recon": DatasetSpec(
        dataset_id="cifar10-recon",
        hf_repo="uoft-cs/cifar10",
        revision="0b2714987fa478483af9968de7c934580d0bb9a2",
        parquet_file="plain_text/test-00000-of-00001.parquet",
        modality="recon",
        num_classes=10,
        n_dev=500,
        n_hidden=1000,
        split_seed=2,
        extra={"size": 64},
    ),
}


# files downloaded with the weights, per loader (prepare.py)
HF_EXTRA_FILES = {
    "hf_sequence_classifier": ["config.json", "vocab.txt", "tokenizer_config.json"],
    "timm_cifar_resnet": ["config.json"],
    "mnist_mlp": ["config.json"],
    "hf_vit": ["config.json"],
    "diffusers_unet": ["config.json"],
    "diffusers_taesd": ["config.json"],
    "simulusion_dit": ["config.yaml"],
    "hf_causal_lm": ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.201)

MODELS: dict[str, ModelSpec] = {
    "resnet18-cifar10": ModelSpec(
        model_id="resnet18-cifar10",
        family="cnn",
        architecture="ResNet-18 (CIFAR variant, 3x3 stem, no max-pool)",
        task="Image classification (CIFAR-10, 10 classes)",
        hf_repo="edadaltocg/resnet18_cifar10",
        revision="ccacbae365101b9bfa45e1f17f4acac30c2601b2",
        weights_file="pytorch_model.bin",
        dataset_id="cifar10",
        loader="timm_cifar_resnet",
        extra={"timm_arch": "resnet18", "num_classes": 10},
    ),
    "resnet18-cifar100": ModelSpec(
        model_id="resnet18-cifar100",
        family="cnn",
        architecture="ResNet-18 (CIFAR variant, 3x3 stem, no max-pool)",
        task="Image classification (CIFAR-100, 100 classes)",
        hf_repo="edadaltocg/resnet18_cifar100",
        revision="9aafc08f99538685c54941ee8a54e942e5d8980e",
        weights_file="pytorch_model.bin",
        dataset_id="cifar100",
        loader="timm_cifar_resnet",
        extra={"timm_arch": "resnet18", "num_classes": 100},
    ),
    "distilbert-sst2": ModelSpec(
        model_id="distilbert-sst2",
        family="transformer",
        architecture="DistilBERT-base-uncased (6-layer Transformer encoder)",
        task="Sentiment classification (SST-2, 2 classes)",
        hf_repo="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0fa89d2f80546fda750413ed43d93601a13",
        weights_file="model.safetensors",
        dataset_id="sst2",
        loader="hf_sequence_classifier",
        extra={"max_length": 66},
    ),
    "mlp-mnist": ModelSpec(
        model_id="mlp-mnist",
        family="mlp",
        architecture="3-layer MLP (784-256-256-10, ReLU)",
        task="Handwritten digit classification (MNIST, 10 classes)",
        hf_repo="dacorvo/mnist-mlp",
        revision="7ed1e96d182395fd7380ea8495b9eb2364ce9dc1",
        weights_file="model.safetensors",
        dataset_id="mnist",
        loader="mnist_mlp",
        extra={"hidden_size": 256},
    ),
    "vit-tiny-cifar10": ModelSpec(
        model_id="vit-tiny-cifar10",
        family="vit",
        architecture="ViT-Tiny/16 (12 layers, width 192, 224x224 input)",
        task="Image classification (CIFAR-10, 10 classes)",
        hf_repo="TestingProfile111/cifar10-vit-tiny",
        revision="5c383b5eb5708ce8a072a6ac958fa2b2329979cf",
        weights_file="model.safetensors",
        dataset_id="cifar10",
        loader="hf_vit",
        extra={"image_size": 224, "mean": 0.5, "std": 0.5},
    ),
    "ddpm-unet-cifar10": ModelSpec(
        model_id="ddpm-unet-cifar10",
        family="unet",
        architecture="DDPM U-Net (diffusers UNet2DModel, 35.7M params, 32x32)",
        task="Diffusion noise prediction on CIFAR-10 (predict the added Gaussian noise)",
        hf_repo="google/ddpm-cifar10-32",
        revision="267b167dc01f0e4e61923ea244e8b988f84deb80",
        weights_file="diffusion_pytorch_model.safetensors",
        dataset_id="cifar10-denoise",
        loader="diffusers_unet",
        objective="denoising",
    ),
    "dit-s4-cifar10": ModelSpec(
        model_id="dit-s4-cifar10",
        family="dit",
        architecture="DiT-S/4 Diffusion Transformer (12 blocks, width 384, adaLN-Zero, class-conditional)",
        task="Class-conditional diffusion noise prediction on CIFAR-10",
        hf_repo="simillusion/dit-s4-cifar10",
        revision="2c099eee901401e7c77335dde3c32de6cf907fb2",
        weights_file="dit_model.pth",
        dataset_id="cifar10-denoise",
        loader="simulusion_dit",
        objective="denoising",
        extra={"img_size": 32, "patch_size": 4, "d_model": 384, "depth": 12, "heads": 6, "num_classes": 10},
    ),
    "taesd-autoencoder": ModelSpec(
        model_id="taesd-autoencoder",
        family="autoencoder",
        architecture="Tiny AutoEncoder for Stable Diffusion (TAESD, conv encoder/decoder, 4-ch latent)",
        task="Image reconstruction (encode -> decode) on CIFAR-10 images upsampled to 64x64",
        hf_repo="madebyollin/taesd",
        revision="614f76814bbe30edbe2e627ace1c2234c81a2c0e",
        weights_file="diffusion_pytorch_model.safetensors",
        dataset_id="cifar10-recon",
        loader="diffusers_taesd",
        objective="reconstruction",
    ),
}

# (metric name, definition) per objective, as the solver sees them.
SCORE_INFO = {
    "classification": ("accuracy", "fraction of correctly classified examples"),
    "denoising": (
        "denoising score",
        "1 - mean squared error between predicted and true noise (eps) at fixed timesteps; "
        "a model that predicts zero noise scores 0",
    ),
    "reconstruction": (
        "reconstruction score",
        "R^2 = 1 - MSE(reconstruction, image) / Var(image); predicting the mean pixel scores 0",
    ),
    "lm": (
        "lm score",
        "negative mean next-token negative log-likelihood (nats) over all 255 predicted positions of each "
        "256-token window; perplexity = exp(-lm score); a uniform predictor over the vocabulary scores -10.83",
    ),
}
