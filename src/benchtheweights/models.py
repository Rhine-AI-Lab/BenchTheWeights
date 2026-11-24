"""Building the registered networks and evaluating them deterministically.

Every network maps a split to outputs (logits, predicted noise, reconstructions, token NLLs) and a
scalar score, higher is better:
  classification  accuracy
  denoising       1 - MSE(eps_pred, eps)              (a zero-noise predictor scores 0)
  reconstruction  1 - MSE(recon, x) / Var(x), R^2     (the mean pixel scores 0)
  lm              minus the mean next-token NLL
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import CIFAR_MEAN, CIFAR_STD, DATASETS, MODELS, ModelSpec


def cache_root() -> Path:
    return Path(os.environ.get("BTW_CACHE", Path(__file__).resolve().parents[2] / "cache"))


def checkpoint_dir(model_id: str) -> Path:
    return cache_root() / "checkpoints" / model_id


def data_path(dataset_id: str, split: str) -> Path:
    return cache_root() / "data" / dataset_id / f"{split}.pt"


def configure_torch() -> None:
    """Fixed threads and deterministic kernels, so every container computes the same scores."""
    torch.set_num_threads(int(os.environ.get("BTW_THREADS", "4")))
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------- builders


def _build_timm_cifar_resnet(spec: ModelSpec) -> nn.Module:
    import timm

    model = timm.create_model(
        spec.extra["timm_arch"], pretrained=False, num_classes=spec.extra["num_classes"]
    )
    # the CIFAR variant of the `detectors` package that trained these weights
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _build_hf_sequence_classifier(spec: ModelSpec) -> nn.Module:
    from transformers import AutoConfig, AutoModelForSequenceClassification

    cfg = AutoConfig.from_pretrained(checkpoint_dir(spec.model_id))
    return AutoModelForSequenceClassification.from_config(cfg, attn_implementation="eager")


class MnistMLP(nn.Module):
    """dacorvo/mnist-mlp (the HF repo's modeling_mlp.py, reimplemented without remote code)."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.input_layer = nn.Linear(784, hidden_size)
        self.mid_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.mid_layer(x))
        return self.output_layer(x)  # logits; the original applies a softmax on top


def _build_mnist_mlp(spec: ModelSpec) -> nn.Module:
    return MnistMLP(spec.extra["hidden_size"])


def _build_hf_vit(spec: ModelSpec) -> nn.Module:
    from transformers import ViTConfig, ViTForImageClassification

    cfg = ViTConfig.from_pretrained(checkpoint_dir(spec.model_id))
    return ViTForImageClassification(cfg)


def _build_diffusers_unet(spec: ModelSpec) -> nn.Module:
    from diffusers import UNet2DModel

    return UNet2DModel.from_config(UNet2DModel.load_config(checkpoint_dir(spec.model_id)))


def _build_diffusers_taesd(spec: ModelSpec) -> nn.Module:
    from diffusers import AutoencoderTiny

    return AutoencoderTiny.from_config(AutoencoderTiny.load_config(checkpoint_dir(spec.model_id)))


def _build_hf_causal_lm(spec: ModelSpec) -> nn.Module:
    from transformers import AutoConfig, GPTNeoXForCausalLM

    cfg = AutoConfig.from_pretrained(checkpoint_dir(spec.model_id))
    cfg.torch_dtype = torch.float32
    return GPTNeoXForCausalLM._from_config(cfg, attn_implementation="eager", torch_dtype=torch.float32)


def _build_simulusion_dit(spec: ModelSpec) -> nn.Module:
    from .arch.dit import DiT

    e = spec.extra
    return DiT(e["img_size"], e["patch_size"], 3, e["d_model"], e["depth"], e["heads"], e["num_classes"])


BUILDERS = {
    "timm_cifar_resnet": _build_timm_cifar_resnet,
    "hf_sequence_classifier": _build_hf_sequence_classifier,
    "mnist_mlp": _build_mnist_mlp,
    "hf_vit": _build_hf_vit,
    "diffusers_unet": _build_diffusers_unet,
    "diffusers_taesd": _build_diffusers_taesd,
    "simulusion_dit": _build_simulusion_dit,
    "hf_causal_lm": _build_hf_causal_lm,
}


def load_state_dict(model_id: str) -> dict[str, torch.Tensor]:
    spec = MODELS[model_id]
    path = checkpoint_dir(model_id) / spec.weights_file
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(path))
        if spec.loader == "hf_causal_lm":
            # the clean weights are the exact fp32 upcast of the stored fp16 tensors
            sd = {k: v.float() for k, v in sd.items()}
        if spec.loader == "diffusers_unet":
            # legacy diffusers attention names, which from_pretrained converts implicitly
            legacy = {".query.": ".to_q.", ".key.": ".to_k.", ".value.": ".to_v.", ".proj_attn.": ".to_out.0."}
            for old, new in legacy.items():
                sd = {k.replace(old, new): v for k, v in sd.items()}
        return sd
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if spec.loader == "simulusion_dit":
        sd = sd["model_state_dict"]
    return sd


@lru_cache(maxsize=4)
def _clean_state(model_id: str) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in load_state_dict(model_id).items()}


def build_model(model_id: str) -> nn.Module:
    """The architecture with the pinned clean checkpoint loaded."""
    spec = MODELS[model_id]
    model = BUILDERS[spec.loader](spec)
    state = _clean_state(model_id)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # HF checkpoints may omit tied or buffer entries; anything else is an error
    missing = [k for k in missing if not k.endswith("position_ids")]
    if spec.loader == "hf_causal_lm":  # the model rebuilds its attention masks and rotary tables
        missing = [k for k in missing if not (k.endswith(".attention.bias") or k.endswith(".masked_bias") or k.endswith("inv_freq"))]
    if missing or unexpected:
        raise RuntimeError(f"{model_id}: missing={missing} unexpected={unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def editable_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """The parameters; buffers such as BatchNorm running statistics are not editable."""
    return dict(model.named_parameters())


def load_start(model_id: str, path: str | Path) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    """The network with a problem's starting weights (a safetensors file of every parameter).
    Buffers keep their clean values."""
    from safetensors.torch import load_file

    model = build_model(model_id)
    params = editable_parameters(model)
    sd = load_file(str(path))
    missing = set(params) - set(sd)
    if missing:
        raise SystemExit(f"{path} lacks {len(missing)} parameters, e.g. {sorted(missing)[:3]}")
    with torch.no_grad():
        for n, p in params.items():
            if tuple(sd[n].shape) != tuple(p.shape):
                raise SystemExit(f"{n}: shape {tuple(sd[n].shape)} != {tuple(p.shape)}")
            p.data.copy_(sd[n])
    return model, params


def load_split(dataset_id: str, split: str) -> dict[str, torch.Tensor]:
    return torch.load(data_path(dataset_id, split), map_location="cpu", weights_only=True)


# ------------------------------------------------------------------ inputs

_MEAN = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)


def _image_batch(images_u8: torch.Tensor) -> torch.Tensor:
    x = images_u8.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - _MEAN) / _STD


def _unit_images(images_u8: torch.Tensor) -> torch.Tensor:
    return images_u8.permute(0, 3, 1, 2).float().div_(255.0)


def _recon_target(images_u8: torch.Tensor, size: int) -> torch.Tensor:
    """The reconstruction target: images in [-1, 1], bilinearly resized to size x size."""
    x = _unit_images(images_u8) * 2 - 1
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def model_inputs(model_id: str, data: dict, sl: slice) -> dict:
    spec = MODELS[model_id]
    if spec.loader == "timm_cifar_resnet":
        return {"x": _image_batch(data["images"][sl])}
    if spec.loader == "hf_sequence_classifier":
        return {"input_ids": data["input_ids"][sl], "attention_mask": data["attention_mask"][sl]}
    if spec.loader == "mnist_mlp":
        x = data["images"][sl].float().div(255.0)
        return {"x": ((x - 0.1307) / 0.3081).flatten(1)}
    if spec.loader == "hf_vit":
        x = _unit_images(data["images"][sl])
        x = F.interpolate(x, size=(spec.extra["image_size"],) * 2, mode="bilinear", align_corners=False)
        return {"pixel_values": (x - spec.extra["mean"]) / spec.extra["std"]}
    if spec.loader in ("diffusers_unet", "simulusion_dit"):
        return {"x_t": data["x_t"][sl], "t": data["t"][sl], "y": data["labels"][sl]}
    if spec.loader == "diffusers_taesd":
        return {"x": _recon_target(data["images"][sl], DATASETS[spec.dataset_id].extra["size"])}
    if spec.loader == "hf_causal_lm":
        return {"input_ids": data["input_ids"][sl]}
    raise KeyError(spec.loader)


def forward(model_id: str, model: nn.Module, inputs: dict) -> torch.Tensor:
    loader = MODELS[model_id].loader
    if loader in ("timm_cifar_resnet", "mnist_mlp"):
        return model(inputs["x"])
    if loader == "hf_sequence_classifier":
        return model(**inputs).logits
    if loader == "hf_vit":
        return model(pixel_values=inputs["pixel_values"]).logits
    if loader == "diffusers_unet":
        return model(inputs["x_t"], inputs["t"]).sample
    if loader == "simulusion_dit":
        return model(inputs["x_t"], inputs["t"], inputs["y"])[:, :3]  # the eps half of the 2C output
    if loader == "diffusers_taesd":
        # encode and decode without the uint8 latent quantization of AutoencoderTiny.forward
        return model.decode(model.encode(inputs["x"]).latents).sample
    if loader == "hf_causal_lm":
        return lm_token_nll(model, inputs["input_ids"])
    raise KeyError(loader)


def lm_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Full next-token logits [B, T, V] (fp32). Only for small batches: V = 50304."""
    return model(input_ids=input_ids).logits.float()


def lm_token_nll(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token NLL [B, T-1] of the shifted targets input_ids[:, 1:]: the output of an LM."""
    logits = lm_logits(model, input_ids)[:, :-1]
    return F.cross_entropy(logits.transpose(1, 2), input_ids[:, 1:], reduction="none")


BATCH_SIZE = {"classification": 250, "denoising": 64, "reconstruction": 100, "lm": 8}


def num_examples(data: dict) -> int:
    return int(data["labels"].shape[0])


@torch.no_grad()
def predict(model: nn.Module, model_id: str, data: dict) -> torch.Tensor:
    bs = BATCH_SIZE[MODELS[model_id].objective]
    n = num_examples(data)
    out = [forward(model_id, model, model_inputs(model_id, data, slice(s, s + bs))) for s in range(0, n, bs)]
    return torch.cat(out).float()


def targets(model_id: str, data: dict) -> torch.Tensor:
    spec = MODELS[model_id]
    if spec.objective == "classification":
        return data["labels"]
    if spec.objective == "denoising":
        return data["eps"]
    if spec.objective == "lm":
        return data["input_ids"][:, 1:]
    return _recon_target(data["images"], DATASETS[spec.dataset_id].extra["size"])


def score_outputs(model_id: str, outputs: torch.Tensor, data: dict) -> float:
    spec = MODELS[model_id]
    tgt = targets(model_id, data)
    if spec.objective == "classification":
        return float((outputs.argmax(-1) == tgt).float().mean())
    if spec.objective == "lm":
        return float(-outputs.mean())
    mse = (outputs - tgt).pow(2).mean()
    if spec.objective == "denoising":
        return float(1.0 - mse)
    return float(1.0 - mse / tgt.var())


def score(model: nn.Module, model_id: str, data: dict) -> float:
    return score_outputs(model_id, predict(model, model_id, data), data)


def floor_score(model_id: str, data: dict) -> float:
    """The score of a trivial predictor: the majority class, a uniform LM, or 0 for the regression objectives."""
    spec = MODELS[model_id]
    if spec.objective == "lm":
        import math

        return -math.log(data["vocab_size"].item())
    if spec.objective != "classification":
        return 0.0
    labels = data["labels"]
    counts = torch.bincount(labels, minlength=DATASETS[spec.dataset_id].num_classes)
    return float(counts.max().item() / labels.numel())

