from pathlib import Path
import torch
from torch import nn


def save_model(model: nn.Module, target_dir: str, model_name: str) -> None:
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end in '.pth' or '.pt'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
