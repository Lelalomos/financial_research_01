from __future__ import annotations

from typing import Any, Dict

import torch


def is_structured_output(output: Any) -> bool:
    return isinstance(output, dict)


def get_prediction_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        prediction = output.get("prediction")
        if prediction is None:
            raise ValueError("Structured model output is missing 'prediction'")
        return prediction
    return output


def get_output_components(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        return dict(output)
    return {"prediction": output}


def compute_batch_loss(model: Any, output: Any, batch: Dict[str, torch.Tensor], criterion) -> torch.Tensor:
    if hasattr(model, "compute_loss"):
        return model.compute_loss(output, batch, criterion)
    return criterion(get_prediction_tensor(output), batch["target"])
