from transformers import (
    CLIPModel,
    CLIPProcessor,
    WhisperModel,
    WhisperProcessor,
)
import torch

from termcolor import colored


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

for name, module in clip_model.named_modules():
    output = colored(
        f"Initialized {name} with {module.__class__.__name__}", "green"
    )
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, torch.nn.Conv2d):
        torch.nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, torch.nn.Conv1d):
        torch.nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, torch.nn.ConvTranspose1d):
        torch.nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    else:
        output = f"Skipped {name} with {module.__class__.__name__}"
        output = colored(output, "yellow")

    print(output)
