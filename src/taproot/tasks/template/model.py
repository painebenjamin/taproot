"""
This is a template model file. It is a simple model that can be used as a starting point for your own models.
It contains a single linear layer and a forward pass function.
It also contains an example function to save the model using safetensors.

Since we're defining a model, we don't lazy import torch here, we import it directly.
That means that all the files that will be in the global namespace that import this file will also import torch,
so they must do so lazily, or CUDA will be initialized multiple times if that process launches subprocesses.
"""
import os
import torch
import safetensors.torch

__all__ = ["TemplateModel"]

class TemplateModel(torch.nn.Module):
    """
    A template model class.
    """
    def __init__(self, n_dim: int = 256) -> None:
        super(TemplateModel, self).__init__()
        self.fc = torch.nn.Linear(n_dim, n_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        return self.fc(x) # type: ignore[no-any-return]

    def save_safetensors(
        self,
        directory_or_file: str,
        file_name: str="template-model",
        use_fp16: bool=True,
    ) -> str:
        """
        Save the model using safetensors.
        """
        import safetensors.torch
        if os.path.isfile(directory_or_file):
            target_path = directory_or_file
        elif use_fp16:
            target_path = os.path.join(directory_or_file, f"{file_name}.fp16.safetensors")
        else:
            target_path = os.path.join(directory_or_file, f"{file_name}.safetensors")

        if use_fp16:
            self.half()

        safetensors.torch.save_file(self.state_dict(), target_path)
        return target_path

# This section of code allows this file to be ran as a script.
# It will save the model in the current working directory.
# Be sure to remove any script code from your model files before submitting.
if __name__ == "__main__":
    model = TemplateModel()
    path = model.save_safetensors(os.getcwd())
    print(f"Safetensors model saved to: {path}")
