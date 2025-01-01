import sys

from typing import cast, Any, Dict, Optional

from taproot import Tap, Task
from taproot.payload import *
from taproot.util import debug_logger

class TorchTest(Task):
    task = "torch_test"

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Returns the required packages for the task.
        """
        return {
            "torch": None,
        }

    @classmethod
    def requires_gpu(cls, **parameters: ParameterMetadataPayload) -> bool:
        """
        Returns whether the task should use the GPU.
        """
        return True

    def __init__(self, config: Optional[Dict[str,Any]]=None) -> None:
        super().__init__(config)

        import torch
        import torch.nn as nn

        class SimpleNN(nn.Module):
            def __init__(self) -> None:
                super(SimpleNN, self).__init__()
                self.fc = nn.Linear(1, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x) # type: ignore[no-any-return]

        self.nn = SimpleNN().to(self.device)

    def __call__(self, x: float) -> float: # type: ignore[override]
        """
        Test function that simply returns the input value.
        """
        import torch
        input_tensor = torch.tensor([x], dtype=torch.float32).to(self.device)
        result_tensor = self.nn(input_tensor)
        return float(result_tensor[0])

def test_torch() -> None:
    """
    Tests parallel torch usage.
    """
    num_requests = 3 if sys.platform == "win32" else 7
    with debug_logger():
        with Tap.local(
            max_workers=num_requests,
            use_multiprocessing=sys.platform != "win32"
        ) as tap:
            results = tap.parallel(*[
                cast(TaskPayload, {"task": "torch_test", "parameters": {"x": i}})
                for i in range(num_requests)
            ])
            assert len(results) == num_requests
            for i, result in enumerate(results):
                assert isinstance(result, float)
                assert -(i+1) <= result <= i+1
