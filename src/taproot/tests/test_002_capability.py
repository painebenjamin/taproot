from taproot.util import MachineCapability

def test_capability() -> None:
    capability = MachineCapability.get_capability()
    assert 0 <= capability.score(use_gpu=False) <= 10000
