import torch
import custom_op_lib
import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vector_add():
    a = torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda')
    b = torch.tensor([4, 5, 6], dtype=torch.float32, device='cuda')
    
    c = custom_op_lib.vector_add(a, b)
    
    expected = torch.tensor([5, 7, 9], dtype=torch.float32, device='cuda')
    
    assert torch.allclose(c, expected), f"Test failed: expected {expected}, got {c}"

def test_placeholder():
    # This is a placeholder test to ensure pytest has something to run
    # on systems without a CUDA device.
    assert True 