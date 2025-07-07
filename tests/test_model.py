import pytest
import torch
from src.model import GPT, GPTConfig

@pytest.fixture
def gpt_config():
    """Provides a basic GPT configuration for testing."""
    return GPTConfig(
        vocab_size=256,  # Vocabulary size for pixel values (0-255)
        block_size=128,  # Max sequence length
        n_layer=2,       # Number of layers
        n_head=4,        # Number of attention heads
        n_embd=128,      # Embedding dimension
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )

def test_gpt_model_creation(gpt_config):
    """Tests if the GPT model can be created without errors."""
    model = GPT(gpt_config)
    assert model is not None

def test_gpt_forward_pass(gpt_config):
    """Tests the forward pass of the GPT model."""
    model = GPT(gpt_config)
    batch_size = 4
    seq_len = gpt_config.block_size

    # Create a dummy input tensor (batch_size, seq_len)
    idx = torch.randint(0, gpt_config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets=idx) # Using idx as targets for simplicity

    # Check output shapes
    assert logits.shape == (batch_size, seq_len, gpt_config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # Loss should be a scalar

def test_gpt_forward_pass_no_targets(gpt_config):
    """Tests the forward pass without targets (for inference)."""
    model = GPT(gpt_config)
    batch_size = 4
    seq_len = gpt_config.block_size

    idx = torch.randint(0, gpt_config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets=None)

    # Check output shapes and that loss is None
    assert logits.shape == (batch_size, seq_len, gpt_config.vocab_size)
    assert loss is None
