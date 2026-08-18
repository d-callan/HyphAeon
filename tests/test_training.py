from types import SimpleNamespace

import torch

from axomeme.model import PhyloAxialTransformer, decode_soft_ordinal_lrt
from train import train_epoch


def test_ordinal_decoder_is_differentiable():
    logits = torch.randn(2, 16, requires_grad=True)
    decoded, _ = decode_soft_ordinal_lrt(logits)
    decoded.sum().backward()

    assert decoded.shape == (2,)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_train_epoch_decodes_logits_for_smooth_l1():
    model = PhyloAxialTransformer(
        embed_dim=8, num_layers=1, num_heads=1, window_size=1
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    batch = {
        'c': torch.randint(0, 61, (1, 3, 1)),
        'a': torch.randint(0, 20, (1, 3, 1)),
        'd': torch.zeros(1, 3, 3),
        'z': torch.zeros(1, 3, 4),
        'target_lrt': torch.tensor([2.0]),
    }

    loss = train_epoch(
        model, [batch], optimizer, scaler, torch.device('cpu'),
        SimpleNamespace(fp16=False)
    )

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
