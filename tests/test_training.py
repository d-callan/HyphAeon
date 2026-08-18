from types import SimpleNamespace

import torch

from axomeme.model import PhyloAxialTransformer, encode_ordinal_lrt_targets
from train import train_epoch


def test_ordinal_lrt_targets_are_monotonic():
    targets = encode_ordinal_lrt_targets(torch.tensor([0.0, 1.0, 100.0]), 16)

    assert targets.shape == (3, 16)
    assert not targets[0].any()
    assert torch.all(targets[:, 1:] <= targets[:, :-1])
    assert targets[2].sum() > targets[1].sum()


def test_train_epoch_accepts_model_training_output():
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
