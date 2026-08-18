from types import SimpleNamespace

import pytest
import torch

from axomeme.model import PhyloAxialTransformer, decode_soft_ordinal_lrt
from train import iter_site_indices, train_epoch


def make_gene(site_count, taxon_count, targets=None, eligible=None, gene_name="gene"):
    if targets is None:
        targets = torch.arange(site_count, dtype=torch.float32)
    if eligible is None:
        eligible = torch.ones(site_count, dtype=torch.bool)
    return {
        "c": torch.randint(0, 61, (site_count, taxon_count, 1)),
        "a": torch.randint(0, 20, (site_count, taxon_count, 1)),
        "d": torch.zeros(taxon_count, taxon_count),
        "z": torch.zeros(taxon_count, 4),
        "target_lrt": targets,
        "eligible_mask": eligible,
        "gene_name": gene_name,
    }


def test_ordinal_decoder_is_differentiable():
    logits = torch.randn(2, 16, requires_grad=True)
    decoded, _ = decode_soft_ordinal_lrt(logits)
    decoded.sum().backward()

    assert decoded.shape == (2,)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_site_batches_visit_every_eligible_site_once():
    eligible = torch.tensor([True, False, True, True, False, True, True])
    original_mask = eligible.clone()

    cases = ((1, [1, 1, 1, 1, 1]), (2, [2, 2, 1]), (20, [5]))
    for batch_size, expected_sizes in cases:
        batches = list(
            iter_site_indices(
                eligible,
                batch_size=batch_size,
                generator=torch.Generator().manual_seed(7),
            )
        )
        visited = torch.cat(batches).tolist()
        assert sorted(visited) == [0, 2, 3, 5, 6]
        assert len(visited) == len(set(visited))
        assert [len(batch) for batch in batches] == expected_sizes
    assert torch.equal(eligible, original_mask)


def test_train_epoch_batches_sites_with_heterogeneous_gene_shapes():
    model = PhyloAxialTransformer(
        embed_dim=8, num_layers=1, num_heads=1, window_size=1
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    genes = [make_gene(3, 3, gene_name="short"), make_gene(5, 5, gene_name="long")]

    loss = train_epoch(
        model,
        genes,
        optimizer,
        scaler,
        torch.device("cpu"),
        SimpleNamespace(fp16=False, batch_size=2),
    )

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_train_epoch_reports_site_weighted_loss():
    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(16))

        def forward(self, c, a, d, z):
            return self.logits.unsqueeze(0).expand(c.shape[0], -1)

    targets = torch.tensor([0.0, 1.0, 10.0])
    gene = make_gene(3, 2, targets=targets)
    model = ConstantModel()
    prediction, _ = decode_soft_ordinal_lrt(model.logits.unsqueeze(0))
    expected = torch.nn.functional.smooth_l1_loss(
        prediction.expand_as(targets), targets, reduction="mean"
    ).item()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    actual = train_epoch(
        model,
        [gene],
        optimizer,
        scaler,
        torch.device("cpu"),
        SimpleNamespace(fp16=False, batch_size=2),
    )

    assert actual == pytest.approx(expected)


def test_batch_size_larger_than_gene_uses_one_step():
    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(16))
            self.batch_sizes = []

        def forward(self, c, a, d, z):
            self.batch_sizes.append(c.shape[0])
            return self.logits.unsqueeze(0).expand(c.shape[0], -1)

    model = CountingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    gene = make_gene(3, 4)

    train_epoch(
        model,
        [gene],
        optimizer,
        scaler,
        torch.device("cpu"),
        SimpleNamespace(fp16=False, batch_size=50),
    )

    assert model.batch_sizes == [3]
