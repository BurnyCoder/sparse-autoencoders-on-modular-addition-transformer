"""Tests for modsae.data: dataset construction and leakage-free splits.

Global context: every later phase (base-model training, SAE training, steering
evaluation) indexes inputs through these functions, so a bug here would leak
SAE-test inputs into training or evaluate on the wrong labels.
"""

import torch

from modsae import data

P = 113


def test_dataset_tokens_and_labels():
    tokens, labels = data.make_dataset(P)
    assert tokens.shape == (P * P, 3)
    assert labels.shape == (P * P,)
    # Index layout matches the TransformerLens demo: idx = a * p + b.
    a, b = data.index_to_ab(torch.arange(P * P), P)
    assert torch.equal(tokens[:, 0], a) and torch.equal(tokens[:, 1], b)
    assert torch.all(tokens[:, 2] == P)  # '=' token id is p
    assert torch.equal(labels, (a + b) % P)


def test_base_split_matches_demo_recipe():
    train, test = data.base_split(P, frac_train=0.3, data_seed=598)
    assert len(train) == int(P * P * 0.3) == 3830
    assert len(test) == P * P - 3830
    assert len(set(train.tolist()) & set(test.tolist())) == 0
    # Reproduces torch.manual_seed(598); torch.randperm(p*p) from the demo.
    torch.manual_seed(598)
    perm = torch.randperm(P * P)
    assert torch.equal(train, perm[:3830])


def test_sae_split_keeps_unordered_pairs_together():
    split = data.sae_split(P, seed=0)
    all_idx = torch.cat([split["train"], split["val"], split["test"]])
    assert sorted(all_idx.tolist()) == list(range(P * P))  # partition of all inputs
    owner = {}
    for name, idx in split.items():
        for i in idx.tolist():
            owner[i] = name
    for i in range(P * P):
        a, b = divmod(i, P)
        assert owner[i] == owner[b * P + a]  # (a,b) and (b,a) share a split
    n_pairs = P * (P + 1) // 2
    frac_train = len({tuple(sorted(divmod(i, P))) for i in split["train"].tolist()}) / n_pairs
    assert abs(frac_train - 0.8) < 0.01


def test_dev_test_split_is_a_partition():
    dev, test = data.dev_test_split(torch.arange(1, P), seed=0)
    assert len(dev) + len(test) == P - 1
    assert set(dev.tolist()).isdisjoint(test.tolist())
    assert set(dev.tolist()) | set(test.tolist()) == set(range(1, P))
