import pytest
import torch

from wall_x.model.qact.qwen2_5.vispruner.algorithms import (
    image_token_lengths,
    select_indices,
)


def test_image_token_lengths_supports_tensor_grid():
    grid = torch.tensor([[1, 18, 18], [1, 18, 18]])
    assert image_token_lengths(grid, spatial_merge_size=2) == [81, 81]


def test_select_indices_is_sorted_unique_and_exact_budget():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    features = torch.eye(8)
    selected = select_indices(scores, features, keep_num=4, important_ratio=0.5)

    assert selected.dtype == torch.long
    assert selected.tolist() == sorted(set(selected.tolist()))
    assert selected.numel() == 4
    assert int(selected.max()) < 8


def test_keep_all_returns_identity_indices():
    scores = torch.randn(5)
    features = torch.randn(5, 3)
    selected = select_indices(scores, features, keep_num=5, important_ratio=0.5)
    torch.testing.assert_close(selected, torch.arange(5))


def test_selection_is_deterministic_for_tied_scores():
    scores = torch.ones(8)
    features = torch.eye(8)
    first = select_indices(scores, features, keep_num=4, important_ratio=1.0)
    second = select_indices(scores, features, keep_num=4, important_ratio=1.0)
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    "grid,merge",
    [([[1, 17, 18]], 2), ([[1, 17, 18]], 3)],
)
def test_invalid_grid_is_rejected(grid, merge):
    with pytest.raises(ValueError):
        image_token_lengths(grid, merge)
