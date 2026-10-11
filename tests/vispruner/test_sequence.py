from types import SimpleNamespace

import torch

from wall_x.model.qact.qwen2_5.vispruner.planner import maybe_prune_images
from wall_x.model.qact.qwen2_5.vispruner.sequence import apply_keep_mask


def test_mrope_positions_follow_the_same_sequence_mask():
    keep_mask = torch.tensor([[True, False, True, False], [True, True, False, False]])
    input_ids = torch.tensor([[10, 99, 11, 99], [20, 21, 99, 99]])
    position_ids = torch.arange(24).reshape(3, 2, 4)

    kept_ids, _, _, _, kept_positions, _ = apply_keep_mask(
        keep_mask=keep_mask,
        input_ids=input_ids,
        attention_mask=None,
        labels=None,
        moe_token_types=None,
        position_ids=position_ids,
        inputs_embeds=None,
    )

    assert kept_ids.tolist() == [[10, 11], [20, 21]]
    torch.testing.assert_close(kept_positions[:, 0], position_ids[:, 0, [0, 2]])
    torch.testing.assert_close(kept_positions[:, 1], position_ids[:, 1, [0, 1]])


def test_two_batch_rows_each_prune_their_own_image_positions():
    image_id = 99
    input_ids = torch.tensor(
        [[10, image_id, image_id, image_id, image_id, 11],
         [20, image_id, image_id, image_id, image_id, 21]]
    )
    image_embeds = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    config = SimpleNamespace(
        image_token_id=image_id,
        pad_token_id=0,
        vispruner_enable=True,
        vispruner_score_layers=[31],
        vispruner_score_layer=31,
        vispruner_score_reduce="norm",
        vispruner_keep_num=2,
        vispruner_important_ratio=1.0,
        vispruner_seed=0,
    )

    plan = maybe_prune_images(
        visual=torch.nn.Identity(),
        image_embeds=image_embeds,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]),
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=None,
        moe_token_types=None,
        position_ids=None,
        inputs_embeds=None,
        spatial_merge_size=2,
        model_config=config,
    )
    kept_ids, *_ = plan.apply(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=None,
        moe_token_types=None,
        position_ids=None,
        inputs_embeds=None,
    )

    assert plan.pruned
    assert plan.image_embeds.shape[0] == 4
    assert kept_ids.tolist() == [
        [10, image_id, image_id, 11],
        [20, image_id, image_id, 21],
    ]
