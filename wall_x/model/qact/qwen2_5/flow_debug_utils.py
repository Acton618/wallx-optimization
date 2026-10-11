import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

DEBUG_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "debug_output"

NOISY_ACTION_LOG_PATH = (
    DEBUG_OUTPUT_DIR / "noisy_action_trajectory.txt"
)

DEBUG_SAVE_ATTENTION = True


SIMULATION_COUNT = 0


def get_simulation_tag():
    global SIMULATION_COUNT

    tag = f"sim_{SIMULATION_COUNT}"
    SIMULATION_COUNT += 1

    return tag


def should_write_header():
    return SIMULATION_COUNT == 0


def dump_noisy_action_trajectory(
    noisy_action_trajectory,
    times=None,
    filepath=NOISY_ACTION_LOG_PATH,
    experiment_tag="exp",
    write_header=True,
):
    """
    Append full noisy action trajectory to a text file.

    Args:
        noisy_action_trajectory:
            List[Tensor], each tensor has shape [B, T, A].

        times:
            Tensor or list with length equal to trajectory length.

        filepath:
            CSV output path.

        experiment_tag:
            Identifier for an experiment or simulation.

        write_header:
            Whether to write the CSV header.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if write_header else "a"

    with filepath.open(mode, newline="") as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow(
                [
                    "experiment",
                    "step",
                    "time",
                    "batch",
                    "action_timestep",
                    "dim",
                    "value",
                ]
            )

        for step_idx, noisy_action in enumerate(
            noisy_action_trajectory
        ):
            if times is not None:
                current_time = times[step_idx]

                if torch.is_tensor(current_time):
                    current_time = current_time.item()

                current_time = float(current_time)
            else:
                current_time = float(step_idx)

            noisy_action = (
                noisy_action.detach().cpu().numpy()
            )

            if noisy_action.ndim != 3:
                raise ValueError(
                    "noisy_action 必须是 [B, T, A]，"
                    f"实际 shape={noisy_action.shape}"
                )

            batch_size, action_steps, action_dim = (
                noisy_action.shape
            )

            for batch_idx in range(batch_size):
                for action_step in range(action_steps):
                    for dim_idx in range(action_dim):
                        writer.writerow(
                            [
                                experiment_tag,
                                step_idx,
                                f"{current_time:.6f}",
                                batch_idx,
                                action_step,
                                dim_idx,
                                f"{noisy_action[batch_idx, action_step, dim_idx]:.6f}",
                            ]
                        )


def _to_pil_rgb_v0(image):
    """
    将 Tensor、NumPy 数组、PIL.Image 或图片路径
    转换成 RGB PIL.Image。
    """
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(
            "图像必须是三维数组，"
            f"期望 [H, W, C]，实际 shape={image.shape}"
        )

    # 支持 CHW。
    if (
        image.shape[0] in (1, 3, 4)
        and image.shape[-1] not in (1, 3, 4)
    ):
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    image = image.astype(np.float32)

    image_min = float(image.min())
    image_max = float(image.max())

    # 兼容：
    # 1. uint8 / [0, 255]
    # 2. [0, 1]
    # 3. [-1, 1]
    if image_min >= -1.0 and image_max <= 1.0:
        if image_min < 0.0:
            image = (image + 1.0) / 2.0

        image = image * 255.0

    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)

    return Image.fromarray(image, mode="RGB")


def export_processor_images(
    images,
    output_dir=DEBUG_OUTPUT_DIR,
    prefix=None,
):
    """
    导出送入 processor 前的图像。

    Args:
        images:
            单张图像或图像列表。
            支持 torch.Tensor、np.ndarray、PIL.Image、
            str、Path。

        output_dir:
            输出目录。

        prefix:
            输出文件名前缀。
            为 None 时自动生成时间戳。

    Returns:
        List[Path]:
            导出的图片路径。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = (
            "processor_input_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )

    if isinstance(
        images,
        (
            torch.Tensor,
            np.ndarray,
            Image.Image,
            str,
            Path,
        ),
    ):
        images = [images]

    if not isinstance(images, (list, tuple)):
        raise TypeError(
            "images 必须是单张图像或图像列表"
        )

    output_paths = []

    for image_idx, image in enumerate(images):
        output_image = _to_pil_rgb(image)

        output_path = (
            output_dir / f"{prefix}_{image_idx:02d}.png"
        )

        output_image.save(output_path)
        output_paths.append(output_path)

        print(
            "processor input image saved to: "
            f"{output_path}, size={output_image.size}"
        )

    return output_paths


def save_attention_heatmap_v0(   ####正式版本
    attention_maps,
    image,
    image_grid_thw,
    output_path,
    step_idx=-1,
    layer_idx=-1,
    spatial_merge_size=1,
    alpha=0.5,
):
    """
    保存 attention heatmap。

    Args:
        attention_maps:
            Tensor，形状为
            [flow_steps, layers, image_tokens]。

        image:
            单张图像或图像列表。
            支持 PIL.Image、str、Path、
            torch.Tensor、np.ndarray。

        image_grid_thw:
            形状为 [3] 或 [num_images, 3]。
            每一行为 [t, grid_h, grid_w]。

        output_path:
            输出文件路径。
            多图时自动追加 _image_0、_image_1。

        step_idx:
            flow step 索引，默认 -1。

        layer_idx:
            layer 索引，默认 -1。

        spatial_merge_size:
            视觉 token 的空间 merge size。

        alpha:
            heatmap 叠加透明度。
    """
    attention_maps = (
        torch.as_tensor(attention_maps)
        .detach()
        .float()
        .cpu()
    )

    image_grid_thw = (
        torch.as_tensor(image_grid_thw)
        .long()
        .cpu()
    )

    if attention_maps.ndim != 3:
        raise ValueError(
            "attention_maps 应为 "
            "[flow_steps, layers, image_tokens]，"
            f"实际 shape={tuple(attention_maps.shape)}"
        )

    if image_grid_thw.ndim == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)

    if (
        image_grid_thw.ndim != 2
        or image_grid_thw.shape[1] != 3
    ):
        raise ValueError(
            "image_grid_thw 应为 [num_images, 3]，"
            f"实际 shape={tuple(image_grid_thw.shape)}"
        )

    num_images = image_grid_thw.shape[0]

    if isinstance(
        image,
        (
            str,
            Path,
            Image.Image,
            torch.Tensor,
            np.ndarray,
        ),
    ):
        images = [image]
    elif isinstance(image, (list, tuple)):
        images = list(image)
    else:
        raise TypeError(
            "image 必须是单张图像或图像列表"
        )

    if len(images) != num_images:
        raise ValueError(
            "图片数量与 image_grid_thw 数量不匹配："
            f"图片数={len(images)}，"
            f"image_grid_thw 数量={num_images}"
        )

    # 统一转换为 PIL RGB。
    images = [
        _to_pil_rgb(current_image)
        for current_image in images
    ]


    # 选择一个 flow step 和一个 layer。
    # 得到所有图片视觉 token 的拼接 attention。
    image_scores = (
        attention_maps[step_idx, layer_idx]
        .flatten()
    )


    image_infos = []
    expected_total_tokens = 0

    for image_idx, grid_info in enumerate(
        image_grid_thw
    ):
        time_dim, raw_grid_h, raw_grid_w = (
            grid_info.tolist()
        )

        if raw_grid_h % spatial_merge_size != 0:
            raise ValueError(
                f"第 {image_idx} 张图片的 grid_h="
                f"{raw_grid_h} 不能被 "
                f"spatial_merge_size="
                f"{spatial_merge_size} 整除"
            )

        if raw_grid_w % spatial_merge_size != 0:
            raise ValueError(
                f"第 {image_idx} 张图片的 grid_w="
                f"{raw_grid_w} 不能被 "
                f"spatial_merge_size="
                f"{spatial_merge_size} 整除"
            )

        grid_h = raw_grid_h // spatial_merge_size
        grid_w = raw_grid_w // spatial_merge_size

        token_count = grid_h * grid_w

        image_infos.append(
            {
                "image_idx": image_idx,
                "time_dim": time_dim,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "token_count": token_count,
            }
        )

        expected_total_tokens += token_count

    actual_total_tokens = image_scores.numel()

    if actual_total_tokens != expected_total_tokens:
        image_shapes = [
            (
                info["grid_h"],
                info["grid_w"],
                info["token_count"],
            )
            for info in image_infos
        ]

        raise ValueError(
            "image token 数量不匹配："
            f"实际={actual_total_tokens}，"
            f"预期={expected_total_tokens}，"
            f"各图片={image_shapes}，"
            f"attention_maps shape="
            f"{tuple(attention_maps.shape)}，"
            f"step_idx={step_idx}，"
            f"layer_idx={layer_idx}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    color_map = plt.get_cmap("jet")

    token_offset = 0
    saved_paths = []

    for info, current_image in zip(
        image_infos,
        images,
    ):
        image_idx = info["image_idx"]
        grid_h = info["grid_h"]
        grid_w = info["grid_w"]
        token_count = info["token_count"]

        current_scores = image_scores[
            token_offset : token_offset + token_count
        ]
        token_offset += token_count

        token_map = current_scores.reshape(
            1,
            1,
            grid_w,
            grid_h,
        )

        width, height = current_image.size

        heatmap = F.interpolate(
            token_map,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()

        heatmap = (
            heatmap - heatmap_min
        ) / (
            heatmap_max - heatmap_min + 1e-8
        )

        image_array = (
            np.asarray(current_image.convert("RGB"))
            .astype(np.float32)
            / 255.0
        )

        colored_heatmap = color_map(heatmap)[..., :3]

        overlay = (
            (1.0 - alpha) * image_array
            + alpha * colored_heatmap
        )

        overlay = np.clip(overlay, 0.0, 1.0)
        overlay = (overlay * 255.0).astype(np.uint8)

        if num_images == 1:
            current_output_path = output_path
        else:
            current_output_path = (
                output_path.parent
                / f"{output_path.stem}_image_{image_idx}"
                f"{output_path.suffix}"
            )

        Image.fromarray(overlay, mode="RGB").save(
            current_output_path
        )

        saved_paths.append(current_output_path)

        print(
            "attention heatmap saved to: "
            f"{current_output_path}"
        )

    return saved_paths




def save_attention_heatmap_test(       #####空间关系调试版本
    attention_maps,
    image,
    image_grid_thw,
    output_path,
    step_idx=-1,
    layer_idx=-1,
    spatial_merge_size=1,
    alpha=0.5,
):
    """
    保存 attention heatmap。

    Args:
        attention_maps:
            Tensor，形状为
            [flow_steps, layers, image_tokens]。

        image:
            单张图像或图像列表。
            支持 PIL.Image、str、Path、
            torch.Tensor、np.ndarray。

        image_grid_thw:
            形状为 [3] 或 [num_images, 3]。
            每一行为 [t, grid_h, grid_w]。

        output_path:
            输出文件路径。
            多图时自动追加 _image_0、_image_1。

        step_idx:
            flow step 索引，默认 -1。

        layer_idx:
            layer 索引，默认 -1。

        spatial_merge_size:
            视觉 token 的空间 merge size。

        alpha:
            heatmap 叠加透明度。
    """
    attention_maps = (
        torch.as_tensor(attention_maps)
        .detach()
        .float()
        .cpu()
    )

    image_grid_thw = (
        torch.as_tensor(image_grid_thw)
        .long()
        .cpu()
    )

    if attention_maps.ndim != 3:
        raise ValueError(
            "attention_maps 应为 "
            "[flow_steps, layers, image_tokens]，"
            f"实际 shape={tuple(attention_maps.shape)}"
        )

    if image_grid_thw.ndim == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)

    if (
        image_grid_thw.ndim != 2
        or image_grid_thw.shape[1] != 3
    ):
        raise ValueError(
            "image_grid_thw 应为 [num_images, 3]，"
            f"实际 shape={tuple(image_grid_thw.shape)}"
        )

    num_images = image_grid_thw.shape[0]

    if isinstance(
        image,
        (
            str,
            Path,
            Image.Image,
            torch.Tensor,
            np.ndarray,
        ),
    ):
        images = [image]
    elif isinstance(image, (list, tuple)):
        images = list(image)
    else:
        raise TypeError(
            "image 必须是单张图像或图像列表"
        )

    if len(images) != num_images:
        raise ValueError(
            "图片数量与 image_grid_thw 数量不匹配："
            f"图片数={len(images)}，"
            f"image_grid_thw 数量={num_images}"
        )

    # 统一转换为 PIL RGB。
    images = [
        _to_pil_rgb(current_image)
        for current_image in images
    ]

    # ============================================================
    # 【对齐验证模式】
    # 1) 暂不取真实 attention（如需恢复数值，取消注释下一行并删掉伪造块）
    # image_scores = (
    #     attention_maps[step_idx, layer_idx]
    #     .flatten()
    # )
    # ============================================================

    image_infos = []
    expected_total_tokens = 0

    for image_idx, grid_info in enumerate(image_grid_thw):
        time_dim, raw_grid_h, raw_grid_w = grid_info.tolist()

        if raw_grid_h % spatial_merge_size != 0:
            raise ValueError(
                f"第 {image_idx} 张图片的 grid_h="
                f"{raw_grid_h} 不能被 "
                f"spatial_merge_size="
                f"{spatial_merge_size} 整除"
            )

        if raw_grid_w % spatial_merge_size != 0:
            raise ValueError(
                f"第 {image_idx} 张图片的 grid_w="
                f"{raw_grid_w} 不能被 "
                f"spatial_merge_size="
                f"{spatial_merge_size} 整除"
            )

        grid_h = raw_grid_h // spatial_merge_size
        grid_w = raw_grid_w // spatial_merge_size
        token_count = grid_h * grid_w

        image_infos.append(
            {
                "image_idx": image_idx,
                "time_dim": time_dim,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "token_count": token_count,
            }
        )

        expected_total_tokens += token_count

    # ============================================================
    # 【对齐验证】用空间递增的伪造 attention 替代真实值
    # 目的：验证 9x9 token -> 256x256 图片 的空间映射是否正确
    # 正确现象：每张图都应是「左上暗 -> 右下亮」的线性渐变
    # ============================================================
    ref = image_infos[0]
    grid_h = ref["grid_h"]
    grid_w = ref["grid_w"]

    per_img = torch.arange(grid_h * grid_w, dtype=torch.float32)
    image_scores = torch.cat(
        [per_img for _ in range(len(image_infos))]
    )

    # 可选：让第二张图渐变方向相反，方便区分 face/wrist 是否对应正确
    if len(image_infos) > 1:
        n = grid_h * grid_w
        image_scores[n:] = torch.flip(image_scores[n:], dims=[0])

    assert image_scores.numel() == expected_total_tokens, (
        f"伪造 token 数 {image_scores.numel()} != 预期 {expected_total_tokens}"
    )
    # ============================================================

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    color_map = plt.get_cmap("jet")

    token_offset = 0
    saved_paths = []

    for info, current_image in zip(image_infos, images):
        image_idx = info["image_idx"]
        grid_h = info["grid_h"]
        grid_w = info["grid_w"]
        token_count = info["token_count"]

        current_scores = image_scores[
            token_offset : token_offset + token_count
        ]
        token_offset += token_count

        token_map = current_scores.reshape(
            1, 1, grid_w , grid_h
        )

        width, height = current_image.size

        heatmap = F.interpolate(
            token_map,
            size=(height, width),
            mode="nearest",
        )[0, 0].numpy()

        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        heatmap = (heatmap - heatmap_min) / (
            heatmap_max - heatmap_min + 1e-8
        )

        image_array = (
            np.asarray(current_image.convert("RGB"))
            .astype(np.float32)
            / 255.0
        )

        colored_heatmap = color_map(heatmap)[..., :3]
        overlay = (
            (1.0 - alpha) * image_array
            + alpha * colored_heatmap
        )
        overlay = np.clip(overlay, 0.0, 1.0)
        overlay = (overlay * 255.0).astype(np.uint8)

        if num_images == 1:
            current_output_path = output_path
        else:
            current_output_path = (
                output_path.parent
                / f"{output_path.stem}_image_{image_idx}"
                f"{output_path.suffix}"
            )

        Image.fromarray(overlay, mode="RGB").save(
            current_output_path
        )
        saved_paths.append(current_output_path)

        print(
            "attention heatmap saved to: "
            f"{current_output_path}"
        )

    return saved_paths

def save_attention_heatmap(
    attention_maps,          # [flow_steps, layers, image_tokens]
    images,
    image_grid_thw,
    output_dir,
    step_idx=-1,
    layer_indices=None,      # ✅ 支持多个 layer
    spatial_merge_size=2,
    alpha=0.5,
    mode="nearest",
    topk_vis=5,              # ✅ 在图上画几个 top token
    token_counts=None,
):
    """
    ✅ WALLOX / Qwen2.5-VL 专用（诊断版）
    ✅ 用于验证模型是否关注「碗」和「拿取点」
    """

    # ========== 基础转换 ==========
    attention_maps = attention_maps.detach().float().cpu()
    image_grid_thw = image_grid_thw.long().cpu()

    if attention_maps.ndim != 3:
        raise ValueError("attention_maps: [flow_steps, layers, image_tokens]")

    if image_grid_thw.ndim == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)

    num_images = image_grid_thw.shape[0]

    if not isinstance(images, (list, tuple)):
        images = [images]
    images = [_to_pil_rgb(img) for img in images]

    if len(images) != num_images:
        raise ValueError("images 数量与 image_grid_thw 不匹配")

    # ========== step / layer 处理 ==========
    if step_idx < 0:
        step_idx = attention_maps.shape[0] - 1

    if layer_indices is None:
        layer_indices = [attention_maps.shape[1] - 1]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 解析 token grid ==========
    # ``image_grid_thw`` describes the unpruned visual grid.  When VisPruner
    # is enabled, ``token_counts`` carries the actual post-pruning count for
    # each image; the original grid can no longer be reshaped directly.
    if token_counts is not None:
        token_counts = [int(count) for count in token_counts]
        if len(token_counts) != num_images:
            raise ValueError("token_counts 数量与 image_grid_thw 不匹配")
        if sum(token_counts) != attention_maps.shape[-1]:
            raise ValueError(
                "token_counts 总数与 attention_maps 不匹配: "
                f"counts={sum(token_counts)}, maps={attention_maps.shape[-1]}"
            )

    def _grid_shape(count, target_h, target_w):
        if count == target_h * target_w:
            return target_h, target_w
        # Pick a factor pair close to the source aspect ratio.  This keeps a
        # compact diagnostic image for budgets such as 20 (4x5) while also
        # handling prime or otherwise irregular token budgets.
        best = (1, count)
        best_error = float("inf")
        for height in range(1, int(count**0.5) + 1):
            if count % height:
                continue
            width = count // height
            error = abs((height / width) - (target_h / target_w))
            if error < best_error:
                best, best_error = (height, width), error
        return best

    image_infos = []
    for i in range(num_images):
        t, patch_h, patch_w = image_grid_thw[i].tolist()
        source_h = patch_h // spatial_merge_size
        source_w = patch_w // spatial_merge_size
        count = (
            token_counts[i]
            if token_counts is not None
            else int(t) * source_h * source_w
        )
        token_h, token_w = _grid_shape(count, source_h, source_w)
        image_infos.append(
            {
                "image_idx": i,
                "token_h": token_h,
                "token_w": token_w,
                "token_count": count,
            }
        )

    # ========== 逐 layer 分析 ==========
    for layer_idx in layer_indices:
        image_scores = attention_maps[step_idx, layer_idx]

        print("\n" + "=" * 60)
        print(f"DEBUG: step={step_idx}, layer={layer_idx}")
        print(f"       min={image_scores.min():.6f}, "
              f"max={image_scores.max():.6f}, "
              f"mean={image_scores.mean():.6f}")

        token_offset = 0
        saved_paths = []

        for info, pil_img in zip(image_infos, images):
            token_h = info["token_h"]
            token_w = info["token_w"]
            count = info["token_count"]

            scores = image_scores[token_offset : token_offset + count]
            token_offset += count

            # ========== Top-k + 空间坐标 ==========
            topk = min(10, count)
            topk_vals, topk_idx = torch.topk(scores, topk)

            print(f"\n[Image {info['image_idx']}] Top-{topk} tokens:")
            for j in range(topk):
                idx = topk_idx[j].item()
                gy, gx = divmod(idx, token_w)  # ✅ row, col
                print(
                    f"  token_{idx:03d}: "
                    f"pos=({gy:02d},{gx:02d}), "
                    f"score={topk_vals[j].item():.6f}"
                )

            # ========== 画 heatmap ==========
            token_map = scores.reshape(token_h, token_w).unsqueeze(0).unsqueeze(0)
            w, h = pil_img.size

            heatmap = F.interpolate(
                token_map, size=(h, w), mode=mode
            )[0, 0].numpy()

            heatmap = (heatmap - heatmap.min()) / (
                heatmap.max() - heatmap.min() + 1e-8
            )

            img_np = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
            heat_np = plt.get_cmap("jet")(heatmap)[..., :3]
            overlay = (1 - alpha) * img_np + alpha * heat_np
            overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

            # ========== ✅ 在图上画 Top token 红点 ==========
            overlay_pil = Image.fromarray(overlay, mode="RGB")
            draw = ImageDraw.Draw(overlay_pil)

            for j in range(min(topk_vis, topk)):
                idx = topk_idx[j].item()
                gy, gx = divmod(idx, token_w)

                # token 中心映射到原图坐标
                cx = int(gx / token_w * w)
                cy = int(gy / token_h * h)

                r = 4
                draw.ellipse(
                    (cx - r, cy - r, cx + r, cy + r),
                    fill=(255, 0, 0),
                    outline=(255, 255, 255),
                )

            # ========== 保存 ==========
            out_name = (
                f"step{step_idx}_layer{layer_idx}_image{info['image_idx']}.png"
            )
            out_path = output_dir / out_name
            overlay_pil.save(out_path)
            saved_paths.append(str(out_path))
            print(f"heatmap saved: {out_path}")

        # ========== Face / Wrist 对比 ==========
        if num_images == 2 and image_scores.numel() == 162:
            face = image_scores[:81]
            wrist = image_scores[81:]
            print(f"\n[Face vs Wrist]")
            print(f"  face : mean={face.mean():.6f}, max={face.max():.6f}")
            print(f"  wrist: mean={wrist.mean():.6f}, max={wrist.max():.6f}")

    print("=" * 60 + "\n")
    return saved_paths


# ========== PIL 辅助 ==========
def _to_pil_rgb(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, np.ndarray):
        if x.dtype != np.uint8:
            x = (x * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(x).convert("RGB")
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        if x.dtype != np.uint8:
            x = (x * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(x).convert("RGB")
    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(x)}")
