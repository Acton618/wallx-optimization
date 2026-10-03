Wall-X 模型复现与 LIBERO 评测说明（当前服务器适配版）
================================================================

本文档适用于当前服务器上的 Wall-X v1_ft checkpoint。
所有命令默认在 bash 中执行，并使用已经存在的 wallx Conda 环境。

当前实际使用的版本
--------------------

代码目录：
  /root/wall-x/wall-x-v0

模型目录：
  /root/wall-x/wall-x-model/wall-oss-0.5_v1_ft

模型评测配置：
  /root/wall-x/wall-x-model/wall-oss-0.5_v1_ft/eval_config.yml

Qwen2.5-VL processor：
  /root/wall-x/wall-x-model/Qwen2.5-VL-3B-Instruct

LIBERO/LeRobot 第三方代码：
  /root/wall-x/wall-x-v0/third_party/LIBERO
  /root/wall-x/wall-x-v0/third_party/lerobot

LeRobot LIBERO 数据集：
  /root/dataset/lerobot/libero

Python 环境：
  /usr/local/miniconda3/envs/wallx

LIBERO 配置：
  /root/.libero/config.yaml

模型目录中已经包含 config.json、model.safetensors、norm_stats.json、
tokenizer 和 processor 相关文件。当前评测不需要重新计算 norm stats。


1. 激活环境和设置路径
-----------------------

每打开一个新的终端，先执行：

source /usr/local/miniconda3/bin/activate wallx

export WALLX_ROOT=/root/wall-x/wall-x-v0
export MODEL_ROOT=/root/wall-x/wall-x-model
export CKPT_DIR=$MODEL_ROOT/wall-oss-0.5_v1_ft
export PROCESSOR_DIR=$MODEL_ROOT/Qwen2.5-VL-3B-Instruct
export LIBERO_DATA=/root/dataset/lerobot/libero
export LIBERO_PATH=$WALLX_ROOT/third_party/LIBERO
export EVAL_CONFIG=$CKPT_DIR/eval_config.yml

# 让 Python 和 LIBERO 使用当前代码树中的第三方代码。
export PYTHONPATH=$LIBERO_PATH:$WALLX_ROOT:${PYTHONPATH:-}

# 无显示器服务器上的 MuJoCo EGL 渲染。
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# 固定 LIBERO 配置位置，避免首次启动时出现交互式询问。
export LIBERO_CONFIG_PATH=/root/.libero

cd $WALLX_ROOT


2. 环境和路径检查
------------------

python - <<'PY'
import torch
import lerobot
import mujoco
import robosuite
import bddl
import libero
import wall_x

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("lerobot: OK")
print("mujoco: OK")
print("robosuite: OK")
print("bddl: OK")
print("libero: OK")
print("wall_x: OK")
PY

test -f "$CKPT_DIR/config.json" && echo "model config OK"
test -f "$CKPT_DIR/model.safetensors" && echo "model weights OK"
test -f "$CKPT_DIR/norm_stats.json" && echo "checkpoint norm stats OK"
test -f "$PROCESSOR_DIR/config.json" && echo "processor OK"
test -f "$LIBERO_DATA/meta/info.json" && echo "dataset OK"
test -f "$EVAL_CONFIG" && echo "eval config OK"
test -f "$LIBERO_CONFIG_PATH/config.yaml" && echo "LIBERO config OK"

说明：robosuite 可能打印 private macro warning。这个 warning 不影响当前
LIBERO 评测；如果希望生成 macro 文件，可执行：

python /usr/local/miniconda3/envs/wallx/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py


3. fake inference
-----------------

fake inference 使用随机图像和随机状态，只检查模型能否加载并输出动作，
不代表真实 LIBERO 任务成功率。

CUDA_VISIBLE_DEVICES=0 \
python scripts/fake_inference.py \
  --checkpoint-path "$CKPT_DIR" \
  --train-config-path "$EVAL_CONFIG" \
  --cam-names face_view right_wrist_view \
  --action-horizon 10 \
  --image-size 256 \
  --instruction "pick up the object and place it in the target location" \
  --seed 10222

成功标志：
  Fake inference succeeded.
  action shape: (10, 26)


4. LIBERO 单任务冒烟测试
------------------------

注意：当前脚本的 SMOKE=1 模式会强制 task 0 运行 5 次，
即使设置 NUM_TRIALS_PER_TASK=1 也仍然是 5 次。因此如果只想运行一次，
使用下面的正式命令，不要加 SMOKE=1。

RUN_DIR=/tmp/wallx_libero_check_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"
set -o pipefail

CUDA_ID=0 \
CHECKPOINT_PATH="$CKPT_DIR" \
TRAIN_CONFIG_PATH="$EVAL_CONFIG" \
TASK_SUITE_NAME=libero_spatial \
TASK_INDICES=0 \
NUM_TRIALS_PER_TASK=1 \
LOG_DIR="$RUN_DIR" \
LIBERO_PATH="$LIBERO_PATH" \
LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
bash "$WALLX_ROOT/scripts/run_libero.sh" 2>&1 | tee "$RUN_DIR/run.log"

rc=${PIPESTATUS[0]}
echo "script exit code: $rc"
test -f "$RUN_DIR/report.json" && cat "$RUN_DIR/report.json"
echo "result directory: $RUN_DIR"

成功标志是 report.json 中：
  "attempted": 1


5. 小规模正式评测并保存视频
----------------------------

以下命令运行 libero_spatial 的 task 0、1、2，每个任务 3 次，
并保存 JSON、日志和视频。3 个任务乘以 3 次，报告中应有 attempted=9。

RUN_TAG=$(date +%Y%m%d_%H%M%S)
SAVE_ROOT="$WALLX_ROOT/eval_results/spatial_3trials_$RUN_TAG"
mkdir -p "$SAVE_ROOT/json" "$SAVE_ROOT/videos"
set -o pipefail

CUDA_ID=0 \
CHECKPOINT_PATH="$CKPT_DIR" \
TRAIN_CONFIG_PATH="$EVAL_CONFIG" \
TASK_SUITE_NAME=libero_spatial \
TASK_INDICES=0,1,2 \
NUM_TRIALS_PER_TASK=3 \
LOG_DIR="$SAVE_ROOT/json" \
WALLX_ROLLOUT_DIR="$SAVE_ROOT/videos" \
LIBERO_PATH="$LIBERO_PATH" \
LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
bash "$WALLX_ROOT/scripts/run_libero.sh" 2>&1 | tee "$SAVE_ROOT/run.log"

rc=${PIPESTATUS[0]}
echo "script exit code: $rc"
cat "$SAVE_ROOT/json/report.json"
echo "result directory: $SAVE_ROOT"

视频在：
  $SAVE_ROOT/videos


6. 完整 LIBERO 评测（四个 suite，共 40 个任务）
------------------------------------------------

下面每个任务运行 1 次，共 40 个 episode。这个测试用于检查完整 suite
是否都能启动并完成，不足以得到稳定的统计成功率。若要稳定统计，
将 NUM_TRIALS_PER_TASK 改为 3 或 5，但运行时间会明显增加。

RUN_TAG=$(date +%Y%m%d_%H%M%S)
SAVE_ROOT="$WALLX_ROOT/eval_results/full_libero_$RUN_TAG"
mkdir -p "$SAVE_ROOT/json" "$SAVE_ROOT/videos"
set -o pipefail

ALL_SUITES=1 \
CUDA_ID=0 \
CHECKPOINT_PATH="$CKPT_DIR" \
TRAIN_CONFIG_PATH="$EVAL_CONFIG" \
NUM_TRIALS_PER_TASK=1 \
LOG_DIR="$SAVE_ROOT/json" \
WALLX_ROLLOUT_DIR="$SAVE_ROOT/videos" \
LIBERO_PATH="$LIBERO_PATH" \
LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
bash "$WALLX_ROOT/scripts/run_libero.sh" 2>&1 | tee "$SAVE_ROOT/run.log"

rc=${PIPESTATUS[0]}
echo "script exit code: $rc"
echo "result directory: $SAVE_ROOT"

查看四个 suite 的报告：

for f in "$SAVE_ROOT"/json/*/report.json; do
  echo "===== $f ====="
  cat "$f"
done

四个标准 suite 是：
  libero_spatial
  libero_object
  libero_goal
  libero_10

当前服务器已完成过一次完整 40 任务测试，结果为 21/40，成功率 52.5%。
该结果是每个任务只运行 1 次的测量，不代表最终稳定成功率。


7. 开环 serving
----------------

启动 WebSocket serving：

cd "$WALLX_ROOT"

bash scripts/run_serving.sh \
  --checkpoint-path "$CKPT_DIR" \
  --train-config-path "$EVAL_CONFIG" \
  --cuda-id 0 \
  --port 32195 \
  --env X2ROBOT

如果需要使用实验性推理引擎和 CUDA graph：

bash scripts/run_serving.sh \
  --checkpoint-path "$CKPT_DIR" \
  --train-config-path "$EVAL_CONFIG" \
  --cuda-id 0 \
  --port 32195 \
  --enable-cuda-graph \
  --enable-experimental-engine

serving 会占用当前终端。停止服务按 Ctrl+C。


8. 新模型版本的目录组织
------------------------

当前版本：
  /root/wall-x/wall-x-model/wall-oss-0.5_v1_ft/
    model.safetensors
    config.json
    config.yml
    eval_config.yml
    norm_stats.json
    tokenizer.json
    ...

以后新增模型时，在 wall-x-model 下创建新目录，例如：
  /root/wall-x/wall-x-model/wall-oss-0.5_v2_ft/

然后把对应 checkpoint、norm_stats 和 eval_config.yml 放在该目录中，
只需修改：

export CKPT_DIR=/root/wall-x/wall-x-model/wall-oss-0.5_v2_ft
export EVAL_CONFIG=$CKPT_DIR/eval_config.yml

其余 LIBERO 和运行命令可以继续复用。


9. 常见问题
------------

1) 出现 custom path for the dataset folder? 提示：
   设置 LIBERO_CONFIG_PATH=/root/.libero，并确认
   /root/.libero/config.yaml 存在，然后重新运行。

2) 运行脚本提示 scripts/run_libero.sh 不存在：
   必须先 cd "$WALLX_ROOT"，或直接使用
   bash "$WALLX_ROOT/scripts/run_libero.sh"。

3) report.json 的 attempted=0：
   可能是复用了旧 LOG_DIR，状态文件把旧 episode 标记为已完成。
   使用新的 SAVE_ROOT 或新的 LOG_DIR 重新运行。

4) 只想运行一个 episode：
   不要使用 SMOKE=1；使用第 4 节的正式命令。

5) 模型输出 fake inference 正常，但 LIBERO 成功率低：
   fake inference 只验证推理链路；真实成功率必须通过 LIBERO 闭环评测统计。

6) 日志中出现 transformers 的 FutureWarning 或 rope_parameters warning：
   这些是当前 transformers 与 checkpoint 配置的兼容性提示，当前不影响
   已验证的推理和 LIBERO 运行。


10. 当前复现结论
-----------------

当前服务器已经完成：
  - 环境和核心依赖导入
  - fake inference
  - LIBERO 闭环仿真
  - 四个 suite 的完整 40 任务单次覆盖

当前完整单次覆盖结果：
  - libero_spatial: 6/10
  - libero_object: 9/10
  - libero_goal: 5/10
  - libero_10: 1/10
  - 总计：21/40，52.5%

完整评测结果目录：
  /root/wall-x/wall-x-v0/eval_results/full_libero_20261003_103711

该结果证明代码、模型、数据和 LIBERO 仿真链路已经打通；模型成功率应通过
多次 trial 继续统计，不能只用单次 40 任务结果代表稳定性能。
