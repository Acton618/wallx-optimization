# Wall-X DaChuang v1 使用说明
#
# 本文档对应的代码目录：
#   /root/wall-x/wall-x-dachuang-v1
#
# 这是基于 wall-x-v0 建立的开发版本。它不复制第三方代码；运行时共享
# v0 中的 LIBERO 和 LeRobot 目录。模型 checkpoint、数据集和 Python 环境
# 也使用服务器上已经验证过的路径。


一、目录和环境
==============

代码：
  /root/wall-x/wall-x-dachuang-v1

稳定版本代码：
  /root/wall-x/wall-x-v0

共享第三方代码：
  /root/wall-x/wall-x-v0/third_party/LIBERO
  /root/wall-x/wall-x-v0/third_party/lerobot

模型 checkpoint 和评测配置：
  /root/wall-x/wall-x-model/wall-oss-0.5_v1_ft

LIBERO LeRobot 数据集：
  /root/dataset/lerobot/libero

Python 环境：
  /usr/local/miniconda3/envs/wallx

LIBERO 全局配置：
  /root/.libero/config.yaml


二、激活环境并设置路径
========================

每次打开新的终端，先执行：

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate wallx

export WALLX_ROOT=/root/wall-x/wall-x-dachuang-v1
export CKPT_DIR=/root/wall-x/wall-x-model/wall-oss-0.5_v1_ft
export PROCESSOR_DIR="$CKPT_DIR"
export EVAL_CONFIG="$CKPT_DIR/eval_config.yml"
export LIBERO_DATA=/root/dataset/lerobot/libero
export LIBERO_PATH=/root/wall-x/wall-x-v0/third_party/LIBERO
export LIBERO_CONFIG_PATH=/root/.libero
export PYTHONPATH="$LIBERO_PATH:$WALLX_ROOT:${PYTHONPATH:-}"

cd "$WALLX_ROOT"

说明：
  1. 当前 eval_config.yml 已经写入上述绝对路径，因此可直接使用。
  2. 如果以后复制代码到新的目录，只需要修改 WALLX_ROOT，并重新执行一次
     pip 的 editable 安装命令（见第九节）。
  3. 不要把 checkpoint 或数据集复制到代码目录；它们由多个代码版本共享。


三、确认关键文件
================

执行：

test -f "$CKPT_DIR/config.json" && echo "model config OK"
test -f "$CKPT_DIR/model.safetensors" && echo "model weights OK"
test -f "$CKPT_DIR/norm_stats.json" && echo "checkpoint norm stats OK"
test -f "$CKPT_DIR/preprocessor_config.json" && echo "processor OK"
test -f "$EVAL_CONFIG" && echo "eval config OK"
test -f "$LIBERO_DATA/meta/info.json" && echo "dataset OK"
test -f "$LIBERO_CONFIG_PATH/config.yaml" && echo "LIBERO config OK"
test -d "$LIBERO_PATH/libero" && echo "LIBERO source OK"

模型目录中的 eval_config.yml、config.json、model.safetensors、norm_stats.json
和 processor 文件必须来自同一套 checkpoint。评测时不需要在数据集目录下再
生成 norm_stats_libero.json；当前配置直接使用 checkpoint 中的 norm_stats.json。


四、检查 Python 包和 GPU
=========================

python - <<'PY'
import torch
import lerobot
import mujoco
import robosuite
import bddl
import wall_x

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("lerobot: OK")
print("mujoco: OK")
print("robosuite: OK")
print("bddl: OK")
print("wall_x:", wall_x.__file__)
PY

最后一行应指向：
  /root/wall-x/wall-x-dachuang-v1/wall_x/__init__.py

robosuite 的 private macro warning 不影响当前 LIBERO 评测。如果需要生成
私有 macro 文件，可以执行：

python /usr/local/miniconda3/envs/wallx/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py


五、模型假推理测试
====================

该测试使用随机图像和随机状态，只检查 checkpoint 能否加载并输出动作，
不代表真实 LIBERO 成功率。

mkdir -p "$WALLX_ROOT/eval_results"
RUN_DIR="$WALLX_ROOT/eval_results/fake_inference_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0 \
python "$WALLX_ROOT/scripts/fake_inference.py" \
  --checkpoint-path "$CKPT_DIR" \
  --train-config-path "$EVAL_CONFIG" \
  --cam-names face_view right_wrist_view \
  --action-horizon 10 \
  --image-size 256 \
  --instruction "pick up the object and place it in the target location" \
  --seed 10222 2>&1 | tee "$RUN_DIR/run.log"

成功标志：
  Fake inference succeeded.
  action shape: (10, 26)


六、LIBERO 单任务快速验证
==========================

这是最适合检查环境、模型和仿真链路的命令：libero_spatial 的 task 0 跑 1 次。
结果会保存在 wall-x-dachuang-v1/eval_results 下。

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

RUN_TAG=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$WALLX_ROOT/eval_results/libero_spatial_task0_$RUN_TAG"
mkdir -p "$RUN_DIR/json" "$RUN_DIR/videos"
set -o pipefail

CUDA_ID=0 \
CHECKPOINT_PATH="$CKPT_DIR" \
TRAIN_CONFIG_PATH="$EVAL_CONFIG" \
TASK_SUITE_NAME=libero_spatial \
TASK_INDICES=0 \
NUM_TRIALS_PER_TASK=1 \
LOG_DIR="$RUN_DIR/json" \
WALLX_ROLLOUT_DIR="$RUN_DIR/videos" \
bash "$WALLX_ROOT/scripts/run_libero.sh" 2>&1 | tee "$RUN_DIR/run.log"

rc=${PIPESTATUS[0]}


七、Predictor 迁移（默认关闭）
===============================

Predictor 使用当前 wall-x 0.5 视觉塔第 8 层的 token 特征，训练一个两层
MLP 去拟合当前 VisPruner 第 31 层的注意力教师分数。旧 wall-x 版本的权重
不直接复用，因为视觉塔和教师选择规则已经变化。代码开关如下，默认值均为
关闭或空值：

  predictor_enable: false
  predictor_checkpoint_path: null
  predictor_early_layer: 16
  predictor_keep_num: 40

先在一个单独的训练配置副本中加入：

  vispruner_enable: true
  vispruner_keep_num: 20
  predictor_enable: true
  predictor_checkpoint_path: /绝对路径/token_score_predictor.pt
  predictor_early_layer: 16
  predictor_keep_num: 40

采集教师数据（数据集是本地 LeRobot 3.0 时脚本直接读取 parquet 和视频）：

  PYTHONPATH="$WALLX_ROOT:$WALLX_ROOT/scripts:${PYTHONPATH:-}" \
  python "$WALLX_ROOT/scripts/collect_predictor_teacher.py" \
    --checkpoint-path "$CKPT_DIR" \
    --train-config-path /绝对路径/teacher_config.yml \
    --dataset-root "$LIBERO_DATA" \
    --image-key observation.images.image observation.images.image2 \
    --num-samples 1000 \
    --output-path "$WALLX_ROOT/eval_results/predictor_teacher.pt"

训练 MLP（正式实验建议使用覆盖多个任务和相机的样本）：

  PYTHONPATH="$WALLX_ROOT:${PYTHONPATH:-}" \
  python "$WALLX_ROOT/scripts/train_predictor.py" \
    --teacher-path "$WALLX_ROOT/eval_results/predictor_teacher.pt" \
    --output-path "$WALLX_ROOT/eval_results/token_score_predictor.pt" \
    --device cuda:0 --epochs 20

训练后的 checkpoint 会检查输入维度、特征来源和 early layer，配置不匹配时
会直接报错。假推理成功标志仍是 `Fake inference succeeded`，LIBERO 仿真
视频位于对应 `eval_results/predictor_task0_*/videos/` 目录。
echo "script exit code: $rc"
test -f "$RUN_DIR/json/report.json" && cat "$RUN_DIR/json/report.json"
echo "result directory: $RUN_DIR"

成功标志：
  script exit code 为 0；
  report.json 存在；
  overall.attempted 为 1。

注意：不要在该命令中加 SMOKE=1。当前脚本的 SMOKE 模式可能强制重复执行
task 0；如果只需要一次试验，使用上面的正式单任务命令。


八、小规模正式评测（推荐）
============================

下面运行 libero_spatial 的 task 0、1、2，每个任务 3 次，共 9 次，并保存
JSON 报告、运行日志和视频。运行时间取决于 GPU 和任务执行速度。

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
bash "$WALLX_ROOT/scripts/run_libero.sh" 2>&1 | tee "$SAVE_ROOT/run.log"

rc=${PIPESTATUS[0]}
echo "run_libero exit code: $rc"
cat "$SAVE_ROOT/json/report.json"
echo "result directory: $SAVE_ROOT"

正常情况下 report.json 中：
  overall.attempted = 9
  overall.successes = 实际成功次数
  overall.success_rate = successes / 9

TASK_SUITE_NAME、TASK_INDICES、NUM_TRIALS_PER_TASK、CUDA_ID 等环境变量会
覆盖脚本中的默认值，可以直接在命令前设置，不必修改 run_libero.sh。


九、完整 LIBERO 评测
======================

以下命令运行四个标准 suite 的全部任务。它用于最终评估，耗时较长，结果会
按 suite 保存在一个新的结果目录中。

RUN_TAG=$(date +%Y%m%d_%H%M%S)
SAVE_ROOT="$WALLX_ROOT/eval_results/full_libero_$RUN_TAG"
mkdir -p "$SAVE_ROOT"
set -o pipefail

CUDA_ID=0 \
CHECKPOINT_PATH="$CKPT_DIR" \
TRAIN_CONFIG_PATH="$EVAL_CONFIG" \
ALL_SUITES=1 \
NUM_TRIALS_PER_TASK=1 \
ROLLOUT_BASE="$SAVE_ROOT" \
bash "$WALLX_ROOT/scripts/run_libero.sh" 2>&1 | tee "$SAVE_ROOT/run.log"

rc=${PIPESTATUS[0]}
echo "run_libero exit code: $rc"
find "$SAVE_ROOT" -name report.json -print
echo "result directory: $SAVE_ROOT"

四个标准 suite 为：
  libero_spatial、libero_object、libero_goal、libero_10


十、开环 serving 推理
======================

如果需要启动 Wall-X serving 服务：

export ENABLE_CUDA_GRAPH=True
export ENABLE_EXPERIMENTAL_INFERENCE_ENGINE=True

cd "$WALLX_ROOT"
bash scripts/run_serving.sh \
  --checkpoint-path "$CKPT_DIR" \
  --train-config-path "$EVAL_CONFIG" \
  --port 32195

服务启动后保持该终端运行。端口可按需要更换，但客户端必须使用相同端口。


十一、开发版本与稳定版本切换
============================

当前 wallx 环境应安装为新版本的 editable package。检查：

python -c "import wall_x; print(wall_x.__file__)"

应显示：
  /root/wall-x/wall-x-dachuang-v1/wall_x/...

如果要切回稳定版 v0：

/usr/local/miniconda3/envs/wallx/bin/python -m pip install \
  --no-deps --no-build-isolation -e /root/wall-x/wall-x-v0

如果要再次切回 DaChuang v1：

/usr/local/miniconda3/envs/wallx/bin/python -m pip install \
  --no-deps --no-build-isolation -e /root/wall-x/wall-x-dachuang-v1

新版本 Git 分支为 dachuang-v1。修改代码后可查看：

cd "$WALLX_ROOT"
git status
git log --oneline --decorate -5

测试结果 eval_results 默认未纳入版本提交；如果需要保留结果记录，可以手动
选择对应的 report.json、run.log 或 summary 文件提交。


十二、常见问题
================

1. 提示 scripts/run_libero.sh 不存在
   必须先执行 cd "$WALLX_ROOT"，或者使用绝对路径：
   bash "$WALLX_ROOT/scripts/run_libero.sh"

2. report.json 不存在
   先检查脚本退出码和 run.log。通常是路径变量未设置、checkpoint 文件缺失，
   或 LIBERO_PATH 没有指向 /root/wall-x/wall-x-v0/third_party/LIBERO。

3. 只想运行一次却执行了多次
   使用第六节的命令，并设置 NUM_TRIALS_PER_TASK=1；不要使用 SMOKE=1。

4. 出现 transformers 的 FutureWarning、rope_parameters warning 或 gym warning
   这些是当前依赖版本的提示。只要脚本最后退出码为 0 且 report.json 正常生成，
   就不影响本次推理结果。

5. 修改第三方代码
   当前 v1 共享 v0 的 third_party。若以后要改 LIBERO 或 LeRobot，先把对应
   目录复制到 v1 的 third_party 下，再调整 LIBERO_PATH、PYTHONPATH，并在该
   目录重新安装需要的 editable package；普通 Wall-X 模型代码修改不需要复制
   third_party。
