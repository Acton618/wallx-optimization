1.环境确认:以下指令用于确认环境内核心包导入是否正常，如果正常报告版本，说明环境内核心包可正常导入
python -c "import torch; print(torch.__version__)"
python -c "import lerobot; print('lerobot ok')"
python -c "import mujoco; print('mujoco ok')"
python -c "import robosuite; print('robosuite ok')"

2.参数配置：相当于配置当前终端下的全局变量，配置后，在执行指令时，可以避免频繁做长路径编辑，直接修改全局配置export即可

##下面的指令配置的是主要的wallox路径，修改为自己文件夹下的wall路径，具体如何修改参考我当前使用的指令以及我的路径层级
export WALLX_ROOT=/opt/data/zhuoziying/wall-x

##下面的指令配置的是使用的模型路径，我们统一使用wall-x-model的v1版本0.5模型以及Qwen2.5
export CKPT_DIR=/opt/data/zhuoziying/wall-x-model/wall-oss-0.5_v1
export PROCESSOR_DIR=/opt/data/zhuoziying/wall-x-model/Qwen2.5-VL-3B-Instruct

##下面的指令配置的是使用的数据集，我们统一使用libero数据集
export LIBERO_DATA=/opt/data/zhuoziying/dataset/lerobot/libero

##下面的指令配置的是使用的config文件，修改为自己文件夹的对应同名文件
export EVAL_CONFIG=/opt/data/zhuoziying/wall-x/workspace/example/libero_v0.yml

与参数验证，在设置完全局变量之后，可以test验证是否正常
test -f "$CKPT_DIR/config.json" && echo "model config OK"
test -f "$CKPT_DIR/model.safetensors" && echo "model weights OK"
test -d "$PROCESSOR_DIR" && echo "processor OK"
test -d "$LIBERO_DATA" && echo "dataset OK"
test -f "$EVAL_CONFIG" && echo "eval config OK"

##下面的指令跑了会报错，正常的，因为我们的norm-state没有放在数据集文件夹下面，我们放在了模型文件夹下面
test -f "$LIBERO_DATA/norm_stats_libero.json" && echo "norm stats OK"    

3.norm_stats：以下指令用于生成对应的norm-state文件，如果没有自己改数据集或者微调需求，不用重跑
python scripts/compute_norm_stats.py \
  --train_config "$EVAL_CONFIG" \
  --data_root "$LIBERO_DATA" \
  --output_path "$LIBERO_DATA/norm_stats_libero.json"


4.fake_inference：假推一轮，看看是否报错，导入是否正常
cd /opt/data/zhuoziying/wall-x

CUDA_VISIBLE_DEVICES=0 \
python scripts/fake_inference.py \
  --checkpoint-path "$CKPT_DIR" \
  --train-config-path "$EVAL_CONFIG" \
  --cam-names face_view right_wrist_view \
  --action-horizon 10 \
  --image-size 256 \
  --instruction "pick up the object and place it in the target location" \
  --seed 10222


5.闭环推理：跑正式的实验代码

##冒烟推理测试：相当于假推一次，不正式跑模型，只是看看有没有报错
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd /opt/data/zhuoziying/wall-x
SMOKE=1 \
CUDA_ID=0 \
CHECKPOINT_PATH="$CKPT_DIR" \
TRAIN_CONFIG_PATH="$EVAL_CONFIG" \
TASK_SUITE_NAME=libero_spatial \
TASK_INDICES=0 \
bash scripts/run_libero.sh 2>&1 | tee libero_smoke.log


##正式推理测试
CUDA_ID=0 \
CHECKPOINT_PATH="$CKPT_DIR" \
TRAIN_CONFIG_PATH="$EVAL_CONFIG" \
TASK_SUITE_NAME=libero_spatial \
TASK_INDICES=0,1,2 \
NUM_TRIALS_PER_TASK=5 \
bash scripts/run_libero.sh 2>&1 | tee libero_spatial_debug.log


##带结果保存的正式推理测试
*提供一条测试指令，可以将对应的输出结果导出到wall-x文件夹下/opt/data/zhuoziying/wall-x/eval_videos/libero_spatial

################################################################################################################################
##以下指令是进入对应工作区，目前是进入我的工作区，修改为自己的工作区
cd /opt/data/zhuoziying/wall-x || exit 1

RUN_TAG=$(date +%Y%m%d_%H%M%S)

##以下指令是设置对应的保存位置，目前是保存在我的路径，需要修改为自己的保存路径
SAVE_ROOT="/opt/data/zhuoziying/wall-x/eval_results/$RUN_TAG"

VIDEO_DIR="$SAVE_ROOT/videos"
LOG_DIR_RUN="$SAVE_ROOT/json"
mkdir -p "$VIDEO_DIR" "$LOG_DIR_RUN"
set -o pipefail
PYTHONUNBUFFERED=1 \
CUDA_ID=1 \                                           ##设置使用的GPU id，在终端中，可以通过nvidia-smi查看目前的GPU使用情况

TASK_SUITE_NAME=libero_spatial \                      ##设置测试任务，一共五个任务
TASK_INDICES=0 \                                  ##设置测试任务的编号，libero_spatial下一共10个任务，这次跑0、1、2三个任务
NUM_TRIALS_PER_TASK=5 \                               ##跑单个任务的次数，这里设置是5次

CHECKPOINT_PATH="$CKPT_DIR" \
TRAIN_CONFIG_PATH="$EVAL_CONFIG" \
LOG_DIR="$LOG_DIR_RUN" \
WALLX_ROLLOUT_DIR="$VIDEO_DIR" \
bash scripts/run_libero.sh 2>&1 | tee "$SAVE_ROOT/run.log"
rc=(${PIPESTATUS[@]})
echo "run_libero exit code: ${rc[0]}"
echo "本次结果目录：$SAVE_ROOT"

**********特别关注，上述指令中TASK_SUITE_NAME等等设置会覆盖执行脚本run_libero.sh内的配置，也可以不在指令中设置，直接进脚本配置
################################################################################################################################



6.开环评测
cd /opt/data/zhuoziying/wall-x

export ENABLE_CUDA_GRAPH=True
export ENABLE_EXPERIMENTAL_INFERENCE_ENGINE=True

bash scripts/run_serving.sh \
  --checkpoint-path "$CKPT_DIR" \
  --train-config-path "$EVAL_CONFIG" \
  --port 32195

















cd /opt/data/zhuoziying/wall-x || exit 1
RUN_TAG=$(date +%Y%m%d_%H%M%S)
SAVE_ROOT="/opt/data/zhuoziying/wall-x/eval_results_v2/$RUN_TAG"
VIDEO_DIR="$SAVE_ROOT/videos"
LOG_DIR_RUN="$SAVE_ROOT/json"
mkdir -p "$VIDEO _DIR" "$LOG_DIR_RUN"
set -o pipefail
PYTHONUNBUFFERED=1 \
CUDA_ID=1 \
CHECKPOINT_PATH="$CKPT_DIR" \
TRAIN_CONFIG_PATH="$EVAL_CONFIG" \
LOG_DIR="$LOG_DIR_RUN" \
WALLX_ROLLOUT_DIR="$VIDEO_DIR" \
TASK_SUITE_NAME=libero_object \
TASK_INDICES=0 \
NUM_TRIALS_PER_TASK=1 \
bash scripts/run_libero.sh 2>&1 | tee "$SAVE_ROOT/run.log"
rc=(${PIPESTATUS[@]})
echo "run_libero exit code: ${rc[0]}"
echo "本次结果目录：$SAVE_ROOT"




export WALLX_ROOT=/opt/data/zhuoziying/wall-x
export CKPT_DIR=/opt/data/zhuoziying/wall-x-model/wall-oss-0.5_v1
export PROCESSOR_DIR=/opt/data/zhuoziying/wall-x-model/Qwen2.5-VL-3B-Instruct
export LIBERO_DATA=/opt/data/zhuoziying/dataset/lerobot/libero
export EVAL_CONFIG=/opt/data/zhuoziying/wall-x/workspace/example/libero_v1.yml



export WALLX_ROOT=/opt/data/zhuoziying/wall-x
export CKPT_DIR=/opt/data/zhuoziying/wall-x-model/wall-oss-0.5_v2
export PROCESSOR_DIR=/opt/data/zhuoziying/wall-x-model/Qwen2.5-VL-3B-Instruct
export LIBERO_DATA=/opt/data/zhuoziying/dataset/lerobot/libero
export EVAL_CONFIG=/opt/data/zhuoziying/wall-x/workspace/example/libero_v2.yml



git tag vis-atten-get