DATA_DIR="./data/collabllm-math-hard-large"
TRAIN_PQ="${DATA_DIR}/train.parquet"
VAL_PQ="${DATA_DIR}/test.parquet"

REWARD_PY="./customized/my_RM_reward.py"
REWARD_FN="compute_score"

MODEL_PATH="/home/ma-user/work/models/Qwen/Qwen3-8B"


export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
source /usr/local/Ascend/nnal/atb/set_env.sh

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_PQ}" \
  data.val_files="${VAL_PQ}" \
  data.prompt_key=prompt \
  data.train_batch_size=16 \
  data.max_prompt_length=5120 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 \
  custom_reward_function.path="${REWARD_PY}" \
  custom_reward_function.name="${REWARD_FN}" \
  trainer.project_name='verl_grpo_collab_math' \
  trainer.experiment_name='qwen3_8b_grpo_baseline_my_RM_reward' \
  trainer.logger=console \
  trainer.critic_warmup=0 \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.device=npu
