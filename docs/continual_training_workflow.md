# Continual Training Multi-Template Workflow

## 概述

本文档描述了支持多模板的continual training完整实验流程，包括数据生成、SFT训练、continual训练等各个阶段。

## 1. 数据生成阶段

### 统一模板数据生成

使用新的统一数据生成脚本，根据模型类型自动选择对应模板：

```bash
# 为Llama instruct模型生成数据
python experiments/continual/data_gen_unified_template.py \
    --model_type llama \
    --group 0 \
    --split train \
    --num_samples 1000 \
    --output_dir /nas/shared/sys2/yuanhangli/tmp/continual_unified_data/llama

# 为Qwen instruct模型生成数据
python experiments/continual/data_gen_unified_template.py \
    --model_type qwen \
    --group 0 \
    --split train \
    --num_samples 1000 \
    --output_dir /nas/shared/sys2/yuanhangli/tmp/continual_unified_data/qwen
```

### 批量生成示例

```bash
# 使用提供的批量生成脚本
bash experiments/continual/generate_unified_data_example.sh
```

### 数据组配置

- **Group 0**: `["+", "-", "*"]` + 4个数字
- **Group 1**: `["+", "*"]` + 5个数字  
- **Group 2**: `["-", "*"]` + 6个数字

### 关键改进

- **统一字段名**: 所有数据集使用 `prompt` 和 `response` 字段
- **自动模板选择**: 根据 `model_type` 参数自动选择对应instruct模板
- **无需额外数据集类**: 现有 `RLHFDataset` 可直接处理

## 2. SFT训练阶段

确保已有SFT检查点：
- Llama: `/nas/shared/sys2/yuanhangli/tmp/llama_instruct_sft_model/global_step_15`
- Qwen: `/nas/shared/sys2/yuanhangli/tmp/qwen_instruct_sft_model/global_step_15`

## 3. Continual训练阶段

### 单数据集多组curriculum训练

```bash
# Llama模型continual训练
bash reset_scripts_llama/train_continual_countdown_3b_curriculum_after_sft_qwen.sh

# Qwen模型continual训练
bash reset_scripts_qwen/train_continual_countdown_3b_curriculum_after_sft_qwen.sh
```

### 跨数据集continual训练

```bash
# Countdown → GSM8K 跨数据集学习
bash reset_scripts_llama/train_continual_cross_dataset_llama.sh
```

## 4. 配置文件

### PPO训练配置 (统一字段名)

- `verl/trainer/config/ppo_trainer_llama.yaml` - Llama instruct模型
- `verl/trainer/config/ppo_trainer_qwen.yaml` - Qwen instruct模型
- `verl/trainer/config/ppo_trainer_continual_llama.yaml` - 跨数据集continual训练

所有配置文件都使用统一字段名：
```yaml
data:
  prompt_key: prompt
  response_key: response
```

## 5. 实验类型

### A. 单数据集多组curriculum

**特点**:
- 在同一数据集内按组递进训练
- 每个组使用不同的运算符组合和数字数量
- 保持优化器状态连续性

**执行**:
```bash
bash reset_scripts_llama/train_continual_countdown_3b_curriculum_after_sft_qwen.sh
```

### B. 跨数据集continual学习

**特点**:
- Phase 1: Countdown数据集训练
- Phase 2: GSM8K数据集训练  
- 自动处理不同数据集的模板差异
- 保持优化器连续性

**执行**:
```bash
bash reset_scripts_llama/train_continual_cross_dataset_llama.sh
```

## 6. 环境变量配置

### 关键环境变量

```bash
# GPU配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=4
export ROLLOUT_TP_SIZE=1

# 检查点路径
export BASE_MODEL=/nas/shared/sys2/yuanhangli/tmp/llama_instruct_sft_model/global_step_15
export CHECKPOINT_BASE_DIR=/nas/shared/sys2/yuanhangli/tmp/checkpoints/continual_llama

# 注意力日志 (Qwen模型需要禁用)
export ATTENTION_LOGGING_ENABLED=false  # 对于Qwen模型

# Layer Reset (可选)
export LAYER_RESET_ENABLE=false
export LAYER_RESET_K_FIRST=2
export LAYER_RESET_K_LAST=2
```

## 7. 监控和日志

### 日志文件位置

- 训练日志: `./llama_logs/` 或 `./qwen_logs/`
- 实验日志: `./llama_logs/experiments/` 
- 主日志: `./llama_logs/Continual_Master.log`

### WandB集成

```bash
export WANDB_MODE=offline  # 或 online
```

## 8. 故障排除

### 常见问题

1. **Qwen注意力日志错误**:
   ```bash
   export ATTENTION_LOGGING_ENABLED=false
   ```

2. **数据字段不匹配**:
   - 确保使用统一数据生成脚本
   - 检查PPO配置中的 `prompt_key` 和 `response_key`

3. **检查点路径错误**:
   - 验证 `BASE_MODEL` 环境变量指向正确的SFT检查点
   - 确保检查点目录存在且可访问

## 9. 下一步开发

- [ ] 测试完整的SFT→continual训练流程
- [ ] 为其他reset_scripts配置多模板支持
- [ ] 添加更多数据集支持(IFEval, DeepMath等)

## 10. 文件结构

```
ContinualCountdown/
├── experiments/continual/
│   ├── data_gen_unified_template.py      # 统一模板数据生成
│   └── generate_unified_data_example.sh  # 批量生成示例
├── reset_scripts_llama/
│   ├── train_continual_countdown_3b_curriculum_after_sft_qwen.sh
│   └── train_continual_cross_dataset_llama.sh
├── reset_scripts_qwen/
│   └── train_continual_countdown_3b_curriculum_after_sft_qwen.sh
└── verl/trainer/config/
    ├── ppo_trainer_llama.yaml
    ├── ppo_trainer_qwen.yaml
    └── ppo_trainer_continual_llama.yaml
```

这个工作流程确保了多模板continual training的无缝集成，支持不同模型类型和数据集的灵活切换。
