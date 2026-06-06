# Meteorite Identification

陨石图像二分类系统，结合 DINOv2 + LoRA 视觉模型与 GPT-4o-mini 大语言模型进行融合推理。

## 项目结构

```
meteorite-identification-G15/
├── fusion_llm_lora.py              # 主程序：LoRA + LLM 融合分类
├── sample_submission.csv           # 提交模板
├── dinov2算法/                      # DINOv2 LoRA 训练管道
│   ├── train_dinov2_lora_v2.py    # 模型训练脚本
│   ├── dinov2_lora_v2_detail.csv   # LoRA 预测概率
│   └── model_checkpoints/          # 训练好的模型权重
└── outputs/                        # 输出结果
    ├── submission_fusion.csv       # 最终提交文件
    └── submission_fusion_detail.csv # 详细融合结果
```

## 技术方案

### 模型架构

| 阶段 | 模型 | 说明 |
|------|------|------|
| 主分类器 | DINOv2 (ViT-B/14) + LoRA | 冻结骨干网络，LoRA 微调适配陨石分类 |
| 不确定样本 | GPT-4o-mini | 分析图像中的陨石特征（fusion crust、regmaglypts、金属颗粒、球粒） |

### 决策策略

```
LoRA 概率 < 0.1  → 直接判定为负
LoRA 概率 > 0.9  → 直接判定为正
LoRA 概率 0.20-0.85 → 调用 GPT-4o-mini，置信度 >= 0.5 判定为正
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 图像尺寸 | 518 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| 目标 blocks | 最后 4 层 |
| 学习率 | 1e-4 |
| Label smoothing | 0.05 |
| Epochs | 15 |
| 验证策略 | 5-fold CV + TTA (水平翻转) |

## 环境依赖

```bash
pip install torch torchvision openai opencv-python pillow numpy pandas scikit-learn tqdm
```

## 运行方式

### 1. 训练 LoRA 模型

```bash
cd dinov2算法
python train_dinov2_lora_v2.py --data-root .. --epochs 15 --folds 5
```

### 2. 运行融合分类

```bash
export OPENAI_API_KEY='your-api-key'
python fusion_llm_lora.py
```

### 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `OPENAI_API_KEY` | 是 | - | OpenAI API 密钥 |
| `OPENAI_BASE_URL` | 否 | OpenAI 官方 | API 端点 |
| `OPENAI_MODEL` | 否 | gpt-4o-mini | 模型名称 |
| `LLM_MAX_WORKERS` | 否 | 8 | 并发 LLM 请求数 |

## 输出文件

- `outputs/submission_fusion.csv` — 最终提交文件
- `outputs/submission_fusion_detail.csv` — 含 LoRA 概率、LLM 置信度、最终标签的详细结果