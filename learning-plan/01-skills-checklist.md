# MiniMind 项目技能清单

> 分析 minimind 项目的核心功能、技术栈和架构设计，提炼关键技术技能

---

## 一、LLM 核心原理技能

### 1.1 归一化机制

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| 梯度消失/爆炸 | 理解深层网络为何难训练，8层网络标准差从1.04衰减到0.016 | `modules/01-foundation/01-normalization/` | ⭐⭐⭐ |
| RMSNorm 实现 | `x_norm = x / sqrt(mean(x²) + eps) * weight`，只做缩放不减均值 | `model/model_minimind.py:95-105` | ⭐⭐⭐ |
| Pre-LN vs Post-LN | Pre-LN残差路径更干净，深层网络更稳定，所有现代LLM使用Pre-LN | `model/model_minimind.py:359-380` | ⭐⭐⭐ |
| 数值稳定性 | bfloat16半精度下RMSNorm比LayerNorm更稳定 | `model/model_minimind.py:103` `.type_as(x)` | ⭐⭐ |

**核心认知**：归一化不是丢失信息，只是控制数值规模

### 1.2 位置编码

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| 排列不变性 | Attention的`Q@K^T`只看向量不看位置，"我喜欢你"和"你喜欢我"无法区分 | knowledge_base.md §2.1 | ⭐⭐⭐ |
| RoPE旋转编码 | 用旋转角度编码位置，相对位置=相对旋转角度 | `model/model_minimind.py:108-128` | ⭐⭐⭐ |
| 多频率机制 | 32个频率组合：高频编码局部(每6.3 token转一圈)，低频编码全局(每6,283,185 token) | `model/model_minimind.py:110-115` | ⭐⭐⭐ |
| 浮点精度限制 | 超低频率相邻位置差(10^-11) < float32精度(10^-7)，导致位置0和1无法区分 | knowledge_base.md Q11 | ⭐⭐ |
| YaRN长文本外推 | factor=16，可从2048外推至32768 | `model/model_minimind.py:116-128` | ⭐⭐ |

**核心认知**：RoPE多频率是数学理论与计算机硬件约束的完美平衡

### 1.3 注意力机制

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| Q/K/V计算 | 同一输入通过三个权重矩阵变换得到三个"视角" | `model/model_minimind.py:159-161` | ⭐⭐⭐ |
| Scaled Dot-Product | `scores = Q @ K^T / sqrt(d_k)` → softmax → weighted sum | `model/model_minimind.py:205-218` | ⭐⭐⭐ |
| Multi-Head Attention | 8个头并行，每个学不同模式(语法/语义/情感/指代) | `model/model_minimind.py:177-220` | ⭐⭐⭐ |
| GQA(Grouped Query Attention) | 8个Q头共享2个KV头，KV Cache内存减少75% | `model/model_minimind.py:155-157` | ⭐⭐⭐ |
| Causal Mask | 语言模型只能看到"过去"的词，下三角掩码 | `model/model_minimind.py:207-210` | ⭐⭐⭐ |
| Flash Attention | 自动检测`F.scaled_dot_product_attention`，大幅加速 | `model/model_minimind.py:205` | ⭐⭐ |

**核心认知**：Attention让模型知道"哪些词相关"，但不知道"如何处理"

### 1.4 前馈网络

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| 扩张-压缩结构 | 768 → 2048 → 768，在高维空间增强表达能力 | `model/model_minimind.py:225-238` | ⭐⭐⭐ |
| SwiGLU激活 | `output = down_proj(SiLU(gate_proj(x)) * up_proj(x))`，三个投影矩阵 | `model/model_minimind.py:232-237` | ⭐⭐⭐ |
| 门控机制 | gate分支控制up分支的信息流，动态选择性传递 | `model/model_minimind.py:235` | ⭐⭐ |
| SiLU激活 | `SiLU(x) = x * sigmoid(x)`，平滑、非单调、梯度更稳定 | `model/model_minimind.py:235` | ⭐⭐ |

**核心认知**：Attention = 开会讨论(信息交换)，FeedForward = 各自思考(深度处理)

### 1.5 Transformer 架构

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| Decoder-Only结构 | 类似GPT/Llama，只有解码器部分 | `model/model_minimind.py:383-471` | ⭐⭐⭐ |
| 残差连接 | `y = x + F(x)`，保底机制+梯度高速公路 | `model/model_minimind.py:370-378` | ⭐⭐⭐ |
| 权重共享 | embed_tokens和lm_head共享权重矩阵，省一半内存(4.9M参数) | `model/model_minimind.py:447` | ⭐⭐⭐ |
| 三层架构 | ForCausalLM(接口层) → MiniMindModel(核心层) → lm_head(输出层) | `model/model_minimind.py:383-471` | ⭐⭐⭐ |
| 自回归生成 | 一个词一个词生成，KV Cache加速增量推理 | `model/model_minimind.py:452-471` | ⭐⭐⭐ |

### 1.6 混合专家模型(MoE)

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| MoE门控路由 | Top-K路由选择专家，scoring_func='softmax' | `model/model_minimind.py:241-280` | ⭐⭐ |
| 共享专家+路由专家 | n_routed_experts=4 + n_shared_experts=1 | `model/model_minimind.py:281-356` | ⭐⭐ |
| 负载均衡损失 | aux_loss_alpha=0.01，避免专家利用不均 | `model/model_minimind.py:273-278` | ⭐⭐ |

---

## 二、模型训练技能

### 2.1 分词器

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| BPE算法 | 字节对编码，迭代合并高频子词对 | `trainer/train_tokenizer.py` | ⭐⭐⭐ |
| 词表构建 | vocab_size=6400，特殊token设计 | `model/tokenizer.json` | ⭐⭐⭐ |
| Chat Template | ChatML格式，Jinja2模板，支持Tool Calling和Reasoning标签 | `model/tokenizer_config.json` | ⭐⭐⭐ |

### 2.2 预训练

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| Next-token prediction | Causal LM目标，预测下一个token | `trainer/train_pretrain.py` | ⭐⭐⭐ |
| DDP多卡训练 | `torchrun --nproc_per_node N`，分布式数据并行 | `trainer/trainer_utils.py` | ⭐⭐⭐ |
| 混合精度训练 | bfloat16/float16自动混合精度 | `trainer/train_pretrain.py` autocast | ⭐⭐⭐ |
| 余弦学习率调度 | `get_lr()`函数，warmup+cosine decay | `trainer/trainer_utils.py` | ⭐⭐⭐ |
| 断点续训 | `lm_checkpoint()`保存/恢复，SkipBatchSampler跳过已训练batch | `trainer/trainer_utils.py` | ⭐⭐ |

### 2.3 监督微调(SFT)

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| Loss Mask | 仅对assistant回复部分计算loss，通过bos/eos标记定位 | `dataset/lm_dataset.py:SFTDataset` | ⭐⭐⭐ |
| 多轮对话处理 | chat_template格式化，支持system/user/assistant角色 | `dataset/lm_dataset.py` | ⭐⭐⭐ |
| 权重加载 | `--from_weight pretrain`加载预训练权重继续训练 | `trainer/train_full_sft.py` | ⭐⭐⭐ |

### 2.4 LoRA微调

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| 低秩分解 | `y = W(x) + B(A(x))`，A高斯初始化，B零初始化 | `model/model_lora.py:LoRA` | ⭐⭐⭐ |
| 参数冻结 | 冻结非LoRA参数，仅训练LoRA参数 | `trainer/train_lora.py` | ⭐⭐⭐ |
| 领域适配 | 医疗/身份等特定领域数据微调 | `dataset/lora_medical.jsonl` | ⭐⭐ |

### 2.5 DPO偏好优化

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| DPO Loss | 从零实现，使用冻结参考模型计算log概率比率 | `trainer/train_dpo.py` | ⭐⭐ |
| Chosen/Rejected对 | 偏好数据格式，good answer vs bad answer | `dataset/dpo.jsonl` | ⭐⭐ |

### 2.6 强化学习(PPO/GRPO/SPO)

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| PPO | Actor+Critic+Old Actor+Reference Model+Reward Model | `trainer/train_ppo.py` | ⭐⭐ |
| GRPO | 无需Critic，组内标准化优势，每个prompt生成多个回复 | `trainer/train_grpo.py` | ⭐⭐ |
| SPO | 自适应价值追踪器(Beta分布)，无需Critic | `trainer/train_spo.py` | ⭐⭐ |
| Reward Model | 外部internlm2-1_8b-reward提供奖励信号 | `trainer/train_ppo.py` | ⭐⭐ |

### 2.7 知识蒸馏

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| 白盒蒸馏 | KL散度蒸馏损失，alpha*CE + (1-alpha)*Distill | `trainer/train_distillation.py` | ⭐⭐ |
| 推理模型蒸馏 | 对`<think/>`/`<answer/>`标记加权loss(10倍)，R1风格 | `trainer/train_reason.py` | ⭐⭐ |

---

## 三、推理与优化技能

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| KV Cache | past_key_value缓存，增量推理避免重复计算 | `model/model_minimind.py` Attention.forward | ⭐⭐⭐ |
| Flash Attention | `F.scaled_dot_product_attention`自动检测 | `model/model_minimind.py:205` | ⭐⭐ |
| 模型格式转换 | PyTorch ↔ Transformers ↔ Llama格式 | `scripts/convert_model.py` | ⭐⭐ |
| 生态兼容 | llama.cpp / vllm / ollama推理引擎 | 项目README | ⭐⭐ |
| 流式输出 | 逐token生成并实时返回 | `eval_llm.py` / `scripts/serve_openai_api.py` | ⭐⭐ |

---

## 四、工程与部署技能

| 技能 | 描述 | 项目代码位置 | 重要度 |
|------|------|------------|--------|
| OpenAI API兼容 | FastAPI+uvicorn，流式/非流式响应 | `scripts/serve_openai_api.py` | ⭐⭐ |
| Streamlit Web UI | 聊天界面，思考链折叠展示 | `scripts/web_demo.py` | ⭐ |
| 训练监控 | SwanLab/WandB集成，训练曲线可视化 | `trainer/trainer_utils.py` | ⭐⭐ |
| 数据处理 | JSONL格式、SimHash/MinHash去重 | `dataset/lm_dataset.py` | ⭐⭐ |
| 命令行推理 | eval_llm.py，自动测试/手动输入/流式输出 | `eval_llm.py` | ⭐⭐ |

---

## 五、数学与理论基础

| 领域 | 核心概念 | 在项目中的应用 | 重要度 |
|------|---------|--------------|--------|
| 线性代数 | 矩阵乘法、向量点积、旋转矩阵 | Q@K^T、RoPE旋转、embed_tokens | ⭐⭐⭐ |
| 概率论 | Softmax、交叉熵损失、KL散度 | Attention权重、训练损失、蒸馏 | ⭐⭐⭐ |
| 优化理论 | 余弦退火、梯度累积、AdamW | 学习率调度、大batch模拟 | ⭐⭐⭐ |
| 信息论 | 困惑度(Perplexity)、信息熵 | 模型评估指标 | ⭐⭐ |
| 浮点数精度 | float32/bfloat16精度限制、数值稳定性 | RoPE多频率设计、RMSNorm半精度 | ⭐⭐ |

---

## 六、技能依赖关系图

```
基础数学（线性代数/概率论/优化）
    ↓
┌───────────────────────────────────────┐
│ Tier 1: 基础组件（必须按顺序学习）      │
│ 1. RMSNorm → 2. RoPE → 3. Attention  │
│                → 4. FeedForward       │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Tier 2: 架构组装                       │
│ 5. Transformer Block（残差+Pre-Norm） │
│ 6. 完整模型（三层架构+权重共享）        │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Tier 3: 训练流程                       │
│ 7. Tokenizer → 8. Pretrain → 9. SFT  │
│              → 10. LoRA               │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Tier 4: 进阶主题                       │
│ 11. DPO → 12. PPO/GRPO/SPO           │
│ 13. 蒸馏 → 14. MoE                    │
│ 15. 推理优化 → 16. 部署服务            │
└───────────────────────────────────────┘
```

---

## 七、技能与岗位对应

| 目标岗位 | 必须掌握(Tier 1-2) | 核心掌握(Tier 3) | 加分项(Tier 4) |
|---------|-------------------|-----------------|---------------|
| LLM训练工程师 | 全部 | Pretrain/SFT/LoRA | DPO/GRPO/蒸馏 |
| 推理优化工程师 | 全部 | - | KV Cache/Flash Attention/量化/格式转换 |
| NLP算法工程师 | 全部 | 全部 | MoE/RL/蒸馏 |
| AI应用开发工程师 | 架构理解 | SFT/LoRA | API服务/Web UI |
| 大模型面试准备 | 全部+数学 | 全部 | 全部(至少理解原理) |
