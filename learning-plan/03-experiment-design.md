# MiniMind 实验练习设计

> 针对学习计划每个阶段设计对应的实验练习，紧密结合 minimind 项目实际功能

---

## 实验总览

| 编号 | 实验名称 | 阶段 | 时长 | 所需工具 | 难度 |
|------|---------|------|------|---------|------|
| 01 | 归一化对照实验 | 阶段1 | 30min | PyTorch | ⭐ |
| 02 | RoPE位置编码实验 | 阶段1 | 40min | PyTorch+Matplotlib | ⭐⭐ |
| 03 | 注意力机制实验 | 阶段1 | 50min | PyTorch+Matplotlib | ⭐⭐ |
| 04 | 前馈网络实验 | 阶段1 | 30min | PyTorch | ⭐ |
| 05 | Transformer Block组装 | 阶段1 | 60min | PyTorch | ⭐⭐⭐ |
| 06 | 预训练实践 | 阶段2 | 4h | PyTorch+GPU | ⭐⭐⭐ |
| 07 | SFT微调实践 | 阶段2 | 3h | PyTorch+GPU | ⭐⭐⭐ |
| 08 | LoRA微调实践 | 阶段2 | 2h | PyTorch+GPU | ⭐⭐ |
| 09 | DPO偏好优化 | 阶段3 | 3h | PyTorch+GPU | ⭐⭐⭐ |
| 10 | GRPO强化学习 | 阶段3 | 3h | PyTorch+GPU+Reward Model | ⭐⭐⭐⭐ |

---

## 阶段 1 实验：基础组件

### 实验 01：归一化对照实验

**目标**：通过对照实验理解为什么深层网络需要归一化，以及RMSNorm的优势

**所需工具**：PyTorch、Python 3.9+、无需GPU

**步骤指导**：

1. **梯度消失演示**（10min）
   ```python
   import torch
   torch.manual_seed(42)
   x = torch.randn(1, 512)
   for i in range(8):
       W = torch.randn(512, 512) * 0.5
       x = x @ W
       x = torch.relu(x)
       print(f"Layer {i}: std={x.std().item():.6f}")
   # 预期：标准差从1.04快速衰减到接近0
   ```

2. **RMSNorm效果验证**（10min）
   ```python
   class RMSNorm(torch.nn.Module):
       def __init__(self, dim, eps=1e-5):
           super().__init__()
           self.eps = eps
           self.weight = torch.nn.Parameter(torch.ones(dim))
       def forward(self, x):
           return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

   x = torch.randn(1, 512)
   norm = RMSNorm(512)
   for i in range(8):
       W = torch.randn(512, 512) * 0.5
       x = norm(x)
       x = x @ W
       x = torch.relu(x)
       print(f"Layer {i}: std={x.std().item():.6f}")
   # 预期：标准差保持稳定在1.0附近
   ```

3. **Pre-LN vs Post-Ln对比**（10min）
   - 修改归一化位置，观察训练稳定性差异
   - 参考 `model/model_minimind.py:359-380` 的Pre-Norm实现

**预期成果**：
- 无归一化：标准差从1.04衰减到0.016（梯度消失）
- 有RMSNorm：标准差保持稳定在1.0附近
- Pre-LN比Post-LN在深层网络中更稳定

---

### 实验 02：RoPE位置编码实验

**目标**：理解RoPE如何编码位置信息，以及多频率机制的必要性

**所需工具**：PyTorch、Matplotlib

**步骤指导**：

1. **排列不变性验证**（10min）
   ```python
   import torch
   x1 = torch.randn(1, 3, 8)
   x2 = x1[:, [2, 0, 1], :]
   W_Q = torch.randn(8, 8)
   W_K = torch.randn(8, 8)
   q1, k1 = x1 @ W_Q, x1 @ W_K
   q2, k2 = x2 @ W_Q, x2 @ W_K
   scores1 = (q1 @ k1.transpose(-2, -1))
   scores2 = (q2 @ k2.transpose(-2, -1))
   print("原始:", scores1)
   print("打乱:", scores2)
   # 预期：分数矩阵只是行列顺序改变，值不变
   ```

2. **RoPE旋转效果**（15min）
   ```python
   def precompute_freqs_cis(dim, end, rope_base=1e6):
       freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
       t = torch.arange(end)
       freqs = torch.outer(t, freqs)
       return torch.cos(freqs), torch.sin(freqs)

   freqs_cos, freqs_sin = precompute_freqs_cis(8, 10)
   print("位置0的cos:", freqs_cos[0])
   print("位置1的cos:", freqs_cos[1])
   print("差值:", (freqs_cos[1] - freqs_cos[0]).abs())
   ```

3. **浮点精度问题验证**（15min）
   - 使用超低频率（rope_base=1e12），观察位置0和1的cos值差异
   - 在float32下，差值小于精度下限，导致位置无法区分
   - 这就是为什么需要多频率机制

**预期成果**：
- 验证Attention确实是排列不变的
- 理解RoPE通过旋转编码位置
- 理解多频率是浮点精度限制的工程解决方案

---

### 实验 03：注意力机制实验

**目标**：理解Q/K/V的作用、Multi-Head Attention和GQA

**所需工具**：PyTorch、Matplotlib

**步骤指导**：

1. **Q/K/V直觉理解**（15min）
   ```python
   import torch
   import torch.nn.functional as F

   sentence = ["我", "爱", "编程"]
   x = torch.randn(3, 8)
   W_Q, W_K, W_V = torch.randn(8, 8), torch.randn(8, 8), torch.randn(8, 8)
   Q, K, V = x @ W_Q, x @ W_K, x @ W_V

   scores = Q @ K.T / (8 ** 0.5)
   weights = F.softmax(scores, dim=-1)
   output = weights @ V

   print("注意力权重:")
   for i, word in enumerate(sentence):
       attn = {sentence[j]: f"{weights[i][j]:.3f}" for j in range(3)}
       print(f"  {word} 关注: {attn}")
   ```

2. **Multi-Head Attention实现**（20min）
   - 参考 `model/model_minimind.py:177-220`
   - 实现8头注意力，观察不同头关注的模式
   - 用Matplotlib可视化注意力热力图

3. **GQA效果对比**（15min）
   - 对比 MHA(8Q+8KV) vs GQA(8Q+2KV) 的KV Cache大小
   - GQA: KV Cache内存减少75%，效果损失很小

**预期成果**：
- 理解Q/K/V是同一输入的三个"视角"
- 理解Multi-Head的"多面性"
- 理解GQA的内存-效果权衡

---

### 实验 04：前馈网络实验

**目标**：理解扩张-压缩机制和SwiGLU激活函数

**所需工具**：PyTorch

**步骤指导**：

1. **直接变换 vs 扩张-压缩**（10min）
   ```python
   import torch
   import torch.nn as nn

   x = torch.randn(1, 10, 64)

   # 直接变换
   direct = nn.Linear(64, 64)
   out_direct = direct(x)

   # 扩张-压缩
   up = nn.Linear(64, 256)
   down = nn.Linear(256, 64)
   out_expand = down(torch.relu(up(x)))

   print(f"直接变换输出范数: {out_direct.norm():.4f}")
   print(f"扩张-压缩输出范数: {out_expand.norm():.4f}")
   ```

2. **SwiGLU vs 普通FFN**（10min）
   - 实现 `gate_proj * SiLU(up_proj)` 门控机制
   - 对比普通FFN和SwiGLU在简单任务上的效果

3. **Attention vs FeedForward分工**（10min）
   - Attention：词与词交互（信息交换）
   - FeedForward：每个词独立处理（深度思考）
   - 两者缺一不可

**预期成果**：
- 理解扩张-压缩比直接变换表达力更强
- 理解SwiGLU的门控机制
- 理解Attention和FeedForward的分工

---

### 实验 05：Transformer Block 组装

**目标**：将所有组件组装成完整的Transformer Block

**所需工具**：PyTorch

**步骤指导**：

1. **实现完整的MiniMindBlock**（30min）
   ```python
   class MiniMindBlock(nn.Module):
       def __init__(self, hidden_size=512, num_heads=8, num_kv_heads=2):
           super().__init__()
           self.input_layernorm = RMSNorm(hidden_size)
           self.self_attn = Attention(hidden_size, num_heads, num_kv_heads)
           self.post_attention_layernorm = RMSNorm(hidden_size)
           self.mlp = FeedForward(hidden_size, hidden_size * 4)

       def forward(self, x, position_embeddings):
           residual = x
           x = self.input_layernorm(x)
           x = self.self_attn(x, position_embeddings)
           x = x + residual
           residual = x
           x = self.post_attention_layernorm(x)
           x = self.mlp(x)
           x = x + residual
           return x
   ```

2. **验证数据流正确性**（15min）
   - 输入: `[batch, seq_len, 512]`
   - 输出: `[batch, seq_len, 512]`（维度不变）
   - 检查残差连接是否正确

3. **完整模型组装**（15min）
   - embed_tokens → 8个Block → RMSNorm → lm_head
   - 实现权重共享：`embed_tokens.weight = lm_head.weight`
   - 验证前向传播能正常输出logits

**预期成果**：
- 能从零实现完整的Transformer Block
- 理解Pre-Norm数据流
- 能组装完整模型并运行前向传播

---

## 阶段 2 实验：训练流水线

### 实验 06：预训练实践

**目标**：从零训练一个小型语言模型，理解预训练的完整流程

**所需工具**：PyTorch、GPU（推荐）、SwanLab/WandB

**步骤指导**：

1. **准备数据**（30min）
   ```bash
   cd /home/shao/workspace/minimind/trainer
   # 使用mini数据集快速训练
   python train_pretrain.py \
       --data_path ../dataset/pretrain_t2t_mini.jsonl \
       --epochs 1 \
       --batch_size 16 \
       --hidden_size 512 \
       --num_hidden_layers 8 \
       --max_seq_len 512
   ```

2. **监控训练**（2h）
   - 观察loss曲线：初始~8.9 → 训练后~4.0
   - 理解学习率调度：warmup → cosine decay
   - 使用 `--use_wandb` 可视化

3. **测试模型**（30min）
   ```bash
   python ../eval_llm.py --load_from ./out/pretrain_512.pth
   ```
   - 预训练模型能生成文本，但不会对话

4. **调试练习**（1h）
   - 故意设置过大学习率，观察NaN
   - 减小batch_size解决OOM
   - 练习断点续训：`--from_resume 1`

**预期成果**：
- 成功训练一个预训练模型
- 理解loss曲线和学习率调度
- 能调试常见的训练问题

---

### 实验 07：SFT微调实践

**目标**：将预训练模型微调为能对话的聊天模型

**所需工具**：PyTorch、GPU

**步骤指导**：

1. **理解SFT数据格式**（30min）
   ```json
   {"conversations": [
     {"role": "user", "content": "你好"},
     {"role": "assistant", "content": "你好！有什么可以帮你的？"}
   ]}
   ```
   - 阅读dataset/lm_dataset.py中SFTDataset的loss mask实现

2. **训练SFT模型**（2h）
   ```bash
   cd /home/shao/workspace/minimind/trainer
   python train_full_sft.py \
       --data_path ../dataset/sft_t2t_mini.jsonl \
       --from_weight pretrain \
       --epochs 2 \
       --batch_size 16
   ```

3. **对比效果**（30min）
   - 对比pretrain模型和SFT模型的对话效果
   - SFT模型能进行基本对话，pretrain模型只会续写文本

**预期成果**：
- SFT模型能进行基本对话
- 理解loss mask机制（仅对assistant部分计算loss）
- 理解pretrain→SFT的权重加载

---

### 实验 08：LoRA微调实践

**目标**：使用LoRA在特定领域数据上微调，理解参数高效微调

**所需工具**：PyTorch、GPU

**步骤指导**：

1. **理解LoRA实现**（30min）
   - 阅读 model/model_lora.py
   - 理解低秩分解：`y = W(x) + B(A(x))`
   - 理解apply_lora()如何注入LoRA到Linear层

2. **训练医疗LoRA**（1.5h）
   ```bash
   cd /home/shao/workspace/minimind/trainer
   python train_lora.py \
       --data_path ../dataset/lora_medical.jsonl \
       --from_weight full_sft \
       --lora_r 8 \
       --lora_alpha 16
   ```

3. **测试LoRA效果**（30min）
   ```bash
   python ../eval_llm.py --weight full_sft --lora_weight lora_medical
   ```
   - 对比base模型和base+lora_medical在医疗问题上的回答

**预期成果**：
- 理解LoRA的数学原理和实现
- 成功训练领域特定的LoRA
- 理解LoRA叠加机制

---

## 阶段 3 实验：进阶主题

### 实验 09：DPO偏好优化

**目标**：使用DPO让模型学会区分好回答和坏回答

**所需工具**：PyTorch、GPU

**步骤指导**：

1. **理解DPO数据格式**（30min）
   ```json
   {
     "chosen": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "good answer"}],
     "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "bad answer"}]
   }
   ```

2. **阅读DPO实现**（30min）
   - 阅读 trainer/train_dpo.py
   - 理解DPO loss：使用冻结参考模型计算log概率比率
   - 理解beta参数的作用（KL惩罚系数）

3. **训练DPO模型**（2h）
   ```bash
   cd /home/shao/workspace/minimind/trainer
   python train_dpo.py \
       --data_path ../dataset/dpo.jsonl \
       --from_weight full_sft \
       --beta 0.1
   ```

**预期成果**：
- 理解DPO的核心思想：直接优化偏好，无需训练Reward Model
- 成功训练DPO模型
- 对比SFT和DPO的回答质量

---

### 实验 10：GRPO强化学习

**目标**：使用GRPO进行强化学习训练，理解RLAIF流程

**所需工具**：PyTorch、GPU、Reward Model（internlm2-1_8b-reward）

**步骤指导**：

1. **前置准备**（1h）
   - 下载Reward Model：`git clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm2-1_8b-reward.git`
   - 学习李宏毅强化学习课程第1-2节

2. **理解GRPO原理**（1h）
   - 阅读 trainer/train_grpo.py
   - 理解组内标准化优势（无需Critic模型）
   - 对比PPO（需要Critic）vs GRPO（不需要Critic）

3. **运行GRPO训练**（3h）
   ```bash
   cd /home/shao/workspace/minimind/trainer
   python train_grpo.py \
       --data_path ../dataset/rlaif-mini.jsonl \
       --from_weight dpo \
       --reward_model_path ../../internlm2-1_8b-reward
   ```

4. **效果对比**（30min）
   - 对比SFT → DPO → GRPO的渐进效果
   - 观察GRPO训练曲线的稳定性

**预期成果**：
- 理解GRPO的核心优势：无需Critic，训练更简单
- 成功运行GRPO训练
- 理解RLHF/RLAIF的完整流程

---

## 实验记录模板

每个实验完成后，使用以下模板记录到 [docs/learning_log.md](docs/learning_log.md)：

```markdown
### YYYY-MM-DD：实验XX - 实验名称

#### 实验目标
[简述实验要验证什么]

#### 实验环境
- 硬件：[CPU/GPU型号]
- 数据集：[使用的数据集]
- 模型配置：[hidden_size/layers/heads等]

#### 实验结果
| 配置 | 指标1 | 指标2 |
|------|-------|-------|
| 基线 | xxx | xxx |
| 改进 | xxx | xxx |

#### 关键发现
1. [发现1]
2. [发现2]

#### 遇到的问题
- [问题描述] → [解决方案]

#### 下一步
- [基于实验结果的后续计划]
```

---

## 实验代码模板

所有实验的代码模板位于 `learning-plan/experiments/` 目录，每个文件包含：
- 实验目标说明
- 完整可运行的代码
- 预期输出和解释
- 扩展练习建议

使用方式：
```bash
cd /home/shao/workspace/minimind/learning-plan/experiments
python exp_01_rmsnorm.py
```
