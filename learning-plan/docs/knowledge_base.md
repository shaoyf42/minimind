# MiniMind 知识库

> 系统化整理 LLM 学习过程中的知识点、概念解释和问答记录

---

## 📑 目录

1. [归一化机制](#1-归一化机制)
2. [位置编码](#2-位置编码)
3. [注意力机制](#3-注意力机制)
4. [前馈网络](#4-前馈网络)
5. [Transformer 架构](#5-transformer-架构)
6. [训练流程](#6-训练流程)
7. [问答记录](#问答记录)

---

## 1. 归一化机制

### 核心问题
深层网络为什么难训练？

### 关键知识点
- 梯度消失：8层网络标准差从1.04衰减到0.016
- 梯度爆炸：数值溢出变成NaN
- RMSNorm：`x_norm = x / sqrt(mean(x²) + eps) * weight`
- Pre-LN vs Post-LN：Pre-LN残差路径更干净

### 对照实验结果
| 配置 | 是否收敛 | 最终 Loss |
|------|---------|-----------|
| NoNorm | 否 | NaN |
| Post-LN | 是 | ~3.5 |
| Pre-LN + RMSNorm | 是 | ~2.7 |

### 待补充
- [ ] 自己运行实验后的观察
- [ ] 对RMSNorm数学原理的理解

---

## 2. 位置编码

### 核心问题
Attention是"排列不变"的，如何让模型知道词序？

### 关键知识点
- RoPE：用旋转角度编码位置，相对位置=相对旋转角度
- 多频率机制：32个频率，高频编码局部，低频编码全局
- 浮点精度限制：超低频率差值(10^-11) < float32精度(10^-7)
- YaRN长文本外推：factor=16，2048→32768

### 待补充
- [ ] 自己运行RoPE实验后的观察
- [ ] 对多频率机制的直觉理解

---

## 3. 注意力机制

### 核心问题
如何让模型理解词与词之间的关系？

### 关键知识点
- Q/K/V：同一输入的三个"视角"
- GQA：8个Q头共享2个KV头，KV Cache减少75%
- Causal Mask：语言模型只能看到"过去"的词
- Flash Attention：`F.scaled_dot_product_attention`

### 待补充
- [ ] 自己运行Attention实验后的观察
- [ ] 对GQA权衡的理解

---

## 4. 前馈网络

### 核心问题
FFN在Transformer中的作用是什么？

### 关键知识点
- 扩张-压缩：768 → 2048 → 768
- SwiGLU：`down_proj(SiLU(gate_proj(x)) * up_proj(x))`
- 门控机制：gate分支控制up分支的信息流
- Attention vs FeedForward：开会讨论 vs 各自思考

### 待补充
- [ ] 自己运行FeedForward实验后的观察

---

## 5. Transformer 架构

### 核心问题
如何将所有组件组装成完整模型？

### 关键知识点
- 三层架构：ForCausalLM → MiniMindModel → lm_head
- 权重共享：embed_tokens和lm_head共享，省一半参数(4.9M)
- Pre-Norm：`x = x + Attention(Norm(x))`
- 自回归生成：一个词一个词生成

### 待补充
- [ ] 自己组装Transformer Block的体会

---

## 6. 训练流程

### 核心问题
如何从零训练一个LLM？

### 关键知识点
- 训练流水线：Tokenizer → Pretrain → SFT → DPO → GRPO
- Loss Mask：SFT仅对assistant部分计算loss
- LoRA：`y = W(x) + B(A(x))`，低秩分解
- DPO：直接偏好优化，无需Reward Model
- GRPO：组内标准化优势，无需Critic

### 待补充
- [ ] 自己训练模型后的经验和教训

---

## 问答记录

### Q1: 为什么RoPE需要多频率？
**A**: 主要原因是float32精度限制。超低频率的相邻位置差值小于精度下限，导致位置0和1无法区分。多频率是数学理论与硬件约束的平衡。

### Q2: RMSNorm比LayerNorm快多少？
**A**: 约7.7倍。RMSNorm省略了减均值操作，只有weight无bias，计算更简单。

### Q3: GQA如何减少KV Cache？
**A**: 8个Q头共享2个KV头（4:1压缩比），KV Cache内存减少75%，效果损失很小。

### 待补充
- [ ] 学习过程中产生的新问题和解答
