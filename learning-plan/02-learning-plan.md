# MiniMind 系统性 LLM 学习计划

> 结合 minimind 项目和 GitHub issue #504 推荐资源制定

---

## 计划总览

| 阶段 | 时长 | 目标 | 理论:实践 | 核心产出 |
|------|------|------|----------|---------|
| 阶段0: 环境准备 | 1-2天 | 搭建环境，跑通模型 | 1:4 | 可运行的推理环境 |
| 阶段1: 基础组件 | 1-2周 | 吃透Transformer每个组件 | 2:3 | 手写实现所有组件 |
| 阶段2: 训练流水线 | 2-3周 | 完整训练pretrain→SFT→LoRA | 1:3 | 可对话的小模型 |
| 阶段3: 进阶主题 | 2-3周 | 掌握RL/蒸馏/MoE | 2:3 | RLHF模型+推理模型 |

**总时长**：6-9周（每天投入3-4小时）

---

## 阶段 0：环境准备与快速体验（1-2 天）

### 目标
搭建开发环境，跑通模型推理，通过3个快速实验建立直觉

### Day 1：环境搭建 + 模型推理（4h）

| 时间 | 内容 | 操作 |
|------|------|------|
| 1h | 克隆项目、安装依赖 | `pip install -r requirements.txt` |
| 1h | 下载模型权重 | `git clone https://huggingface.co/jingyaogong/MiniMind2` |
| 1h | 运行命令行推理 | `python eval_llm.py --load_from ./MiniMind2` |
| 1h | 启动Web UI和API服务 | `streamlit run scripts/web_demo.py` / `python scripts/serve_openai_api.py` |

**常见问题预案**（来自 [docs/learning_log.md](docs/learning_log.md) 经验）：
- Python 3.13 兼容性：将 `requirements.txt` 中 `==` 改为 `>=`
- Git LFS 大文件：`brew install git-lfs && git lfs pull`
- 多Python环境：始终使用虚拟环境 `python3 -m venv venv`

### Day 2：快速实验建立直觉（4h）

| 时间 | 内容 | 操作 |
|------|------|------|
| 1.5h | 实验1：为什么需要归一化？ | `python experiments/exp_01_rmsnorm.py` |
| 1.5h | 实验2：为什么用RoPE？ | `python experiments/exp_02_rope.py` |
| 1h | 实验3：Attention如何工作？ | `python experiments/exp_03_attention.py` |

### 完成标准
- [ ] 模型能正常对话
- [ ] Web UI 和 API 服务正常运行
- [ ] 3个快速实验全部跑通并理解结果

### 推荐资源
- [docs/ROADMAP.md](docs/ROADMAP.md) 的「快速体验」路径
- [docs/learning_log.md](docs/learning_log.md) 2025-11-06 的环境搭建经验
- [MiniMind 文档站](https://minimind.readthedocs.io)

---

## 阶段 1：Transformer 基础组件深度学习（1-2 周）

### 目标
逐个吃透Transformer的每个组件，不仅知道"是什么"，更理解"为什么这样设计"

### 学习方法论
遵循**对照实验驱动**方法论（详见 [docs/ROADMAP.md](docs/ROADMAP.md)）：
```
实验 → 直觉 → 理论 → 代码
10分钟 → 20分钟 → 30分钟 → 10分钟
```

### 1.1 归一化模块（2-3 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 1h | 阅读 [docs/knowledge_base.md](docs/knowledge_base.md) §1 归一化章节 | 3:1 |
| D1 | 2h | 运行对照实验：NoNorm vs Post-LN vs Pre-LN+RMSNorm | 1:3 |
| D2 | 1h | 阅读 model/model_minimind.py:95-105 源码 | 2:1 |
| D2 | 1h | 从零手写 RMSNorm 实现 | 1:2 |
| D3 | 1h | 完成自测题 | 2:1 |

**必须能回答的问题**：
1. 8层网络无归一化时，标准差从1.04衰减到多少？→ 0.016（梯度消失）
2. RMSNorm比LayerNorm快多少？→ 7.7倍
3. 为什么现代LLM都用Pre-LN？→ 残差路径更干净，深层网络更稳定
4. RMSNorm省略减均值为什么可行？→ 深度网络激活值通常已接近零均值

### 1.2 位置编码模块（2-3 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 1.5h | 阅读 [docs/knowledge_base.md](docs/knowledge_base.md) §2 位置编码章节 | 3:1 |
| D1 | 2h | 运行RoPE实验：基础原理+多频率+浮点精度 | 1:3 |
| D2 | 1h | 阅读 model/model_minimind.py:108-137 源码 | 2:1 |
| D2 | 1h | 手写RoPE预计算和应用函数 | 1:2 |
| D3 | 0.5h | 完成自测 | 2:1 |

**必须能回答的问题**：
1. Attention为什么是"排列不变"的？→ Q@K^T只看向量不看位置
2. 为什么需要32个频率而不是一个超低频率？→ float32精度限制（差值10^-11 < 精度10^-7）
3. RoPE只旋转Q和K不旋转V，为什么？→ 位置信息只影响"匹配度"，不影响"内容"
4. RoPE是否丢失了绝对位置信息？→ 没有！每个词被旋转到特定角度

### 1.3 注意力机制模块（3-4 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 2h | 阅读 [docs/knowledge_base.md](docs/knowledge_base.md) §3 注意力章节 | 3:1 |
| D1 | 2h | 运行Attention实验：QKV+Multi-Head+可视化 | 1:3 |
| D2 | 1.5h | 阅读 model/model_minimind.py:139-220 源码 | 2:1 |
| D2 | 1.5h | 手写Multi-Head Attention + GQA实现 | 1:2 |
| D3 | 1h | 深入理解Causal Mask和Flash Attention | 2:1 |
| D4 | 1h | 完成自测 | 2:1 |

**必须能回答的问题**：
1. Q/K/V的数据库查询类比？→ Query(搜索条件)/Key(索引标签)/Value(数据值)
2. GQA如何减少KV Cache？→ 8个Q头共享2个KV头，内存减少75%
3. Causal Mask的作用？→ 语言模型只能看到"过去"的词
4. Multi-Head的"多面性"是什么？→ 就像用8副不同的眼镜看同一句话

### 1.4 前馈网络 + 架构组装（2-3 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 1h | 阅读 [docs/knowledge_base.md](docs/knowledge_base.md) §4-5 前馈网络+架构章节 | 3:1 |
| D1 | 1.5h | 运行FeedForward实验+理解SwiGLU | 1:2 |
| D2 | 1.5h | 阅读 model/model_minimind.py:225-380 源码 | 2:1 |
| D2 | 1.5h | 手写完整Transformer Block（含残差连接） | 1:2 |
| D3 | 1h | 理解完整三层架构+权重共享 | 2:1 |

**必须能回答的问题**：
1. 为什么"扩张-压缩"比直接768→768好？→ 高维空间有更多自由度做非线性变换
2. 残差连接的"保底机制"？→ `y = x + F(x)`，即使F学不到东西，至少还有x
3. 权重共享省多少参数？→ embed_tokens和lm_head共享，省一半(4.9M)
4. Pre-Norm vs Post-Norm？→ Pre-Norm梯度流更平滑，所有现代LLM使用

### 阶段1推荐外部资源

| 资源 | 用途 | 链接 |
|------|------|------|
| happy-llm | 理论补充，系统化LLM原理 | https://github.com/datawhalechina/happy-llm |
| MiniMind-in-Depth | 源码逐行解读 | https://github.com/hans0809/MiniMind-in-Depth |
| from-minimind-to-more | 架构算法详解+面试经验 | https://github.com/Tongyun1/from-minimind-to-more |
| 3Blue1Brown Transformer可视化 | 直觉理解 | YouTube: Attention in transformers, visually explained |

### 阶段1完成标准
- [ ] 能从零手写 RMSNorm、RoPE、Multi-Head Attention、SwiGLU
- [ ] 能画出Pre-LN Transformer Block完整数据流图
- [ ] 通过各模块的自测题（参考 [docs/ref/modules/](docs/ref/modules/) 目录下的 quiz.md）
- [ ] 所有关键问题都能清晰回答

---

## 阶段 2：完整训练流水线实践（2-3 周）

### 目标
从Tokenizer到SFT/LoRA，完整走一遍训练流程，获得"从零训练一个LLM"的实战经验

### 2.1 数据准备与Tokenizer（3-4 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 2h | 学习BPE算法原理，阅读 trainer/train_tokenizer.py | 2:1 |
| D1 | 2h | 运行Tokenizer训练，理解词表构建过程 | 1:3 |
| D2 | 2h | 阅读 dataset/lm_dataset.py，理解4种数据集类 | 2:1 |
| D2 | 2h | 理解数据格式：Pretrain/SFT/DPO/RLAIF | 2:1 |
| D3 | 2h | 准备自己的小数据集（个人文档/特定领域文本） | 1:3 |
| D4 | 2h | 数据清洗和格式转换实践 | 1:3 |

**推荐资源**：
- CS336课程的BPE作业（issue #504推荐）
- dataset/dataset.md 数据集详细说明

### 2.2 预训练实践（3-4 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 2h | 阅读 trainer/train_pretrain.py 源码，理解训练循环 | 2:1 |
| D1 | 1h | 理解训练工具：init_model/get_lr/lm_checkpoint | 2:1 |
| D2 | 4h | 使用 pretrain_t2t_mini.jsonl 训练Small模型(512维,8层) | 1:4 |
| D3 | 2h | 监控训练曲线(loss/lr)，理解余弦学习率调度 | 2:1 |
| D3 | 2h | 使用 eval_llm.py 测试预训练模型效果 | 1:3 |
| D4 | 2h | 调试常见问题(NaN/OOM)并记录解决方案 | 1:2 |

**关键学习点**：
- Causal Language Modeling目标函数
- 混合精度训练(bfloat16)
- 断点续训机制(`--from_resume 1`)
- 常见问题：NaN loss→学习率过大；OOM→减小batch_size

### 2.3 SFT监督微调（2-3 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 1.5h | 阅读 trainer/train_full_sft.py，理解loss mask机制 | 2:1 |
| D1 | 3h | 使用 sft_t2t_mini.jsonl 进行SFT训练 | 1:3 |
| D2 | 1.5h | 对比pretrain vs SFT模型的对话效果 | 1:2 |
| D2 | 1.5h | 理解chat_template格式化和多轮对话处理 | 2:1 |

**关键学习点**：
- SFT仅对assistant回复部分计算loss
- chat_template格式化
- pretrain → SFT的权重加载

### 2.4 LoRA微调（2-3 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 1.5h | 阅读 model/model_lora.py，理解从零实现的LoRA | 2:1 |
| D1 | 2h | 使用 lora_medical.jsonl 训练医疗领域LoRA | 1:3 |
| D2 | 1h | 测试LoRA叠加效果(base + lora_medical) | 1:2 |
| D2 | 1h | 对比不同rank/alpha参数的效果 | 1:2 |

### 阶段2完成标准
- [ ] 成功训练 pretrain → SFT → LoRA 完整流水线
- [ ] 预训练模型困惑度 < 5.0（Small模型）
- [ ] SFT模型能进行基本对话
- [ ] LoRA能在特定领域表现提升

---

## 阶段 3：进阶主题与强化学习（2-3 周）

### 目标
掌握RLHF/RLAIF、蒸馏、MoE等进阶技术

### 3.1 DPO直接偏好优化（2-3 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 2h | 学习DPO论文核心思想 | 3:1 |
| D1 | 2h | 阅读 trainer/train_dpo.py，理解从零实现的DPO loss | 2:1 |
| D2 | 3h | 使用 dpo.jsonl 训练DPO模型 | 1:3 |
| D3 | 1h | 对比SFT vs DPO模型的回答质量 | 1:2 |

### 3.2 PPO/GRPO强化学习（3-4 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 3h | 学习李宏毅强化学习课程(策略梯度/PPO原理) | 4:1 |
| D2 | 2h | 阅读 trainer/train_grpo.py，理解组内标准化优势 | 2:1 |
| D2 | 2h | 阅读 trainer/train_ppo.py，理解Actor-Critic架构 | 2:1 |
| D3 | 3h | 运行GRPO训练(无需Critic，比PPO更简单) | 1:3 |
| D4 | 2h | 对比DPO vs GRPO的效果和训练稳定性 | 1:2 |

**推荐资源**：
- 李宏毅强化学习课程（issue #504多次推荐）
- [hello-agents](https://github.com/datawhalechina/hello-agents)（DataWhale智能体教程）

### 3.3 知识蒸馏与推理模型（2-3 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 1.5h | 阅读 trainer/train_distillation.py，理解白盒蒸馏 | 2:1 |
| D1 | 1.5h | 阅读 trainer/train_reason.py，理解R1风格推理蒸馏 | 2:1 |
| D2 | 2h | 运行推理模型测试，观察思考链输出 | 1:2 |
| D2 | 1h | 理解推理模型的loss加权(思考标签10倍权重) | 2:1 |

### 3.4 MoE混合专家模型（2 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 2h | 阅读 model/model_minimind.py:241-356，理解MoE架构 | 2:1 |
| D1 | 2h | 理解门控路由、共享专家、负载均衡损失 | 2:1 |
| D2 | 2h | 对比Dense vs MoE模型的参数量和效果 | 1:2 |

### 3.5 推理优化与部署（2 天）

| 天 | 时间 | 内容 | 理论:实践 |
|----|------|------|----------|
| D1 | 2h | 理解KV Cache原理，对比有/无Cache的推理速度 | 2:1 |
| D1 | 2h | 模型格式转换(PyTorch→Llama→ollama) | 1:3 |
| D2 | 2h | 部署OpenAI API兼容服务 | 1:3 |
| D2 | 1h | 使用vllm/ollama加速推理 | 1:2 |

### 阶段3完成标准
- [ ] 成功训练DPO和GRPO模型
- [ ] 理解PPO/GRPO/SPO的区别和适用场景
- [ ] 能解释推理模型的训练流程和思考链机制
- [ ] 理解MoE架构和负载均衡
- [ ] 能部署API服务并使用第三方推理引擎

---

## 学习节奏建议

### 每日时间分配（3-4小时/天）

| 时间段 | 内容 | 时长 |
|--------|------|------|
| 上午 | 理论学习（阅读文档/论文/源码） | 1.5h |
| 下午 | 动手实践（运行实验/训练模型） | 2h |
| 晚上 | 总结记录（更新笔记/复习） | 0.5h |

### 每周节奏

| 日 | 内容 |
|----|------|
| 周一-周五 | 按计划学习新内容 |
| 周六 | 复习本周内容，完成自测题 |
| 周日 | 自由探索，阅读扩展资源 |

### 学习记录规范

每完成一个学习阶段，更新以下文件：
1. **[docs/learning_log.md](docs/learning_log.md)**：添加日期条目，记录完成事项、问题、思考
2. **[docs/knowledge_base.md](docs/knowledge_base.md)**：添加新知识点和Q&A
3. **Git提交**：`git commit -m "学习 XXX 原理"`

---

## 时间安排总览

```
第1周   │ 阶段0(2天) + 阶段1.1归一化(3天) + 阶段1.2位置编码(2天)
第2周   │ 阶段1.2位置编码(1天) + 阶段1.3注意力(4天) + 复习(2天)
第3周   │ 阶段1.4前馈网络+架构(3天) + 阶段2.1数据准备(4天)
第4周   │ 阶段2.2预训练(4天) + 阶段2.3 SFT(3天)
第5周   │ 阶段2.4 LoRA(3天) + 阶段3.1 DPO(4天)
第6周   │ 阶段3.2 PPO/GRPO(4天) + 阶段3.3蒸馏(3天)
第7周   │ 阶段3.4 MoE(2天) + 阶段3.5部署(2天) + 总结(3天)
第8-9周 │ 自由探索：自定义数据集训练/参加Kaggle/面试准备
```

**弹性调整**：
- 如果某阶段耗时超预期，不要跳过，慢下来吃透
- 阶段3的各子模块相对独立，可根据兴趣调整顺序
- 有GPU条件时优先做训练实践，无GPU时侧重源码阅读和理论
