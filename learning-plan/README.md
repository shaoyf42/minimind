# MiniMind 学习体系总览

> 基于 minimind 项目和 GitHub issue #504 推荐资源构建的完整学习体系

## 文件结构

```
learning-plan/
├── CLAUDE.md                    # AI 助手指南（Claude Code 自动加载）
├── README.md                    # 本文件：总览与导航
├── 01-skills-checklist.md       # 技能清单：项目涉及的关键技术技能
├── 02-learning-plan.md          # 学习计划：四阶段系统性 LLM 学习路线
├── 03-experiment-design.md      # 实验设计：每个阶段的动手实验练习
├── 04-learning-methodology.md   # 方法指导：如何有效利用 AI 助手辅助学习
├── docs/                        # 学习记录与参考
│   ├── learning_log.md          # 学习日志（按日期，本项目使用）
│   ├── knowledge_base.md        # 知识库（按主题+Q&A，本项目使用）
│   └── ref/                     # 参考归档（源自 minimind-notes，仅供查阅）
│       ├── CLAUDE_ref.md        # 原 AI 助手指南
│       ├── knowledge_base_ref.md # 原知识库（含完整19个Q&A）
│       ├── learning_log_ref.md  # 原学习日志（2025-11 ~ 2026-01）
│       ├── ROADMAP_ref.md       # 原路线图（30min/6h/30h+）
│       └── modules/             # 原模块化教学文档
│           └── 01-foundation/   # 归一化/注意力/前馈网络（teaching/quiz/code_guide）
└── experiments/                 # 实验代码模板目录
    ├── exp_01_rmsnorm.py        # 实验1：归一化对照实验
    ├── exp_02_rope.py           # 实验2：RoPE 位置编码实验
    ├── exp_03_attention.py      # 实验3：注意力机制实验
    ├── exp_04_feedforward.py    # 实验4：前馈网络实验
    ├── exp_05_transformer_block.py  # 实验5：Transformer Block 组装
    ├── exp_06_pretrain.py       # 实验6：预训练实践
    ├── exp_07_sft.py            # 实验7：SFT 微调实践
    ├── exp_08_lora.py           # 实验8：LoRA 微调实践
    ├── exp_09_dpo.py            # 实验9：DPO 偏好优化
    └── exp_10_grpo.py           # 实验10：GRPO 强化学习
```

## 本项目使用文件

| 文档 | 位置 | 用途 |
|------|------|------|
| 学习日志 | [docs/learning_log.md](docs/learning_log.md) | 记录每日学习进度、问题、思考 |
| 知识库 | [docs/knowledge_base.md](docs/knowledge_base.md) | 按主题整理知识点和Q&A（待学习过程中填充） |
| AI助手指南 | [../CLAUDE.md](../CLAUDE.md) | 项目根目录，Claude Code 自动加载（覆盖源码+学习计划） |

## 参考归档（仅供查阅）

| 文档 | 位置 | 说明 |
|------|------|------|
| 原知识库 | [docs/ref/knowledge_base_ref.md](docs/ref/knowledge_base_ref.md) | minimind-notes 完整知识库（含19个Q&A），可作为学习参考 |
| 原学习日志 | [docs/ref/learning_log_ref.md](docs/ref/learning_log_ref.md) | 他人学习历程（2025-11-06 ~ 2026-01-18），可参考经验 |
| 原路线图 | [docs/ref/ROADMAP_ref.md](docs/ref/ROADMAP_ref.md) | 三条路径（30min/6h/30h+），与本项目02-learning-plan互补 |
| 原AI指南 | [docs/ref/CLAUDE_ref.md](docs/ref/CLAUDE_ref.md) | minimind-notes 的 Claude Code 指南 |
| 教学模块 | [docs/ref/modules/](docs/ref/modules/) | 归一化/注意力/前馈网络的 teaching/quiz/code_guide |

## 信息来源

1. **minimind 项目源码**：`/home/shao/workspace/minimind/`
2. **GitHub issue #504 推荐内容合集**：
   - [MiniMind-in-Depth](https://github.com/hans0809/MiniMind-in-Depth) —— 源码逐行解读
   - [happy-llm](https://github.com/datawhalechina/happy-llm) —— DataWhale 大语言模型原理教程
   - [hello-agents](https://github.com/datawhalechina/hello-agents) —— DataWhale 智能体教程
   - [from-minimind-to-more](https://github.com/Tongyun1/from-minimind-to-more) —— 架构算法详解+面试经验
   - [minimind-ascend](https://github.com/fzkun/minimind-ascend) —— 华为昇腾910B复现+Function Calling
   - [MiniMind 文档站](https://minimind.readthedocs.io)
   - [MiniMind 官网](https://jingyaogong.github.io/minimind)

## 快速导航

| 想要... | 阅读 |
|---------|------|
| 了解项目涉及哪些技能 | [01-skills-checklist.md](01-skills-checklist.md) |
| 制定学习路线和时间安排 | [02-learning-plan.md](02-learning-plan.md) |
| 找到具体的实验练习 | [03-experiment-design.md](03-experiment-design.md) |
| 学习如何利用 AI 辅助 | [04-learning-methodology.md](04-learning-methodology.md) |
| 记录学习进度 | [docs/learning_log.md](docs/learning_log.md) |
| 整理知识点 | [docs/knowledge_base.md](docs/knowledge_base.md) |
| 查阅参考知识库 | [docs/ref/knowledge_base_ref.md](docs/ref/knowledge_base_ref.md) |
| 获取实验代码模板 | [experiments/](experiments/) |
