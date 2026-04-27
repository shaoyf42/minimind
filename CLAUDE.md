# MiniMind AI 助手指南

> 本文件位于 minimind 项目根目录，Claude Code 会自动加载此文件。
> 涵盖项目源码和学习计划两部分，辅助 LLM 学习。

---

## 项目结构

```
minimind/
├── CLAUDE.md                    # 本文件：AI 助手指南（自动加载）
├── eval_llm.py                  # 推理/对话入口
├── model/                       # 模型定义
│   ├── model_minimind.py        # 核心模型（RMSNorm/RoPE/Attention/FFN/MoE/TransformerBlock）
│   ├── model_lora.py            # LoRA 低秩适配器
│   └── tokenizer.json           # BPE 分词器
├── trainer/                     # 训练脚本
│   ├── train_pretrain.py        # 预训练
│   ├── train_full_sft.py        # 监督微调
│   ├── train_lora.py            # LoRA 微调
│   ├── train_dpo.py             # DPO 偏好优化
│   ├── train_ppo.py             # PPO
│   ├── train_grpo.py            # GRPO
│   ├── train_spo.py             # SPO
│   ├── train_reason.py          # 推理模型蒸馏
│   ├── train_distillation.py    # 白盒知识蒸馏
│   ├── train_tokenizer.py       # Tokenizer 训练
│   └── trainer_utils.py         # 训练工具函数
├── dataset/                     # 数据集
│   ├── lm_dataset.py            # 4种数据集类（Pretrain/SFT/DPO/RLAIF）
│   └── *.jsonl                  # 训练数据文件
├── scripts/                     # 工具脚本
│   ├── web_demo.py              # Streamlit Web UI
│   ├── serve_openai_api.py      # OpenAI API 兼容服务
│   ├── chat_openai_api.py       # API 客户端
│   └── convert_model.py         # 模型格式转换
├── model_weight/                # 预训练权重
├── learning-plan/               # 学习计划体系
│   ├── README.md                # 总览与导航
│   ├── 01-skills-checklist.md   # 技能清单
│   ├── 02-learning-plan.md      # 四阶段学习计划
│   ├── 03-experiment-design.md  # 实验设计
│   ├── 04-learning-methodology.md # 方法指导
│   ├── docs/
│   │   ├── learning_log.md      # 学习日志（本项目使用）
│   │   ├── knowledge_base.md    # 知识库（本项目使用）
│   │   └── ref/                 # 参考归档（仅供查阅）
│   └── experiments/             # 实验代码模板
│       ├── exp_01_rmsnorm.py ~ exp_10_grpo.py
```

## 学习笔记维护

### 笔记系统

| 文件 | 用途 | 更新时机 |
|------|------|---------|
| `learning-plan/docs/learning_log.md` | 按日期的学习记录 | 每次学习会话结束 |
| `learning-plan/docs/knowledge_base.md` | 按主题的知识点和Q&A | 每学新知识点 |

### 更新 learning_log.md

每次学习会话结束，添加格式如下的条目：

```markdown
### YYYY-MM-DD：学习主题

#### ✅ 完成事项
- [x] 完成的项目1

#### 🐛 问题与解决方案
**问题**：描述
**解决方案**：描述

#### 💭 思考与收获
- 收获1

#### 📋 下次计划
- 计划1
```

### 更新 knowledge_base.md

学完一个知识点后：
1. 在对应章节补充"关键知识点"
2. 在"对照实验结果"中记录自己的实验数据
3. 在"问答记录"中添加新Q&A
4. 勾选"待补充"中的已完成项

## 学习进度跟踪

学习阶段进度表在 `learning-plan/docs/learning_log.md` 的"学习阶段进度"部分，每完成一个阶段更新状态：
- ⬜ 未开始
- 🔄 进行中
- ✅ 已完成

## 关键命令参考

### 运行学习实验
```bash
cd /home/shao/workspace/minimind/learning-plan/experiments
python exp_01_rmsnorm.py
```

### 推理测试
```bash
python eval_llm.py --load_from ./MiniMind2
```

### 实际训练（需要GPU）
```bash
cd /home/shao/workspace/minimind/trainer
python train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl
python train_full_sft.py --data_path ../dataset/sft_t2t_mini.jsonl --from_weight pretrain
python train_lora.py --data_path ../dataset/lora_medical.jsonl --from_weight full_sft
```

### Web 服务
```bash
streamlit run scripts/web_demo.py
python scripts/serve_openai_api.py
```

## 参考资源

- 学习计划详情：`learning-plan/02-learning-plan.md`
- 方法指导：`learning-plan/04-learning-methodology.md`
- 参考归档：`learning-plan/docs/ref/`（含原 minimind-notes 知识库、学习日志、教学模块）
- GitHub issue #504 推荐资源：MiniMind-in-Depth / happy-llm / from-minimind-to-more
