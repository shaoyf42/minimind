"""
实验09：DPO偏好优化
目标：使用DPO让模型学会区分好回答和坏回答
参考：trainer/train_dpo.py

实际训练命令：
cd /home/shao/workspace/minimind/trainer
python train_dpo.py --data_path ../dataset/dpo.jsonl --from_weight full_sft --beta 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def exp1_dpo_loss():
    print("=" * 60)
    print("实验9.1：DPO Loss 计算演示")
    print("=" * 60)
    beta = 0.1
    chosen_logps = torch.tensor([-2.0, -1.5, -3.0])
    rejected_logps = torch.tensor([-3.0, -2.5, -4.0])
    ref_chosen_logps = torch.tensor([-2.1, -1.6, -3.1])
    ref_rejected_logps = torch.tensor([-2.9, -2.4, -3.9])

    chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    print(f"Chosen log概率: {chosen_logps.tolist()}")
    print(f"Rejected log概率: {rejected_logps.tolist()}")
    print(f"Chosen奖励: {chosen_rewards.tolist()}")
    print(f"Rejected奖励: {rejected_rewards.tolist()}")
    print(f"DPO Loss: {loss.item():.6f}")
    print("\n结论：DPO通过比较chosen和rejected的log概率比率来优化偏好")


def exp2_dpo_vs_sft():
    print("\n" + "=" * 60)
    print("实验9.2：DPO vs SFT 对比")
    print("=" * 60)
    print("SFT: 只学习'好回答'长什么样")
    print("  Loss = -log P(good_answer | question)")
    print("  问题：不知道'坏回答'长什么样")
    print()
    print("DPO: 同时学习'好回答'和'坏回答'的区别")
    print("  Loss = -log σ(β * (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))")
    print("  优势：显式区分好坏，对齐人类偏好")
    print()
    print("训练流程对比:")
    print("  SFT:  question → good_answer → 最小化交叉熵")
    print("  DPO:  question → (good_answer, bad_answer) → 最大化好/坏比率")


def exp3_beta_effect():
    print("\n" + "=" * 60)
    print("实验9.3：Beta参数效果")
    print("=" * 60)
    log_ratio_diff = torch.tensor([0.5, 1.0, 2.0, 5.0])
    print("log(π_chosen/π_ref) - log(π_rejected/π_ref) 的差值:")
    print(f"  {log_ratio_diff.tolist()}")
    print()
    for beta in [0.01, 0.1, 0.5, 1.0]:
        rewards = beta * log_ratio_diff
        loss = -F.logsigmoid(rewards).mean()
        print(f"β={beta:.2f}: rewards={rewards.tolist()}, loss={loss.item():.6f}")
    print("\n结论：β越大，对偏好差异越敏感，但也可能导致训练不稳定")


if __name__ == "__main__":
    exp1_dpo_loss()
    exp2_dpo_vs_sft()
    exp3_beta_effect()
