"""
实验10：GRPO强化学习
目标：使用GRPO进行强化学习训练，理解RLAIF流程
参考：trainer/train_grpo.py

实际训练命令（需要Reward Model）：
cd /home/shao/workspace/minimind/trainer
python train_grpo.py --data_path ../dataset/rlaif-mini.jsonl --from_weight dpo --reward_model_path ../../internlm2-1_8b-reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def exp1_grpo_advantage():
    print("=" * 60)
    print("实验10.1：GRPO组内标准化优势")
    print("=" * 60)
    torch.manual_seed(42)
    num_responses = 4
    rewards = torch.tensor([0.3, 0.7, 0.5, 0.1])
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    advantages = (rewards - mean_reward) / (std_reward + 1e-8)
    print(f"4个回复的奖励: {rewards.tolist()}")
    print(f"组均值: {mean_reward:.4f}")
    print(f"组标准差: {std_reward:.4f}")
    print(f"标准化优势: {advantages.tolist()}")
    print()
    print("优势含义:")
    for i, adv in enumerate(advantages):
        label = "高于平均" if adv > 0 else "低于平均"
        print(f"  回复{i}: reward={rewards[i]:.2f}, advantage={adv:.4f} ({label})")
    print("\n结论：GRPO通过组内标准化，无需Critic模型即可估计优势")


def exp2_grpo_vs_ppo():
    print("\n" + "=" * 60)
    print("实验10.2：GRPO vs PPO 对比")
    print("=" * 60)
    print("PPO (Proximal Policy Optimization):")
    print("  需要: Actor + Critic + Old Actor + Reference Model + Reward Model")
    print("  优势函数: Critic网络估计V(s)，A = R - V(s)")
    print("  复杂度: 高（5个模型）")
    print()
    print("GRPO (Group Relative Policy Optimization):")
    print("  需要: Actor + Reference Model + Reward Model")
    print("  优势函数: 组内标准化，A = (R - mean(R_group)) / std(R_group)")
    print("  复杂度: 低（3个模型，无需Critic）")
    print()
    print("关键区别:")
    print("  PPO: 需要训练Critic来估计baseline → 额外参数+训练不稳定")
    print("  GRPO: 用同一prompt的多个回复作为baseline → 简单且有效")


def exp3_rlhf_pipeline():
    print("\n" + "=" * 60)
    print("实验10.3：完整RLHF/RLAIF流水线")
    print("=" * 60)
    print("MiniMind的完整训练流水线:")
    print()
    print("  Step 1: Pretrain (学习语言模式)")
    print("    ↓")
    print("  Step 2: SFT (学习对话格式)")
    print("    ↓")
    print("  Step 3: DPO (学习偏好，区分好坏回答)")
    print("    ↓")
    print("  Step 4: PPO/GRPO/SPO (强化学习，进一步优化)")
    print("    ↓")
    print("  Step 5 (可选): Reason蒸馏 (训练推理模型)")
    print()
    print("各阶段模型权重依赖:")
    print("  pretrain.pth → full_sft.pth → dpo.pth → grpo.pth/ppo.pth")
    print("                                        → reason.pth")


if __name__ == "__main__":
    exp1_grpo_advantage()
    exp2_grpo_vs_ppo()
    exp3_rlhf_pipeline()
