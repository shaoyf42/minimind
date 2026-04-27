"""
实验07：SFT微调实践
目标：将预训练模型微调为能对话的聊天模型
参考：trainer/train_full_sft.py, dataset/lm_dataset.py

实际训练命令：
cd /home/shao/workspace/minimind/trainer
python train_full_sft.py --data_path ../dataset/sft_t2t_mini.jsonl --from_weight pretrain --epochs 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def exp1_loss_mask():
    print("=" * 60)
    print("实验7.1：SFT Loss Mask 机制演示")
    print("=" * 60)
    vocab_size = 50
    hidden_size = 32
    embed = nn.Embedding(vocab_size, hidden_size)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    bos_id, eos_id, pad_id = 1, 2, 0
    im_start, im_end = 3, 4

    conversation = [bos_id, im_start] + [10, 11] + [im_end] + \
                   [im_start] + [20, 21, 22] + [im_end] + [eos_id]
    input_ids = torch.tensor([conversation])
    print(f"对话token序列: {conversation}")
    print(f"  BOS={bos_id}, <|im_start|>={im_start}, <|im_end|>={im_end}, EOS={eos_id}")

    loss_mask = torch.zeros_like(input_ids, dtype=torch.float)
    in_assistant = False
    for i in range(len(conversation)):
        if conversation[i] == im_start:
            in_assistant = True
            continue
        if conversation[i] == im_end:
            in_assistant = False
            continue
        if in_assistant:
            loss_mask[0, i] = 1.0

    print(f"\nLoss Mask: {loss_mask[0].tolist()}")
    print("只有assistant部分的token参与loss计算")

    hidden = embed(input_ids)
    logits = lm_head(hidden)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()
    per_token_loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), reduction='none')
    masked_loss = (per_token_loss * shift_mask.view(-1)).sum() / shift_mask.sum()
    print(f"\n全token loss: {per_token_loss.mean():.4f}")
    print(f"仅assistant loss: {masked_loss:.4f}")
    print("\n结论：SFT只对assistant回复部分计算loss，避免学习用户输入")


def exp2_chat_template():
    print("\n" + "=" * 60)
    print("实验7.2：Chat Template 格式化")
    print("=" * 60)
    template = "<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    formatted = template.format(user_msg="你好", assistant_msg="你好！有什么可以帮你的？")
    print("格式化后的对话:")
    print(formatted)
    print("\n结论：Chat Template统一了对话格式，便于模型学习角色区分")


if __name__ == "__main__":
    exp1_loss_mask()
    exp2_chat_template()
