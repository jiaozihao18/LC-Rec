import argparse
import os
import re
import math
import random
import numpy as np
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import ConcatDataset

from utils import *
from prompt import sft_prompt

os.environ['WANDB_MODE'] = 'disabled'


def extract_semantic_id_from_text(text):
    """
    从文本中提取语义id（response部分）
    文本格式: "Below is an instruction...\n\n### Instruction:\n...\n\n### Response:{response}"
    """
    # 提取Response后面的内容
    response_pattern = r'### Response:\s*(.+?)(?:\n|$)'
    match = re.search(response_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_semantic_id_from_completion(completion):
    """
    从生成的completion中提取语义id
    """
    # 清理completion，去除可能的换行和引号
    completion = completion.strip().strip('"').strip("'").strip("\n")
    return completion


def check_format(semantic_id):
    """
    检查语义id格式是否为 <a_xxx><b_xxx><c_xxx>，xxx是数字
    返回: (是否匹配, 匹配的完整字符串)
    """
    # 正则表达式匹配 <a_数字><b_数字><c_数字>
    pattern = r'<a_\d+><b_\d+><c_\d+>'
    match = re.search(pattern, semantic_id)
    if match:
        return True, match.group(0)
    return False, None


def accuracy_reward(prompts, completions, targets):
    """
    准确性奖励：生成正确的语义id
    Args:
        prompts: 输入prompt列表
        completions: 生成的completion列表
        targets: 目标语义id列表（从labels中提取）
    Returns:
        rewards: 奖励列表，正确为1.0，错误为0.0
    """
    rewards = []
    for i, completion in enumerate(completions):
        semantic_id = extract_semantic_id_from_completion(completion)
        target_id = targets[i].strip() if i < len(targets) else ""
        
        # 比较是否完全匹配
        if semantic_id == target_id:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


def format_reward(prompts, completions, targets):
    """
    格式奖励：语义id的格式为<a_xxx><b_xxx><c_xxx>，xxx是数字
    Args:
        prompts: 输入prompt列表
        completions: 生成的completion列表
        targets: 目标语义id列表（未使用，但保持接口一致）
    Returns:
        rewards: 奖励列表，格式正确为1.0，错误为0.0
    """
    rewards = []
    for completion in completions:
        semantic_id = extract_semantic_id_from_completion(completion)
        is_valid, matched = check_format(semantic_id)
        
        if is_valid:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


def ndcg_accuracy_reward(prompts, completions, targets, num_generations):
    """
    NDCG准确性奖励：鼓励正确答案排序靠前
    如果一组里都没有正确答案，那就奖励都是0
    Args:
        prompts: 输入prompt列表
        completions: 生成的completion列表
        targets: 目标语义id列表
        num_generations: 每个prompt生成的候选数量
    Returns:
        rewards: 奖励列表，使用NDCG权重
    """
    # 计算NDCG权重：位置越靠前，权重越大
    ndcg_weights = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
    # 归一化，使得权重和为1
    total_weight = sum(ndcg_weights)
    ndcg_weights = [-w / total_weight for w in ndcg_weights]
    
    rewards = []
    num_prompts = len(prompts) // num_generations
    
    for prompt_idx in range(num_prompts):
        # 获取该prompt对应的所有completions
        start_idx = prompt_idx * num_generations
        end_idx = start_idx + num_generations
        prompt_completions = completions[start_idx:end_idx]
        target_id = targets[start_idx].strip()  # 同一prompt的target相同
        
        # 检查是否有正确答案
        has_correct = False
        correct_positions = []
        
        for i, completion in enumerate(prompt_completions):
            semantic_id = extract_semantic_id_from_completion(completion)
            if semantic_id == target_id:
                has_correct = True
                correct_positions.append(i)
        
        # 如果一组里都没有正确答案，奖励都是0
        if not has_correct:
            rewards.extend([0.0] * num_generations)
        else:
            # 有正确答案，根据位置分配NDCG权重
            for i in range(num_generations):
                if i in correct_positions:
                    # 正确答案获得NDCG权重
                    rewards.append(ndcg_weights[i])
                else:
                    rewards.append(0.0)
    
    return rewards


def combined_reward(prompts, completions, targets, num_generations, 
                   accuracy_weight=1.0, format_weight=0.5, ndcg_weight=1.0):
    """
    组合奖励函数：结合准确性、格式和NDCG奖励
    Args:
        prompts: 输入prompt列表
        completions: 生成的completion列表
        targets: 目标语义id列表
        num_generations: 每个prompt生成的候选数量
        accuracy_weight: 准确性奖励权重
        format_weight: 格式奖励权重
        ndcg_weight: NDCG奖励权重
    Returns:
        rewards: 组合后的奖励列表
    """
    acc_rewards = accuracy_reward(prompts, completions, targets)
    fmt_rewards = format_reward(prompts, completions, targets)
    ndcg_rewards = ndcg_accuracy_reward(prompts, completions, targets, num_generations)
    
    # 组合奖励
    combined = []
    for i in range(len(acc_rewards)):
        reward = (accuracy_weight * acc_rewards[i] + 
                 format_weight * fmt_rewards[i] + 
                 ndcg_weight * ndcg_rewards[i])
        combined.append(reward)
    
    return combined


def prepare_dataset_for_grpo(train_data):
    """
    将数据集转换为GRPO训练所需的格式
    GRPO需要的数据格式：包含"prompt"字段的字典列表
    同时建立prompt到target的映射
    """
    grpo_data = []
    prompt_to_target = {}
    
    for item in train_data:
        # 优先从input_ids获取prompt（这是真正的输入）
        prompt = item.get("input_ids", "")
        
        # 如果没有input_ids，尝试从labels中提取prompt部分
        if not prompt:
            full_text = item.get("labels", "")
            prompt_match = re.search(r'(.*?### Instruction:\s*.+?)\n\n### Response:', full_text, re.DOTALL)
            if prompt_match:
                prompt = prompt_match.group(1) + "\n\n### Response:"
        
        # 从labels中提取target（response部分）
        full_text = item.get("labels", "")
        target = extract_semantic_id_from_text(full_text)
        
        # 建立prompt到target的映射（使用规范化后的prompt作为key）
        prompt_key = prompt.strip()
        prompt_to_target[prompt_key] = target
        
        grpo_data.append({"prompt": prompt})
    
    train_dataset = Dataset.from_list(grpo_data)
    
    return train_dataset, prompt_to_target


def create_reward_fn(prompt_to_target, reward_type, num_generations):
    """
    创建reward函数
    Args:
        prompt_to_target: prompt到target的映射字典
        reward_type: reward类型
        num_generations: 每个prompt生成的候选数量
    """
    def reward_fn(prompts, completions, **kwargs):
        """
        GRPO的reward函数接口
        Args:
            prompts: prompt列表
            completions: completion列表（每个prompt对应num_generations个completion）
        Returns:
            rewards: 奖励列表或tensor
        """
        # 根据prompts找到对应的targets
        targets = []
        for prompt in prompts:
            # 从映射中获取target（使用规范化后的prompt作为key）
            prompt_key = prompt.strip()
            target = prompt_to_target.get(prompt_key, "")
            # 如果直接匹配失败，尝试更宽松的匹配
            if not target:
                for key, value in prompt_to_target.items():
                    # 尝试匹配prompt的核心部分（去掉前后空白）
                    if key.strip() == prompt_key or key == prompt_key:
                        target = value
                        break
            targets.append(target)
        
        if reward_type == "accuracy":
            rewards = accuracy_reward(prompts, completions, targets)
        elif reward_type == "format":
            rewards = format_reward(prompts, completions, targets)
        elif reward_type == "ndcg":
            rewards = ndcg_accuracy_reward(prompts, completions, targets, num_generations)
        elif reward_type == "combined":
            rewards = combined_reward(prompts, completions, targets, num_generations)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        # 转换为tensor（如果需要）
        if isinstance(rewards, list):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        
        return rewards
    
    return reward_fn


def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    
    # 加载数据集（使用finetune.py中的方式）
    # 注意：GRPO只需要训练集，不需要验证集
    train_data, _ = load_datasets(args)
    
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
    )
    
    # 添加新token（如果需要）
    if hasattr(train_data, 'datasets') and len(train_data.datasets) > 0:
        new_tokens = train_data.datasets[0].get_new_tokens()
        if new_tokens:
            num_added = tokenizer.add_tokens(new_tokens)
            if num_added > 0:
                model.resize_token_embeddings(len(tokenizer))
                print(f"Added {num_added} new tokens.")
    
    # 准备GRPO数据集
    # 将PyTorch数据集转换为HuggingFace Dataset格式
    train_dataset_list = []
    for i in range(len(train_data)):
        item = train_data[i]
        train_dataset_list.append(item)
    
    # 转换为Dataset格式并准备GRPO格式
    train_data_hf = Dataset.from_list(train_dataset_list)
    
    # 转换为GRPO格式并建立prompt到target的映射
    train_dataset, prompt_to_target = prepare_dataset_for_grpo(train_data_hf)
    
    # 创建reward函数
    reward_fn = create_reward_fn(prompt_to_target, args.reward_type, args.num_generations)
    
    # 配置GRPO训练参数
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_step,
        save_steps=args.save_and_eval_steps,
        save_strategy=args.save_and_eval_strategy,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        fp16=args.fp16,
        optim=args.optim,
        weight_decay=args.weight_decay,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.beta,
        report_to=None,
    )
    
    # 创建GRPO Trainer
    # 注意：GRPO通常不需要eval_dataset
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRPO Training with TRL')
    
    # 使用finetune.py中的参数解析
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    
    # GRPO特定参数
    parser.add_argument("--reward_type", type=str, default="combined",
                       choices=["accuracy", "format", "ndcg", "combined"],
                       help="Reward function type")
    parser.add_argument("--num_generations", type=int, default=4,
                       help="Number of generations per prompt for GRPO")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for generation")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="Beta parameter for GRPO")
    parser.add_argument("--max_completion_length", type=int, default=128,
                       help="Maximum completion length")
    
    args = parser.parse_args()
    
    train(args)

