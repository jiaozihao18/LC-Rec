import argparse
import json
import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import parse_global_args, parse_dataset_args, parse_test_args, set_seed
from prompt import sft_prompt, all_prompt


def load_indices(args):
    """加载item索引文件"""
    index_file = os.path.join(args.data_path, args.dataset, args.dataset + args.index_file)
    with open(index_file, 'r') as f:
        indices = json.load(f)
    return indices


def convert_semantic_ids_to_items(semantic_ids, indices, max_his_len=-1, add_prefix=False, his_sep=", "):
    """
    将语义id列表转换为item字符串列表
    
    Args:
        semantic_ids: 语义id列表，例如 [1, 2, 3, 4]
        indices: 索引字典，将id映射到token列表
        max_his_len: 最大历史长度，-1表示不限制
        add_prefix: 是否添加前缀（如 "1. ", "2. "）
        his_sep: 历史序列分隔符wo
    
    Returns:
        items: item字符串列表
        items_str: 格式化后的历史序列字符串
    """
    # 将语义id转换为item字符串
    items = ["".join(indices[str(i)]) for i in semantic_ids]
    
    # 处理历史序列
    history = items
    if max_his_len > 0:
        history = history[-max_his_len:]
    
    if add_prefix:
        history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
    
    items_str = his_sep.join(history)
    
    return items, items_str


def prepare_input_data(semantic_ids, prompt_id, indices, args):
    """
    根据语义id和prompt_id准备输入数据（仅支持seqrec任务）
    输入的semantic_ids就是完整的历史序列，让模型预测下一个item
    
    Args:
        semantic_ids: 语义id列表（完整的历史序列）
        prompt_id: prompt id
        indices: 索引字典
        args: 参数对象
    
    Returns:
        data: 用于格式化prompt的数据字典
        prompt: prompt字典
    """
    # 获取seqrec的prompt
    prompts = all_prompt["seqrec"]
    
    if prompt_id >= len(prompts):
        raise ValueError(f"prompt_id {prompt_id} out of range (max: {len(prompts)-1})")
    
    prompt = prompts[prompt_id]
    
    # 输入的semantic_ids就是完整的历史序列，全部作为历史
    items, items_str = convert_semantic_ids_to_items(
        semantic_ids,
        indices,
        max_his_len=args.max_his_len,
        add_prefix=args.add_prefix,
        his_sep=args.his_sep
    )
    data = {
        "inters": items_str
    }
    
    return data, prompt


def generate_input_text(data, prompt):
    """
    根据data和prompt生成输入文本
    
    Args:
        data: 数据字典
        prompt: prompt字典
    
    Returns:
        input_text: 输入文本
    """
    instruction = prompt["instruction"].format(**data)
    input_text = sft_prompt.format(instruction=instruction, response="")
    return input_text


def simple_test(semantic_ids, prompt_ids, args):
    """
    简单的测试函数（仅支持seqrec任务）
    
    Args:
        semantic_ids: 语义id列表，例如 [1, 2, 3, 4]
        prompt_ids: prompt id列表，例如 [0, 1, 2] 或单个id
        args: 参数对象
    """
    set_seed(args.seed)
    
    # 加载模型和tokenizer
    device_map = {"": args.gpu_id}
    device = torch.device("cuda", args.gpu_id)
    
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    
    if args.lora:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
    
    model.eval()
    
    # 加载索引
    indices = load_indices(args)
    
    # 处理prompt_ids
    if isinstance(prompt_ids, str):
        if prompt_ids == "all":
            prompt_ids = range(len(all_prompt["seqrec"]))
        else:
            prompt_ids = [int(_) for _ in prompt_ids.split(",")]
    elif isinstance(prompt_ids, int):
        prompt_ids = [prompt_ids]
    
    # 对每个prompt_id生成输出
    results = []
    with torch.no_grad():
        for prompt_id in prompt_ids:
            # 准备输入数据
            data, prompt = prepare_input_data(semantic_ids, prompt_id, indices, args)
            
            # 生成输入文本
            input_text = generate_input_text(data, prompt)
            
            print(f"\n{'='*60}")
            print(f"Task: seqrec, Prompt ID: {prompt_id}")
            print(f"Input text:\n{input_text}")
            print(f"{'='*60}\n")
            
            # Tokenize输入
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(device)
            
            # 生成输出（不使用beam search和前缀约束）
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                do_sample=False,  # 使用greedy decoding
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            
            # 解码输出
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # 只提取生成的部分（去掉输入部分）
            generated_text = output_text[len(input_text):].strip()
            
            print(f"Generated output: {generated_text}")
            print(f"{'='*60}\n")
            
            results.append({
                "prompt_id": prompt_id,
                "input_text": input_text,
                "generated_output": generated_text,
                "full_output": output_text
            })
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple test for LC-Rec")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    
    # 添加自定义参数
    parser.add_argument("--semantic_ids", type=str, required=True,
                        help="语义id列表，用逗号分隔，例如: 1,2,3,4")
    parser.add_argument("--prompt_ids", type=str, default="0",
                        help="prompt id，可以是单个数字、逗号分隔的多个数字，或'all'")
    
    args = parser.parse_args()
    
    # 解析语义id
    semantic_ids = [int(_) for _ in args.semantic_ids.split(",")]
    
    # 运行测试
    results = simple_test(semantic_ids, args.prompt_ids, args)
    
    # 保存结果（可选）
    if args.results_file:
        with open(args.results_file, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nResults saved to {args.results_file}")

