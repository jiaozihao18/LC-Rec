"""
基于vLLM进行在线推理（prompt_only模式）
读取prompt文件，调用vLLM API，保存结果
"""

import argparse
import os
import json
import re
from typing import Optional
from openai import OpenAI


def extract_json_from_text(text: str) -> Optional[str]:
    """
    从文本中提取JSON字符串（可能包含markdown代码块或其他文本）
    
    Args:
        text: 可能包含JSON的文本
    
    Returns:
        提取的JSON字符串，如果提取失败返回None
    """
    if not text:
        return None
    
    # 尝试直接解析（如果整个文本就是JSON）
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
    
    # 尝试从markdown代码块中提取
    # 匹配 ```json ... ``` 或 ``` ... ```
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 尝试找到第一个 { 到最后一个 } 之间的内容
    start_idx = text.find('{')
    if start_idx != -1:
        # 从后往前找最后一个 }
        end_idx = text.rfind('}')
        if end_idx != -1 and end_idx > start_idx:
            json_candidate = text[start_idx:end_idx+1]
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass
    
    return None


def vllm_inference(client, model_name, prompt_list, system_message="You are a helpful assistant.", 
                   enable_thinking=None, max_tokens=1024):
    """
    使用vLLM进行推理（prompt_only模式）
    
    Args:
        client: OpenAI客户端
        model_name: 模型名称
        prompt_list: prompt列表
        system_message: 系统消息
        enable_thinking: 是否启用thinking功能（None表示使用默认，False表示禁用）
        max_tokens: 最大token数
    
    Returns:
        output_list: 输出列表（原始文本）
    """
    # 准备extra_body参数（用于thinking控制）
    extra_body = {}
    if enable_thinking is False:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    
    # 构建messages列表
    messages_list = [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompt_list
    ]
    
    # 逐个调用
    output_list = []
    total = len(messages_list)
    for i, messages in enumerate(messages_list):
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"Processing {i+1}/{total}")
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.4,
                max_tokens=max_tokens,
                extra_body=extra_body if extra_body else None
            )
            output = completion.choices[0].message.content.strip()
            output_list.append(output)
        except Exception as e:
            print(f"API call failed for prompt {i+1}: {e}")
            output_list.append(None)
    
    return output_list


def load_prompts_file(prompt_file):
    """
    加载prompt文件
    
    Args:
        prompt_file: prompt文件路径（JSONL格式）
    
    Returns:
        prompts: prompt记录列表
    """
    prompts = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def save_results(results, output_file):
    """
    保存推理结果
    
    Args:
        results: 结果列表，每个元素包含原始记录和推理结果
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(results)} results to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='基于vLLM进行在线推理')
    parser.add_argument('--prompt_file', type=str, required=True, help='输入的prompt文件路径（JSONL格式）')
    parser.add_argument('--base_url', type=str, required=True, help='vLLM服务的基础URL，例如: http://localhost:8000/v1')
    parser.add_argument('--api_key', type=str, default='dummy', help='API密钥（vLLM可以使用dummy）')
    parser.add_argument('--model_name', type=str, required=True, help='模型名称')
    parser.add_argument('--output', type=str, default='', help='输出文件路径（如果不指定，使用默认路径）')
    parser.add_argument('--max_tokens', type=int, default=1024, help='最大token数')
    parser.add_argument('--enable_thinking', type=str, default=None,
                       help='是否启用thinking功能。可选值: true/1/yes, false/0/no, 或不设置（None，使用默认行为）')
    parser.add_argument('--extract_json', action='store_true', 
                       help='是否从输出中提取JSON（如果输出包含JSON）')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 解析enable_thinking参数
    enable_thinking = None
    if args.enable_thinking:
        v_lower = args.enable_thinking.lower()
        if v_lower in ('true', '1', 'yes'):
            enable_thinking = True
        elif v_lower in ('false', '0', 'no'):
            enable_thinking = False
    
    # 创建客户端
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    
    # 加载prompt文件
    print(f"Loading prompts from {args.prompt_file}...")
    prompts = load_prompts_file(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts")
    
    # 提取prompt列表
    prompt_list = [record["prompt"] for record in prompts]
    
    # 进行推理
    print("Starting inference...")
    outputs = vllm_inference(
        client=client,
        model_name=args.model_name,
        prompt_list=prompt_list,
        enable_thinking=enable_thinking,
        max_tokens=args.max_tokens
    )
    
    # 构建结果
    results = []
    for record, output in zip(prompts, outputs):
        result = {
            "id": record.get("id"),
            "user": record.get("user"),
            "item": record.get("item"),
            "history": record.get("history"),
            "prompt": record.get("prompt"),
            "output": output
        }
        
        # 如果需要提取JSON
        if args.extract_json and output:
            json_str = extract_json_from_text(output)
            if json_str:
                try:
                    result["extracted_json"] = json.loads(json_str)
                except json.JSONDecodeError:
                    result["extracted_json"] = None
            else:
                result["extracted_json"] = None
        
        results.append(result)
    
    # 确定输出文件路径
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.prompt_file)[0]
        output_file = f"{base_name}_results.jsonl"
    
    # 保存结果
    save_results(results, output_file)
    print("Inference completed!")

