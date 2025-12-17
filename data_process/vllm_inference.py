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


def vllm_inference_streaming(client, model_name, prompts, output_file, system_message="You are a helpful assistant.", 
                             enable_thinking=None, max_tokens=1024, extract_json=False):
    """
    使用vLLM进行推理（prompt_only模式），每10个结果就写入文件
    
    Args:
        client: OpenAI客户端
        model_name: 模型名称
        prompts: prompt记录列表（包含id, user, item, history, prompt等）
        output_file: 输出文件路径
        system_message: 系统消息
        enable_thinking: 是否启用thinking功能（None表示使用默认，False表示禁用）
        max_tokens: 最大token数
        extract_json: 是否从输出中提取JSON
    """
    # 准备extra_body参数（用于thinking控制）
    extra_body = {}
    if enable_thinking is False:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    
    total = len(prompts)
    buffer = []  # 缓冲区，每10个写入一次
    
    # 打开文件（追加模式，如果文件已存在则追加）
    file_mode = 'a' if os.path.exists(output_file) else 'w'
    
    with open(output_file, file_mode, encoding='utf-8') as f:
        for i, record in enumerate(prompts):
            prompt = record["prompt"]
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
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
            except Exception as e:
                print(f"API call failed for prompt {i+1}: {e}")
                output = None
            
            # 构建结果
            result = {
                "id": record.get("id"),
                "user": record.get("user"),
                "item": record.get("item"),
                "history": record.get("history"),
                "prompt": record.get("prompt"),
                "output": output
            }
            
            # 如果需要提取JSON
            if extract_json and output:
                json_str = extract_json_from_text(output)
                if json_str:
                    try:
                        result["extracted_json"] = json.loads(json_str)
                    except json.JSONDecodeError:
                        result["extracted_json"] = None
                else:
                    result["extracted_json"] = None
            
            buffer.append(result)
            
            # 每10个或最后一个就写入文件
            if len(buffer) >= 10 or (i + 1) == total:
                for res in buffer:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')
                f.flush()  # 确保立即写入磁盘
                buffer = []
    
    print(f"Saved {total} results to {output_file}")


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
    
    # 确定输出文件路径
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.prompt_file)[0]
        output_file = f"{base_name}_results.jsonl"
    
    # 加载prompt文件
    print(f"Loading prompts from {args.prompt_file}...")
    prompts = load_prompts_file(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts")
    
    # 进行推理并实时写入文件
    print("Starting inference...")
    vllm_inference_streaming(
        client=client,
        model_name=args.model_name,
        prompts=prompts,
        output_file=output_file,
        enable_thinking=enable_thinking,
        max_tokens=args.max_tokens,
        extract_json=args.extract_json
    )
    
    print("Inference completed!")

