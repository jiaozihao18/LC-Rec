
import argparse
import os
import json
import re
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
from utils import get_res_batch, load_json, unified_user_analysis_prompt, amazon18_dataset2fullname, write_json_file, _is_vllm_service


def format_history_items(item_list, item2feature, max_his_len):
    """格式化历史商品信息"""
    formatted_items = []
    for j, item_id in enumerate(item_list):
        item_feat = item2feature[str(item_id)]
        item_str = f"{j+1}. Title: {item_feat['title']}"
        if item_feat.get('description'):
            item_str += f"\n   Description: {item_feat['description'][:200]}..."  # 截断过长的描述
        if item_feat.get('brand'):
            item_str += f"\n   Brand: {item_feat['brand']}"
        formatted_items.append(item_str)
    
    if max_his_len > 0 and len(formatted_items) > max_his_len:
        formatted_items = formatted_items[-max_his_len:]
    
    return "\n".join(formatted_items)


# 定义输出结构的Pydantic模型
class UserAnalysisResponse(BaseModel):
    """用户分析和意图提取的响应结构"""
    general_preference: str = Field(description="用户整体偏好的简要第三人称总结")
    long_term_preference: str = Field(description="用户长期偏好，反映在所有购买中的固有特征")
    short_term_preference: str = Field(description="用户短期偏好，反映在最近购买中的偏好")
    user_related_intention: str = Field(description="用户相关意图，以第一人称描述用户的个人偏好、需求和动机")
    item_related_intention: str = Field(description="商品特征，以第三人称客观描述商品本身的特征、功能和属性")


# 默认值常量
DEFAULT_PREFERENCES = {
    "general_preference": "The user enjoys high-quality items.",
    "long_term_preference": "The user prefers high-quality items.",
    "short_term_preference": "The user has been focusing on quality items recently.",
    "user_related_intention": "I enjoy high-quality items.",
    "item_related_intention": "High-quality item with good features."
}




def generate_batch_submission_file(args, inters, item2feature, reviews, api_info=None, mode='train'):
    """
    生成批量提交文件（JSONL格式）用于离线批量推理
    
    Args:
        args: 命令行参数
        inters: 交互数据
        item2feature: 商品特征字典
        reviews: 评论数据
        api_info: (可选) API配置信息，用于检测是否为vLLM服务
        mode: 'train' 或 'test'
    """
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()
    
    prompt_list = []
    user_data_list = []  # 存储(user, item, history)用于后续处理
    
    for user, item_list in inters.items():
        user = int(user)
        
        # 根据模式选择目标item
        if mode == 'train':
            if len(item_list) < 3:
                continue
            target_item = int(item_list[-3])
            history = item_list[:-3]
        else:  # test
            if len(item_list) < 1:
                continue
            target_item = int(item_list[-1])
            history = item_list[:-1]
        
        # 获取目标商品特征
        item_feat = item2feature[str(target_item)]
        item_title = item_feat.get('title', '')
        item_description = item_feat.get('description') or 'N/A'
        
        # 获取review（如果存在）
        review = reviews.get(str((user, target_item)), {}).get('review', '')
        review_section = f"\nUser Review: {review}" if review else "\nNote: No user review available."
        
        # 对于preference，使用去掉最后3个的历史
        preference_history = item_list[:-3] if len(item_list) >= 3 else []
        preference_history_items = format_history_items(preference_history, item2feature, args.max_his_len) if preference_history else "No purchase history available."
        
        # 构建prompt
        prompt = unified_user_analysis_prompt.format(
            dataset_full_name=dataset_full_name,
            history_items=preference_history_items,
            item_title=item_title,
            item_description=item_description,
            review_section=review_section
        )
        
        prompt_list.append(prompt)
        user_data_list.append((user, target_item, history))
    
    # 准备系统消息（json_object模式需要在消息中包含JSON关键词，prompt中已有）
    system_message = "You are a helpful assistant."
    
    # 检测是否为vLLM服务
    use_vllm = False
    if api_info:
        base_url = api_info.get("base_url")
        use_vllm = api_info.get("use_vllm", _is_vllm_service(base_url) if base_url else False)
    
    # 根据响应格式类型和是否使用vLLM设置request_body
    json_schema = UserAnalysisResponse.model_json_schema()
    guided_decoding_backend = api_info.get("guided_decoding_backend", "outlines") if api_info else "outlines"
    
    # 生成JSONL文件
    output_file = os.path.join(args.root, f'{args.dataset}_{mode}_batch_submission.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, prompt in enumerate(prompt_list):
            custom_id = f"{mode}_{idx}"
            
            if use_vllm:
                # vLLM使用extra_body传递guided_json实现结构化输出
                # vLLM不支持标准的response_format={"type": "json_object"}
                # 对于json_object模式，也使用guided_json（使用完整的JSON Schema）
                request_body = {
                    "model": args.model_name,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    "extra_body": {
                        "guided_json": json_schema,
                        "guided_decoding_backend": guided_decoding_backend
                    }
                }
            else:
                # OpenAI兼容API使用response_format
                if args.response_format == 'prompt_only':
                    # prompt_only模式：不设置response_format，仅通过prompt引导
                    request_body = {
                        "model": args.model_name,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ]
                    }
                elif args.response_format == 'json_object':
                    response_format_config = {"type": "json_object"}
                    request_body = {
                        "model": args.model_name,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": response_format_config
                    }
                else:  # json_schema
                    response_format_config = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "user_analysis_response",
                            "strict": True,
                            "schema": json_schema
                        }
                    }
                    request_body = {
                        "model": args.model_name,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": response_format_config
                    }
            
            request_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body
            }
            
            f.write(json.dumps(request_obj, ensure_ascii=False) + '\n')
    
    print(f"Generated batch submission file: {output_file} ({len(prompt_list)} requests)")
    return output_file, user_data_list


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


def parse_json_object_response(json_string: str) -> Optional[UserAnalysisResponse]:
    """
    解析并验证JSON Object模式的响应
    
    Args:
        json_string: JSON字符串
    
    Returns:
        解析后的Pydantic对象，如果解析或验证失败返回None
    """
    if not json_string:
        return None
    
    try:
        # 解析JSON字符串
        json_data = json.loads(json_string)
        # 使用Pydantic模型验证数据
        return UserAnalysisResponse(**json_data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Failed to parse/validate JSON response: {e}")
        return None


def generate_user_data(args, inters, item2feature, reviews, api_info, mode='train'):
    """
    生成用户意图数据（train或test模式）
    mode: 'train' 使用倒数第三个item, 'test' 使用最后一个item
    """
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()
    
    prompt_list = []
    user_data_list = []  # 存储(user, item, history)用于后续处理
    
    for user, item_list in inters.items():
        user = int(user)
        
        # 根据模式选择目标item
        if mode == 'train':
            if len(item_list) < 3:
                continue  # 跳过历史不足的用户
            target_item = int(item_list[-3])
            history = item_list[:-3]
        else:  # test
            if len(item_list) < 1:
                continue
            target_item = int(item_list[-1])
            history = item_list[:-1]
        
        # 获取目标商品特征
        item_feat = item2feature[str(target_item)]
        item_title = item_feat.get('title', '')
        item_description = item_feat.get('description') or 'N/A'
        
        # 获取review（如果存在）
        review = reviews.get(str((user, target_item)), {}).get('review', '')
        review_section = f"\nUser Review: {review}" if review else "\nNote: No user review available."
        
        # 对于preference，使用去掉最后3个的历史
        preference_history = item_list[:-3] if len(item_list) >= 3 else []
        preference_history_items = format_history_items(preference_history, item2feature, args.max_his_len) if preference_history else "No purchase history available."
        
        # 构建prompt
        prompt = unified_user_analysis_prompt.format(
            dataset_full_name=dataset_full_name,
            history_items=preference_history_items,
            item_title=item_title,
            item_description=item_description,
            review_section=review_section
        )
        
        prompt_list.append(prompt)
        user_data_list.append((user, target_item, history))
    
    # 批量调用大模型
    results = {}
    st = 0
    while st < len(prompt_list):
        print(f"Processing {mode} data: {st}/{len(prompt_list)}")
        
        # 检测是否为vLLM服务
        base_url = api_info.get("base_url") if api_info else None
        use_vllm = api_info.get("use_vllm", _is_vllm_service(base_url) if base_url else False) if api_info else False
        
        # 根据响应格式类型和是否使用vLLM调用不同的API
        # 注意：vLLM不支持标准的json_object模式，两种模式都使用guided_json
        if args.response_format == 'prompt_only':
            # prompt_only模式：不依赖API的结构化输出，仅通过prompt引导，然后代码解析
            res = get_res_batch(args.model_name, prompt_list[st:st+args.batchsize], api_info, 
                              response_format=None, use_json_object=False, use_prompt_only=True)
            
            # 解析JSON字符串并验证
            parsed_responses = []
            for json_str in res:
                if json_str is None:
                    parsed_responses.append(None)
                else:
                    # 尝试从文本中提取JSON（可能包含markdown代码块或其他文本）
                    json_str = extract_json_from_text(json_str)
                    parsed_response = parse_json_object_response(json_str) if json_str else None
                    parsed_responses.append(parsed_response)
            res = parsed_responses
        elif args.response_format == 'json_object' and not use_vllm:
            # 使用json_object模式（仅适用于OpenAI兼容API），需要手动解析和验证
            res = get_res_batch(args.model_name, prompt_list[st:st+args.batchsize], api_info, 
                              response_format=None, use_json_object=True)
            
            # 解析JSON字符串并验证
            parsed_responses = []
            for json_str in res:
                if json_str is None:
                    parsed_responses.append(None)
                else:
                    parsed_response = parse_json_object_response(json_str)
                    parsed_responses.append(parsed_response)
            res = parsed_responses
        else:
            # 使用json_schema模式（默认），或者vLLM使用json_object模式时也使用完整的schema
            # 对于vLLM，json_object和json_schema都使用guided_json + 完整schema
            res = get_res_batch(args.model_name, prompt_list[st:st+args.batchsize], api_info, 
                              response_format=UserAnalysisResponse)
        
        for i, parsed_response in enumerate(res):
            user, target_item, history = user_data_list[st + i]
            
            # 使用解析后的响应（已经是Pydantic对象）
            if parsed_response is None:
                print(f"Empty response for user {user}, using defaults")
                parsed_data = DEFAULT_PREFERENCES.copy()
            else:
                # 将Pydantic对象转换为字典
                parsed_data = parsed_response.model_dump()
            
            results[user] = {
                "item": target_item,
                "inters": history,
                **{k: parsed_data.get(k, DEFAULT_PREFERENCES[k]) for k in DEFAULT_PREFERENCES.keys()}
            }
        
        st += args.batchsize
    
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments', help='Instruments / Arts / Games')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--api_info', type=str, default='./api_info.json')
    parser.add_argument('--model_name', type=str, default='qwen-plus', help='模型名称，如qwen-plus')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--max_his_len', type=int, default=20)
    parser.add_argument('--generate_batch_file', action='store_true', help='生成批量提交文件而不是直接调用API')
    parser.add_argument('--response_format', type=str, default='json_schema', 
                       choices=['json_schema', 'json_object', 'prompt_only'],
                       help='响应格式类型: json_schema (使用严格的JSON Schema), json_object (使用JSON Object模式), 或 prompt_only (仅通过prompt引导，代码解析)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.root = os.path.join(args.root, args.dataset)

    api_info = load_json(args.api_info)
    # 兼容旧的api_key_list格式
    if "api_key" not in api_info and "api_key_list" in api_info:
        api_key_list = api_info["api_key_list"]
        api_info["api_key"] = api_key_list[0] if isinstance(api_key_list, list) else api_key_list
    if "api_key" not in api_info:
        raise ValueError("api_info must contain 'api_key'")

    # 加载数据
    inter_path = os.path.join(args.root, f'{args.dataset}.inter.json')
    inters = load_json(inter_path)

    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)

    reviews_path = os.path.join(args.root, f'{args.dataset}.review.json')
    if os.path.exists(reviews_path):
        reviews = load_json(reviews_path)
    else:
        print(f"Warning: {reviews_path} not found, using empty reviews")
        reviews = {}

    # 根据参数决定是生成批量提交文件还是直接调用API
    if args.generate_batch_file:
        # 生成批量提交文件
        print("Generating batch submission files...")
        train_file, train_user_data = generate_batch_submission_file(args, inters, item2feature, reviews, api_info, mode='train')
        test_file, test_user_data = generate_batch_submission_file(args, inters, item2feature, reviews, api_info, mode='test')
        
        # 保存用户数据映射文件（用于后续解析结果）
        mapping_file = os.path.join(args.root, f'{args.dataset}_batch_mapping.json')
        mapping_data = {
            "train": {f"train_{i}": {"user": user, "item": item, "history": history} 
                     for i, (user, item, history) in enumerate(train_user_data)},
            "test": {f"test_{i}": {"user": user, "item": item, "history": history} 
                    for i, (user, item, history) in enumerate(test_user_data)}
        }
        write_json_file(mapping_data, mapping_file)
        print(f"Generated mapping file: {mapping_file}")
        print("Batch submission files generated. Please submit them for batch inference.")
    else:
        # 直接调用API生成数据
        print("Generating train data...")
        train_data = generate_user_data(args, inters, item2feature, reviews, api_info, mode='train')
        
        print("Generating test data...")
        test_data = generate_user_data(args, inters, item2feature, reviews, api_info, mode='test')

        # 构建最终的用户字典
        user_dict = {
            "user_explicit_preference": {},
            "user_vague_intention": {
                "train": {},
                "test": {}
            }
        }

        # 处理训练集数据（preference从train数据中提取，但每个用户只保存一次）
        for user, data in train_data.items():
            user_dict["user_explicit_preference"][user] = [
                data["general_preference"],
                data["long_term_preference"],
                data["short_term_preference"]
            ]
            user_dict["user_vague_intention"]["train"][user] = {
                "item": data["item"],
                "inters": data["inters"],
                "querys": [data["user_related_intention"], data["item_related_intention"]]
            }

        # 处理测试集数据（preference已经在train中设置，这里只设置intention）
        for user, data in test_data.items():
            # 如果用户不在训练集中，也设置preference
            if user not in user_dict["user_explicit_preference"]:
                user_dict["user_explicit_preference"][user] = [
                    data["general_preference"],
                    data["long_term_preference"],
                    data["short_term_preference"]
                ]
            
            user_dict["user_vague_intention"]["test"][user] = {
                "item": data["item"],
                "inters": data["inters"],
                "querys": [data["user_related_intention"], data["item_related_intention"]]
            }

        # 直接输出最终文件
        output_file = os.path.join(args.root, f'{args.dataset}.user.json')
        write_json_file(user_dict, output_file)
        print(f"Successfully generated {output_file}")
