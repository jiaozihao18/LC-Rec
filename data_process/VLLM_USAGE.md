# vLLM 结构化输出使用指南

本指南说明如何在 LC-Rec 项目中使用本地部署的 vLLM 进行结构化输出推理。

## vLLM 结构化输出支持

vLLM 支持使用引导解码（guided decoding）生成结构化输出，支持的引导解码后端包括：
- `outlines` (默认，推荐)
- `lm-format-enforcer`
- `xgrammar`

## 配置说明

### 1. API 配置文件 (api_info.json)

创建或修改 `api_info.json` 文件，配置 vLLM 服务信息：

```json
{
    "api_key": "dummy",
    "base_url": "http://localhost:8000/v1",
    "use_vllm": true,
    "guided_decoding_backend": "outlines"
}
```

**配置项说明：**
- `api_key`: vLLM 通常不需要真实的 API key，可以使用 "dummy" 或任意字符串
- `base_url`: vLLM 服务的地址，格式为 `http://host:port/v1`
  - 默认: `http://localhost:8000/v1`
  - 如果 vLLM 运行在其他机器，使用对应的 IP 地址，如 `http://0.0.0.0:8000/v1` 或 `http://192.168.1.100:8000/v1`
- `use_vllm`: (可选) 显式指定使用 vLLM。如果不设置，系统会根据 `base_url` 自动检测（检测 localhost/127.0.0.1/0.0.0.0）
- `guided_decoding_backend`: (可选) 引导解码后端，默认为 "outlines"

### 2. 启动 vLLM 服务

首先启动 vLLM 服务（如果还未启动）：

```bash
# 使用 outlines 后端（推荐）
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --dtype float16

# 或者使用其他模型
vllm serve Qwen/Qwen2.5-3B-Instruct \
    --port 8000 \
    --dtype float16
```

注意：确保 vLLM 版本支持结构化输出功能（通常需要较新的版本）。

## 使用方法

### 1. 直接调用 API 生成数据

使用配置好的 `api_info.json` 文件运行脚本：

```bash
python data_process/get_llm_output.py \
    --dataset Instruments \
    --api_info ./api_info.json \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --response_format json_schema \
    --batchsize 16
```

**参数说明：**
- `--response_format json_schema`: 使用 JSON Schema 结构化输出（推荐）
- `--response_format json_object`: 使用 JSON Object 模式
- `--model_name`: 模型名称，需要与启动 vLLM 服务时使用的模型名称一致

### 2. 生成批量提交文件

如果需要生成批量提交文件用于离线批量推理：

```bash
python data_process/get_llm_output.py \
    --dataset Instruments \
    --api_info ./api_info.json \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --response_format json_schema \
    --generate_batch_file
```

这会生成 JSONL 格式的批量提交文件，可以直接提交到 vLLM 服务进行批量推理。

## 工作原理

### JSON Schema 模式（推荐）

当使用 `response_format=json_schema` 时：

1. **代码自动检测 vLLM 服务**：通过检查 `base_url` 是否包含 localhost/127.0.0.1/0.0.0.0 来判断
2. **使用 guided_json**：将 Pydantic 模型的 JSON Schema 通过 `extra_body` 的 `guided_json` 参数传递给 vLLM
3. **自动解析**：优先尝试使用 `client.beta.chat.completions.parse()` 方法自动解析响应为 Pydantic 对象
4. **回退机制**：如果自动解析失败，会回退到使用 `create()` 方法配合 `guided_json`，然后手动解析 JSON 字符串

### JSON Object 模式

当使用 `response_format=json_object` 时：

- vLLM 使用标准的 `response_format={"type": "json_object"}` 参数
- 返回 JSON 字符串，需要手动解析

## 代码实现细节

### 1. vLLM 检测函数

`utils.py` 中的 `_is_vllm_service()` 函数用于检测是否为 vLLM 服务：

```python
def _is_vllm_service(base_url):
    """检测是否为vLLM服务（本地部署）"""
    if base_url is None:
        return False
    vllm_indicators = ['localhost', '127.0.0.1', '0.0.0.0']
    base_url_lower = base_url.lower()
    return any(indicator in base_url_lower for indicator in vllm_indicators)
```

### 2. get_res_batch 函数

`get_res_batch()` 函数会自动处理 vLLM 和 OpenAI 兼容 API 的差异：

- **vLLM**: 使用 `extra_body` 传递 `guided_json` 和 `guided_decoding_backend`
- **OpenAI 兼容 API**: 使用标准的 `response_format` 参数

### 3. Pydantic 模型

代码中定义了 `UserAnalysisResponse` Pydantic 模型来描述输出结构：

```python
class UserAnalysisResponse(BaseModel):
    general_preference: str
    long_term_preference: str
    short_term_preference: str
    user_related_intention: str
    item_related_intention: str
```

vLLM 会将输出严格遵循这个模型的 JSON Schema。

## 故障排查

### 1. 连接错误

如果出现连接错误，检查：
- vLLM 服务是否正在运行
- `base_url` 是否正确
- 端口号是否匹配

### 2. 结构化输出失败

如果结构化输出失败：
- 确保 vLLM 版本支持结构化输出（需要较新版本）
- 检查 `guided_decoding_backend` 是否已安装（如 outlines）
- 尝试使用不同的 `guided_decoding_backend`（如 `lm-format-enforcer`）

### 3. 解析错误

如果出现 JSON 解析错误：
- 检查模型输出是否符合 JSON Schema
- 查看错误日志中的原始输出内容
- 确保 Pydantic 模型定义正确

## 示例

完整的使用示例：

```python
from data_process.utils import get_res_batch
from data_process.get_llm_output import UserAnalysisResponse

# 配置 vLLM
api_info = {
    "api_key": "dummy",
    "base_url": "http://localhost:8000/v1",
    "use_vllm": True,
    "guided_decoding_backend": "outlines"
}

# 准备 prompt
prompts = ["分析用户的购买历史和偏好..."]

# 调用 API（自动使用 vLLM 结构化输出）
results = get_res_batch(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    prompt_list=prompts,
    api_info=api_info,
    response_format=UserAnalysisResponse  # 使用 Pydantic 模型
)

# results 是 UserAnalysisResponse 对象的列表
for result in results:
    print(result.general_preference)
    print(result.long_term_preference)
```

## 注意事项

1. **模型一致性**：确保 `--model_name` 参数与启动 vLLM 服务时使用的模型名称一致
2. **性能**：结构化输出可能会稍微降低推理速度，但能保证输出格式的准确性
3. **兼容性**：代码同时支持 vLLM 和 OpenAI 兼容 API，可以通过 `base_url` 自动切换
4. **版本要求**：需要较新版本的 vLLM 和 OpenAI Python SDK 才能支持结构化输出功能
