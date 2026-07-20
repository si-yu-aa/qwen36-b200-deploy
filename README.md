# Qwen3.6-35B-A3B on Vast.ai B200

在一台全新的 Vast.ai B200 上部署 Qwen3.6-35B-A3B NVFP4 + DFlash B8，并通过带 Bearer 鉴权的公网 OpenAI 兼容接口提供服务。默认不需要 SSH 端口转发。

## 已固定的运行栈

- SGLang：`ff6bbeb0055234774be911d9f76788393731ca85`
- DFlash 修复：HH1162 `fix-draft-unquant-override` 分支
- ModelOpt mixed loader：SGLang PR #30078，补丁 SHA256 固定并在部署时校验
- SGLang 包：`0.5.15.post1`
- 草稿模型：`z-lab/Qwen3.6-35B-A3B-DFlash@f181eece...`
- DFlash block size：8
- 主模型注意力：`trtllm_mha`
- 草稿注意力：FA4
- NVFP4 GEMM / MoE：`flashinfer_trtllm`
- KV cache：BF16；这也是 mmangkad checkpoint 与 DFlash FA4 兼容所需的设置
- TP：1

主模型支持两个固定版本：

| `MAIN_MODEL_VARIANT` | Hugging Face checkpoint | SGLang 量化路径 |
|---|---|---|
| `nvidia`（默认） | `nvidia/Qwen3.6-35B-A3B-NVFP4@491c2f1e...` | `modelopt_mixed` |
| `mmangkad` | `mmangkad/Qwen3.6-35B-A3B-NVFP4@4384ac3b...` | `modelopt_fp4` |

## 网络结构

```text
公网客户端
  -> 0.0.0.0:8000  Bearer 鉴权代理
  -> 127.0.0.1:30000  SGLang

本机计时工具
  -> 127.0.0.1:30001  Bearer 鉴权计时代理
  -> 127.0.0.1:30000  SGLang
```

API key 只由代理从权限为 `0600` 的文件读取，不再作为 SGLang 命令行参数，因此不会出现在 SGLang 的 `server_args` 日志中。两个代理的 HTTP keep-alive 默认是 60 秒。

## 创建 Vast 实例

必须在创建实例时为容器端口 `8000/tcp` 分配公网映射；Vast 不支持实例创建后再增加端口。部署脚本会读取 `VAST_TCP_PORT_8000` 或 `vast-capabilities` 验证映射，没有映射时提前报错。

建议配置：

- NVIDIA B200，约 180 GB VRAM；
- 至少 100 GB 磁盘；如果同时下载两个主模型，建议 150 GB 以上；
- 带 `/etc/vast-agents-guide.md`、Supervisor、Python 3.12 和 `uv` 的 Vast base image；
- 将 `8000/tcp` 加入实例端口配置。

`/workspace` 只有在 `vast-capabilities | jq '.instance.workspace_is_volume'` 返回 `true` 时才是持久卷。普通实例的 `destroy/recycle` 会删除源码、模型和 API key；`stop/start` 会保留。

## 一键部署

```bash
git clone git@github.com:si-yu-aa/qwen36-b200-deploy.git
cd qwen36-b200-deploy

# 公共模型可匿名下载；设置 token 可减少限流。
export HF_TOKEN='hf_...'

# 默认 NVIDIA checkpoint。
./deploy.sh

# 或部署 mmangkad checkpoint。
# MAIN_MODEL_VARIANT=mmangkad ./deploy.sh
```

首次未设置 `QWEN36_API_KEY` 时，脚本会自动生成 64 字符密钥：

```bash
cat /workspace/.qwen36_api_key
```

也可以在首次部署前自行指定：

```bash
read -rsp 'Qwen API key: ' QWEN36_API_KEY
echo
export QWEN36_API_KEY
./deploy.sh
```

脚本会：

1. 校验 B200 和显存；
2. 校验 `8000/tcp` 公网映射；
3. checkout 可公开获取的 SGLang 基线；
4. 下载并校验 PR #30078 补丁，再只应用运行时文件；
5. 创建 Python 3.12 venv，并允许安装固定版本所需的 prerelease CUDA 依赖；
6. 下载固定 revision 的主模型和 DFlash 草稿模型；
7. 安装三个 Supervisor 服务并等待 CUDA autotune/CUDA Graph 完成；
8. 验证参数、鉴权边界和真实生成请求。

第一次启动通常需要 5–20 分钟，主要耗时是 FlashInfer/TRTLLM autotune、PTX 编译和 CUDA Graph。后续启动会复用编译缓存，但 CUDA Graph 仍需重新捕获。

常用覆盖项：

```bash
MAIN_MODEL_VARIANT=mmangkad ./deploy.sh
SKIP_MODEL_DOWNLOAD=1 ./deploy.sh
SKIP_PYTHON_INSTALL=1 ./deploy.sh
DFLASH_BLOCK_SIZE=16 ./deploy.sh
PROXY_KEEP_ALIVE_SECONDS=120 ./deploy.sh
```

`SKIP_PUBLIC_PORT_CHECK=1` 仅用于确认没有公网映射的私有测试实例；代理仍会绑定容器端口 8000。

## 调用公网 API

公网端口不是容器内的 `8000`，而是 Vast 分配的 `$VAST_TCP_PORT_8000`：

```bash
echo "http://${PUBLIC_IPADDR}:${VAST_TCP_PORT_8000}/v1"
```

测试：

```bash
API_KEY=$(</workspace/.qwen36_api_key)
curl -N "http://${PUBLIC_IPADDR}:${VAST_TCP_PORT_8000}/v1/chat/completions" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}'
```

OpenAI Python client 应作为长生命周期对象复用，不要每个请求重新创建：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://PUBLIC_IP:PUBLIC_PORT/v1",
    api_key="YOUR_KEY",
)

response = client.chat.completions.create(
    model="Qwen3.6-35B-A3B",
    messages=[{"role": "user", "content": "你好"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

## 验证和版本检查

在 B200 上：

```bash
./verify.sh
supervisorctl status qwen36-nvfp4 qwen36-timing-proxy qwen36-public-api
nvidia-smi
```

`verify.sh` 会确认：

- 模型路径、DFlash B8、FA4、`trtllm_mha`、FlashInfer 和 BF16 KV 参数；
- API key 没有传给 SGLang；
- `30001` 和 `8000` 无密钥均返回 `401`；
- 公网代理绑定 `0.0.0.0:8000`；
- 通过鉴权代理可以完成真实生成。

不租 GPU 也可以验证所有远程 pin 是否仍然可获取：

```bash
./scripts/check-pins.sh
```

## 真实业务流式压测

CSV 需要 `prompts` 列（OpenAI messages JSON），可选 `tools` 和 `trace_id` 列：

```bash
mkdir -p benchmark-output
uv run --with httpx python bench/bench_business_stream.py \
  --csv /path/to/business.csv \
  --base-url "http://${PUBLIC_IPADDR}:${VAST_TCP_PORT_8000}/v1" \
  --api-key-file /workspace/.qwen36_api_key \
  --concurrency 1,2,4,8 \
  --requests 20 \
  --max-tokens 1024 \
  --output benchmark-output/run.json
```

原始业务 CSV、请求、输出、API key、模型、日志和 benchmark JSON 均被 `.gitignore` 排除，不应提交到仓库。

## 已修复的历史问题

- 不再引用远程仓库不存在的本地提交 `5b386b4d...`；
- 明确复现 `ff6bbeb... + PR #30078`；
- 安装 SGLang 时加入 `--prerelease=allow`；
- 兼容新版 Vast `logging.sh` 的参数约定；
- 增加公网 Supervisor 代理和 Bearer 鉴权；
- API key 不再进入 SGLang argv/日志；
- mmangkad 使用 BF16 KV，避免 FP8 KV 与 DFlash FA4 的 dtype 冲突；
- 非流式响应由代理直接返回，流式响应继续逐 token 转发；
- 代理 keep-alive 从默认 5 秒提升到 60 秒。
