# Qwen3.6-35B-A3B on Vast.ai B200

在一台新的 Vast.ai B200 上重建经过实测的 Qwen3.6-35B-A3B 推理服务。部署形态是：

- 主模型：`nvidia/Qwen3.6-35B-A3B-NVFP4`
- 草稿模型：`z-lab/Qwen3.6-35B-A3B-DFlash`
- SGLang + DFlash，TP=1
- NVFP4 GEMM / MoE：`flashinfer_trtllm`
- 主模型注意力：`trtllm_mha`
- 草稿注意力：FA4
- DFlash block size：8
- Prefix caching：启用（SGLang radix cache 默认开启，没有传 `--disable-radix-cache`）
- OpenAI 兼容 API：`127.0.0.1:30000`
- 可选分段计时代理：`127.0.0.1:30001`

这套配置固定了 2026-07-16 实际在线机器的 SGLang commit、模型 revision 和核心 CUDA/Python 包版本。它复现的是实测的 **B8** 配置，不是早期设想的 B16。

## 新机器一键部署

前提：

- Vast.ai NVIDIA B200，约 180 GB VRAM；
- 使用带 `/etc/vast-agents-guide.md`、Supervisor、CUDA 13、Python 3.12 和 `uv` 的 Vast base image；
- 建议至少 80 GB 磁盘。模型本身约 23 GB，Python/CUDA 依赖还会占用较多空间；
- 私有 GitHub 仓库的读取权限。

SSH 进入新实例后执行：

```bash
git clone git@github.com:si-yu-aa/qwen36-b200-deploy.git
cd qwen36-b200-deploy

read -rsp 'Qwen API key: ' QWEN36_API_KEY
echo
export QWEN36_API_KEY

# 公共模型通常可匿名下载；设置 token 可减少限流。
export HF_TOKEN='hf_...'

./deploy.sh
```

`deploy.sh` 会：

1. 确认机器是 B200 且显存足够；
2. 检出固定的 SGLang fork 和 commit；
3. 创建隔离 venv 并安装固定的 SGLang/CUDA 13 依赖；
4. 按固定 Hugging Face revision 下载主模型和 DFlash 草稿模型；
5. 把 API key 写入权限为 0600 的实例本地文件，不写入仓库；
6. 安装并启动两个 Supervisor 服务；
7. 等待模型 ready，核对实际进程参数并发送一次 smoke request。

部署和下载可重复执行。已有模型、源码和 API key 会复用。常用开关：

```bash
SKIP_MODEL_DOWNLOAD=1 ./deploy.sh   # 外部 Volume 已有模型时
SKIP_PYTHON_INSTALL=1 ./deploy.sh   # venv 已经完整时
DFLASH_BLOCK_SIZE=16 ./deploy.sh    # 仅用于明确的 B8/B16 A/B 测试
```

## 从业务机访问

服务默认不暴露公网端口。在业务机建立 SSH 隧道：

```bash
ssh -N \
  -L 31000:127.0.0.1:30000 \
  -L 31001:127.0.0.1:30001 \
  -p <VAST_SSH_PORT> root@<VAST_HOST>
```

业务配置：

```text
base_url = http://127.0.0.1:31000/v1
model    = Qwen3.6-35B-A3B
api_key  = 部署时的 QWEN36_API_KEY
```

对当前 TongAI 项目，API 地址就是 `http://127.0.0.1:31000/v1`。SSH 隧道断开时本地端口会停止服务，因此生产使用建议通过 systemd 或 autossh 保活。

## 验证和运维

在 B200 上：

```bash
cd qwen36-b200-deploy
./verify.sh
supervisorctl status qwen36-nvfp4 qwen36-timing-proxy
nvidia-smi
```

`verify.sh` 不会打印 API key；它会确认 NVFP4、DFlash、B8、FA4、`trtllm_mha` 和 FlashInfer 参数确实出现在运行进程中，并执行真实生成请求。

如需改启动参数，修改 [scripts/qwen36-nvfp4-sglang.sh](scripts/qwen36-nvfp4-sglang.sh)，提交后在实例上 pull 并重跑 `./deploy.sh`。所有版本入口集中在 [versions.env](versions.env)。

## 真实业务流式压测

计时代理只记录时间戳、字节数和请求 ID，不保存请求正文或模型输出。先把本地 `31001` 隧道连接到远端计时代理，然后运行：

```bash
mkdir -p benchmark-output
uv run --with httpx python bench/bench_business_timed_ssh.py \
  --csv /path/to/business.csv \
  --base-url http://127.0.0.1:31001/v1 \
  --api-key-file /secure/path/qwen-api-key \
  --concurrency 1,2,4,8 \
  --requests 20 \
  --max-tokens 1024 \
  --output benchmark-output/run.json
```

CSV 需要 `prompts` 列（OpenAI messages JSON），可选 `tools` 和 `trace_id` 列。CSV 和原始 JSON 输出都被 `.gitignore` 排除。

分段指标的直白含义：

- `uplink_to_b200_full_body_est`：本机发请求到 B200 收完请求体；
- `b200_body_complete_to_first_token`：B200 收完请求到第一个 token 准备好；
- `first_token_downlink_est`：B200 发第一个 token 到本机收到；
- `b200_generation_first_to_done`：B200 上从首 token 到生成结束；
- `last_token_downlink_est`：B200 发完最后数据到本机收完；
- `round_trip_link_and_client_overhead`：端到端耗时扣掉 B200 代理内服务耗时后的网络与客户端总开销。

单向网络时间依赖客户端/服务器时钟校准，只能视为估计值；报告里的 `one_way_uncertainty_bound_ms` 给出由最佳 RTT 推导的误差上界。端到端、B200 服务端耗时和双向网络总开销更可靠。

## 数据与密钥边界

仓库只保存代码、固定 revision、聚合后的性能结论和无敏感信息的配置模板。以下内容不会提交：

- Qwen API key、Hugging Face token；
- 模型权重、venv、服务日志；
- 业务 CSV、prompt、模型输出；
- 请求级 benchmark JSON/JSONL、SQLite。

Vast 实例的 `/workspace` 是否持久化取决于是否挂载了 Volume。Git 仓库负责重建代码和环境；如果希望避免每次重新下载约 23 GB 模型，应把 `/workspace/models` 放到 Vast Volume，并在新机器上设置相同挂载路径。
