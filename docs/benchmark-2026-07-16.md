# 2026-07-16 B200 真实业务基线

## 测试对象

- GPU：NVIDIA B200，183359 MiB
- 模型：Qwen3.6-35B-A3B NVFP4
- 引擎：SGLang 0.5.15.post1，commit `5b386b4d68142e350563315bc46b0f16a9534cfb`
- DFlash：block size 8
- 后端：`trtllm_mha` + FA4 + `flashinfer_trtllm`
- TP：1
- Prefix cache：开启
- 样本：28 条真实业务请求
- 统计范围：B200 SGLang 服务端，不含业务机到 Vast 的网络

## 聚合结果

| 指标 | 结果 | 含义 |
|---|---:|---|
| B200 处理耗时 P50 | 2.06 s | queue + forward |
| B200 处理耗时 P95 | 4.35 s | 95% 请求不超过该值 |
| 有效输出速度 P50 | 532 tok/s | output tokens / B200 总处理时间，含 Prefill |
| 最大排队 | 0.99 ms | 当时没有 GPU 排队瓶颈 |
| 热身后 Prefix Cache 命中 | 82.3% | 排除前两条冷前缀后，按输入 token 加权 |
| DFlash 平均接受长度 | 2.95 token/step | Decode 日志观测 |
| DFlash 平均接受率 | 27.8% | Decode 日志观测 |
| 单请求内部 Decode P50 | 562 tok/s | 去掉 Prefill/空闲低速样本 |
| 双请求聚合 Decode P50 | 996 tok/s | 仅 21 个自然业务日志样本 |

## 解释

28 条请求中 23 条在 3 秒内完成，3 条超过 4 秒。服务端耗时和输出长度的相关系数为 0.94；最慢请求输出 4098 token、处理 4.82 秒，但仍达到约 850 tok/s。因此这批流量的尾延迟主要来自生成内容长，不是 GPU 排队。

这里的 532 tok/s 是包含 Prefill 的“整条请求有效速度”，不是纯 Decode kernel 速度，也不包含业务机网络时间。双并发 996 tok/s 的样本很少，只能作为线上观察，不能当容量上限。

本仓库不保存这批请求的 prompt、输出、原始日志或请求级明细。
