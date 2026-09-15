# vLLM Gaudi - 推理优化与基准测试

本项目基于 vLLM，面向 Intel Gaudi 加速卡开展推理优化、模型适配与性能基准测试，旨在持续提升大语言模型（LLM）在 Gaudi 平台上的推理效率与吞吐量。

## 📌 项目简介

本仓库包含 `vllm-gaudi` 的定制化源码，以及一套可扩展的基准测试框架，主要聚焦于以下方向：

- **推理优化**：结合 Gaudi 硬件特性，对 vLLM 推理性能进行调优。
- **模型适配**：完成主流大语言模型在 Gaudi 平台上的 FP8 / BF16 等精度适配与验证。
- **基准测试**：提供多卡并行（如 Gaudi 4 卡）场景下的标准化 Benchmark 配置。

## 📂 目录结构

```text
.
├── benchmarks/       # 基准测试脚本与各模型配置文件
├── vllm-gaudi/       # vLLM Gaudi 核心源码及项目骨架
├── README.md         # 项目说明文档
└── .gitignore        # Git 忽略文件
