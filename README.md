# vLLM Gaudi (wzx) - Inference Optimization & Benchmark

本项目基于 vLLM 针对 Intel Gaudi 加速卡进行推理优化、模型适配与性能基准测试。旨在探索并提升大语言模型（LLM）在 Gaudi 平台上的推理效率与吞吐量。

## 📌 项目简介

本仓库包含 `vllm-gaudi` 的定制化源码以及一套可扩展的基准测试框架。主要聚焦于：
- **推理优化**：针对 Gaudi 硬件特性的 vLLM 推理性能调优。
- **模型适配**：主流大语言模型在 Gaudi 平台上的 FP8/BF16 等精度适配与验证。
- **基准测试**：提供多卡并行（如 Gaudi 4卡）环境下的标准化 Benchmark 配置。

## 📂 目录结构

```text
.
├── benchmarks/       # 基准测试脚本与各模型的配置文件
├── vllm-gaudi/       # vLLM Gaudi 核心源码及项目骨架
└── .gitignore        # Git 忽略文件
