# VLLM\-Gaudi 性能压测工程 README

## 一、工程简介

本项目为 **Gaudi2 \+ VLLM\-FP8** 推理性能自动化压测体系，针对 Qwen3\.5\-122B\-A10B\-FP8 大模型深度适配，彻底解决传统脚本硬编码、不可复用、参数混乱、无法批量寻优的问题。

核心设计：**JSON配置驱动 \+ 统一Python调度器**

零代码修改，仅通过修改配置文件即可完成三类核心测试场景，全程自动化：容器启停、服务等待、预热、压测、日志记录、参数遍历、结果汇总。

适配三种工作流，完全匹配日常迭代需求：

1. **基础冒烟验收**：模型部署、代码合并、版本升级后快速验证基线性能，可直接输出汇报指标。

2. **单场景并发遍历**：固定推理参数，遍历并发梯度，测出场景性能上限与吞吐曲线。

3. **超参自动寻优**：遍历Chunked Prefill、投机解码、max\-seqs、batch\-tokens等核心参数，自动启停容器消除缓存干扰，输出最优推理配置。

## 二、目录结构

```shell
benchmarks/
├── README.md                 # 工程使用文档（本文档）
├── run_bench.py              # 统一压测调度入口（核心脚本）
├── configs/                  # 全套场景配置文件
│   ├── base_smoke.json       # 场景1：基础冒烟验收配置
│   ├── scene_sweep.json      # 场景2：单场景并发遍历配置
│   └── param_search.json     # 场景3：超参自动寻优配置
└── perf_logs/                # 自动生成：日志、结果、汇总文件
```

## 三、环境依赖

### 1\. 基础依赖

- Python3\.8\+

- evalscope（压测核心工具）

- docker（容器启停、模型服务部署）

### 2\. 镜像与模型

- Gaudi镜像：`vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/vllm-0.26.0-ptupstream-2.11.0:latest`

- 测试模型：Qwen3\.5\-122B\-A10B\-FP8

## 四、核心功能特性

- **全配置化**：模型路径、端口、HPU卡号、推理参数、压测用例全部JSON配置，无代码硬编码。

- **智能服务就绪检测**：自动轮询`/v1/models`接口，替代固定休眠，精准等待服务启动完成。

- **自适应预热与样本数**：根据输入长度、并发数自动计算warmup次数、测试样本数，也支持手动覆盖。

- **两种运行模式**：
        

    - 常驻容器模式：单次启容器，批量跑多用例（适合冒烟、并发遍历）

    - 单参数单容器模式：每组参数独立启停容器，彻底消除Recipe缓存、显存残留干扰（适合参数寻优）

- **安全机制**：自动清理历史容器、单用例冷却、失败不中断批量任务、Dry\-run预校验。

- **结构化日志**：所有日志带时间戳，参数寻优自动生成`summary.json`汇总结果。

## 五、快速使用教程

### 通用前置操作

所有命令均在项目根目录执行，支持**预校验\+正式运行**两步操作。

#### 1\. 基础冒烟验收（日常迭代必跑）

适用场景：新模型部署、vllm源码更新、补丁合并、版本升级后的基线验证，输出可汇报的基础性能指标。

```shell
# 1. 预校验：预览容器命令、压测命令，检查配置是否正确
python benchmarks/run_bench.py --config benchmarks/configs/base_smoke.json --dry-run

# 2. 正式执行自动化压测
python benchmarks/run_bench.py --config benchmarks/configs/base_smoke.json
```

覆盖场景：纯解码、长短Prefill、混合推理、50K超长上下文，全覆盖验收。

#### 2\. 单场景并发遍历（性能摸底）

适用场景：固定最优推理参数，遍历1/2/4/8/16/32梯度并发，测试超长上下文场景吞吐上限、稳定性。

```shell
# 预校验
python benchmarks/run_bench.py --config benchmarks/configs/scene_sweep.json --dry-run

# 正式运行
python benchmarks/run_bench.py --config benchmarks/configs/scene_sweep.json
```

#### 3\. 超参自动寻优（最优参数调研）

适用场景：迭代优化推理性能，批量测试Chunked Prefill、投机解码、max\-num\-seqs、batch\-tokens等核心参数组合。

自动流程：清旧容器→启新容器→等待服务就绪→压测→销毁容器→下一组参数循环，全程无人值守。

```shell
# 预校验
python benchmarks/run_bench.py --config benchmarks/configs/param_search.json --dry-run

# 正式运行参数寻优
python benchmarks/run_bench.py --config benchmarks/configs/param_search.json
```

## 六、配置文件详解（可自定义修改）

### 1\. 全局通用配置

- `docker_image`：Gaudi\-vllm镜像地址，固定无需修改

- `log_dir`：日志输出目录，默认`./perf_logs`

- `server_ready_timeout`：服务启动超时时间，默认600s

### 2\. 模型配置 model

- `name/path/tokenizer_path`：模型名称、本地路径、分词器路径

- `dtype/quantization`：数据精度、量化方式（当前固定bf16\+FP8）

- `tensor_parallel_size`：张量并行卡数，默认4卡

- `vllm_args`：vllm启动超参（上下文长度、批次参数、功能开关等）

### 3\. 硬件环境配置 env

包含HPU设备绑定、缓存策略、FP8开关、分桶策略等Gaudi专属环境变量，根据硬件环境微调即可。

### 4\. 压测用例 cases

支持四种场景类型：`decode_only/prefill_only/mixed/long_ctx`

参数：并发数、输入长度、输出长度，脚本自动适配预热次数与采样数，支持手动覆盖。

### 5\. 参数寻优矩阵 param\_matrix

仅`param_search.json`生效，可自由扩展：

- 环境变量开关：Chunked Prefill、投机解码、KV Cache FP8等

- 推理超参：max\-num\-seqs、max\-batched\-tokens、显存利用率等

## 七、输出结果说明

1. **时序日志文件**：`perf_logs/xxx_20260915_xxxxxx.log`，完整保存所有容器启动、压测、命令输出日志，可追溯。

2. **参数寻优汇总文件**：`perf_logs/param_search_summary.json`，记录每一组参数的执行状态、是否报错，方便批量对比。

3. **核心指标**：天然输出 TTFT、首包延迟、Prefill吞吐、Decode吞吐、平均token耗时等evalscope标准指标。

## 八、常见问题与注意事项

- **端口冲突**：修改配置中`server.port`即可，自动适配API地址。

- **HPU设备占用**：脚本自动清理同名容器，避免设备占用，若异常残留可手动执行`docker rm -f 容器名`。

- **服务启动超时**：超大模型首次启动、缓存初始化较慢，可适当调大`server_ready_timeout`。

- **压测失败**：参数寻优模式下单组失败不中断任务，全部跑完后可通过summary文件定位异常参数。

## 九、迭代扩展方向

- 新增指标自动解析脚本，批量提取数据生成CSV/图表对比报表

- 新增多模型适配，一键切换不同模型压测配置

- 新增定时回归任务，适配CI/CD自动化性能回归测试

- 新增失败重试、资源监控、显存占用统计功能

## 十、Git 提交规范

功能迭代/配置更新统一提交说明：

- 新增脚本/功能：`bench: xxx feature support`

- 修改配置参数：`config: update xxx params for performance test`

- 修复bug：`fix: solve xxx benchmark error`

> （注：部分内容可能由 AI 生成）
