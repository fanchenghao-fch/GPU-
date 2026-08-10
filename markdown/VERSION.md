# GPU 卡数计算器 — 版本说明

> 版本：v1.1.0  
> 日期：2026-08-10  
> 状态：生产就绪（Vercel 已部署）

---

## 一、项目概述

GPU 卡数计算器是一个轻量级 Web 工具，面向销售人员快速估算"跑某个 AI 模型需要多少张 GPU 卡"。纯前端 SPA，无后端/登录/数据库/分享。

**核心公式**：`卡数 = ceil(模型所需显存 / 单卡可用显存)`

- 部署地址：[https://gpu-card-calculator.vercel.app](https://gpu-card-calculator.vercel.app)
- 仓库地址：[https://github.com/fanchenghao-fch/GPU-](https://github.com/fanchenghao-fch/GPU-)

---

## 二、技术栈

| 层 | 选型 | 版本 |
|----|------|------|
| 运行时 | Node.js (ESM) | 20+ |
| 构建 | Vite | 5.4+ |
| UI 框架 | React | 18.3 |
| CSS | Tailwind CSS | 3.4 |
| 测试 | Node.js 原生 test runner | — |
| 部署 | Vercel | — |

---

## 三、文件清单（25 个源文件）

```
项目根目录/
├── calculator/                         # 计算引擎（纯 JS，零运行时依赖）
│   ├── index.js                        # 主入口 calculate() / getSelectOptions()
│   ├── index.test.js                   # 103 项自动化测试
│   ├── constants/
│   │   ├── gpu-data.js                 # GPU 产品常量（2 款：N300 / C600）
│   │   ├── model-data.js               # 模型常量（30 个，覆盖 6 种注意力架构）
│   │   └── precision.js                # 精度常量（7 种：FP32→INT4）
│   ├── formulas/
│   │   ├── model-memory.js             # 模型显存计算（权重 + KV Cache + 其他）
│   │   └── gpu-memory.js               # GPU 显存计算（标称 → 驱动可见 → 可用）
│   └── presets/
│       └── sla-presets.js              # SLA 预设（5 推理 + 2 训练）
├── src/                                # UI 层（React + Tailwind）
│   ├── main.jsx                        # ReactDOM 入口
│   ├── index.css                       # Tailwind directives
│   ├── App.jsx                         # 根组件：居中布局
│   ├── hooks/
│   │   └── useCalculator.js            # 状态管理 hook（useState + useMemo）
│   └── components/
│       ├── Calculator.jsx              # 主毛玻璃卡片容器
│       ├── Selector.jsx                # 定制 <select> 下拉控件
│       ├── ResultCard.jsx              # 结果面板（卡数/服务器/利用率）
│       ├── MemoryBreakdown.jsx         # 显存明细柱状图
│       └── FormulaSteps.jsx            # 计算步骤编号列表
├── index.html                          # Vite 入口 HTML
├── vite.config.js                      # Vite 配置（@calculator 别名）
├── tailwind.config.js                  # Tailwind 配置
├── postcss.config.js                   # PostCSS 配置
├── vercel.json                         # Vercel 部署配置
├── package.json                        # 项目元信息与脚本
├── .gitignore                          # Git 忽略规则
├── ui-design.md                        # UI 层设计文档（参考）
└── VERSION.md                          # 本文件
```

---

## 四、计算引擎能力

### 4.1 计算公式

来源：`calculator/formulas/gpu-memory.js` + `calculator/formulas/model-memory.js`

核心链路：

```
标称显存 → 驱动可见 → GPU 可用显存
                              ↓
模型参数 × 精度 → 权重显存 ──→ 模型总显存 → 卡数
                    KV Cache ──→
                    其他显存 ──→
```

---

#### 4.1.1 GPU 显存计算链（gpu-memory.js）

**第一层 — 驱动可见显存：**

`driverVisible = nominalGB × driverEfficiency`

- 标称显存扣除 ECC 预留、GPU 固件、显存位宽损耗（约 3%）
- driverEfficiency 典型值 = 0.97

**第二层 — 模型可用显存：**

`usable = driverVisible × inferenceRatio`

- 扣除 CUDA Context / 驱动运行时开销（kernel launch、cuDNN/cuBLAS 内部缓存）
- 推理场景 inferenceRatio 典型值 = 0.85–0.90

**第三层 — 一步计算（calcGPUUsableMemory）：**

`usable = nominalGB × driverEfficiency × inferenceRatio`

- 合并系数 = driverEfficiency × inferenceRatio（典型值 0.97 × 0.90 = 0.873）
- N300: `48 × 0.873 = 41.90 GB`
- C600: `144 × 0.873 = 125.71 GB`

> ⚠️ GPU 侧开销（此处 10–15%）与模型侧"其他显存"是**独立的两块**：
> - GPU 侧 → CUDA context / 驱动运行时（固定开销，不随模型变化）
> - 模型侧 → 激活值 / 临时缓冲区（随模型和 batch 变化，见 4.1.4）

---

#### 4.1.2 模型权重显存（calcWeightMemory）

`weightGB = paramsB × 10^9 × bytesPerParam / 1024^3`

- paramsB = 模型总参数量（B = 十亿），**MoE 按全量专家计算**
- bytesPerParam 由精度决定（详见 4.4 精度表）
- 例：DeepSeek-R1-Distill-Llama-70B（70.6B）@ FP16
  → `70.6 × 10^9 × 2 / 1024^3 ≈ 131.5 GB`

---

#### 4.1.3 KV Cache 显存（calcKVCache）

统一换算：

`kvCacheGB = elements × bytesPerKV / 1024^3`

- bytesPerKV 默认与权重精度一致（当前版本不分离指定 KV 精度）
- elements 按 `attnArch` 分支计算：

| 架构 | elements 公式 | 说明 |
|------|-------------|------|
| **standard** | `2 × numLayers × numKVHeads × headDim × ctx × batch` | ×2 覆盖 K 和 V；GQA 模型 kvDim < hiddenDim |
| **mla** | `numLayers × (kvLoraRank + qkRopeHeadDim) × ctx × batch` | 无 ×2，K/V 压缩为联合隐向量 |
| **cla** | `2 × (numLayers / claShareFactor) × numKVHeads × headDim × ctx × batch` | 每 claShareFactor 层共享一份 KV Cache |
| **linear_hybrid** | `2 × fullAttnLayers × numKVHeads × headDim × ctx × batch` | 仅全注意力层产生 KV Cache；M2.5 全 Lightning 层(fullAttnLayers=0)，KV Cache=0 |
| **kda_mla** | `fullAttnLayers × (kvLoraRank + qkRopeHeadDim) × ctx × batch` | KDA 层无 KV Cache |
| **hca_mla** | `effectiveKVDim × ctx × batch` | 逐层不等压缩比，使用预计算等效 KV 维度 |

计算示例：

- **standard** — R1-Distill-Llama-70B @ FP16, ctx=8192, batch=1  
  kvDim = 8 × 128 = 1024  
  elements = `2 × 80 × 1024 × 8192 × 1 = 1,342,177,280`  
  kvCache = `1,342,177,280 × 2 / 1024^3 ≈ 2.50 GB`

- **mla** — DeepSeek-V3.2 @ FP16, ctx=8192, batch=1  
  mlaDim = 512 + 64 = 576  
  elements = `61 × 576 × 8192 × 1 = 287,670,272`  
  kvCache = `287,670,272 × 2 / 1024^3 ≈ 0.54 GB`

- **hca_mla** — DeepSeek V4 Flash @ FP16, ctx=8192, batch=1  
  elements = `4176 × 8192 × 1 = 34,209,792`  
  kvCache = `34,209,792 × 2 / 1024^3 ≈ 0.064 GB`

---

#### 4.1.4 其他显存（calcOtherMemory）

`otherGB = (weightGB + kvCacheGB) × overheadRatio`

- 包含：激活值临时缓存 + 算子缓冲区 + 内存分配器碎片
- 推理场景默认 overheadRatio = 0.10（vLLM/TensorRT-LLM 实测 5–15%）
- 各 SLA 预设使用不同 overheadRatio（见 4.5）

---

#### 4.1.5 模型总显存（calcModelMemory）

`totalGB = round2(weightGB) + round2(kvCacheGB) + round2(otherGB)`

- 先各自 round2（保留 2 位小数），再求和 — 确保前端展示明细可加得上
- 例：R1-Distill-Llama-70B @ FP16, ctx=8192, batch=1, overhead=0.10  
  weight = 131.50 GB  
  kvCache = 2.50 GB  
  other = (131.50 + 2.50) × 0.10 = 13.40 GB  
  total = 131.50 + 2.50 + 13.40 = **147.40 GB**

---

#### 4.1.6 最终卡数（calculate）

```
cards       = ceil(totalGB / singleGPUUsableGB)
servers     = ceil(cards / cardsPerServer)
utilization = totalGB / (cards × singleGPUUsableGB)
```

- singleGPUUsableGB 来自 4.1.1 GPU 显存计算链
- cardsPerServer 来自 GPU 产品配置（N300=16, C600=8）
- 例：147.40 GB ÷ 41.90 GB = 3.52 → ceil = **4 张 N300**，1 台服务器，利用率 88.0%

---

### 4.2 GPU 产品（2 款）

| ID | 名称 | 标称显存 | 单台卡数 | 驱动效率 | 推理可用率 | 单卡可用 |
|----|------|----------|----------|----------|------------|----------|
| n300 | N300 | 48 GB | 16 | 0.97 | 0.90 | 41.90 GB |
| c600 | C600 | 144 GB | 8 | 0.97 | 0.90 | 125.71 GB |

### 4.3 模型库（30 个）

按注意力架构分类：

| 架构类型 | 数量 | 代表模型 |
|----------|------|----------|
| standard (标准 GQA) | 17 | R1-Distill 系列, Qwen3-235B, GLM-4.5/4.5-Air, Hunyuan-A13B, Hy3 |
| mla (Multi-head Latent Attention) | 4 | DeepSeek-V3.2, DeepSeek-V4.1, Kimi-K2.6, GLM-5.1/5.2 |
| linear_hybrid (混合线性注意力) | 6 | MiniMax-M1/M2.5/M2.7/M3, Qwen3.5 系列, Qwen3.6 系列 |
| kda_mla (KDA + Gated MLA) | 1 | Kimi K3 |
| hca_mla (HCA + MLA) | 2 | DeepSeek V4 Flash, V4 Pro |

覆盖参数量范围：0.8B（Qwen3.5-0.8B）→ 2.8T（Kimi K3），含 Dense 和 MoE 两种参数架构。

### 4.4 精度（7 种）

| 精度 | 字节/参数 | 典型场景 |
|------|-----------|----------|
| FP32 | 4 | 传统训练 |
| TF32 | 2 | NVIDIA Ampere+ 训练 |
| BF16 | 2 | 现代大模型训练 |
| FP16 | 2 | 推理常用 |
| FP8 | 1 | H100+ 推理 |
| INT8 | 1 | 量化推理 |
| INT4 | 0.5 | 极端压缩推理 |

### 4.5 SLA 预设（7 个）

**推理类（5 个）：**

| ID | 名称 | 上下文 | 并发 | 开销 |
|----|------|--------|------|------|
| inference-light | 轻量推理 | 4K | 1 | 5% |
| inference-standard | 标准推理 | 8K | 1 | 10% |
| inference-batch | 批量推理 | 8K | 8 | 12% |
| inference-long | 长上下文推理 | 128K | 1 | 12% |
| inference-xlong | 超长上下文推理 | 256K | 1 | 12% |

**训练类（2 个）：**

| ID | 名称 | 上下文 | 并发 | 开销 | 备注 |
|----|------|--------|------|------|------|
| training-lora | LoRA 微调 | 4K | 1 | 20% | ⚠️ 粗略估算 |
| training-full | 全量微调 | 4K | 1 | 30% | ⚠️ 偏保守 |

---

## 五、UI 设计

### 5.1 视觉风格

Apple 毛玻璃风格（Frosted Glass），三层透明度：

| 层级 | 元素 | 背景 | 模糊 |
|------|------|------|------|
| 页面背景 | `<body>` | `gradient: slate-100 → sky-50 → indigo-100` | — |
| 主卡片 | Calculator | `bg-white/40` | `backdrop-blur-[40px]` |
| 子面板 | ResultCard / MemoryBreakdown / FormulaSteps | `bg-white/25` | `backdrop-blur` |
| 下拉 | `<select>` | `bg-white/25` | `backdrop-blur` |

统一边框：`border border-white/30 shadow-xl`

### 5.2 组件结构

```
App (min-h-screen, items-start, justify-center)
└── Calculator (max-w-lg, 主毛玻璃卡片)
    ├── Selector ×4 (模型 / GPU / SLA 预设 / 精度)
    ├── ResultCard (卡数大号数字 + 服务器/利用率三列)
    ├── MemoryBreakdown (权重/KV Cache/其他 柱状图)
    └── FormulaSteps (计算步骤编号列表)
```

### 5.3 交互设计

- **默认值**：四个下拉均默认选中第一项 → 页面加载即有计算结果，无空白态
- **实时更新**：任一选择变化即时重算（`useMemo` 响应式）
- **错误处理**：`try/catch` 包裹计算，异常展示为红色提示文本
- **响应式**：375px+ 视口适配，移动端 `px-4` 边距 + 触控友好的 44px+ 交互区域

---

## 六、测试覆盖（103 项）

| 测试分组 | 数量 | 覆盖内容 |
|----------|------|----------|
| 精度常量 | 7 | 所有精度 + 未知精度异常 |
| 权重显存 | 5 | Dense/MoE @ FP16/INT8/INT4 |
| KV Cache (standard) | 5 | 上下文/批次的线性缩放 |
| KV Cache (MLA) | 7 | DeepSeek-V2/V3 + 辅助函数 |
| KV Cache (CLA) | 2 | Hunyuan-Large 跨层共享 |
| KV Cache (linear_hybrid) | 4 | MiniMax-M1 / Qwen3.6 |
| KV Cache (kda_mla) | 4 | Kimi K3 KDA+MLA |
| KV Cache (hca_mla) | 4 | DeepSeek V4 Flash + V4 Pro HCA |
| GPU 显存 | 4 | N300 / C600 可用显存 |
| 模型总显存 | 2 | 组合计算校验 |
| 端到端 calculate() | 16 | 覆盖所有 6 种架构 + 2 款 GPU |
| SLA 预设 | 13 | 7 预设校验 + 排序 + 批次 |
| 新预设端到端 | 6 | 128K/256K + 各架构长上下文 |
| 边界条件与回归 | 9 | 利用率/服务器/显存一致性/30 模型全量 |
| 一致性检查 | 8 | 线性缩放 + 字段完整性 |
| **合计** | **103** | |

---

## 七、部署信息

| 项目 | 详情 |
|------|------|
| 平台 | Vercel |
| 地址 | [https://gpu-card-calculator.vercel.app](https://gpu-card-calculator.vercel.app) |
| 构建命令 | `npm run build` |
| 输出目录 | `dist` |
| 框架 | 无（静态 SPA） |
| 自动部署 | 待手动连接 GitHub |

---

## 八、版本历史

| 提交 | 日期 | 说明 |
|------|------|------|
| `c80aef4` | 2026-08-09 | 初始化 — 公式模块 v1.0（6 种 KV Cache 架构 + 24 模型 + 2 GPU + 7 精度 + 7 SLA 预设 + 104 测试） |
| `040c18c` | 2026-08-09 | 文档 — UI 层设计方案（ui-design.md） |
| `dab9a3c` | 2026-08-09 | 实现 — React + Vite + Tailwind 毛玻璃风格 UI（5 组件 + 1 hook） |
| `477804c` | 2026-08-09 | 维护 — 添加 .vercel 到 .gitignore |
| `e165bac` | 2026-08-10 | 修复 — HF config.json 全量核验，修正 12 款模型参数（层数/hiddenDim/KV heads 等）；模型库扩充至 30 款 |

---

## 九、已知限制

1. **训练显存估算较粗略**：仅包含权重+梯度+优化器状态，不含完整激活值反传，建议实测。LoRA 微调与全量微调预设已标注 ⚠️ 警告。
2. **KV Cache 为理论下限**：不计 attention mask 零头、padding 对齐、内存分配器碎片等实现细节（通常 ≤5% 误差）。
3. **KV Cache 使用权重精度**：实际部署中可分离指定 KV 精度（如权重 INT4 + KV FP16），当前版本统一使用选择的精度。可保守地手动选择较高精度覆盖此场景。
4. **不计分布式策略**：不建模 tensor/pipeline/expert parallelism 或序列并行的通信开销。大规模部署请参考 vLLM/TensorRT-LLM 实际基准，本工具提供初筛而非精确值。
5. **MoE 全量加载**：权重显存按总参数量（非激活参数量）计算，所有专家权重均计入。不计 expert offloading 等优化。
6. **无持久化/分享功能**：纯前端工具，无登录、无数据库、无分享链接。用户需截图保存结果。
7. **MiniMax-M2.5 KV Cache 为零**：M2.5 全部 62 层使用 Lightning 线性注意力（HF config `attn_type_list` 全为 1），无标准 full-attention 层，因此 `fullAttnLayers=0`，KV Cache=0 GB。线性注意力使用固定大小循环状态（不随 context 增长），其显存开销已包含在"其他显存"的 overhead 中。