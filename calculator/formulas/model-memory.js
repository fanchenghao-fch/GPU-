/**
 * 模型显存计算公式
 *
 * 核心公式：模型所需显存 = 权重显存 + KV Cache 显存 + 其他显存
 *
 * 三项含义：
 * 1. 权重显存：模型参数本身占用的显存 = params × bytesPerParam
 * 2. KV Cache： 推理时缓存 Key/Value 矩阵，避免每步重新计算
 * 3. 其他显存：激活值临时缓存 + 算子缓冲区 + 分配器碎片
 *
 * ── 训练模式（TODO：后续版本）──
 * 训练显存 = 权重 + 梯度 + 优化器状态 + 激活值（远大于推理，需单独建模）
 */

import { bytesPerParam } from '../constants/precision.js';
import { getKVDim, getMLADim } from '../constants/model-data.js';

/**
 * 计算模型权重显存（GB）
 *
 * @param {number} paramsB     - 参数量（B = 十亿）
 * @param {string} precision   - 精度（如 'FP16'、'INT8'）
 * @returns {number} 权重显存（GB）
 *
 * 例：Llama-3.1-70B（Dense, 70.6B params）@ FP16
 *   → 70.6 × 10⁹ × 2 bytes = 141.2 GB
 *
 * 例：Mixtral 8×7B（MoE, 46.7B params, all experts）@ FP16
 *   → 46.7 × 10⁹ × 2 bytes = 93.4 GB
 *
 * ⚠️ MoE 说明：
 *   MoE 模型虽然每次推理只激活部分专家，但所有专家的权重都需加载到显存中。
 *   因此使用总参数量（非激活参数量）计​​算。
 */
export function calcWeightMemory(paramsB, precision) {
  const b = bytesPerParam(precision);
  // paramsB × 10^9 params × b bytes/param → GB
  const bytes = paramsB * 1e9 * b;
  return bytes / (1024 ** 3);
}

/**
 * 计算 KV Cache 显存（GB）—— 仅推理
 *
 * 根据模型的 attnArch 字段自动选择正确的计算公式：
 *
 *   standard: KV Cache = 2 × numLayers × kvDim × contextLen × batchSize × bytesPerKV
 *     - kvDim = numKVHeads × headDim
 *     - 适用于 Llama / Qwen / Mistral / Mixtral / ChatGLM 等标准 Transformer
 *
 *   mla:      KV Cache = numLayers × (kvLoraRank + qkRopeHeadDim) × contextLen × batchSize × bytesPerKV
 *     - 没有 ×2，因为 MLA 将 K 和 V 压缩为一个联合隐向量
 *     - 适用于 DeepSeek-V2/V3/R1、GLM-5.2（Multi-head Latent Attention）
 *
 *   cla:      KV Cache = 2 × (numLayers / claShareFactor) × kvDim × contextLen × batchSize × bytesPerKV
 *     - 有效层数 = numLayers / claShareFactor
 *     - 适用于 Hunyuan-Large（Cross-Layer Attention）
 *
 *   linear_hybrid: KV Cache = 2 × fullAttnLayers × kvDim × contextLen × batchSize × bytesPerKV
 *     - 仅 full-attention 层产生 KV Cache，线性注意力层使用固定大小状态（不随 ctx 增长）
 *     - 适用于 MiniMax-M1/M3（Lightning/MSA 混合）、Qwen3.5/3.6（Gated DeltaNet 混合）
 *     - 特例：MiniMax-M2.5 全 Lightning 层(fullAttnLayers=0)，KV Cache=0
 *
 *   kda_mla:  KV Cache = fullAttnLayers × (kvLoraRank + qkRopeHeadDim) × contextLen × batchSize × bytesPerKV
 *     - KDA（Kimi Dynamic Attention）层无 KV Cache，Gated MLA 层使用压缩 KV
 *     - 适用于 Kimi K3（KDA + Gated MLA 混合）
 *
 *   hca_mla:  KV Cache = effectiveKVDim × contextLen × batchSize × bytesPerKV
 *     - HCA（Hybrid Compression Attention）逐层不等压缩比，使用预计算等效 KV 维度
 *     - 适用于 DeepSeek V4 Flash/Pro（HCA + MLA + DSA）
 *
 * @param {object} model        - 模型配置对象（来自 model-data.js）
 * @param {number} contextLen   - 上下文长度（token 数）
 * @param {number} batchSize    - 并发数（同时处理的请求数）
 * @param {string} precision    - KV Cache 使用的精度（通常与权重精度一致）
 * @returns {number} KV Cache 显存（GB）
 *
 * 例：Llama-3.1-70B（standard）@ FP16, context=8192, batch=1
 *   kvDim = 8 × 128 = 1024
 *   KV Cache = 2 × 80 × 1024 × 8192 × 1 × 2 bytes
 *            = 2,684,354,560 bytes ≈ 2.50 GB
 *
 * 例：DeepSeek-V3（mla）@ FP16, context=8192, batch=1
 *   mlaDim = 512 + 64 = 576
 *   KV Cache = 61 × 576 × 8192 × 1 × 2 bytes
 *            = 575,340,544 bytes ≈ 0.54 GB
 */
export function calcKVCache(model, contextLen, batchSize, precision) {
  const b = bytesPerParam(precision);
  const attnArch = model.attnArch || 'standard';
  let elements;

  switch (attnArch) {
    case 'mla': {
      // DeepSeek MLA: K 和 V 压缩为联合隐向量，没有 ×2
      const mlaDim = getMLADim(model);
      elements = model.numLayers * mlaDim * contextLen * batchSize;
      break;
    }
    case 'cla': {
      // Hunyuan-Large CLA: 每 claShareFactor 层共享一份 KV Cache
      const effLayers = model.numLayers / model.claShareFactor;
      const kvDim = getKVDim(model);
      elements = 2 * effLayers * kvDim * contextLen * batchSize;
      break;
    }
    case 'linear_hybrid': {
      // MiniMax-M1 / Qwen3.6: 仅 full-attention 层有 KV Cache
      // Lightning/linear attention 层使用固定大小循环状态，不随 context 增长
      const kvDim = getKVDim(model);
      elements = 2 * model.fullAttnLayers * kvDim * contextLen * batchSize;
      break;
    }
    case 'kda_mla': {
      // Kimi K3: KDA 层无 KV Cache + Gated MLA 层使用 MLA 压缩
      const mlaDim = getMLADim(model);
      elements = model.fullAttnLayers * mlaDim * contextLen * batchSize;
      break;
    }
    case 'hca_mla': {
      // DeepSeek V4 Flash: HCA 逐层不等压缩比
      // 使用预计算等效 KV 维度（≈vLLM 实测 8.7× 节省 vs 标准 attention）
      elements = model.effectiveKVDim * contextLen * batchSize;
      break;
    }
    default: {
      // standard / gqa-head-dim 等标准模型
      const kvDim = getKVDim(model);
      elements = 2 * model.numLayers * kvDim * contextLen * batchSize;
      break;
    }
  }

  const bytes = elements * b;
  return bytes / (1024 ** 3);
}

/**
 * 计算"其他显存占用"（GB）
 *
 * 包含：激活值临时缓存 + 算子缓冲区 + 内存分配器碎片
 *
 * 这部分与模型架构、推理框架实现有关，无法精确计算。
 * 采用经验比例：取 (权重 + KV Cache) 的 overheadRatio 倍。
 *
 * 推理场景典型 overheadRatio = 0.10（即权重+KV Cache 的 10%）
 *
 * 这是保守估计，大部分推理框架（vLLM、TensorRT-LLM）的额外开销在 5-15% 之间。
 *
 * ⚠️ 注意：这个"其他显存" ≠ GPU 侧预留的 10-15%
 *   - GPU 侧预留 → CUDA context、驱动运行时（固定开销，跟模型无关）
 *   - 模型侧其他 → 激活值、临时缓冲区（随模型和 batch 大小变化）
 *   两者是独立的。GPU 侧开销已在 gpu-memory.js 中处理。
 */
export function calcOtherMemory(weightMemoryGB, kvCacheGB, overheadRatio = 0.10) {
  return (weightMemoryGB + kvCacheGB) * overheadRatio;
}

/**
 * 计算模型推理所需总显存（GB）
 *
 * @param {object} model        - 模型配置对象
 * @param {string} precision    - 精度
 * @param {number} contextLen   - 上下文长度
 * @param {number} batchSize    - 并发数
 * @param {number} overheadRatio - 其他显存比例（默认 0.10）
 * @returns {{ weightGB: number, kvCacheGB: number, otherGB: number, totalGB: number }}
 */
export function calcModelMemory(model, precision, contextLen, batchSize, overheadRatio = 0.10) {
  const weightGB = calcWeightMemory(model.paramsB, precision);
  const kvCacheGB = calcKVCache(model, contextLen, batchSize, precision);
  const otherGB = calcOtherMemory(weightGB, kvCacheGB, overheadRatio);

  // 先各自舍入，再求和 — 确保前端展示的明细加得上
  const w = round2(weightGB);
  const kv = round2(kvCacheGB);
  const o = round2(otherGB);
  const total = round2(w + kv + o);

  return {
    weightGB:   w,
    kvCacheGB:  kv,
    otherGB:    o,
    totalGB:    total,
    breakdown: `[${model.attnArch || 'standard'}] 权重=${weightGB.toFixed(1)}GB + KV Cache=${kvCacheGB.toFixed(1)}GB + 其他=${otherGB.toFixed(1)}GB`,
  };
}

function round2(v) {
  return Math.round(v * 100) / 100;
}
