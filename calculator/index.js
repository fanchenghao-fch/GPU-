/**
 * GPU 卡数计算器 —— 主入口
 *
 * 使用方式：
 *   import { calculate } from './calculator/index.js';
 *   const result = calculate('llama3.1-70b', 'mxn200', 'inference-standard', 'FP16');
 *   // → { cards: 4, servers: 1, modelMemoryGB: ..., gpuUsableGB: ..., ... }
 *
 * 核心公式：
 *   卡数 = ceil(模型所需显存 / 单卡可用显存)
 *   服务器数 = ceil(卡数 / 单台服务器卡数)
 *
 *   模型所需显存 = 权重 + KV Cache + 其他
 *   单卡可用显存 = 标称 × 驱动可见率 × 推理可用率
 */

import { getModelById, getAvailableModels } from './constants/model-data.js';
import { getGPUById, getAvailableGPUs } from './constants/gpu-data.js';
import { getPresetById, SLA_PRESETS } from './presets/sla-presets.js';
import { BYTES_PER_PARAM } from './constants/precision.js';
import { calcModelMemory } from './formulas/model-memory.js';
import { calcGPUUsableMemory } from './formulas/gpu-memory.js';

/**
 * 计算 GPU 卡数和服务器数
 *
 * @param {string} modelId    - 模型 ID（来自 model-data.js）
 * @param {string} gpuId      - GPU ID（来自 gpu-data.js）
 * @param {string} presetId   - SLA 预设 ID（来自 sla-presets.js）
 * @param {string} precision  - 精度（如 'FP16'）
 * @returns {{
 *   cards: number,
 *   servers: number,
 *   cardsPerServer: number,
 *   modelMemoryGB: { weightGB, kvCacheGB, otherGB, totalGB, breakdown },
 *   gpuUsableGB: number,
 *   gpuNominalGB: number,
 *   utilization: number,          // 实际显存利用率
 *   modelName: string,
 *   gpuName: string,
 *   presetName: string,
 *   precision: string,
 * }}
 *
 * 例：
 *   calculate('llama3.1-70b', 'mxn200', 'inference-standard', 'FP16')
 *   // Llama-3.1-70B @ FP16, 标准推理(8K ctx, batch=1), MXN200(64GB, 8卡/台)
 *   // → 需要 4 张卡，1 台服务器
 */
export function calculate(modelId, gpuId, presetId, precision) {
  // ── 加载配置 ──
  const model = getModelById(modelId);
  const gpu = getGPUById(gpuId);
  const preset = getPresetById(presetId);

  return calculateFromConfig(model, gpu, preset, precision);
}

/**
 * 从配置对象计算（供内部测试使用）
 */
export function calculateFromConfig(model, gpu, preset, precision) {
  // ── 模型侧：计算所需显存 ──
  const modelMemory = calcModelMemory(
    model,
    precision,
    preset.contextLen,
    preset.batchSize,
    preset.overheadRatio,
  );

  // ── GPU 侧：计算单卡可用显存 ──
  const gpuUsableGB = calcGPUUsableMemory(
    gpu.memoryGB,
    gpu.driverEfficiency,
    gpu.inferenceRatio,
  );

  // ── 计算 ──
  const cards = Math.ceil(modelMemory.totalGB / gpuUsableGB);
  const servers = Math.ceil(cards / gpu.cardsPerServer);
  const utilization = round2(modelMemory.totalGB / (cards * gpuUsableGB));

  return {
    // 结果
    cards,
    servers,
    cardsPerServer: gpu.cardsPerServer,

    // 显存明细
    modelMemoryGB: modelMemory,
    gpuUsableGB: round2(gpuUsableGB),
    gpuNominalGB: gpu.memoryGB,
    utilization,

    // 溯源信息
    modelName: model.displayName,
    gpuName: gpu.displayName,
    presetName: preset.displayName,
    precision: precision.toUpperCase(),
    architecture: model.architecture,

    // 诊断公式
    formula: [
      `模型显存 = ${modelMemory.breakdown} = ${modelMemory.totalGB.toFixed(1)}GB`,
      `单卡可用 = ${gpu.memoryGB}GB × ${gpu.driverEfficiency} × ${gpu.inferenceRatio} = ${gpuUsableGB.toFixed(1)}GB`,
      `卡数 = ceil(${modelMemory.totalGB.toFixed(1)} / ${gpuUsableGB.toFixed(1)}) = ${cards}`,
      `服务器 = ceil(${cards} / ${gpu.cardsPerServer}) = ${servers}`,
    ].join('\n'),
  };
}

/**
 * 便捷方法：快速获取几个常用输入选项（供前端初始化下拉列表）
 */
export function getSelectOptions() {
  return {
    models: getAvailableModels(),
    gpus: getAvailableGPUs(),
    presets: SLA_PRESETS,
    precisions: Object.keys(BYTES_PER_PARAM),
  };
}

function round2(v) {
  return Math.round(v * 100) / 100;
}
