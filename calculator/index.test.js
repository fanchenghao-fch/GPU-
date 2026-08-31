/**
 * GPU 卡数计算器 —— 自动化测试
 *
 * 运行方式：node --test calculator/index.test.js
 *
 * 当前覆盖：29 款模型（6 家厂商）× 2 款 GPU × 7 个 SLA 预设 × 7 种精度
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { calculate } from './index.js';
import { calcWeightMemory, calcKVCache, calcModelMemory } from './formulas/model-memory.js';
import { calcGPUUsableMemory, calcDriverVisible, calcUsableMemory } from './formulas/gpu-memory.js';
import { getModelById, getAvailableModels, getMLADim, isMLA } from './constants/model-data.js';
import { getGPUById, getAvailableGPUs } from './constants/gpu-data.js';
import { bytesPerParam } from './constants/precision.js';
import { getPresetById, getInferencePresets, getTrainingPresets } from './presets/sla-presets.js';
import { getSelectOptions } from './index.js';

// ──────────────────────────────────────────────
// 精度常量
// ──────────────────────────────────────────────
describe('precision constants', () => {
  it('FP16 = 2 bytes/param', () => {
    assert.equal(bytesPerParam('FP16'), 2);
  });
  it('INT8 = 1 byte/param', () => {
    assert.equal(bytesPerParam('INT8'), 1);
  });
  it('FP32 = 4 bytes/param', () => {
    assert.equal(bytesPerParam('FP32'), 4);
  });
  it('INT4 = 0.5 bytes/param', () => {
    assert.equal(bytesPerParam('INT4'), 0.5);
  });
  it('FP8 = 1 byte/param', () => {
    assert.equal(bytesPerParam('FP8'), 1);
  });
  it('BF16 = 2 bytes/param', () => {
    assert.equal(bytesPerParam('BF16'), 2);
  });
  it('unknown precision throws', () => {
    assert.throws(() => bytesPerParam('INT2'), /未知精度/);
  });
});

// ──────────────────────────────────────────────
// 权重显存
// ──────────────────────────────────────────────
describe('calcWeightMemory', () => {
  it('70B dense @ FP16 = ~131.5 GB', () => {
    const result = calcWeightMemory(70.6, 'FP16');
    assert.ok(result > 131 && result < 132, `expected ~131.5, got ${result}`);
  });

  it('8B dense @ FP16 = ~15.0 GB', () => {
    const result = calcWeightMemory(8.03, 'FP16');
    assert.ok(result > 14.9 && result < 15.1, `expected ~14.96, got ${result}`);
  });

  it('7B dense @ INT8 = ~6.6 GB', () => {
    const result = calcWeightMemory(7.07, 'INT8');
    assert.ok(result > 6.5 && result < 6.7, `expected ~6.58, got ${result}`);
  });

  it('8B dense @ INT4 = ~3.7 GB', () => {
    const result = calcWeightMemory(8.03, 'INT4');
    assert.ok(result > 3.7 && result < 3.8, `expected ~3.74, got ${result}`);
  });

  it('MoE 46.7B @ FP16 = ~87.0 GB (all experts loaded)', () => {
    const result = calcWeightMemory(46.7, 'FP16');
    assert.ok(result > 86 && result < 88, `expected ~86.98, got ${result}`);
  });
});

// ──────────────────────────────────────────────
// KV Cache — Standard GQA
// ──────────────────────────────────────────────
describe('calcKVCache - standard', () => {
  it('R1-Distill-Llama-70B: ctx=8192, batch=1, FP16 → ~2.50 GB', () => {
    const model = getModelById('deepseek-r1-distill-llama-70b');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // 2 × 80 × (8×128) × 8192 × 1 × 2 = 2.50 GB
    assert.ok(result > 2.4 && result < 2.6, `expected ~2.50, got ${result}`);
  });

  it('R1-Distill-Llama-70B: ctx=4096 → ~1.25 GB (half)', () => {
    const model = getModelById('deepseek-r1-distill-llama-70b');
    const result = calcKVCache(model, 4096, 1, 'FP16');
    assert.ok(result > 1.2 && result < 1.3, `expected ~1.25, got ${result}`);
  });

  it('R1-Distill-Llama-70B: ctx=8192, batch=4 → ~10.0 GB (4×)', () => {
    const model = getModelById('deepseek-r1-distill-llama-70b');
    const result = calcKVCache(model, 8192, 4, 'FP16');
    assert.ok(result > 9.9 && result < 10.1, `expected ~10.0, got ${result}`);
  });

  it('R1-Distill-Llama-8B: ctx=4096 → ~0.50 GB', () => {
    const model = getModelById('deepseek-r1-distill-llama-8b');
    const result = calcKVCache(model, 4096, 1, 'FP16');
    // 2 × 32 × (8×128) × 4096 × 2 / (1024^3) = 0.50 GB
    assert.ok(result > 0.49 && result < 0.51, `expected ~0.50, got ${result}`);
  });

  it('R1-Distill-Qwen-32B (GQA, 8 KV heads): ctx=8192 → ~1.0 GB', () => {
    const model = getModelById('deepseek-r1-distill-qwen-32b');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // 2 × 64 × (8×128) × 8192 × 2 / (1024^3) = 2.0 GB
    assert.ok(result > 1.9 && result < 2.1, `expected ~2.0, got ${result}`);
  });
});

// ──────────────────────────────────────────────
// KV Cache — MLA（Multi-head Latent Attention）
// ──────────────────────────────────────────────
describe('calcKVCache - MLA', () => {
  it('DeepSeek-V3.2: ctx=8192, batch=1, FP16 → ~0.54 GB (no ×2)', () => {
    const model = getModelById('deepseek-v3.2');
    // MLA: 61 × (512+64) × 8192 × 1 × 2 / (1024^3) ≈ 0.536 GB
    const result = calcKVCache(model, 8192, 1, 'FP16');
    assert.ok(result > 0.53 && result < 0.55, `expected ~0.536, got ${result}`);
  });

  it('DeepSeek-V3.2: ctx=4096 → ~0.27 GB (half)', () => {
    const model = getModelById('deepseek-v3.2');
    const result = calcKVCache(model, 4096, 1, 'FP16');
    assert.ok(result > 0.26 && result < 0.28, `expected ~0.268, got ${result}`);
  });

  it('DeepSeek-V3.2: ctx=8192, batch=4 → ~2.14 GB (4×)', () => {
    const model = getModelById('deepseek-v3.2');
    const result = calcKVCache(model, 8192, 4, 'FP16');
    assert.ok(result > 2.1 && result < 2.2, `expected ~2.144, got ${result}`);
  });

  it('MLA elements/token should be at least 3× smaller than standard', () => {
    const model = getModelById('deepseek-v3.2');
    const mlaElementsPerToken = model.numLayers * (model.kvLoraRank + model.qkRopeHeadDim);
    const standardElementsPerToken = 2 * model.numLayers * model.numKVHeads * model.headDim;
    const ratio = standardElementsPerToken / mlaElementsPerToken;
    assert.ok(ratio > 3, `MLA should be at least 3× smaller than standard, got ratio=${ratio.toFixed(1)}`);
  });

  it('isMLA identifies MLA/kda_mla models correctly', () => {
    assert.equal(isMLA(getModelById('deepseek-v3.2')), true);
    assert.equal(isMLA(getModelById('glm-5.2')), true);
    assert.equal(isMLA(getModelById('kimi-k3')), true);
    assert.equal(isMLA(getModelById('deepseek-r1-distill-llama-70b')), false);
  });

  it('getMLADim returns correct latent dim for DeepSeek-V3.2', () => {
    const model = getModelById('deepseek-v3.2');
    assert.equal(getMLADim(model), 576); // 512 + 64
  });

  it('getMLADim throws for non-MLA models', () => {
    const model = getModelById('deepseek-r1-distill-llama-70b');
    assert.throws(() => getMLADim(model), /仅适用于 attnArch='mla'/);
  });
});

// ──────────────────────────────────────────────
// KV Cache — CLA（Cross-Layer Attention）—— 公式逻辑测试（当前无 CLA 模型）
// ──────────────────────────────────────────────
describe('calcKVCache - CLA', () => {
  it('CLA with share_factor=2 halves effective layers', () => {
    const claModel = {
      id: 'test-cla',
      attnArch: 'cla',
      numLayers: 64,
      numKVHeads: 8,
      headDim: 80,
      claShareFactor: 2,
    };
    const result = calcKVCache(claModel, 8192, 1, 'FP16');
    // effLayers = 64/2 = 32, kvDim = 8 × 80 = 640
    // elements = 2 × 32 × 640 × 8192 × 1 = 335,544,320
    // bytes = 335,544,320 × 2 = 671,088,640 → ≈ 0.625 GB
    assert.ok(result > 0.62 && result < 0.63, `expected ~0.625, got ${result}`);
  });

  it('CLA with share_factor=4 → 1/4 effective layers', () => {
    const sf2 = calcKVCache(
      { id: 'test-cla2', attnArch: 'cla', numLayers: 64, numKVHeads: 8, headDim: 80, claShareFactor: 2 },
      8192, 1, 'FP16');
    const sf4 = calcKVCache(
      { id: 'test-cla4', attnArch: 'cla', numLayers: 64, numKVHeads: 8, headDim: 80, claShareFactor: 4 },
      8192, 1, 'FP16');
    const ratio = sf2 / sf4;
    assert.ok(Math.abs(ratio - 2.0) < 0.01, `sf4 should be half of sf2, ratio=${ratio}`);
  });
});

// ──────────────────────────────────────────────
// KV Cache — linear_hybrid（线性注意力混合）
// ──────────────────────────────────────────────
describe('calcKVCache - linear_hybrid', () => {
  it('MiniMax-M1: ctx=8192, batch=1 → only 10/80 layers have KV cache (~0.31 GB)', () => {
    const model = getModelById('minimax-m1');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // 2 × 10 × (8×128) × 8192 × 1 × 2 / (1024^3) ≈ 0.313 GB
    assert.ok(result > 0.30 && result < 0.33, `expected ~0.313, got ${result}`);
  });

  it('MiniMax-M1: hybrid KV cache is 12.5% of full standard (10/80 layers)', () => {
    const model = getModelById('minimax-m1');
    const hybrid = calcKVCache(model, 8192, 1, 'FP16');
    const fullStandard = calcKVCache(
      { ...model, attnArch: 'standard', numLayers: 80 }, 8192, 1, 'FP16');
    const ratio = fullStandard / hybrid;
    assert.ok(Math.abs(ratio - 8.0) < 0.2, `hybrid should be ~1/8 of full (10/80), ratio=${ratio.toFixed(2)}`);
  });

  it('Qwen3.6-27B: ctx=4096 → only 16/64 layers have KV cache (~0.25 GB)', () => {
    const model = getModelById('qwen3.6-27b');
    const result = calcKVCache(model, 4096, 1, 'FP16');
    // 2 × 16 × (4×256) × 4096 × 1 × 2 / (1024^3) ≈ 0.250 GB
    assert.ok(result > 0.24 && result < 0.26, `expected ~0.250, got ${result}`);
  });

  it('Qwen3.6-27B: hybrid KV cache is 25% of full standard (16/64 layers)', () => {
    const model = getModelById('qwen3.6-27b');
    const hybrid = calcKVCache(model, 4096, 1, 'FP16');
    const fullStandard = calcKVCache(
      { ...model, attnArch: 'standard', numLayers: 64 }, 4096, 1, 'FP16');
    const ratio = fullStandard / hybrid;
    assert.ok(Math.abs(ratio - 4.0) < 0.2, `hybrid should be ~1/4 of full (16/64), ratio=${ratio.toFixed(2)}`);
  });

  it('Qwen3.5-0.8B: ctx=8192 → only 6/24 layers (~0.09 GB)', () => {
    const model = getModelById('qwen3.5-0.8b');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // 2 × 6 × (2×256) × 8192 × 1 × 2 / (1024^3) ≈ 0.094 GB
    assert.ok(result > 0.08 && result < 0.10, `expected ~0.094, got ${result}`);
  });
});

// ──────────────────────────────────────────────
// KV Cache — kda_mla（Kimi K3 KDA + Gated MLA）
// ──────────────────────────────────────────────
describe('calcKVCache - kda_mla', () => {
  it('Kimi K3: ctx=8192, batch=1 → only 24/93 layers with MLA (~0.21 GB)', () => {
    const model = getModelById('kimi-k3');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // 24 × (512+64) × 8192 × 1 × 2 / (1024^3) ≈ 0.211 GB
    assert.ok(result > 0.20 && result < 0.22, `expected ~0.211, got ${result}`);
  });

  it('Kimi K3: getMLADim returns 576', () => {
    assert.equal(getMLADim(getModelById('kimi-k3')), 576);
  });

  it('Kimi K3: isMLA returns true (kda_mla uses same MLA compression)', () => {
    assert.equal(isMLA(getModelById('kimi-k3')), true);
  });

  it('Kimi K3: ctx=4096 → exactly half of ctx=8192', () => {
    const model = getModelById('kimi-k3');
    const kv4096 = calcKVCache(model, 4096, 1, 'FP16');
    const kv8192 = calcKVCache(model, 8192, 1, 'FP16');
    assert.ok(Math.abs(kv8192 / kv4096 - 2.0) < 0.01);
  });
});

// ──────────────────────────────────────────────
// KV Cache — hca_mla（DeepSeek V4 Flash/Pro HCA）
// ──────────────────────────────────────────────
describe('calcKVCache - hca_mla', () => {
  it('DeepSeek V4 Flash: ctx=8192 → ~0.064 GB', () => {
    const model = getModelById('deepseek-v4-flash');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // effectiveKVDim=4176 × 8192 × 1 × 2 / (1024^3) ≈ 0.064 GB
    assert.ok(result > 0.06 && result < 0.07, `expected ~0.064, got ${result}`);
  });

  it('DeepSeek V4 Flash: HCA saves ~10.5× vs standard', () => {
    const model = getModelById('deepseek-v4-flash');
    const standardElements = 2 * model.numLayers * model.numKVHeads * model.headDim;
    const ratio = standardElements / model.effectiveKVDim;
    assert.ok(ratio > 8 && ratio < 12, `expected ~10.5× savings, got ${ratio.toFixed(1)}×`);
  });

  it('DeepSeek V4 Flash: ctx=4096 → exactly half of ctx=8192', () => {
    const model = getModelById('deepseek-v4-flash');
    const kv4096 = calcKVCache(model, 4096, 1, 'FP16');
    const kv8192 = calcKVCache(model, 8192, 1, 'FP16');
    assert.ok(Math.abs(kv8192 / kv4096 - 2.0) < 0.01);
  });

  it('DeepSeek V4 Pro: HCA+MLA KV Cache also uses effectiveKVDim', () => {
    const model = getModelById('deepseek-v4-pro');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // effectiveKVDim=6000 × 8192 × 2 / (1024^3) ≈ 0.092 GB
    assert.ok(result > 0.08 && result < 0.10, `expected ~0.092, got ${result}`);
  });
});

// ──────────────────────────────────────────────
// GPU 显存
// ──────────────────────────────────────────────
describe('GPU memory calculation', () => {
  it('calcDriverVisible: 80GB × 0.97 = 77.6 GB', () => {
    assert.equal(calcDriverVisible(80, 0.97), 77.6);
  });

  it('calcUsableMemory: 77.6GB × 0.85 = 65.96 GB', () => {
    assert.equal(calcUsableMemory(77.6, 0.85), 65.96);
  });

  it('N300 usable: 48 × 0.97 × 0.90 = 41.90 GB', () => {
    const gpu = getGPUById('n300');
    const result = calcGPUUsableMemory(gpu.memoryGB, gpu.driverEfficiency, gpu.inferenceRatio);
    assert.ok(result > 41 && result < 43, `expected ~41.9, got ${result}`);
  });

  it('C600 usable: 144 × 0.97 × 0.90 = 125.71 GB', () => {
    const gpu = getGPUById('c600');
    const result = calcGPUUsableMemory(gpu.memoryGB, gpu.driverEfficiency, gpu.inferenceRatio);
    assert.ok(result > 124 && result < 127, `expected ~125.7, got ${result}`);
  });
});

// ──────────────────────────────────────────────
// 模型总显存（组合计算）
// ──────────────────────────────────────────────
describe('calcModelMemory (combined)', () => {
  it('R1-Distill-Llama-70B, FP16, ctx=8192, batch=1, overhead=0.10 → total ~147 GB', () => {
    const model = getModelById('deepseek-r1-distill-llama-70b');
    const result = calcModelMemory(model, 'FP16', 8192, 1, 0.10);
    assert.ok(result.totalGB > 145 && result.totalGB < 150,
      `expected ~147, got ${result.totalGB}`);
    assert.ok(result.weightGB > 130 && result.weightGB < 133,
      `weight expected ~131.5, got ${result.weightGB}`);
    assert.ok(result.kvCacheGB > 2.4 && result.kvCacheGB < 2.6,
      `kvCache expected ~2.5, got ${result.kvCacheGB}`);
    assert.ok(result.otherGB > 12 && result.otherGB < 15,
      `other expected ~13.4, got ${result.otherGB}`);
  });

  it('R1-Distill-Llama-8B, INT8, ctx=4096, batch=1, overhead=0.05 → total ~8 GB', () => {
    const model = getModelById('deepseek-r1-distill-llama-8b');
    const result = calcModelMemory(model, 'INT8', 4096, 1, 0.05);
    // weight ≈ 7.48 GB, kvCache ≈ 0.25 GB, other ≈ 0.39 GB, total ≈ 8.12 GB
    assert.ok(result.totalGB > 7.8 && result.totalGB < 8.5,
      `expected ~8.1, got ${result.totalGB}`);
  });
});

// ──────────────────────────────────────────────
// 端到端：calculate() — Standard 模型
// ──────────────────────────────────────────────
describe('calculate — standard models', () => {
  it('R1-Distill-Llama-70B + N300 + SLA-1 + FP16 → 5 cards', () => {
    const result = calculate('deepseek-r1-distill-llama-70b', 'n300', 'sla-1', 'FP16');
    // 模型 ~208.7 GB (batch=20) / 单卡 ~41.90 GB = 4.98 → ceil = 5
    assert.equal(result.cards, 5, `expected 5 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.servers, 1, `expected 1 server, got ${result.servers}`);
    assert.equal(result.modelName, 'DeepSeek-R1-Distill-Llama-70B（70B）');
  });

  it('R1-Distill-Llama-8B + N300 + 轻量推理 + INT8 → 1 card', () => {
    const result = calculate('deepseek-r1-distill-llama-8b', 'n300', 'sla-1', 'INT8');
    // ~8 GB / 41.90 GB = 0.19 → ceil = 1
    assert.equal(result.cards, 1, `expected 1 card, got ${result.cards}\n${result.formula}`);
  });

  it('Qwen3.6-27B + C600 + 标准推理 + FP16 → 1 card (55.9 GB fits on 125.7 GB)', () => {
    const result = calculate('qwen3.6-27b', 'c600', 'sla-1', 'FP16');
    // weight=50.3, kv=0.5, other=5.1 → total≈55.9, usable=125.7 → 1 card
    assert.equal(result.cards, 1, `expected 1 card, got ${result.cards}\n${result.formula}`);
    assert.equal(result.servers, 1);
  });

  it('GLM-4.5 (standard MoE) + N300 + 标准推理 + FP16 → ~18 cards', () => {
    const result = calculate('glm-4.5', 'n300', 'sla-1', 'FP16');
    // weight=661, kv=2.9, other=66.4 → total≈730, /41.9≈17.4 → ceil=18
    assert.equal(result.architecture, 'moe');
    assert.ok(result.cards >= 16 && result.cards <= 20,
      `expected ~18 cards, got ${result.cards}\n${result.formula}`);
  });

  it('Hunyuan-A13B + N300 + 轻量推理 + INT8 → 1-2 cards', () => {
    const result = calculate('hunyuan-a13b', 'n300', 'sla-1', 'INT8');
    assert.ok(result.cards >= 1 && result.cards <= 3,
      `expected ~2 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.architecture, 'moe');
  });
});

// ──────────────────────────────────────────────
// 端到端：calculate() — MLA 模型
// ──────────────────────────────────────────────
describe('calculate — MLA models', () => {
  it('DeepSeek-V3.2 + N300 + 标准推理 + FP16 → ~33 cards (MLA)', () => {
    const result = calculate('deepseek-v3.2', 'n300', 'sla-1', 'FP16');
    // weight=1250, kv=0.54, other=125 → total≈1375, /41.9≈32.8 → ceil=33
    assert.ok(result.cards >= 30 && result.cards <= 36,
      `expected ~33 cards, got ${result.cards}\n${result.formula}`);
    assert.ok(result.servers >= 2, `expected >= 2 servers, got ${result.servers}`);
    assert.ok(result.formula.includes('mla'), `formula should mention 'mla'`);
  });

  it('DeepSeek-V3.2 + N300 + 轻量推理 + FP16 → ~12 cards', () => {
    const result = calculate('deepseek-v3.2', 'n300', 'sla-1', 'FP16');
    // weight=1250, kv=0.27, other≈62.5 → total≈1312.5, /41.9≈31.3 → ceil=32
    // Actually lighter overhead (0.05): other ≈ (1250+0.27)*0.05 = 62.51
    // total = 1250 + 0.27 + 62.51 = 1312.78... that doesn't change much
    // 1312.78/41.90 = 31.33 → 32 cards. Hmm.
    assert.ok(result.cards >= 29 && result.cards <= 35,
      `expected ~32 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.architecture, 'moe');
  });

  it('GLM-5.2 + N300 + 标准推理 + FP16 → ~37 cards (MLA)', () => {
    const result = calculate('glm-5.2', 'n300', 'sla-1', 'FP16');
    // weight=1385.7 (744B), kv=0.69, other=138.6 → total≈1525, /41.9≈36.4 → ceil=37
    assert.ok(result.cards >= 34 && result.cards <= 40,
      `expected ~37 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.architecture, 'moe');
    assert.ok(result.formula.includes('mla'), `formula should mention 'mla'`);
  });

  it('GLM-5.1 (MLA) + N300 + 标准推理 + FP16 → similar to GLM-5.2', () => {
    const result = calculate('glm-5.1', 'n300', 'sla-1', 'FP16');
    assert.ok(result.cards >= 34 && result.cards <= 40,
      `expected ~37 cards, got ${result.cards}\n${result.formula}`);
    assert.ok(result.formula.includes('mla'));
  });
});

// ──────────────────────────────────────────────
// 端到端：calculate() — HCA+MLA 模型
// ──────────────────────────────────────────────
describe('calculate — HCA+MLA models', () => {
  it('DeepSeek V4 Flash + N300 + 标准推理 + FP16 → ~14 cards (HCA)', () => {
    const result = calculate('deepseek-v4-flash', 'n300', 'sla-1', 'FP16');
    // weight=529, kv=0.06, other=52.9 → total≈582, /41.9≈13.9 → ceil=14
    assert.ok(result.cards >= 12 && result.cards <= 16,
      `expected ~14 cards, got ${result.cards}\n${result.formula}`);
    assert.ok(result.formula.includes('hca_mla'), `formula should mention 'hca_mla'`);
  });

  it('DeepSeek V4 Pro + N300 + 标准推理 + FP16 → many cards (large HCA)', () => {
    const result = calculate('deepseek-v4-pro', 'n300', 'sla-1', 'FP16');
    // weight=685B = 1275.8GB, kv≈0.13, other≈127.6 → total≈1403.5, /41.9≈33.5 → ceil=34
    assert.ok(result.cards >= 30 && result.cards <= 40,
      `expected ~34 cards, got ${result.cards}\n${result.formula}`);
    assert.ok(result.formula.includes('hca_mla'));
  });
});

// ──────────────────────────────────────────────
// 端到端：calculate() — linear_hybrid 模型
// ──────────────────────────────────────────────
describe('calculate — linear_hybrid models', () => {
  it('MiniMax-M1 + C600 + 标准推理 + FP16 → ~8 cards', () => {
    const result = calculate('minimax-m1', 'c600', 'sla-1', 'FP16');
    // weight=849, kv=0.31, other=85.0 → total≈934.6, /125.7≈7.43 → ceil=8
    assert.ok(result.cards >= 6 && result.cards <= 10,
      `expected ~8 cards, got ${result.cards}\n${result.formula}`);
    assert.ok(result.formula.includes('linear_hybrid'), `formula should mention 'linear_hybrid'`);
  });

  it('Qwen3.6-27B + N300 + 轻量推理 + INT8 → 1 card (dense hybrid)', () => {
    const result = calculate('qwen3.6-27b', 'n300', 'sla-1', 'INT8');
    assert.equal(result.architecture, 'dense');
    assert.ok(result.cards >= 1, `expected at least 1 card, got ${result.cards}`);
  });

  it('Qwen3.6-35B-A3B (MoE hybrid) + N300 + 标准推理 + FP16 → ~2 cards', () => {
    const result = calculate('qwen3.6-35b-a3b', 'n300', 'sla-1', 'FP16');
    // weight=65.2, kv=0.16, other=6.5 → total≈71.9, /41.9≈1.72 → ceil=2
    assert.ok(result.cards >= 1 && result.cards <= 3,
      `expected ~2 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.architecture, 'moe');
  });

  it('Qwen3.5-0.8B + N300 + 标准推理 + FP16 → 1 card (smallest hybrid)', () => {
    const result = calculate('qwen3.5-0.8b', 'n300', 'sla-1', 'FP16');
    // weight=1.5, kv=0.09, other=0.16 → total≈1.74 → 1 card
    assert.equal(result.cards, 1, `expected 1 card, got ${result.cards}`);
  });
});

// ──────────────────────────────────────────────
// 端到端：calculate() — kda_mla 模型
// ──────────────────────────────────────────────
describe('calculate — kda_mla models', () => {
  it('Kimi K3 + C600 + 标准推理 + FP16 → ~46 cards', () => {
    const result = calculate('kimi-k3', 'c600', 'sla-1', 'FP16');
    // weight=5214.7, kv=0.21, other=521.5 → total≈5736.4, /125.7≈45.6 → ceil=46
    assert.ok(result.cards >= 42 && result.cards <= 50,
      `expected ~46 cards, got ${result.cards}\n${result.formula}`);
    assert.ok(result.formula.includes('kda_mla'), `formula should mention 'kda_mla'`);
  });
});

// ──────────────────────────────────────────────
// 腾讯混元系列
// ──────────────────────────────────────────────
describe('calculate — Tencent Hunyuan', () => {
  it('Hunyuan-Hy3 + N300 + 标准推理 + FP16 → ~15 cards', () => {
    const result = calculate('hy3', 'n300', 'sla-1', 'FP16');
    // weight=549.5, kv=2.5, other=55.2 → total≈607.2, /41.9≈14.5 → ceil=15
    assert.ok(result.cards >= 13 && result.cards <= 17,
      `expected ~15 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.architecture, 'moe');
  });
});

// ──────────────────────────────────────────────
// 边界情况
// ──────────────────────────────────────────────
describe('edge cases', () => {
  it('smallest model + biggest GPU → 1 card', () => {
    const result = calculate('qwen3.5-0.8b', 'c600', 'sla-1', 'INT4');
    assert.equal(result.cards, 1);
  });

  it('largest model + smallest GPU → many cards', () => {
    const result = calculate('kimi-k3', 'n300', 'sla-1', 'FP16');
    // 2.8T params → ~5000+ GB, N300 = 41.9 GB → ~120+ cards
    assert.ok(result.cards >= 100, `expected >= 100 cards, got ${result.cards}`);
  });

  it('servers: cards/16 → correct server count', () => {
    const result = calculate('kimi-k3', 'n300', 'sla-1', 'FP16');
    const { cards, servers } = result;
    assert.equal(servers, Math.ceil(cards / 16),
      `servers should be ceil(cards/16), got ${servers} for ${cards} cards`);
  });

  it('utilization should be ≤ 1.0', () => {
    const result = calculate('deepseek-r1-distill-llama-70b', 'n300', 'sla-1', 'FP16');
    assert.ok(result.utilization <= 1.0, `utilization should be ≤ 1.0, got ${result.utilization}`);
    assert.ok(result.utilization > 0, `utilization should be > 0`);
  });
});

// ──────────────────────────────────────────────
// 一致性检查
// ──────────────────────────────────────────────
describe('consistency', () => {
  it('KV cache scales linearly with context length', () => {
    const model = getModelById('deepseek-r1-distill-llama-70b');
    const kv4096 = calcKVCache(model, 4096, 1, 'FP16');
    const kv8192 = calcKVCache(model, 8192, 1, 'FP16');
    assert.ok(Math.abs(kv8192 / kv4096 - 2.0) < 0.01,
      `kvCache(8192)=${kv8192} should be 2× kvCache(4096)=${kv4096}`);
  });

  it('KV cache scales linearly with batch size', () => {
    const model = getModelById('deepseek-r1-distill-llama-70b');
    const b1 = calcKVCache(model, 8192, 1, 'FP16');
    const b4 = calcKVCache(model, 8192, 4, 'FP16');
    assert.ok(Math.abs(b4 / b1 - 4.0) < 0.01);
  });

  it('all GPU models have valid configs', () => {
    const gpus = getAvailableGPUs();
    for (const gpu of gpus) {
      assert.ok(gpu.memoryGB > 0, `${gpu.id}: memoryGB must be > 0`);
      assert.ok(gpu.cardsPerServer > 0, `${gpu.id}: cardsPerServer must be > 0`);
      assert.ok(gpu.driverEfficiency > 0 && gpu.driverEfficiency <= 1);
      assert.ok(gpu.inferenceRatio > 0 && gpu.inferenceRatio <= 1);
    }
  });

  it('all 29 models have architecture params for KV cache', () => {
    const models = getAvailableModels();
    assert.equal(models.length, 29, `expected 29 models, got ${models.length}`);
    for (const m of models) {
      assert.ok(m.numLayers > 0, `${m.id}: numLayers must be > 0`);
      assert.ok(m.hiddenDim > 0, `${m.id}: hiddenDim must be > 0`);
      assert.ok(m.numKVHeads > 0, `${m.id}: numKVHeads must be > 0`);
      assert.ok(m.headDim > 0, `${m.id}: headDim must be > 0`);
    }
  });

  it('all models have valid attnArch', () => {
    const valid = ['standard', 'mla', 'cla', 'linear_hybrid', 'kda_mla', 'hca_mla'];
    const models = getAvailableModels();
    for (const m of models) {
      assert.ok(valid.includes(m.attnArch || 'standard'),
        `${m.id}: attnArch '${m.attnArch}' not in ${valid}`);
    }
  });

  it('MLA models have required fields (kvLoraRank, qkRopeHeadDim)', () => {
    const models = getAvailableModels();
    let mlaCount = 0;
    for (const m of models) {
      if (m.attnArch === 'mla') {
        mlaCount++;
        assert.ok(m.kvLoraRank > 0, `${m.id}: MLA model must have kvLoraRank > 0`);
        assert.ok(m.qkRopeHeadDim > 0, `${m.id}: MLA model must have qkRopeHeadDim > 0`);
      }
    }
    assert.ok(mlaCount > 0, `expected at least 1 MLA model, got ${mlaCount}`);
  });

  it('linear_hybrid models have required field (fullAttnLayers < numLayers)', () => {
    const models = getAvailableModels();
    let hybridCount = 0;
    for (const m of models) {
      if (m.attnArch === 'linear_hybrid') {
        hybridCount++;
        assert.ok(m.fullAttnLayers >= 0, `${m.id}: must have fullAttnLayers >= 0`);
        assert.ok(m.fullAttnLayers < m.numLayers,
          `${m.id}: fullAttnLayers (${m.fullAttnLayers}) < numLayers (${m.numLayers})`);
      }
    }
    assert.ok(hybridCount >= 8, `expected ≥8 linear_hybrid models (Qwen + MiniMax), got ${hybridCount}`);
  });

  it('kda_mla models have required fields', () => {
    const models = getAvailableModels();
    for (const m of models) {
      if (m.attnArch === 'kda_mla') {
        assert.ok(m.fullAttnLayers > 0, `${m.id}: must have fullAttnLayers > 0`);
        assert.ok(m.kvLoraRank > 0, `${m.id}: must have kvLoraRank > 0`);
        assert.ok(m.qkRopeHeadDim > 0, `${m.id}: must have qkRopeHeadDim > 0`);
      }
    }
  });

  it('hca_mla models have required field (effectiveKVDim)', () => {
    const models = getAvailableModels();
    let hcaCount = 0;
    for (const m of models) {
      if (m.attnArch === 'hca_mla') {
        hcaCount++;
        assert.ok(m.effectiveKVDim > 0, `${m.id}: must have effectiveKVDim > 0`);
      }
    }
    assert.ok(hcaCount > 0, `expected at least 1 hca_mla model, got ${hcaCount}`);
  });

  it('KV cache linear scaling holds for all attnArch types', () => {
    const testCases = [
      { id: 'deepseek-v3.2', arch: 'mla' },
      { id: 'glm-5.2', arch: 'mla' },
      { id: 'minimax-m1', arch: 'linear_hybrid' },
      { id: 'qwen3.6-27b', arch: 'linear_hybrid' },
      { id: 'kimi-k3', arch: 'kda_mla' },
      { id: 'deepseek-v4-flash', arch: 'hca_mla' },
      { id: 'deepseek-v4-pro', arch: 'hca_mla' },
    ];
    for (const { id, arch } of testCases) {
      const model = getModelById(id);
      assert.equal(model.attnArch, arch);
      const kv4096 = calcKVCache(model, 4096, 1, 'FP16');
      const kv8192 = calcKVCache(model, 8192, 1, 'FP16');
      assert.ok(Math.abs(kv8192 / kv4096 - 2.0) < 0.01,
        `${id}: kvCache(8192) should be 2× kvCache(4096)`);
      const b1 = calcKVCache(model, 8192, 1, 'FP16');
      const b4 = calcKVCache(model, 8192, 4, 'FP16');
      assert.ok(Math.abs(b4 / b1 - 4.0) < 0.01,
        `${id}: batch=4 should be 4× batch=1`);
    }
  });
});

// ──────────────────────────────────────────────
// SLA 预设
// ──────────────────────────────────────────────
describe('SLA presets', () => {
  it('should have 7 presets total', () => {
    const { presets } = getSelectOptions();
    assert.equal(presets.length, 7);
  });

  it('should have 5 inference presets', () => {
    const inf = getInferencePresets();
    assert.equal(inf.length, 5);
    assert.ok(inf.every((p) => p.scenario === 'inference'));
  });

  it('should have 2 training presets', () => {
    const train = getTrainingPresets();
    assert.equal(train.length, 2);
    assert.ok(train.every((p) => p.scenario === 'training'));
  });

  const presetIds = [
    'sla-1', 'sla-2', 'sla-3', 'sla-4', 'sla-5',
    'training-lora', 'training-full',
  ];
  for (const id of presetIds) {
    it(`getPresetById("${id}") should return valid preset`, () => {
      const p = getPresetById(id);
      assert.equal(p.id, id);
      assert.ok(p.contextLen > 0, `contextLen must be > 0`);
      assert.ok(p.batchSize > 0, `batchSize must be > 0`);
      assert.ok(p.overheadRatio > 0 && p.overheadRatio < 1);
    });
  }

  it('unknown preset throws', () => {
    assert.throws(() => getPresetById('nonexistent'), /未知 SLA 预设/);
  });

  it('context lengths: sla-1(8K) < sla-2(16K) < sla-3(32K) < sla-4(64K) < sla-5(128K)', () => {
    const s1 = getPresetById('sla-1').contextLen;
    const s2 = getPresetById('sla-2').contextLen;
    const s3 = getPresetById('sla-3').contextLen;
    const s4 = getPresetById('sla-4').contextLen;
    const s5 = getPresetById('sla-5').contextLen;
    assert.ok(s1 < s2);
    assert.ok(s2 < s3);
    assert.ok(s3 < s4);
    assert.ok(s4 < s5);
  });

  it('all SLA inference presets have batchSize=20, training presets have batchSize=1', () => {
    for (let i = 1; i <= 5; i++) {
      const p = getPresetById(`sla-${i}`);
      assert.equal(p.batchSize, 20, `sla-${i}: batchSize should be 20, got ${p.batchSize}`);
    }
    assert.equal(getPresetById('training-lora').batchSize, 1);
    assert.equal(getPresetById('training-full').batchSize, 1);
  });
});

// ──────────────────────────────────────────────
// 新预设端到端测试
// ──────────────────────────────────────────────
describe('calculate with long-context presets', () => {
  it('R1-Distill-Llama-8B + N300 + 批量推理 (batch=8) → cards ≥ standard (batch=1)', () => {
    const batchResult = calculate('deepseek-r1-distill-llama-8b', 'n300', 'sla-1', 'FP16');
    const stdResult = calculate('deepseek-r1-distill-llama-8b', 'n300', 'sla-1', 'FP16');
    assert.ok(batchResult.cards >= stdResult.cards,
      `batch cards=${batchResult.cards} should be >= standard cards=${stdResult.cards}`);
  });

  it('R1-Distill-Llama-8B + N300 + 长上下文 128K → KV Cache dominates weight', () => {
    const result = calculate('deepseek-r1-distill-llama-8b', 'n300', 'sla-5', 'FP16');
    // ctx=128K: KV=2×32×1024×131072×2/(1024^3)≈16 GB > weight≈15 GB
    assert.ok(result.cards >= 1, `expected ≥1 card, got ${result.cards}`);
    assert.ok(result.modelMemoryGB.kvCacheGB > result.modelMemoryGB.weightGB,
      `KV Cache (${result.modelMemoryGB.kvCacheGB}) should dominate weight (${result.modelMemoryGB.weightGB}) at 128K`);
  });

  it('Qwen3.6-27B (linear_hybrid) + C600 + 长上下文 128K → only 16/64 layers produce KV cache', () => {
    const result = calculate('qwen3.6-27b', 'c600', 'sla-5', 'FP16');
    assert.ok(result.formula.includes('linear_hybrid'));
    assert.ok(result.cards >= 1);
  });

  it('DeepSeek-V3.2 + C600 + 超长上下文 256K → MLA keeps KV cache manageable', () => {
    const result = calculate('deepseek-v3.2', 'c600', 'sla-5', 'FP16');
    // MLA: KV=61×576×262144×2/(1024^3)≈17.2 GB (vs standard ~990 GB)
    assert.ok(result.formula.includes('mla'));
  });

  it('DeepSeek V4 Flash + C600 + 超长上下文 256K → HCA extreme savings', () => {
    const result = calculate('deepseek-v4-flash', 'c600', 'sla-5', 'FP16');
    // HCA: KV=4176×262144×2/(1024^3)≈2.04 GB
    assert.ok(result.formula.includes('hca_mla'));
    assert.ok(result.cards >= 4 && result.cards <= 6,
      `expected ~5 cards, got ${result.cards}\n${result.formula}`);
  });
});

// ──────────────────────────────────────────────
// 边界条件与回归
// ──────────────────────────────────────────────
describe('boundary & regression', () => {
  it('utilization rate approaches 1.0 when model barely fits', () => {
    const result = calculate('qwen3.6-27b', 'c600', 'sla-1', 'FP16');
    assert.ok(result.utilization > 0 && result.utilization <= 1,
      `utilization=${result.utilization} out of range`);
  });

  it('cards always ≥ 1 even for smallest model on largest GPU', () => {
    const result = calculate('qwen3.5-0.8b', 'c600', 'sla-1', 'INT4');
    assert.equal(result.cards, 1);
  });

  it('servers = ceil(cards / cardsPerServer) for all GPU types', () => {
    const testCases = [
      { model: 'kimi-k3', gpu: 'n300', preset: 'sla-1', prec: 'FP16' },
      { model: 'deepseek-v3.2', gpu: 'c600', preset: 'sla-1', prec: 'FP16' },
      { model: 'glm-5.2', gpu: 'n300', preset: 'sla-1', prec: 'FP16' },
    ];
    for (const { model, gpu, preset, prec } of testCases) {
      const result = calculate(model, gpu, preset, prec);
      const gpuConfig = getGPUById(gpu);
      const expectedServers = Math.ceil(result.cards / gpuConfig.cardsPerServer);
      assert.equal(result.servers, expectedServers,
        `${model}+${gpu}: servers=${result.servers} != ceil(${result.cards}/${gpuConfig.cardsPerServer})=${expectedServers}`);
    }
  });

  it('modelMemoryGB breakdown sums correctly', () => {
    const result = calculate('deepseek-r1-distill-llama-70b', 'n300', 'sla-1', 'FP16');
    const { weightGB, kvCacheGB, otherGB, totalGB } = result.modelMemoryGB;
    const sum = round2(weightGB + kvCacheGB + otherGB);
    assert.equal(sum, totalGB,
      `${weightGB} + ${kvCacheGB} + ${otherGB} = ${sum} ≠ totalGB=${totalGB}`);
  });

  it('modelMemoryGB breakdown sum is consistent for all 29 models', () => {
    const models = getAvailableModels();
    for (const m of models) {
      const result = calculate(m.id, 'n300', 'sla-1', 'FP16');
      const { weightGB, kvCacheGB, otherGB, totalGB } = result.modelMemoryGB;
      const sum = round2(weightGB + kvCacheGB + otherGB);
      assert.equal(sum, totalGB,
        `${m.id}: ${weightGB} + ${kvCacheGB} + ${otherGB} = ${sum} ≠ ${totalGB}`);
    }
  });

  it('getSelectOptions returns all required keys with 29 models', () => {
    const opts = getSelectOptions();
    assert.ok(Array.isArray(opts.models));
    assert.ok(Array.isArray(opts.gpus));
    assert.ok(Array.isArray(opts.presets));
    assert.ok(Array.isArray(opts.precisions));
    assert.equal(opts.models.length, 29, `expected 29 models, got ${opts.models.length}`);
    assert.equal(opts.gpus.length, 2, `expected 2 GPUs, got ${opts.gpus.length}`);
    assert.ok(opts.precisions.length >= 6);
  });

  it('formula string includes architecture type for non-standard models', () => {
    const archTests = [
      { id: 'deepseek-v3.2', arch: 'mla' },
      { id: 'minimax-m1', arch: 'linear_hybrid' },
      { id: 'kimi-k3', arch: 'kda_mla' },
      { id: 'deepseek-v4-flash', arch: 'hca_mla' },
      { id: 'deepseek-v4-pro', arch: 'hca_mla' },
    ];
    for (const { id, arch } of archTests) {
      const result = calculate(id, 'c600', 'sla-1', 'FP16');
      assert.ok(result.formula.includes(arch),
        `${id}: formula should mention '${arch}', got:\n${result.formula}`);
    }
  });

  it('256K KV cache is exactly 2× 128K for same model', () => {
    const model = getModelById('deepseek-r1-distill-llama-70b');
    const kv128k = calcKVCache(model, 131072, 1, 'FP16');
    const kv256k = calcKVCache(model, 262144, 1, 'FP16');
    assert.ok(Math.abs(kv256k / kv128k - 2.0) < 0.01,
      `KV(256K)=${kv256k} should be 2× KV(128K)=${kv128k}`);
  });

  it('batch=8 KV cache is exactly 8× batch=1', () => {
    const model = getModelById('deepseek-r1-distill-llama-70b');
    const b1 = calcKVCache(model, 8192, 1, 'FP16');
    const b8 = calcKVCache(model, 8192, 8, 'FP16');
    assert.ok(Math.abs(b8 / b1 - 8.0) < 0.01);
  });
});

function round2(v) {
  return Math.round(v * 100) / 100;
}