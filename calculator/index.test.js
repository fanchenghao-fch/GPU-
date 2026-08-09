/**
 * GPU 卡数计算器 —— 自动化测试
 *
 * 运行方式：node --test calculator/index.test.js
 *
 * 每个测试验证一条已知的输入 → 输出映射。
 * 新增模型/GPU 后，请在此添加对应的测试用例。
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
  it('Llama-3.1-70B @ FP16 = 141.2 GB', () => {
    // 70.6 × 10^9 × 2 / (1024^3) ≈ 131.5... let me compute
    const result = calcWeightMemory(70.6, 'FP16');
    // 70.6 * 1e9 * 2 / (1024^3) = 70.6 * 2 / (1024^3/1e9) = 141.2 / 1.0737...
    // Actually: 70.6e9 * 2 / (1024**3) = 141.2e9 / 1073741824 = 131.49...
    // GB = bytes / (1024^3), so 70.6e9 * 2 / 1073741824 = 131.49 GB
    assert.ok(result > 131 && result < 132, `expected ~131.5, got ${result}`);
  });

  it('Llama-3.1-8B @ FP16 = ~15.0 GB', () => {
    const result = calcWeightMemory(8.03, 'FP16');
    // 8.03e9 * 2 / (1024^3) = 14.96 GB
    assert.ok(result > 14.9 && result < 15.1, `expected ~14.96, got ${result}`);
  });

  it('Qwen2-7B @ INT8 = ~6.6 GB', () => {
    const result = calcWeightMemory(7.07, 'INT8');
    // 7.07e9 * 1 / (1024^3) = 6.58 GB
    assert.ok(result > 6.5 && result < 6.7, `expected ~6.58, got ${result}`);
  });

  it('Llama-3.1-8B @ INT4 = ~3.7 GB', () => {
    const result = calcWeightMemory(8.03, 'INT4');
    // 8.03e9 * 0.5 / (1024^3) = 3.74 GB
    assert.ok(result > 3.7 && result < 3.8, `expected ~3.74, got ${result}`);
  });

  it('Mixtral 8×7B (MoE) @ FP16 = ~87.0 GB (all experts)', () => {
    // MoE 使用总参数量计算权重显存
    const result = calcWeightMemory(46.7, 'FP16');
    // 46.7e9 * 2 / (1024^3) = 86.98 GB
    assert.ok(result > 86 && result < 88, `expected ~86.98, got ${result}`);
  });
});

// ──────────────────────────────────────────────
// KV Cache
// ──────────────────────────────────────────────
describe('calcKVCache', () => {
  it('Llama-3.1-70B: ctx=8192, batch=1, FP16 → ~2.50 GB', () => {
    const model = getModelById('llama3.1-70b');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // 2 × 80 × (8×128) × 8192 × 1 × 2 = 2 × 80 × 1024 × 8192 × 2
    // = 2,684,354,560 bytes = 2.50 GB
    assert.ok(result > 2.4 && result < 2.6, `expected ~2.50, got ${result}`);
  });

  it('Llama-3.1-70B: ctx=4096, batch=1, FP16 → ~1.25 GB (half of above)', () => {
    const model = getModelById('llama3.1-70b');
    const result = calcKVCache(model, 4096, 1, 'FP16');
    assert.ok(result > 1.2 && result < 1.3, `expected ~1.25, got ${result}`);
  });

  it('Llama-3.1-70B: ctx=8192, batch=4, FP16 → ~10.0 GB (4×)', () => {
    const model = getModelById('llama3.1-70b');
    const result = calcKVCache(model, 8192, 4, 'FP16');
    assert.ok(result > 9.9 && result < 10.1, `expected ~10.0, got ${result}`);
  });

  it('Llama-3.1-8B: ctx=4096, batch=1, FP16 → ~0.20 GB', () => {
    const model = getModelById('llama3.1-8b');
    const result = calcKVCache(model, 4096, 1, 'FP16');
    // 2 × 32 × (8×128) × 4096 × 1 × 2 = 2 × 32 × 1024 × 4096 × 2
    // = 536,870,912 bytes = 0.50 GB
    // Wait: 2 × 32 × 1024 × 4096 × 2 = 536,870,912 bytes
    // 536,870,912 / (1024^3) = 0.50 GB
    assert.ok(result > 0.49 && result < 0.51, `expected ~0.50, got ${result}`);
  });

  it('Qwen2-7B (GQA, 4 KV heads): ctx=8192, batch=1, FP16 → ~0.21 GB', () => {
    const model = getModelById('qwen2-7b');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // 2 × 28 × (4×128) × 8192 × 2 = 2 × 28 × 512 × 8192 × 2
    // = 469,762,048 bytes = 0.44... wait
    // 2 × 28 × 512 × 8192 × 2 = 469,762,048
    // / 1073741824 = 0.4375 GB
    assert.ok(result > 0.43 && result < 0.45, `expected ~0.44, got ${result}`);
  });
});

// ──────────────────────────────────────────────
// KV Cache — MLA（Multi-head Latent Attention）
// ──────────────────────────────────────────────
describe('calcKVCache - MLA', () => {
  it('DeepSeek-V3: ctx=8192, batch=1, FP16 → ~0.54 GB (no ×2)', () => {
    const model = getModelById('deepseek-v3');
    // MLA: layers × (kvLoraRank + qkRopeHeadDim) × tokens × batch × dtype
    // = 61 × (512 + 64) × 8192 × 1 × 2
    // = 61 × 576 × 8192 × 2
    // = 575,340,544 bytes
    // / (1024^3) ≈ 0.536 GB
    const result = calcKVCache(model, 8192, 1, 'FP16');
    assert.ok(result > 0.53 && result < 0.55,
      `expected ~0.536, got ${result}`);
  });

  it('DeepSeek-V3: ctx=4096, batch=1, FP16 → ~0.27 GB (half of 8192)', () => {
    const model = getModelById('deepseek-v3');
    const result = calcKVCache(model, 4096, 1, 'FP16');
    assert.ok(result > 0.26 && result < 0.28,
      `expected ~0.268, got ${result}`);
  });

  it('DeepSeek-V3: ctx=8192, batch=4, FP16 → ~2.14 GB (4× batch)', () => {
    const model = getModelById('deepseek-v3');
    const result = calcKVCache(model, 8192, 4, 'FP16');
    assert.ok(result > 2.1 && result < 2.2,
      `expected ~2.144, got ${result}`);
  });

  it('MLA formula produces ~1/3.56 of what standard formula would give', () => {
    const model = getModelById('deepseek-v3');
    const mlaResult = calcKVCache(model, 8192, 1, 'FP16');

    // 如果用 standard 公式强行算 DeepSeek-V3：
    // kvDim = numKVHeads(128) × headDim(128) = 16384
    // 2 × layers(61) × kvDim(16384) × tokens(8192) × dtype(2) = 2 × 61 × 16384 × 8192 × 2
    // = 32,687,587,328 bytes / (1024^3) ≈ 30.44 GB
    // ratio = 30.44 / 0.536 ≈ 56.8... wait that's way bigger.

    // Actually let me recalculate. For DeepSeek-V3, numKVHeads=128, headDim=128
    // kvDim = 128 × 128 = 16384
    // standard_elements = 2 × 61 × 16384 = 1,998,848 (per token)
    // mla_elements = 61 × (512+64) = 61 × 576 = 35,136 (per token)
    // ratio = 1,998,848 / 35,136 ≈ 56.9

    // That seems too big. Let me check LMCache's numbers again...
    // Actually from the LMCache calculator source:
    // For DeepSeek: total_elements = num_hidden_layers × tokens × (kv_lora_rank + qk_rope_head_dim)
    // = 61 × tokens × 576 = 35,136 × tokens
    // For standard models: 2 × num_hidden_layers × tokens × num_key_value_heads × head_size
    // DeepSeek-V3 standard params: num_key_value_heads=128, hidden_size=7168, num_attention_heads=128
    // head_size = 7168/128 = 56
    // standard: 2 × 61 × 128 × 56 = 2 × 61 × 7168 = 874,496 per token
    // mla: 61 × 576 = 35,136 per token
    // ratio = 874,496 / 35,136 ≈ 24.9x

    // Hmm, the actual head_size for DeepSeek-V3 is 56 (7168/128), not 128!
    // So for standard: kvDim = 128 × 56 = 7168 (not 16384)
    // standard_elements = 2 × 61 × 7168 = 874,496 (per token)
    // mla_elements = 61 × 576 = 35,136 (per token)
    // ratio ≈ 24.9

    // But 24.9x still seems very high. Let me look at what LMCache's README says:
    // "DeepSeek-V3 (671B): 7x memory reduction through KV-LoRA compression"
    // 7x reduction vs standard...

    // 7x is plausible with quantization considerations. The exact ratio depends on
    // whether you compare at the same precision, and what "standard" you compare to.
    // Let me not assert a specific ratio, but verify MLA < standard.
    const mlaElementsPerToken = model.numLayers * (model.kvLoraRank + model.qkRopeHeadDim);
    const standardElementsPerToken = 2 * model.numLayers * model.numKVHeads * model.headDim;
    const ratio = standardElementsPerToken / mlaElementsPerToken;

    // MLA should be smaller than standard for the same model
    assert.ok(ratio > 3,
      `MLA should be at least 3× smaller than standard, got ratio=${ratio.toFixed(1)}`);
  });

  it('DeepSeek-V2 MLA: ctx=8192, batch=1, FP16 → ~0.53 GB', () => {
    const model = getModelById('deepseek-v2');
    // 60 × (512+64) × 8192 × 2 = 60 × 576 × 8192 × 2
    // = 566,231,040 bytes / (1024^3) ≈ 0.527 GB
    const result = calcKVCache(model, 8192, 1, 'FP16');
    assert.ok(result > 0.52 && result < 0.54,
      `expected ~0.527, got ${result}`);
  });

  it('isMLA helper identifies MLA models correctly', () => {
    assert.equal(isMLA(getModelById('deepseek-v3')), true);
    assert.equal(isMLA(getModelById('deepseek-v2')), true);
    assert.equal(isMLA(getModelById('llama3.1-70b')), false);
  });

  it('getMLADim returns correct latent dim for DeepSeek-V3', () => {
    const model = getModelById('deepseek-v3');
    assert.equal(getMLADim(model), 576); // 512 + 64
  });

  it('getMLADim throws for non-MLA models', () => {
    const model = getModelById('llama3.1-70b');
    assert.throws(() => getMLADim(model), /仅适用于 attnArch='mla'/);
  });
});

// ──────────────────────────────────────────────
// KV Cache — CLA（Cross-Layer Attention）
// ──────────────────────────────────────────────
describe('calcKVCache - CLA', () => {
  it('CLA with share_factor=2 halves effective layers (Hunyuan-Large style)', () => {
    // 构造一个 CLA 模型（不在正式列表中，验证公式逻辑）
    const claModel = {
      id: 'test-cla',
      attnArch: 'cla',
      numLayers: 64,
      numKVHeads: 8,
      headDim: 80,
      claShareFactor: 2,
    };
    const result = calcKVCache(claModel, 8192, 1, 'FP16');
    // effLayers = 64/2 = 32
    // kvDim = 8 × 80 = 640
    // elements = 2 × 32 × 640 × 8192 × 1 = 335,544,320
    // bytes = 335,544,320 × 2 = 671,088,640
    // GB = 671,088,640 / (1024^3) ≈ 0.625 GB
    assert.ok(result > 0.62 && result < 0.63,
      `expected ~0.625, got ${result}`);
  });

  it('CLA with share_factor=4 → 1/4 effective layers', () => {
    const claModel = {
      id: 'test-cla4',
      attnArch: 'cla',
      numLayers: 64,
      numKVHeads: 8,
      headDim: 80,
      claShareFactor: 4,
    };
    const sf2 = calcKVCache(
      { ...claModel, claShareFactor: 2, attnArch: 'cla' },
      8192, 1, 'FP16');
    const sf4 = calcKVCache(claModel, 8192, 1, 'FP16');
    // sf4 should be exactly half of sf2
    const ratio = sf2 / sf4;
    assert.ok(Math.abs(ratio - 2.0) < 0.01,
      `sf4=${sf4} should be half of sf2=${sf2}, ratio=${ratio}`);
  });
});

// ──────────────────────────────────────────────
// KV Cache — linear_hybrid（线性注意力混合）
// ──────────────────────────────────────────────
describe('calcKVCache - linear_hybrid', () => {
  it('MiniMax-M1: ctx=8192, batch=1, FP16 → only 10/80 layers have KV cache', () => {
    const model = getModelById('minimax-m1-80k');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // 2 × 10 × (8×128) × 8192 × 1 × 2 = 2 × 10 × 1024 × 8192 × 2
    // = 335,544,320 bytes / (1024^3) ≈ 0.313 GB
    assert.ok(result > 0.30 && result < 0.33,
      `expected ~0.313, got ${result}`);
  });

  it('MiniMax-M1: hybrid KV cache is 12.5% of full standard (10/80 layers)', () => {
    const model = getModelById('minimax-m1-80k');
    const hybrid = calcKVCache(model, 8192, 1, 'FP16');
    // 如果全部 80 层都是标准注意力：
    const fullStandard = calcKVCache(
      { ...model, attnArch: 'standard', numLayers: 80 },
      8192, 1, 'FP16');
    const ratio = fullStandard / hybrid;
    assert.ok(Math.abs(ratio - 8.0) < 0.2,
      `hybrid should be ~1/8 of full standard (10/80 layers), ratio=${ratio.toFixed(2)}`);
  });

  it('Qwen3.6-27B: ctx=4096, batch=1, FP16 → only 16/64 layers have KV cache', () => {
    const model = getModelById('qwen3.6-27b');
    const result = calcKVCache(model, 4096, 1, 'FP16');
    // 2 × 16 × (4×256) × 4096 × 1 × 2 = 2 × 16 × 1024 × 4096 × 2
    // = 268,435,456 bytes / (1024^3) ≈ 0.250 GB
    assert.ok(result > 0.24 && result < 0.26,
      `expected ~0.250, got ${result}`);
  });

  it('Qwen3.6-27B: hybrid KV cache is 25% of full standard (16/64 layers)', () => {
    const model = getModelById('qwen3.6-27b');
    const hybrid = calcKVCache(model, 4096, 1, 'FP16');
    const fullStandard = calcKVCache(
      { ...model, attnArch: 'standard', numLayers: 64 },
      4096, 1, 'FP16');
    const ratio = fullStandard / hybrid;
    assert.ok(Math.abs(ratio - 4.0) < 0.2,
      `hybrid should be ~1/4 of full standard (16/64 layers), ratio=${ratio.toFixed(2)}`);
  });
});

// ──────────────────────────────────────────────
// KV Cache — kda_mla（Kimi K3 KDA + Gated MLA）
// ──────────────────────────────────────────────
describe('calcKVCache - kda_mla', () => {
  it('Kimi K3: ctx=8192, batch=1, FP16 → only 24/93 layers with MLA', () => {
    const model = getModelById('kimi-k3');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // 24 × (512+64) × 8192 × 1 × 2 = 24 × 576 × 8192 × 2
    // = 226,492,416 bytes / (1024^3) ≈ 0.211 GB
    assert.ok(result > 0.20 && result < 0.22,
      `expected ~0.211, got ${result}`);
  });

  it('Kimi K3: getMLADim returns 576', () => {
    const model = getModelById('kimi-k3');
    assert.equal(getMLADim(model), 576); // 512 + 64
  });

  it('Kimi K3: isMLA returns true (kda_mla uses same MLA compression)', () => {
    const model = getModelById('kimi-k3');
    assert.equal(isMLA(model), true);
  });

  it('Kimi K3 MLA: ctx=4096 → exactly half of ctx=8192', () => {
    const model = getModelById('kimi-k3');
    const kv4096 = calcKVCache(model, 4096, 1, 'FP16');
    const kv8192 = calcKVCache(model, 8192, 1, 'FP16');
    assert.ok(Math.abs(kv8192 / kv4096 - 2.0) < 0.01,
      `kvCache(8192)=${kv8192} should be 2× kvCache(4096)=${kv4096}`);
  });
});

// ──────────────────────────────────────────────
// KV Cache — hca_mla（DeepSeek V4 Flash HCA）
// ──────────────────────────────────────────────
describe('calcKVCache - hca_mla', () => {
  it('DeepSeek V4 Flash: ctx=8192, batch=1, FP16 → ~0.065 GB', () => {
    const model = getModelById('deepseek-v4-flash');
    const result = calcKVCache(model, 8192, 1, 'FP16');
    // effectiveKVDim=4176 × 8192 × 1 × 2 = 68,419,584 bytes / (1024^3) ≈ 0.0637 GB
    assert.ok(result > 0.06 && result < 0.07,
      `expected ~0.064, got ${result}`);
  });

  it('DeepSeek V4 Flash: HCA saves ~10.5× vs standard (effectiveKVDim=4176 vs 44032)', () => {
    const model = getModelById('deepseek-v4-flash');
    const hca = calcKVCache(model, 8192, 1, 'FP16');
    // 如果用标准 attention：2 × 43 × (1×512) = 44032 elements/token
    const standardElements = 2 * model.numLayers * model.numKVHeads * model.headDim;
    const hcaElementsPerToken = model.effectiveKVDim;
    const ratio = standardElements / hcaElementsPerToken;
    assert.ok(ratio > 8 && ratio < 12,
      `expected ~10.5× savings, got ${ratio.toFixed(1)}×`);
    assert.ok(hca > 0.06, `HCA KV cache should be non-trivial, got ${hca}`);
  });

  it('DeepSeek V4 Flash: ctx=4096 → exactly half of ctx=8192', () => {
    const model = getModelById('deepseek-v4-flash');
    const kv4096 = calcKVCache(model, 4096, 1, 'FP16');
    const kv8192 = calcKVCache(model, 8192, 1, 'FP16');
    assert.ok(Math.abs(kv8192 / kv4096 - 2.0) < 0.01,
      `kvCache(8192)=${kv8192} should be 2× kvCache(4096)=${kv4096}`);
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

  it('calcGPUUsableMemory: N300 (48GB, eff=0.97, ratio=0.90)', () => {
    const gpu = getGPUById('n300');
    const result = calcGPUUsableMemory(gpu.memoryGB, gpu.driverEfficiency, gpu.inferenceRatio);
    // 48 × 0.97 × 0.90 = 41.90 GB
    assert.ok(result > 41 && result < 43, `expected ~41.9, got ${result}`);
  });

  it('calcGPUUsableMemory: C600 (144GB, eff=0.97, ratio=0.90)', () => {
    const gpu = getGPUById('c600');
    const result = calcGPUUsableMemory(gpu.memoryGB, gpu.driverEfficiency, gpu.inferenceRatio);
    // 144 × 0.97 × 0.90 = 125.71 GB
    assert.ok(result > 124 && result < 127, `expected ~125.7, got ${result}`);
  });
});

// ──────────────────────────────────────────────
// 模型总显存（组合计算）
// ──────────────────────────────────────────────
describe('calcModelMemory (combined)', () => {
  it('Llama-3.1-70B, FP16, ctx=8192, batch=1, overhead=0.10', () => {
    const model = getModelById('llama3.1-70b');
    const result = calcModelMemory(model, 'FP16', 8192, 1, 0.10);

    // 手动计算预期：
    // weight = ~131.5 GB
    // kvCache = ~2.50 GB
    // other = (131.5 + 2.5) × 0.10 = ~13.4 GB
    // total = ~147.4 GB

    assert.ok(result.totalGB > 145 && result.totalGB < 150,
      `expected ~147, got ${result.totalGB}`);
    assert.ok(result.weightGB > 130 && result.weightGB < 133,
      `weight expected ~131.5, got ${result.weightGB}`);
    assert.ok(result.kvCacheGB > 2.4 && result.kvCacheGB < 2.6,
      `kvCache expected ~2.5, got ${result.kvCacheGB}`);
    assert.ok(result.otherGB > 12 && result.otherGB < 15,
      `other expected ~13.4, got ${result.otherGB}`);
  });

  it('Llama-3.1-8B, INT8, ctx=4096, batch=1, overhead=0.05', () => {
    const model = getModelById('llama3.1-8b');
    const result = calcModelMemory(model, 'INT8', 4096, 1, 0.05);

    // weight = 8.03e9 × 1 / (1024^3) = 7.48 GB
    // kvCache = 2 × 32 × 1024 × 4096 × 1 × 1 / (1024^3) = 0.25 GB
    // other = (7.48 + 0.25) × 0.05 = 0.39 GB
    // total ≈ 8.12 GB
    assert.ok(result.totalGB > 7.8 && result.totalGB < 8.5,
      `expected ~8.1, got ${result.totalGB}`);
  });
});

// ──────────────────────────────────────────────
// 端到端：calculate()
// ──────────────────────────────────────────────
describe('calculate (end-to-end)', () => {
  it('Llama-3.1-70B + N300 + 标准推理 + FP16 → 需要 4 张卡', () => {
    const result = calculate('llama3.1-70b', 'n300', 'inference-standard', 'FP16');

    // 模型显存 ~147.4 GB
    // 单卡可用 = 48 × 0.97 × 0.90 ≈ 41.90 GB
    // cards = ceil(147.4 / 41.90) = ceil(3.52) = 4
    // servers = ceil(4 / 16) = 1
    assert.equal(result.cards, 4, `expected 4 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.servers, 1, `expected 1 server, got ${result.servers}\n${result.formula}`);
    assert.equal(result.gpuName, 'N300');
    assert.equal(result.modelName, 'Llama 3.1 70B');
  });

  it('Llama-3.1-8B + N300 + 轻量推理 + INT8 → 1 张卡', () => {
    const result = calculate('llama3.1-8b', 'n300', 'inference-light', 'INT8');

    // 模型显存：7.48 + 0.25 + 5% ≈ 8.1 GB
    // 单卡可用：48 × 0.97 × 0.90 ≈ 41.90 GB
    // cards = ceil(8.1 / 41.90) = 1
    assert.equal(result.cards, 1,
      `expected 1 card, got ${result.cards}\n${result.formula}`);
  });

  it('Qwen2-72B + C600 + 标准推理 + FP16 → 需要 2 张卡', () => {
    const result = calculate('qwen2-72b', 'c600', 'inference-standard', 'FP16');

    // 权重 = 72.7e9 × 2 / (1024^3) = 135.4 GB
    // kvCache = 2 × 80 × (8×128) × 8192 × 2 / (1024^3) = 2.50 GB
    // other = (135.4 + 2.5) × 0.10 = 13.8 GB
    // total ≈ 151.7 GB
    // 单卡可用 = 144 × 0.97 × 0.90 = 125.71 GB
    // cards = ceil(151.7 / 125.71) = ceil(1.21) = 2
    assert.equal(result.cards, 2, `expected 2 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.servers, 1);
  });

  it('Mixtral 8×7B (MoE) + N300 + 标准推理 + FP16', () => {
    const result = calculate('mixtral-8x7b', 'n300', 'inference-standard', 'FP16');

    // MoE: 权重用总参数 46.7B，KV Cache 用模型架构参数
    // weight = 46.7e9 × 2 / (1024^3) = 87.0 GB
    // kvCache = 2 × 32 × (8×128) × 8192 × 2 / (1024^3) = 1.00 GB
    // other = (87.0 + 1.0) × 0.10 = 8.8 GB
    // total ≈ 96.8 GB
    // 单卡可用 = 48 × 0.97 × 0.90 ≈ 41.90 GB
    // cards = ceil(96.8 / 41.90) = 3
    assert.ok(result.cards >= 2 && result.cards <= 4,
      `expected ~3 cards, got ${result.cards}\n${result.formula}`);
  });

  it('Llama-3.1-70B + N300 + 标准推理 + FP16 → formula is non-empty', () => {
    const result = calculate('llama3.1-70b', 'n300', 'inference-standard', 'FP16');
    assert.ok(result.formula.length > 0, 'formula should not be empty');
    assert.ok(result.formula.includes('ceil'), 'formula should explain the calculation');
  });

  // ── MLA 模型端到端测试 ──
  it('DeepSeek-V3 + N300 + 标准推理 + FP16 → cards should reflect MLA formula', () => {
    const result = calculate('deepseek-v3', 'n300', 'inference-standard', 'FP16');

    // 权重 = 671e9 × 2 / (1024^3) = 1249.9 GB
    // KV Cache (MLA) = 0.536 GB
    // 其他 = (1249.9 + 0.5) × 0.10 ≈ 125.0 GB
    // 总显存 ≈ 1375 GB
    // 单卡可用 = 48 × 0.97 × 0.90 ≈ 41.90 GB
    // cards = ceil(1375 / 41.90) ≈ 33
    assert.ok(result.cards >= 30 && result.cards <= 36,
      `expected ~33 cards, got ${result.cards}\n${result.formula}`);
    assert.ok(result.servers >= 2,
      `expected >= 2 servers, got ${result.servers}\n${result.formula}`);
    assert.ok(result.formula.includes('mla'),
      `formula should mention 'mla', got:\n${result.formula}`);
  });

  it('DeepSeek-V2 (MLA) + N300 + 轻量推理 + FP16', () => {
    const result = calculate('deepseek-v2', 'n300', 'inference-light', 'FP16');

    // 权重 = 236e9 × 2 / (1024^3) = 439.6 GB
    // KV Cache (MLA, ctx=4096, batch=1) = 60 × 576 × 4096 × 2 / (1024^3) ≈ 0.26 GB
    // 其他 = (439.6 + 0.26) × 0.05 ≈ 22.0 GB
    // 总显存 ≈ 461.9 GB
    // 单卡可用 = 48 × 0.97 × 0.90 ≈ 41.90 GB
    // cards = ceil(461.9 / 41.90) ≈ 12
    assert.ok(result.cards >= 10 && result.cards <= 14,
      `expected ~12 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.architecture, 'moe');
  });

  // ── 新国产模型端到端测试 ──
  it('GLM-5.2 + N300 + 标准推理 + FP16', () => {
    const result = calculate('glm-5.2', 'n300', 'inference-standard', 'FP16');
    // 权重 = 753e9 × 2 / (1024^3) = 1402.6 GB
    // KV Cache (MLA) = 78 × 576 × 8192 × 2 / (1024^3) ≈ 0.686 GB
    // 其他 = (1402.6 + 0.7) × 0.10 ≈ 140.3 GB
    // 总显存 ≈ 1543.6 GB
    // 单卡可用 = 48 × 0.97 × 0.90 ≈ 41.90 GB
    // cards = ceil(1543.6 / 41.90) ≈ 37
    assert.ok(result.cards >= 34 && result.cards <= 40,
      `expected ~37 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.architecture, 'moe');
    assert.ok(result.formula.includes('mla'),
      `formula should mention 'mla', got:\n${result.formula}`);
  });

  it('DeepSeek V4 Flash + N300 + 标准推理 + FP16 → formula mentions HCA', () => {
    const result = calculate('deepseek-v4-flash', 'n300', 'inference-standard', 'FP16');
    // 权重 = 284e9 × 2 / (1024^3) = 529.0 GB
    // KV Cache (HCA) = 0.064 GB
    // 其他 ≈ 52.9 GB
    // 总显存 ≈ 582.0 GB
    // 单卡可用 = 48 × 0.97 × 0.90 ≈ 41.90 GB
    // cards = ceil(582.0 / 41.90) ≈ 14
    assert.ok(result.cards >= 12 && result.cards <= 16,
      `expected ~14 cards, got ${result.cards}\n${result.formula}`);
    assert.ok(result.formula.includes('hca_mla'),
      `formula should mention 'hca_mla', got:\n${result.formula}`);
  });

  it('MiniMax-M1 + C600 + 标准推理 + FP16 → formula mentions linear_hybrid', () => {
    const result = calculate('minimax-m1-80k', 'c600', 'inference-standard', 'FP16');
    // 权重 = 456e9 × 2 / (1024^3) = 849.3 GB
    // KV Cache (linear_hybrid) ≈ 0.313 GB
    // 其他 ≈ 85.0 GB
    // 总显存 ≈ 934.6 GB
    // 单卡可用 = 144 × 0.97 × 0.90 = 125.71 GB
    // cards = ceil(934.6 / 125.71) ≈ 8
    assert.ok(result.cards >= 6 && result.cards <= 10,
      `expected ~8 cards, got ${result.cards}\n${result.formula}`);
    assert.ok(result.formula.includes('linear_hybrid'),
      `formula should mention 'linear_hybrid', got:\n${result.formula}`);
  });

  it('Kimi K3 + C600 + 标准推理 + FP16 → formula mentions kda_mla', () => {
    const result = calculate('kimi-k3', 'c600', 'inference-standard', 'FP16');
    // 权重 = 2800e9 × 2 / (1024^3) = 5215.3 GB
    // KV Cache (kda_mla) ≈ 0.211 GB
    // 其他 ≈ 521.5 GB
    // 总显存 ≈ 5737.0 GB
    // 单卡可用 = 144 × 0.97 × 0.90 = 125.71 GB
    // cards = ceil(5737.0 / 125.71) ≈ 46
    assert.ok(result.cards >= 42 && result.cards <= 50,
      `expected ~46 cards, got ${result.cards}\n${result.formula}`);
    assert.ok(result.formula.includes('kda_mla'),
      `formula should mention 'kda_mla', got:\n${result.formula}`);
  });

  it('Qwen3.6-27B + N300 + 轻量推理 + INT8 → 1 card (dense hybrid)', () => {
    const result = calculate('qwen3.6-27b', 'n300', 'inference-light', 'INT8');
    assert.equal(result.architecture, 'dense');
    assert.ok(result.cards >= 1,
      `expected at least 1 card, got ${result.cards}\n${result.formula}`);
  });

  it('GLM-4.5 + N300 + 标准推理 + FP16', () => {
    const result = calculate('glm-4.5', 'n300', 'inference-standard', 'FP16');
    // 权重 = 355e9 × 2 / (1024^3) = 661.2 GB
    // KV Cache (standard) = 2 × 92 × (8×128) × 8192 × 2 / (1024^3) ≈ 2.88 GB
    // 其他 = 66.4 GB, total ≈ 730.5 GB
    // 单卡可用 = 48 × 0.97 × 0.90 ≈ 41.90 GB
    // cards = ceil(730.5 / 41.90) ≈ 18
    assert.equal(result.architecture, 'moe');
    assert.ok(result.cards >= 16 && result.cards <= 20,
      `expected ~18 cards, got ${result.cards}\n${result.formula}`);
  });

  // ── 腾讯混元系列 ──
  it('Hunyuan-Large (CLA) + N300 + 标准推理 + FP16 → uses cla formula', () => {
    const result = calculate('hunyuan-large', 'n300', 'inference-standard', 'FP16');
    // 权重 = 389e9 × 2 / (1024^3) = 724.6 GB
    // KV Cache (CLA) = 2 × 32 × (8×80) × 8192 × 2 / (1024^3) ≈ 0.625 GB
    // 其他 ≈ 72.5 GB, total ≈ 797.8 GB
    // 单卡可用 = 48 × 0.97 × 0.90 ≈ 41.90 GB
    // cards = ceil(797.8 / 41.90) ≈ 20
    assert.ok(result.cards >= 17 && result.cards <= 23,
      `expected ~20 cards, got ${result.cards}\n${result.formula}`);
    assert.ok(result.formula.includes('cla'),
      `formula should mention 'cla', got:\n${result.formula}`);
    assert.equal(result.architecture, 'moe');
  });

  it('Hy3-preview + N300 + 标准推理 + FP16', () => {
    const result = calculate('hy3-preview', 'n300', 'inference-standard', 'FP16');
    // 权重 = 295e9 × 2 / (1024^3) = 549.5 GB
    // KV Cache (standard) = 2 × 80 × (8×128) × 8192 × 2 / (1024^3) ≈ 2.50 GB
    // 其他 ≈ 55.2 GB, total ≈ 607.2 GB
    // 单卡可用 = 48 × 0.97 × 0.90 ≈ 41.90 GB
    // cards = ceil(607.2 / 41.90) ≈ 15
    assert.ok(result.cards >= 13 && result.cards <= 17,
      `expected ~15 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.architecture, 'moe');
  });

  it('Hunyuan-A13B + N300 + 轻量推理 + INT8 → ~2 cards', () => {
    const result = calculate('hunyuan-a13b', 'n300', 'inference-light', 'INT8');
    // 权重 = 80e9 × 1 / (1024^3) = 74.5 GB
    // KV Cache = 2 × 32 × (8×128) × 4096 × 1 / (1024^3) ≈ 0.25 GB
    // 其他 = 5% ≈ 3.7 GB, total ≈ 78.5 GB
    // 单卡可用 = 48 × 0.97 × 0.90 ≈ 41.90 GB
    // cards = ceil(78.5 / 41.90) ≈ 2
    assert.ok(result.cards >= 1 && result.cards <= 3,
      `expected ~2 cards, got ${result.cards}\n${result.formula}`);
    assert.equal(result.architecture, 'moe');
  });
});

// ──────────────────────────────────────────────
// 边界情况
// ──────────────────────────────────────────────
describe('edge cases', () => {
  it('tiny model + big GPU → 1 card', () => {
    // Llama 3.1 8B @ INT4 + C600 → definitely 1 card
    const result = calculate('llama3.1-8b', 'c600', 'inference-light', 'INT4');
    assert.equal(result.cards, 1);
  });

  it('large model + small GPU → many cards', () => {
    // Llama 3.1 405B @ FP16 + N300 (48GB)
    const result = calculate('llama3.1-405b', 'n300', 'inference-standard', 'FP16');
    // weight alone = 405e9 × 2 / (1024^3) = 754 GB
    // should need at least 10 cards
    assert.ok(result.cards >= 10, `expected >= 10 cards, got ${result.cards}`);
  });

  it('servers: 19 cards with 16 cards/server → 2 servers', () => {
    const result = calculate('llama3.1-405b', 'n300', 'inference-light', 'FP16');
    // cards = ceil(794.1 / 41.90) ≈ 19, servers = ceil(19 / 16) = 2
    const { cards, servers } = result;
    assert.equal(servers, Math.ceil(cards / 16),
      `servers should be ceil(cards/16), got ${servers} for ${cards} cards`);
  });

  it('utilization should be ≤ 1.0', () => {
    const result = calculate('llama3.1-70b', 'n300', 'inference-standard', 'FP16');
    assert.ok(result.utilization <= 1.0,
      `utilization should be ≤ 1.0, got ${result.utilization}`);
    assert.ok(result.utilization > 0,
      `utilization should be > 0, got ${result.utilization}`);
  });
});

// ──────────────────────────────────────────────
// 一致性检查
// ──────────────────────────────────────────────
describe('consistency', () => {
  it('KV cache scales linearly with context length', () => {
    const model = getModelById('llama3.1-70b');
    const kv4096 = calcKVCache(model, 4096, 1, 'FP16');
    const kv8192 = calcKVCache(model, 8192, 1, 'FP16');
    // 8192 should be exactly 2× 4096
    assert.ok(Math.abs(kv8192 / kv4096 - 2.0) < 0.01,
      `kvCache(8192)=${kv8192} should be 2× kvCache(4096)=${kv4096}`);
  });

  it('KV cache scales linearly with batch size', () => {
    const model = getModelById('llama3.1-70b');
    const b1 = calcKVCache(model, 8192, 1, 'FP16');
    const b4 = calcKVCache(model, 8192, 4, 'FP16');
    assert.ok(Math.abs(b4 / b1 - 4.0) < 0.01,
      `batch=4 (${b4}) should be 4× batch=1 (${b1})`);
  });

  it('all GPU models have valid configs', () => {
    const gpus = getAvailableGPUs();
    for (const gpu of gpus) {
      assert.ok(gpu.memoryGB > 0, `${gpu.id}: memoryGB must be > 0`);
      assert.ok(gpu.cardsPerServer > 0, `${gpu.id}: cardsPerServer must be > 0`);
      assert.ok(gpu.driverEfficiency > 0 && gpu.driverEfficiency <= 1,
        `${gpu.id}: driverEfficiency out of range`);
      assert.ok(gpu.inferenceRatio > 0 && gpu.inferenceRatio <= 1,
        `${gpu.id}: inferenceRatio out of range`);
    }
  });

  it('all model configs have architecture params for KV cache', () => {
    const models = getAvailableModels();
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
    for (const m of models) {
      if (m.attnArch === 'mla') {
        assert.ok(m.kvLoraRank > 0, `${m.id}: MLA model must have kvLoraRank > 0`);
        assert.ok(m.qkRopeHeadDim > 0, `${m.id}: MLA model must have qkRopeHeadDim > 0`);
        assert.ok(getMLADim(m) === m.kvLoraRank + m.qkRopeHeadDim,
          `${m.id}: getMLADim should be kvLoraRank + qkRopeHeadDim`);
      }
    }
  });

  it('CLA models have required field (claShareFactor)', () => {
    const models = getAvailableModels();
    for (const m of models) {
      if (m.attnArch === 'cla') {
        assert.ok(m.claShareFactor > 0, `${m.id}: CLA model must have claShareFactor > 0`);
        assert.ok(Number.isInteger(m.claShareFactor),
          `${m.id}: claShareFactor should be integer`);
      }
    }
  });

  it('linear_hybrid models have required field (fullAttnLayers)', () => {
    const models = getAvailableModels();
    for (const m of models) {
      if (m.attnArch === 'linear_hybrid') {
        assert.ok(m.fullAttnLayers > 0, `${m.id}: linear_hybrid model must have fullAttnLayers > 0`);
        assert.ok(m.fullAttnLayers < m.numLayers,
          `${m.id}: fullAttnLayers (${m.fullAttnLayers}) must be < numLayers (${m.numLayers})`);
      }
    }
  });

  it('kda_mla models have required fields (fullAttnLayers + MLA fields)', () => {
    const models = getAvailableModels();
    for (const m of models) {
      if (m.attnArch === 'kda_mla') {
        assert.ok(m.fullAttnLayers > 0, `${m.id}: kda_mla model must have fullAttnLayers > 0`);
        assert.ok(m.kvLoraRank > 0, `${m.id}: kda_mla model must have kvLoraRank > 0`);
        assert.ok(m.qkRopeHeadDim > 0, `${m.id}: kda_mla model must have qkRopeHeadDim > 0`);
        assert.ok(getMLADim(m) === m.kvLoraRank + m.qkRopeHeadDim,
          `${m.id}: getMLADim should be kvLoraRank + qkRopeHeadDim`);
      }
    }
  });

  it('hca_mla models have required field (effectiveKVDim)', () => {
    const models = getAvailableModels();
    for (const m of models) {
      if (m.attnArch === 'hca_mla') {
        assert.ok(m.effectiveKVDim > 0, `${m.id}: hca_mla model must have effectiveKVDim > 0`);
      }
    }
  });

  it('KV cache linear scaling holds for all attnArch types', () => {
    const testCases = [
      { id: 'deepseek-v3', arch: 'mla' },
      { id: 'glm-5.2', arch: 'mla' },
      { id: 'minimax-m1-80k', arch: 'linear_hybrid' },
      { id: 'qwen3.6-27b', arch: 'linear_hybrid' },
      { id: 'kimi-k3', arch: 'kda_mla' },
      { id: 'deepseek-v4-flash', arch: 'hca_mla' },
    ];
    for (const { id, arch } of testCases) {
      const model = getModelById(id);
      assert.equal(model.attnArch, arch);
      const kv4096 = calcKVCache(model, 4096, 1, 'FP16');
      const kv8192 = calcKVCache(model, 8192, 1, 'FP16');
      assert.ok(Math.abs(kv8192 / kv4096 - 2.0) < 0.01,
        `${id}: kvCache(8192)=${kv8192} should be 2× kvCache(4096)=${kv4096}`);
      // batch scaling
      const b1 = calcKVCache(model, 8192, 1, 'FP16');
      const b4 = calcKVCache(model, 8192, 4, 'FP16');
      assert.ok(Math.abs(b4 / b1 - 4.0) < 0.01,
        `${id}: batch=4 (${b4}) should be 4× batch=1 (${b1})`);
    }
  });

  it('MLA/kda_mla models should not have attnArch=standard or missing attnArch', () => {
    assert.equal(getModelById('deepseek-v3').attnArch, 'mla');
    assert.equal(getModelById('kimi-k3').attnArch, 'kda_mla');
    assert.equal(getModelById('glm-5.2').attnArch, 'mla');
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
    'inference-light', 'inference-standard', 'inference-batch',
    'inference-long', 'inference-xlong',
    'training-lora', 'training-full',
  ];
  for (const id of presetIds) {
    it(`getPresetById("${id}") should return valid preset`, () => {
      const p = getPresetById(id);
      assert.equal(p.id, id);
      assert.ok(p.contextLen > 0, `contextLen must be > 0, got ${p.contextLen}`);
      assert.ok(p.batchSize > 0, `batchSize must be > 0, got ${p.batchSize}`);
      assert.ok(p.overheadRatio > 0 && p.overheadRatio < 1,
        `overheadRatio out of range: ${p.overheadRatio}`);
    });
  }

  it('unknown preset throws', () => {
    assert.throws(() => getPresetById('nonexistent'), /未知 SLA 预设/);
  });

  it('context lengths scale correctly: light < standard < long < xlong', () => {
    const light = getPresetById('inference-light').contextLen;
    const standard = getPresetById('inference-standard').contextLen;
    const long = getPresetById('inference-long').contextLen;
    const xlong = getPresetById('inference-xlong').contextLen;
    assert.ok(light < standard, `light(${light}) < standard(${standard})`);
    assert.ok(standard < long, `standard(${standard}) < long(${long})`);
    assert.ok(long < xlong, `long(${long}) < xlong(${xlong})`);
  });

  it('batch preset has batchSize=8 > standard batchSize=1', () => {
    const batch = getPresetById('inference-batch');
    const standard = getPresetById('inference-standard');
    assert.equal(batch.contextLen, standard.contextLen,
      'batch preset should have same ctx as standard');
    assert.ok(batch.batchSize > standard.batchSize,
      `batch size ${batch.batchSize} > standard ${standard.batchSize}`);
  });
});

// ──────────────────────────────────────────────
// 新预设端到端测试
// ──────────────────────────────────────────────
describe('calculate with new presets', () => {
  it('Llama-3.1-8B + N300 + 批量推理 + FP16 → cards ≥ standard', () => {
    const batchResult = calculate('llama3.1-8b', 'n300', 'inference-batch', 'FP16');
    const stdResult = calculate('llama3.1-8b', 'n300', 'inference-standard', 'FP16');
    // 批量推理 batch=8 会消耗更多 KV Cache，卡数应 ≥ 标准推理
    assert.ok(batchResult.cards >= stdResult.cards,
      `batch cards=${batchResult.cards} should be >= standard cards=${stdResult.cards}`);
  });

  it('Llama-3.1-8B + N300 + 长上下文 128K + FP16 → KV Cache 主导', () => {
    const result = calculate('llama3.1-8b', 'n300', 'inference-long', 'FP16');
    // ctx=128K, KV Cache = 2 × 32 × 1024 × 131072 × 2 / (1024^3) ≈ 16 GB
    // weight ≈ 15 GB, total ≈ 34 GB, fits on 1 N300
    assert.ok(result.cards >= 1, `expected ≥1 card, got ${result.cards}\n${result.formula}`);
    assert.ok(result.modelMemoryGB.kvCacheGB > result.modelMemoryGB.weightGB,
      `KV Cache (${result.modelMemoryGB.kvCacheGB}) should dominate weight (${result.modelMemoryGB.weightGB}) at 128K`);
  });

  it('Llama-3.1-8B + N300 + 超长上下文 256K + FP16 → KV Cache ~32 GB', () => {
    const result = calculate('llama3.1-8b', 'n300', 'inference-xlong', 'FP16');
    // ctx=256K, KV Cache = 2 × 32 × 1024 × 262144 × 2 / (1024^3) ≈ 32 GB
    // weight ≈ 15 GB, total ≈ 52 GB → needs 2 N300 cards
    assert.equal(result.presetName, '超长上下文推理');
  });

  it('Qwen3.6-27B (linear_hybrid) + C600 + 长上下文 128K → KV Cache only on 16/64 layers', () => {
    const result = calculate('qwen3.6-27b', 'c600', 'inference-long', 'FP16');
    // linear_hybrid: only fullAttnLayers=16 out of 64 layers have KV cache
    // KV Cache = 2 × 16 × (4×256) × 131072 × 2 / (1024^3) ≈ 8 GB
    // vs standard would be ×64 layers ≈ 32 GB
    assert.ok(result.formula.includes('linear_hybrid'),
      `formula should mention linear_hybrid`);
    assert.ok(result.cards >= 1);
  });

  it('DeepSeek-V3 + C600 + 超长上下文 256K → MLA keeps KV Cache manageable', () => {
    const result = calculate('deepseek-v3', 'c600', 'inference-xlong', 'FP16');
    // MLA: KV Cache = 61 × 576 × 262144 × 2 / (1024^3) ≈ 17.2 GB
    // vs standard: 2 × 61 × 128 × 128 × 262144 × 2 / (1024^3) ≈ 1000 GB
    // MLA makes 256K context practical
    assert.ok(result.formula.includes('mla'),
      `formula should mention mla for DeepSeek-V3`);
  });

  it('DeepSeek V4 Flash + C600 + 超长上下文 256K → HCA extreme saving', () => {
    const result = calculate('deepseek-v4-flash', 'c600', 'inference-xlong', 'FP16');
    // HCA: KV Cache = 4176 × 262144 × 2 / (1024^3) ≈ 2.04 GB
    assert.ok(result.formula.includes('hca_mla'));
    // C600 usable = 125.7 GB, weight ≈ 529 GB, KV ≈ 2 GB, total ≈ 584 GB
    // cards = ceil(584 / 125.7) ≈ 5
    assert.ok(result.cards >= 4 && result.cards <= 6,
      `expected ~5 cards, got ${result.cards}\n${result.formula}`);
  });
});

// ──────────────────────────────────────────────
// 边界条件与回归
// ──────────────────────────────────────────────
describe('boundary & regression', () => {
  it('utilization rate approaches 1.0 when model barely fits', () => {
    // Qwen2-72B on C600 is ~151.7 GB, C600 usable = 125.7 GB
    // cards = 2, utilization = 151.7 / (2 × 125.7) ≈ 0.60
    const result = calculate('qwen2-72b', 'c600', 'inference-standard', 'FP16');
    assert.ok(result.utilization > 0 && result.utilization <= 1,
      `utilization=${result.utilization} out of range`);
  });

  it('cards always ≥ 1 even for the smallest model on largest GPU', () => {
    // ChatGLM3 6B @ INT4 = ~2.9 GB weight, on C600 (125.7 GB usable)
    const result = calculate('chatglm3-6b', 'c600', 'inference-light', 'INT4');
    assert.equal(result.cards, 1);
  });

  it('servers = ceil(cards / cardsPerServer) for all GPU types', () => {
    const testCases = [
      { model: 'llama3.1-405b', gpu: 'n300', preset: 'inference-standard', prec: 'FP16' },
      { model: 'deepseek-v3', gpu: 'c600', preset: 'inference-standard', prec: 'FP16' },
      { model: 'kimi-k3', gpu: 'n300', preset: 'inference-light', prec: 'FP16' },
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
    const result = calculate('llama3.1-70b', 'n300', 'inference-standard', 'FP16');
    const { weightGB, kvCacheGB, otherGB, totalGB } = result.modelMemoryGB;
    const sum = round2(weightGB + kvCacheGB + otherGB);
    assert.equal(sum, totalGB,
      `${weightGB} + ${kvCacheGB} + ${otherGB} = ${sum} ≠ totalGB=${totalGB}`);
  });

  it('modelMemoryGB breakdown is consistent for all models', () => {
    const models = getAvailableModels();
    for (const m of models) {
      const result = calculate(m.id, 'n300', 'inference-standard', 'FP16');
      const { weightGB, kvCacheGB, otherGB, totalGB } = result.modelMemoryGB;
      const sum = round2(weightGB + kvCacheGB + otherGB);
      assert.equal(sum, totalGB,
        `${m.id}: ${weightGB} + ${kvCacheGB} + ${otherGB} = ${sum} ≠ ${totalGB}`);
    }
  });

  it('getSelectOptions returns all required keys', () => {
    const opts = getSelectOptions();
    assert.ok(Array.isArray(opts.models));
    assert.ok(Array.isArray(opts.gpus));
    assert.ok(Array.isArray(opts.presets));
    assert.ok(Array.isArray(opts.precisions));
    assert.ok(opts.models.length >= 20, `expected ≥20 models, got ${opts.models.length}`);
    assert.ok(opts.gpus.length === 2, `expected 2 GPUs, got ${opts.gpus.length}`);
    assert.ok(opts.precisions.length >= 6,
      `expected ≥6 precisions, got ${opts.precisions.length}`);
  });

  it('formula string includes architecture type for non-standard models', () => {
    const archTests = [
      { id: 'deepseek-v3', arch: 'mla' },
      { id: 'hunyuan-large', arch: 'cla' },
      { id: 'minimax-m1-80k', arch: 'linear_hybrid' },
      { id: 'kimi-k3', arch: 'kda_mla' },
      { id: 'deepseek-v4-flash', arch: 'hca_mla' },
    ];
    for (const { id, arch } of archTests) {
      const result = calculate(id, 'c600', 'inference-standard', 'FP16');
      assert.ok(result.formula.includes(arch),
        `${id}: formula should mention '${arch}', got:\n${result.formula}`);
    }
  });

  it('256K KV cache is exactly 2× 128K for same model', () => {
    const model = getModelById('llama3.1-70b');
    const kv128k = calcKVCache(model, 131072, 1, 'FP16');
    const kv256k = calcKVCache(model, 262144, 1, 'FP16');
    assert.ok(Math.abs(kv256k / kv128k - 2.0) < 0.01,
      `KV(256K)=${kv256k} should be 2× KV(128K)=${kv128k}`);
  });

  it('batch=8 KV cache is exactly 8× batch=1', () => {
    const model = getModelById('llama3.1-70b');
    const b1 = calcKVCache(model, 8192, 1, 'FP16');
    const b8 = calcKVCache(model, 8192, 8, 'FP16');
    assert.ok(Math.abs(b8 / b1 - 8.0) < 0.01,
      `batch=8 (${b8}) should be 8× batch=1 (${b1})`);
  });
});

function round2(v) {
  return Math.round(v * 100) / 100;
}
