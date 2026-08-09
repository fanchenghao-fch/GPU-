/**
 * 模型常量数据
 *
 * ⚠️ 维护说明（产品部同事）：
 * 新主流模型发布时，在 MODEL_DATA 数组中新增一条记录即可。
 *
 * 字段说明：
 * - id:              内部标识
 * - displayName:     前端展示的模型名称（如 'Llama-3.1-70B'）
 * - paramsB:         总参数量（B = 十亿），MoE 模型为全部专家之和
 * - architecture:    'dense' | 'moe' （影响权重显存计算）
 * - attnArch:        'standard' | 'mla' | 'cla' （影响 KV Cache 计算公式，见下方）
 * - numLayers:       Transformer 层数
 * - hiddenDim:       隐藏层维度
 * - numKVHeads:      KV head 数量（GQA 模型此值 < numAttentionHeads）
 * - headDim:         每个 attention head 的维度
 *   ─ 以下字段仅在特定 attnArch 时需要 ─
 * - kvLoraRank:      [MLA] KV LoRA 隐空间维度（DeepSeek 使用）
 * - qkRopeHeadDim:   [MLA] QK RoPE head 维度（DeepSeek 使用）
 * - claShareFactor:  [CLA] 多少层共享一份 KV Cache（Hunyuan-Large 使用）
 * - vocabSize:       词表大小（可选，影响 Embedding 层显存，通常占比极小可忽略）
 * - available:       是否可用（设为 false 不展示）
 *
 * KV Cache 计算公式（按 attnArch 分支）：
 *
 *   standard:  KV Cache = 2 × numLayers × numKVHeads × headDim × contextLen × batchSize × bytesPerKV
 *              适用于 Llama、Qwen2、Mixtral、Qwen3、GLM4、Hunyuan Dense 等标准 Transformer 模型
 *
 *   mla:       KV Cache = numLayers × (kvLoraRank + qkRopeHeadDim) × contextLen × batchSize × bytesPerKV
 *              适用于 DeepSeek-V2/V3/R1、GLM-5.2 等 MLA（Multi-head Latent Attention）模型
 *              注意：没有 ×2，因为 K 和 V 被压缩为一个联合隐向量
 *
 *   cla:       KV Cache = 2 × (numLayers / claShareFactor) × numKVHeads × headDim × contextLen × batchSize × bytesPerKV
 *              适用于 Hunyuan-Large 等 CLA（Cross-Layer Attention）模型
 *              每 claShareFactor 层共享一份 KV Cache，有效层数 = numLayers / claShareFactor
 *   linear_hybrid: KV Cache = 2 × fullAttnLayers × numKVHeads × headDim × contextLen × batchSize × bytesPerKV
 *              适用于 MiniMax-M1、Qwen3.6 等线性注意力+标准注意力混合模型
 *              仅 full-attention 层产生 KV Cache；线性注意力层不随 ctx 增长
 *
 *   kda_mla:   KV Cache = fullAttnLayers × (kvLoraRank + qkRopeHeadDim) × contextLen × batchSize × bytesPerKV
 *              适用于 Kimi K3（KDA 动态注意力 + Gated MLA 混合）模型
 *              KDA 层无 KV Cache，MLA 层使用压缩 KV
 *
 *   hca_mla:   KV Cache = effectiveKVDim × contextLen × batchSize × bytesPerKV
 *              适用于 DeepSeek V4 Flash/Pro（HCA 混合压缩注意力 + MLA）模型
 *              逐层不等压缩比，使用预计算等效 KV 维度
 */

export const MODEL_DATA = [
  // ========== Dense 模型 ==========
  {
    id: 'llama3.1-8b',
    displayName: 'Llama 3.1 8B',
    paramsB: 8.03,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 32,
    hiddenDim: 4096,
    numKVHeads: 8,
    headDim: 128,
    available: true,
  },
  {
    id: 'llama3.1-70b',
    displayName: 'Llama 3.1 70B',
    paramsB: 70.6,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 80,
    hiddenDim: 8192,
    numKVHeads: 8,
    headDim: 128,
    available: true,
  },
  {
    id: 'llama3.1-405b',
    displayName: 'Llama 3.1 405B',
    paramsB: 405,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 126,
    hiddenDim: 16384,
    numKVHeads: 8,
    headDim: 128,
    available: true,
  },
  {
    id: 'qwen2-7b',
    displayName: 'Qwen2 7B',
    paramsB: 7.07,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 28,
    hiddenDim: 3584,
    numKVHeads: 4,
    headDim: 128,
    available: true,
  },
  {
    id: 'qwen2-72b',
    displayName: 'Qwen2 72B',
    paramsB: 72.7,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 80,
    hiddenDim: 8192,
    numKVHeads: 8,
    headDim: 128,
    available: true,
  },
  {
    id: 'qwen3-8b',
    displayName: 'Qwen3 8B',
    paramsB: 8.2,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 36,
    hiddenDim: 4096,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: 'Qwen3 有显式 head_dim=128，与 hidden_dim/num_heads 一致，使用 standard 公式',
  },
  {
    id: 'qwen3-235b',
    displayName: 'Qwen3 235B',
    paramsB: 235,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 94,
    hiddenDim: 14336,
    numKVHeads: 8,
    headDim: 128,
    available: true,
  },
  {
    id: 'yi-34b',
    displayName: 'Yi 34B',
    paramsB: 34.4,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 60,
    hiddenDim: 7168,
    numKVHeads: 8,
    headDim: 128,
    available: true,
  },
  {
    id: 'chatglm3-6b',
    displayName: 'ChatGLM3 6B',
    paramsB: 6.2,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 28,
    hiddenDim: 4096,
    numKVHeads: 8,
    headDim: 128,
    available: true,
  },

  // ========== MoE 模型（标准 Attention）==========
  {
    id: 'mixtral-8x7b',
    displayName: 'Mixtral 8×7B (MoE)',
    paramsB: 46.7,
    architecture: 'moe',
    attnArch: 'standard',
    numLayers: 32,
    hiddenDim: 4096,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '总参数 46.7B（8 位专家 × 7B + 共享参数），激活参数约 12.9B',
  },
  {
    id: 'mixtral-8x22b',
    displayName: 'Mixtral 8×22B (MoE)',
    paramsB: 141,
    architecture: 'moe',
    attnArch: 'standard',
    numLayers: 56,
    hiddenDim: 6144,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '总参数 141B，激活参数约 39B',
  },
  {
    id: 'qwen2.5-72b-moe',
    displayName: 'Qwen2.5 72B MoE',
    paramsB: 72.7,
    architecture: 'moe',
    attnArch: 'standard',
    numLayers: 48,
    hiddenDim: 5120,
    numKVHeads: 4,
    headDim: 128,
    available: true,
    note: 'MoE 架构，激活参数约 13B',
  },

  // ========== MLA 模型（DeepSeek MLA Attention）==========
  {
    id: 'deepseek-v3',
    displayName: 'DeepSeek-V3 (MLA)',
    paramsB: 671,
    architecture: 'moe',
    attnArch: 'mla',
    numLayers: 61,
    hiddenDim: 7168,
    numKVHeads: 128,
    headDim: 128,
    kvLoraRank: 512,
    qkRopeHeadDim: 64,
    available: true,
    note: '总参数 671B（MoE），激活参数约 37B；MLA 架构，KV Cache ≈ 标准公式的 1/3.6',
  },
  {
    id: 'deepseek-v2',
    displayName: 'DeepSeek-V2 (MLA)',
    paramsB: 236,
    architecture: 'moe',
    attnArch: 'mla',
    numLayers: 60,
    hiddenDim: 5120,
    numKVHeads: 128,
    headDim: 128,
    kvLoraRank: 512,
    qkRopeHeadDim: 64,
    available: true,
    note: '总参数 236B（MoE），激活参数约 21B；MLA 架构',
  },

  // ========== GLM-5.2（MLA + DSA）==========
  {
    id: 'glm-5.2',
    displayName: 'GLM-5.2 (MLA)',
    paramsB: 753,
    architecture: 'moe',
    attnArch: 'mla',
    numLayers: 78,
    hiddenDim: 6144,
    numKVHeads: 64,
    headDim: 192,
    kvLoraRank: 512,
    qkRopeHeadDim: 64,
    available: true,
    note: '总参数 753B（MoE），激活参数约 40B；MLA + DSA 架构，KV Cache 使用 MLA 压缩',
  },

  // ========== DeepSeek V4 Flash（HCA + MLA）==========
  {
    id: 'deepseek-v4-flash',
    displayName: 'DeepSeek V4 Flash (HCA)',
    paramsB: 284,
    architecture: 'moe',
    attnArch: 'hca_mla',
    numLayers: 43,
    hiddenDim: 4096,
    numKVHeads: 1,
    headDim: 512,
    effectiveKVDim: 4176,
    available: true,
    note: '总参数 284B（MoE），激活参数约 13B；HCA+MLA，KV Cache ≈ 标准公式的 1/10.5（vLLM 实测 ~8.7× 节省）',
  },

  // ========== MiniMax-M1（Linear Hybrid）==========
  {
    id: 'minimax-m1-80k',
    displayName: 'MiniMax-M1 80K (Hybrid)',
    paramsB: 456,
    architecture: 'moe',
    attnArch: 'linear_hybrid',
    numLayers: 80,
    hiddenDim: 6144,
    numKVHeads: 8,
    headDim: 128,
    fullAttnLayers: 10,
    available: true,
    note: '总参数 456B（MoE），激活参数约 45.9B；80 层中仅 10 层为标准注意力，其余为 Lightning 线性注意力',
  },

  // ========== Kimi K3（KDA + Gated MLA）==========
  {
    id: 'kimi-k3',
    displayName: 'Kimi K3 (KDA+MLA)',
    paramsB: 2800,
    architecture: 'moe',
    attnArch: 'kda_mla',
    numLayers: 93,
    hiddenDim: 7168,
    numKVHeads: 96,
    headDim: 128,
    kvLoraRank: 512,
    qkRopeHeadDim: 64,
    fullAttnLayers: 24,
    available: true,
    note: '总参数 2.8T（MoE），激活参数约 104B；93 层中 24 层为 Gated MLA，69 层为 KDA 线性注意力',
  },

  // ========== Qwen3.6-27B（Linear Hybrid）==========
  {
    id: 'qwen3.6-27b',
    displayName: 'Qwen3.6 27B (Hybrid)',
    paramsB: 27,
    architecture: 'dense',
    attnArch: 'linear_hybrid',
    numLayers: 64,
    hiddenDim: 5120,
    numKVHeads: 4,
    headDim: 256,
    fullAttnLayers: 16,
    available: true,
    note: '64 层中每 4 层 1 个 full attention（16 层），其余为 linear attention；Qwen3.8-27B 待权重发布后追加',
  },

  // ========== GLM-4.5（Standard MoE）==========
  {
    id: 'glm-4.5',
    displayName: 'GLM-4.5 (MoE)',
    paramsB: 355,
    architecture: 'moe',
    attnArch: 'standard',
    numLayers: 92,
    hiddenDim: 5120,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '总参数 355B（MoE），激活参数约 32B；标准 attention 架构，面向 Agent 场景',
  },

  // ========== 腾讯混元系列 ==========
  {
    id: 'hunyuan-large',
    displayName: 'Hunyuan-Large (CLA)',
    paramsB: 389,
    architecture: 'moe',
    attnArch: 'cla',
    numLayers: 64,
    hiddenDim: 6400,
    numKVHeads: 8,
    headDim: 80,
    claShareFactor: 2,
    available: true,
    note: '总参数 389B（MoE），激活参数约 52B；CLA 架构，每 2 层共享 KV Cache，有效层数 = 64/2 = 32',
  },
  {
    id: 'hy3-preview',
    displayName: 'Hy3-preview (MoE)',
    paramsB: 295,
    architecture: 'moe',
    attnArch: 'standard',
    numLayers: 80,
    hiddenDim: 4096,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '总参数 295B（MoE），激活参数约 21B；192 专家，标准 GQA attention，Apache 2.0',
  },
  {
    id: 'hunyuan-a13b',
    displayName: 'Hunyuan-A13B (MoE)',
    paramsB: 80,
    architecture: 'moe',
    attnArch: 'standard',
    numLayers: 32,
    hiddenDim: 4096,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '总参数 80B（MoE），激活参数约 13B；64 专家，标准 GQA attention',
  },
];

/**
 * 根据 ID 查找模型配置
 * @param {string} id - 模型 ID
 * @returns {object} 模型配置对象
 */
export function getModelById(id) {
  const model = MODEL_DATA.find((m) => m.id === id);
  if (!model) {
    throw new Error(`未知模型: "${id}"，可选: ${MODEL_DATA.map((m) => m.id).join(', ')}`);
  }
  return model;
}

/**
 * 获取所有可用模型（供前端下拉列表使用）
 * @param {'all'|'dense'|'moe'} architecture - 按架构过滤
 */
export function getAvailableModels(architecture = 'all') {
  const available = MODEL_DATA.filter((m) => m.available);
  if (architecture === 'dense') return available.filter((m) => m.architecture === 'dense');
  if (architecture === 'moe') return available.filter((m) => m.architecture === 'moe');
  return available;
}

/**
 * KV Cache 的 KV 投影维度
 *
 * 对于 standard / cla 模型：
 *   kvDim = numKVHeads × headDim
 *   GQA（Grouped Query Attention）下，kvDim < hiddenDim
 *
 * 对于 MLA 模型：
 *   请使用 getMLADim() 获取 MLA 特有的隐向量维度
 */
export function getKVDim(model) {
  return model.numKVHeads * model.headDim;
}

/**
 * MLA（Multi-head Latent Attention）的 KV 联合隐向量维度
 * = kvLoraRank + qkRopeHeadDim
 *
 * 仅在 attnArch === 'mla' 时有效
 */
export function getMLADim(model) {
  if (model.attnArch !== 'mla' && model.attnArch !== 'kda_mla') {
    throw new Error(`getMLADim 仅适用于 attnArch='mla' 或 'kda_mla' 的模型，当前模型为 '${model.attnArch}'`);
  }
  return model.kvLoraRank + model.qkRopeHeadDim;
}

/**
 * 检查模型是否使用 MLA 或 KDA+MLA 压缩 KV
 */
export function isMLA(model) {
  return model.attnArch === 'mla' || model.attnArch === 'kda_mla';
}
