/**
 * 模型常量数据
 *
 * ⚠️ 维护说明（产品部同事）：
 * 新主流模型发布时，在 MODEL_DATA 数组中新增一条记录即可。
 *
 * 当前覆盖：阿里云/通义千问、DeepSeek、智谱AI/Z.ai、MiniMax、月之暗面、腾讯（6家厂商，30款模型）
 *
 * 字段说明：
 * - id:              内部标识
 * - displayName:     前端展示名称，格式为 "模型型号（参数量）"
 *                      Dense 模型：如 "Qwen3.5-0.8B（0.8B）"
 *                      MoE 模型：如 "DeepSeek-V3.2（671B-A37B）"
 * - paramsB:         总参数量（B = 十亿），MoE 模型为全部专家之和
 * - architecture:    'dense' | 'moe' （影响权重显存计算）
 * - attnArch:        'standard' | 'mla' | 'cla' | 'linear_hybrid' | 'kda_mla' | 'hca_mla'
 * - numLayers:       Transformer 层数
 * - hiddenDim:       隐藏层维度
 * - numKVHeads:      KV head 数量（GQA 模型此值 < numAttentionHeads）
 * - headDim:         每个 attention head 的维度
 *   ─ 以下字段仅在特定 attnArch 时需要 ─
 * - kvLoraRank:      [mla/kda_mla] KV LoRA 隐空间维度
 * - qkRopeHeadDim:   [mla/kda_mla] QK RoPE head 维度
 * - claShareFactor:  [cla] 多少层共享一份 KV Cache
 * - fullAttnLayers:  [linear_hybrid/kda_mla] 产生 KV Cache 的全注意力层数
 * - effectiveKVDim:  [hca_mla] 预计算等效 KV 维度
 * - available:       是否可用（设为 false 不展示）
 *
 * KV Cache 计算公式（按 attnArch 分支）：
 *
 *   standard:  KV Cache = 2 × numLayers × numKVHeads × headDim × contextLen × batchSize × bytesPerKV
 *              适用于 Llama、Qwen2/3、R1-Distill、GLM4、Hunyuan Dense 等标准 GQA 模型
 *
 *   mla:       KV Cache = numLayers × (kvLoraRank + qkRopeHeadDim) × contextLen × batchSize × bytesPerKV
 *              适用于 DeepSeek-V2/V3/R1/V4.1、GLM-5.x、Kimi-K2.6 等 MLA 模型
 *              注意：没有 ×2，因为 K 和 V 被压缩为一个联合隐向量
 *
 *   cla:       KV Cache = 2 × (numLayers / claShareFactor) × numKVHeads × headDim × contextLen × batchSize × bytesPerKV
 *              适用于 Hunyuan-Large 等 CLA（Cross-Layer Attention）模型
 *
 *   linear_hybrid: KV Cache = 2 × fullAttnLayers × numKVHeads × headDim × contextLen × batchSize × bytesPerKV
 *              适用于 MiniMax-M1/M3（Lightning/MSA 混合）、Qwen3.5/3.6（Gated DeltaNet 混合）
 *              仅 full-attention 层产生 KV Cache；线性注意力层使用固定大小状态
 *              特例：MiniMax-M2.5 全 Lightning 层(fullAttnLayers=0)，KV Cache=0
 *
 *   kda_mla:   KV Cache = fullAttnLayers × (kvLoraRank + qkRopeHeadDim) × contextLen × batchSize × bytesPerKV
 *              适用于 Kimi K3（KDA 动态注意力 + Gated MLA 混合）模型
 *
 *   hca_mla:   KV Cache = effectiveKVDim × contextLen × batchSize × bytesPerKV
 *              适用于 DeepSeek V4 Flash/Pro（HCA 混合压缩注意力）模型
 */

export const MODEL_DATA = [
  // ═══════════════════════════════════════════════════════════════════
  // 阿里云 / 通义千问（9 款）
  // ═══════════════════════════════════════════════════════════════════

  // ── Qwen3.5 Dense 系列（Gated DeltaNet + Full Attention 混合）──
  {
    id: 'qwen3.5-0.8b',
    displayName: 'Qwen3.5-0.8B（0.8B）',
    paramsB: 0.8,
    architecture: 'dense',
    attnArch: 'linear_hybrid',
    numLayers: 24,
    hiddenDim: 1024,
    numKVHeads: 2,
    headDim: 256,
    fullAttnLayers: 6,
    available: true,
    note: '6×(3×Gated DeltaNet → 1×Gated Attention), head_dim=256',
  },
  {
    id: 'qwen3.5-2b',
    displayName: 'Qwen3.5-2B（2B）',
    paramsB: 2.0,
    architecture: 'dense',
    attnArch: 'linear_hybrid',
    numLayers: 24,
    hiddenDim: 2048,
    numKVHeads: 2,
    headDim: 256,
    fullAttnLayers: 6,
    available: true,
    note: '6×(3×Gated DeltaNet → 1×Gated Attention), head_dim=256',
  },
  {
    id: 'qwen3.5-4b',
    displayName: 'Qwen3.5-4B（4B）',
    paramsB: 4.0,
    architecture: 'dense',
    attnArch: 'linear_hybrid',
    numLayers: 32,
    hiddenDim: 2560,
    numKVHeads: 4,
    headDim: 256,
    fullAttnLayers: 8,
    available: true,
    note: '8×(3×Gated DeltaNet → 1×Gated Attention), 配置为根据同系列模型推算',
  },
  {
    id: 'qwen3.5-9b',
    displayName: 'Qwen3.5-9B（9B）',
    paramsB: 9.0,
    architecture: 'dense',
    attnArch: 'linear_hybrid',
    numLayers: 32,
    hiddenDim: 4096,
    numKVHeads: 4,
    headDim: 256,
    fullAttnLayers: 8,
    available: true,
    note: '8×(3×Gated DeltaNet → 1×Gated Attention), 16 Q heads / 4 KV heads',
  },

  // ── Qwen3.6 MoE 系列（Gated DeltaNet + MoE + Full Attention 混合）──
  {
    id: 'qwen3.6-27b',
    displayName: 'Qwen3.6-27B（27B）',
    paramsB: 27,
    architecture: 'dense',
    attnArch: 'linear_hybrid',
    numLayers: 64,
    hiddenDim: 5120,
    numKVHeads: 4,
    headDim: 256,
    fullAttnLayers: 16,
    available: true,
    note: '16×(3×Gated DeltaNet → 1×Gated Attention), Dense',
  },
  {
    id: 'qwen3.6-35b-a3b',
    displayName: 'Qwen3.6-35B-A3B（35B-A3B）',
    paramsB: 35,
    architecture: 'moe',
    attnArch: 'linear_hybrid',
    numLayers: 40,
    hiddenDim: 2048,
    numKVHeads: 2,
    headDim: 256,
    fullAttnLayers: 10,
    available: true,
    note: '256 专家，每 token 激活 8 routed + 1 shared；40 层中 10 层全注意力',
  },

  // ── Qwen3.5 大 MoE 系列 ──
  {
    id: 'qwen3.5-122b-a10b',
    displayName: 'Qwen3.5-122B-A10B（122B-A10B）',
    paramsB: 122,
    architecture: 'moe',
    attnArch: 'linear_hybrid',
    numLayers: 48,
    hiddenDim: 3072,
    numKVHeads: 2,
    headDim: 256,
    fullAttnLayers: 12,
    available: true,
    note: '48 层中 12 层全注意力（每 4 层一次），256 专家，激活 8 routed，HF 核实',
  },
  {
    id: 'qwen3.5-397b-a17b',
    displayName: 'Qwen3.5-397B-A17B（397B-A17B）',
    paramsB: 397,
    architecture: 'moe',
    attnArch: 'linear_hybrid',
    numLayers: 60,
    hiddenDim: 4096,
    numKVHeads: 2,
    headDim: 256,
    fullAttnLayers: 15,
    available: true,
    note: '60 层中 15 层全注意力（每 4 层一次），512 专家，激活 10 routed，HF 核实',
  },

  // ── Qwen3 系列（Standard GQA）──
  {
    id: 'qwen3-235b-a22b',
    displayName: 'Qwen3-235B-A22B（235B-A22B）',
    paramsB: 235,
    architecture: 'moe',
    attnArch: 'standard',
    numLayers: 94,
    hiddenDim: 4096,
    numKVHeads: 4,
    headDim: 128,
    available: true,
    note: 'Qwen3 代际，标准 GQA，128 专家 MoE；94 层 64 attention heads / 4 KV heads，HF 核实',
  },

  // ═══════════════════════════════════════════════════════════════════
  // DeepSeek（9 款）
  // ═══════════════════════════════════════════════════════════════════

  // ── R1 Distill 系列（基于 Llama/Qwen 基座，Standard GQA）──
  {
    id: 'deepseek-r1-distill-llama-8b',
    displayName: 'DeepSeek-R1-Distill-Llama-8B（8B）',
    paramsB: 8.03,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 32,
    hiddenDim: 4096,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '基于 Llama-3.1-8B 基座，标准 GQA',
  },
  {
    id: 'deepseek-r1-distill-qwen-14b',
    displayName: 'DeepSeek-R1-Distill-Qwen-14B（14B）',
    paramsB: 14.2,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 48,
    hiddenDim: 5120,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '基于 Qwen2.5-14B 基座，标准 GQA，40 attention heads / 8 KV heads，HF 核实',
  },
  {
    id: 'deepseek-r1-distill-qwen-14b-dense',
    displayName: 'DeepSeek-R1-Distill-Qwen-14B-Dense（14B）',
    paramsB: 14.2,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 48,
    hiddenDim: 5120,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '基于 Qwen2.5-14B 基座（Dense 变体），标准 GQA，HF 核实',
  },
  {
    id: 'deepseek-r1-distill-qwen-32b',
    displayName: 'DeepSeek-R1-Distill-Qwen-32B（32B）',
    paramsB: 32.5,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 64,
    hiddenDim: 5120,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '基于 Qwen2.5-32B 基座，标准 GQA',
  },
  {
    id: 'deepseek-r1-distill-llama-70b',
    displayName: 'DeepSeek-R1-Distill-Llama-70B（70B）',
    paramsB: 70.6,
    architecture: 'dense',
    attnArch: 'standard',
    numLayers: 80,
    hiddenDim: 8192,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '基于 Llama-3.1-70B 基座，标准 GQA',
  },

  // ── V3/V4 系列（MLA / HCA+MLA）──
  {
    id: 'deepseek-v3.2',
    displayName: 'DeepSeek-V3.2（671B-A37B）',
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
    note: 'MLA 架构，KV Cache ≈ 标准公式的 1/24.9；V3.2 为 V3 权重重训版本',
  },
  {
    id: 'deepseek-v4-flash',
    displayName: 'DeepSeek-V4-Flash（284B）',
    paramsB: 284,
    architecture: 'moe',
    attnArch: 'hca_mla',
    numLayers: 43,
    hiddenDim: 4096,
    numKVHeads: 1,
    headDim: 512,
    effectiveKVDim: 4176,
    available: true,
    note: 'HCA+MLA，激活参数约 13B；vLLM 实测 ~8.7× KV Cache 节省 vs 标准',
  },
  {
    id: 'deepseek-v4-pro',
    displayName: 'DeepSeek-V4-Pro（685B）',
    paramsB: 685,
    architecture: 'moe',
    attnArch: 'hca_mla',
    numLayers: 61,
    hiddenDim: 7168,
    numKVHeads: 1,
    headDim: 512,
    effectiveKVDim: 6000,
    available: true,
    note: 'HCA+MLA，384 专家；effectiveKVDim 基于 V4 Flash 压缩比(~10.5×)推算(62464/10.5≈5949→6000)，HF 核实基础参数',
  },
  {
    id: 'deepseek-v4.1',
    displayName: 'DeepSeek-V4.1（671B-A47B）',
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
    note: 'V3.2 后继版本，激活参数提升至 47B，MLA 架构',
  },

  // ═══════════════════════════════════════════════════════════════════
  // 智谱AI / Z.ai（4 款）
  // ═══════════════════════════════════════════════════════════════════

  {
    id: 'glm-4.5',
    displayName: 'GLM-4.5（355B）',
    paramsB: 355,
    architecture: 'moe',
    attnArch: 'standard',
    numLayers: 92,
    hiddenDim: 5120,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '标准 attention，面向 Agent 场景，激活参数约 32B',
  },
  {
    id: 'glm-4.5-air',
    displayName: 'GLM-4.5-Air（305B）',
    paramsB: 305,
    architecture: 'moe',
    attnArch: 'standard',
    numLayers: 46,
    hiddenDim: 4096,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: 'GLM-4.5 的轻量版本，标准 attention，HF 核实',
  },
  {
    id: 'glm-5.1',
    displayName: 'GLM-5.1（744B）',
    paramsB: 744,
    architecture: 'moe',
    attnArch: 'mla',
    numLayers: 78,
    hiddenDim: 6144,
    numKVHeads: 64,
    headDim: 256,
    kvLoraRank: 512,
    qkRopeHeadDim: 64,
    available: true,
    note: 'MLA + DSA 架构，激活参数约 40B，256 专家；q_lora_rank=2048，qk_head_dim=256，HF 核实',
  },
  {
    id: 'glm-5.2',
    displayName: 'GLM-5.2（744B）',
    paramsB: 744,
    architecture: 'moe',
    attnArch: 'mla',
    numLayers: 78,
    hiddenDim: 6144,
    numKVHeads: 64,
    headDim: 256,
    kvLoraRank: 512,
    qkRopeHeadDim: 64,
    available: true,
    note: 'MLA + DSA 架构，256 专家；HF config 无 kv_lora_rank 字段(q_lora_rank=2048)，暂沿用 GLM-5.1 的 kv_lora_rank=512 待技术报告核实',
  },

  // ═══════════════════════════════════════════════════════════════════
  // MiniMax（4 款）
  // ═══════════════════════════════════════════════════════════════════

  {
    id: 'minimax-m1',
    displayName: 'MiniMax-M1（456B-A45.9B）',
    paramsB: 456,
    architecture: 'moe',
    attnArch: 'linear_hybrid',
    numLayers: 80,
    hiddenDim: 6144,
    numKVHeads: 8,
    headDim: 128,
    fullAttnLayers: 10,
    available: true,
    note: '80 层中仅 10 层为全注意力，其余为 Lightning 线性注意力；参数量已按核验结果修正',
  },
  {
    id: 'minimax-m2.5',
    displayName: 'MiniMax-M2.5（230B）',
    paramsB: 230,
    architecture: 'moe',
    attnArch: 'linear_hybrid',
    numLayers: 62,
    hiddenDim: 3072,
    numKVHeads: 8,
    headDim: 128,
    fullAttnLayers: 0,
    available: true,
    note: '256 专家，62 层全部为 Lightning 线性注意力(attn_type=1)，无全注意力层，KV Cache=0；HF 核实',
  },
  {
    id: 'minimax-m2.7',
    displayName: 'MiniMax-M2.7（230B）',
    paramsB: 230,
    architecture: 'moe',
    attnArch: 'linear_hybrid',
    numLayers: 64,
    hiddenDim: 5120,
    numKVHeads: 8,
    headDim: 128,
    fullAttnLayers: 8,
    available: true,
    note: 'M2.5 的改进版本，配置为根据同系列推算',
  },
  {
    id: 'minimax-m3',
    displayName: 'MiniMax-M3（350B）',
    paramsB: 350,
    architecture: 'moe',
    attnArch: 'linear_hybrid',
    numLayers: 60,
    hiddenDim: 6144,
    numKVHeads: 8,
    headDim: 128,
    fullAttnLayers: 8,
    available: true,
    note: 'MiniMax Sparse Attention (MSA)，128 专家，激活 4 routed；暂以 linear_hybrid 近似 KV Cache(8 层全注意力)，HF 核实基础参数',
  },

  // ═══════════════════════════════════════════════════════════════════
  // 月之暗面（2 款）
  // ═══════════════════════════════════════════════════════════════════

  {
    id: 'kimi-k2.6',
    displayName: 'Kimi-K2.6（1000B）',
    paramsB: 1000,
    architecture: 'moe',
    attnArch: 'mla',
    numLayers: 61,
    hiddenDim: 7168,
    numKVHeads: 64,
    headDim: 128,
    kvLoraRank: 512,
    qkRopeHeadDim: 64,
    available: true,
    note: '使用 configuration_deepseek.py，MLA 架构，384 专家，激活 8 routed；v_head_dim=128，HF 核实',
  },
  {
    id: 'kimi-k3',
    displayName: 'Kimi-K3（2800B-A104B）',
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
    note: '93 层中 24 层为 Gated MLA，69 层为 KDA 线性注意力；激活参数已核验为 104B',
  },

  // ═══════════════════════════════════════════════════════════════════
  // 腾讯（2 款）
  // ═══════════════════════════════════════════════════════════════════

  {
    id: 'hunyuan-a13b',
    displayName: 'Hunyuan-A13B（80B-A13B）',
    paramsB: 80,
    architecture: 'moe',
    attnArch: 'standard',
    numLayers: 32,
    hiddenDim: 4096,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '64 专家 MoE，激活参数约 13B，标准 GQA attention',
  },
  {
    id: 'hy3',
    displayName: 'Hunyuan-Hy3（295B）',
    paramsB: 295,
    architecture: 'moe',
    attnArch: 'standard',
    numLayers: 80,
    hiddenDim: 4096,
    numKVHeads: 8,
    headDim: 128,
    available: true,
    note: '192 专家 MoE，激活参数约 21B，标准 GQA attention，Apache 2.0',
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
 * 对于 standard / cla / linear_hybrid 模型：
 *   kvDim = numKVHeads × headDim
 *   GQA（Grouped Query Attention）下，kvDim < hiddenDim
 *
 * 对于 MLA / kda_mla 模型：
 *   请使用 getMLADim() 获取 MLA 特有的隐向量维度
 *
 * 对于 hca_mla 模型：
 *   请直接使用 model.effectiveKVDim
 */
export function getKVDim(model) {
  return model.numKVHeads * model.headDim;
}

/**
 * MLA（Multi-head Latent Attention）的 KV 联合隐向量维度
 * = kvLoraRank + qkRopeHeadDim
 *
 * 仅在 attnArch === 'mla' 或 'kda_mla' 时有效
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