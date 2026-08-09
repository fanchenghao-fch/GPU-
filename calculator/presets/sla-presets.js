/**
 * SLA 预设 —— 推理/训练场景的默认参数
 *
 * 这些预设档位由产品部结合业内共识定义。
 * 销售只需选择档位，无需手动调上下文长度、并发数等参数。
 *
 * 每个预设包含：
 * - id:            内部标识
 * - displayName:   前端展示名称
 * - description:   对销售的简要说明（他们会看到）
 * - scenario:      'inference' | 'training'
 * - contextLen:    上下文长度（tokens）
 * - batchSize:     并发数（同时处理的请求数）
 * - overheadRatio: 其他显存比例（相对于 权重+KV Cache）
 */

export const SLA_PRESETS = [
  {
    id: 'inference-light',
    displayName: '轻量推理',
    description: '个人使用 / 低并发，上下文 4K',
    scenario: 'inference',
    contextLen: 4096,
    batchSize: 1,
    overheadRatio: 0.05,
  },
  {
    id: 'inference-standard',
    displayName: '标准推理',
    description: '企业应用 / 中等负载，上下文 8K',
    scenario: 'inference',
    contextLen: 8192,
    batchSize: 1,
    overheadRatio: 0.10,
  },
  {
    id: 'inference-batch',
    displayName: '批量推理',
    description: '离线评估 / API 高并发，上下文 8K，并发 8',
    scenario: 'inference',
    contextLen: 8192,
    batchSize: 8,
    overheadRatio: 0.12,
  },
  {
    id: 'inference-long',
    displayName: '长上下文推理',
    description: 'RAG / 文档分析 / 法律合同，上下文 128K',
    scenario: 'inference',
    contextLen: 131072,
    batchSize: 1,
    overheadRatio: 0.12,
  },
  {
    id: 'inference-xlong',
    displayName: '超长上下文推理',
    description: '全量代码库 / 整书分析 / 海量对话，上下文 256K',
    scenario: 'inference',
    contextLen: 262144,
    batchSize: 1,
    overheadRatio: 0.12,
  },
  {
    id: 'training-lora',
    displayName: 'LoRA 微调',
    description: 'LoRA/QLoRA 参数高效微调，反传梯度显存较小',
    scenario: 'training',
    contextLen: 4096,
    batchSize: 1,
    overheadRatio: 0.20,
    note: '⚠️ 训练显存估算较粗略，仅包含权重+梯度+优化器状态（不含完整激活值反传），建议实测验证',
  },
  {
    id: 'training-full',
    displayName: '全量微调',
    description: '全参数 SFT 微调，显存需求显著大于推理',
    scenario: 'training',
    contextLen: 4096,
    batchSize: 1,
    overheadRatio: 0.30,
    note: '⚠️ 全量训练显存可能达到推理的 4-6 倍，此估算偏保守，建议实测',
  },
];

/**
 * 根据 ID 查找 SLA 预设
 */
export function getPresetById(id) {
  const preset = SLA_PRESETS.find((p) => p.id === id);
  if (!preset) {
    throw new Error(`未知 SLA 预设: "${id}"`);
  }
  return preset;
}

/**
 * 获取推理类预设
 */
export function getInferencePresets() {
  return SLA_PRESETS.filter((p) => p.scenario === 'inference');
}

/**
 * 获取训练类预设
 */
export function getTrainingPresets() {
  return SLA_PRESETS.filter((p) => p.scenario === 'training');
}
