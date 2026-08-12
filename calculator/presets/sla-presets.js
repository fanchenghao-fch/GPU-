/**
 * SLA 预设 —— 推理/训练场景的默认参数
 *
 * 推理预设依据《本地部署大模型服务等级（SLA）》定义。
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
    id: 'sla-1',
    displayName: 'SLA-1 实时交互型',
    description: '产品问答/销售话术/简单查询，8K 上下文 × 20 并发',
    scenario: 'inference',
    contextLen: 8192,
    batchSize: 20,
    overheadRatio: 0.15,
  },
  {
    id: 'sla-2',
    displayName: 'SLA-2 标准业务型',
    description: '多轮对话/客户画像/CRM 分析，16K 上下文 × 20 并发',
    scenario: 'inference',
    contextLen: 16384,
    batchSize: 20,
    overheadRatio: 0.15,
  },
  {
    id: 'sla-3',
    displayName: 'SLA-3 深度业务型',
    description: '客户全景分析/复杂方案/多资料问答，32K 上下文 × 20 并发',
    scenario: 'inference',
    contextLen: 32768,
    batchSize: 20,
    overheadRatio: 0.15,
  },
  {
    id: 'sla-4',
    displayName: 'SLA-4 长文档分析型',
    description: '会议纪要/合同/方案/标书分析，64K 上下文 × 20 并发',
    scenario: 'inference',
    contextLen: 65536,
    batchSize: 20,
    overheadRatio: 0.15,
  },
  {
    id: 'sla-5',
    displayName: 'SLA-5 超长文档分析型',
    description: '整份标书/合同集/长录音转写/多文档联合，128K 上下文 × 20 并发',
    scenario: 'inference',
    contextLen: 131072,
    batchSize: 20,
    overheadRatio: 0.15,
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
