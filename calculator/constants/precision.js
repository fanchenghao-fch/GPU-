/**
 * 精度常量 —— 每种精度对应的每参数字节数
 *
 * 说明：
 * - 模型推理/训练时，参数以特定精度存储，占用的显存量 = 参数量 × bytesPerParam
 * - 这是"模型权重显存"计算的基础
 */

export const BYTES_PER_PARAM = {
  FP32: 4,    // 32位浮点 — 训练常用
  TF32: 2,    // TensorFloat-32 — NVIDIA 特有，部分训练场景
  BF16: 2,    // Brain Floating Point 16 — 训练常用（大部分新卡）
  FP16: 2,    // 16位浮点 — 推理常用
  FP8: 1,     // 8位浮点（E4M3/E5M2）— H100/H200/B200 推理
  INT8: 1,    // 8位整数量化 — 推理常用
  INT4: 0.5,  // 4位整数量化 — 极端压缩推理
};

/**
 * 精度 → 推荐场景
 */
export const PRECISION_USAGE = {
  FP32: { scenario: 'training', note: '传统训练精度，显存占用大，已逐渐被 BF16 替代' },
  TF32: { scenario: 'training', note: 'NVIDIA Ampere+ 专有，训练速度与 FP16 相当、精度接近 FP32' },
  BF16: { scenario: 'training', note: '现代大模型训练主流精度，动态范围与 FP32 相同' },
  FP16: { scenario: 'inference', note: '推理最常用精度，显存与计算效率均衡' },
  FP8: { scenario: 'inference', note: 'H100/H200/B200 支持，推理吞吐翻倍，需硬件支持' },
  INT8: { scenario: 'inference', note: 'INT8 量化推理，显存减半，精度略有损失' },
  INT4: { scenario: 'inference', note: 'INT4 量化推理，显存为 FP16 的 1/4，精度损失较明显' },
};

/**
 * 获取指定精度的每参数字节数
 * @param {string} precision - 精度名称（如 'FP16'）
 * @returns {number} bytes per param
 */
export function bytesPerParam(precision) {
  const val = BYTES_PER_PARAM[precision.toUpperCase()];
  if (val == null) {
    throw new Error(`未知精度: "${precision}"，可选: ${Object.keys(BYTES_PER_PARAM).join(', ')}`);
  }
  return val;
}
