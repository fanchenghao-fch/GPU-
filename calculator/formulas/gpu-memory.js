/**
 * GPU 显存计算公式
 *
 * 单卡可用显存计算链路：
 *
 *   标称显存 (memoryGB)
 *     │  ── 规格书标注的显存，如 32GB / 64GB / 80GB
 *     │
 *     ├── 减去：驱动/Firmware 占用
 *     │     └── ECC 预留、GPU 固件、显存位宽损耗 (约 3%)
 *     │
 *     ▼
 *   驱动可见显存 (driverVisibleGB)
 *     │  ── Host 端能看到的显存量（nvidia-smi 显示的值）
 *     │
 *     ├── 减去：CUDA Context / 驱动运行时开销
 *     │     └── kernel launch、cuDNN/cuBLAS 内部缓存 (约 10-15%)
 *     │
 *     ▼
 *   模型可用显存 (usableGB)
 *     ── 模型权重、KV Cache、激活值等可实际使用的显存
 *
 * ⚠️ 与 model-memory.js 中"其他显存占用"的关系：
 *   - GPU 侧这 10-15% → CUDA context / 驱动运行时固定开销（不随模型变化）
 *   - 模型侧"其他显存" → 激活值/临时缓冲区（随模型和 batch 变大）
 *   两者是独立的两块开销，各管各的，不存在重复计算。
 */

/**
 * 计算驱动可见显存（GB）
 *
 * 标称显存 × 驱动可见率
 * 驱动可见率典型值 = 0.97（即 3% 被固件/ECC 等占用）
 *
 * @param {number} nominalGB      - 标称显存（GB）
 * @param {number} driverEfficiency - 驱动可见率 (0–1)，默认 0.97
 * @returns {number} driverVisibleGB
 *
 * 例：80GB 标称 × 0.97 = 77.6GB 驱动可见
 */
export function calcDriverVisible(nominalGB, driverEfficiency = 0.97) {
  return nominalGB * driverEfficiency;
}

/**
 * 计算单卡模型可用显存（GB）
 *
 * 驱动可见显存 × 推理可用率
 * 推理可用率典型值 = 0.85–0.90（NVIDIA 推荐）
 *
 * @param {number} driverVisibleGB - 驱动可见显存（GB）
 * @param {number} inferenceRatio   - 推理可用率 (0–1)，默认 0.85
 * @returns {number} 模型实际可用的显存（GB）
 *
 * 例：77.6GB × 0.85 = 65.96GB 模型可用
 */
export function calcUsableMemory(driverVisibleGB, inferenceRatio = 0.85) {
  return driverVisibleGB * inferenceRatio;
}

/**
 * 从标称显存一步计算单卡可用显存（GB）
 *
 * usable = nominal × driverEfficiency × inferenceRatio
 *       = nominal × memoryUtilizationRate
 *
 * 其中 memoryUtilizationRate = driverEfficiency × inferenceRatio
 *   典型值 = 0.97 × 0.85 ≈ 0.825
 *
 * @param {number} nominalGB       - 标称显存
 * @param {number} driverEfficiency - 驱动可见率，默认 0.97
 * @param {number} inferenceRatio   - 推理可用率，默认 0.85
 * @returns {number} 单卡可用显存
 */
export function calcGPUUsableMemory(nominalGB, driverEfficiency = 0.97, inferenceRatio = 0.85) {
  const driverVisible = calcDriverVisible(nominalGB, driverEfficiency);
  const usable = calcUsableMemory(driverVisible, inferenceRatio);
  return round2(usable);
}

function round2(v) {
  return Math.round(v * 100) / 100;
}
