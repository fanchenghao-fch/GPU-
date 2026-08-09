/**
 * GPU 常量数据
 *
 * ⚠️ 维护说明（产品部同事）：
 * 新卡上市时，在 GPU_DATA 数组中新增一条记录即可。
 * 字段说明见下方 JSDoc。
 *
 * 字段说明：
 * - id:            内部标识（不会显示在 UI 上）
 * - displayName:   前端展示的 GPU 名称
 * - memoryGB:      标称显存（GB）—— 产品规格书上的数值
 * - cardsPerServer:一台标准服务器最多插几张该型号 GPU
 * - driverEfficiency: 驱动可见率（标称显存 → 驱动可见显存），典型值 0.97
 * - inferenceRatio:   推理可用率（驱动可见显存 → 模型可用的显存），NVIDIA 推荐 0.85–0.90
 * - available:     是否在当前产品线中（下架产品设为 false，前端不展示但历史计算结果仍可追溯）
 */

export const GPU_DATA = [
  {
    id: 'n300',
    displayName: 'N300',
    memoryGB: 48,
    cardsPerServer: 16,
    driverEfficiency: 0.97,
    inferenceRatio: 0.90,
    available: true,
  },
  {
    id: 'c600',
    displayName: 'C600',
    memoryGB: 144,
    cardsPerServer: 8,
    driverEfficiency: 0.97,
    inferenceRatio: 0.90,
    available: true,
  },
];

/**
 * 根据 ID 查找 GPU 配置
 * @param {string} id - GPU ID（如 'mxn100'）
 * @returns {object} GPU 配置对象
 */
export function getGPUById(id) {
  const gpu = GPU_DATA.find((g) => g.id === id);
  if (!gpu) {
    throw new Error(`未知 GPU 型号: "${id}"，可选: ${GPU_DATA.map((g) => g.id).join(', ')}`);
  }
  return gpu;
}

/**
 * 获取所有可用的 GPU 型号（供前端下拉列表使用）
 */
export function getAvailableGPUs() {
  return GPU_DATA.filter((g) => g.available);
}
