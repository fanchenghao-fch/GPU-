import { useState, useMemo } from 'react';
import { calculate, getSelectOptions } from '@calculator/index.js';

export function useCalculator() {
  const options = useMemo(() => getSelectOptions(), []);

  const [modelId, setModelId] = useState(options.models[0]?.id ?? '');
  const [gpuId, setGpuId] = useState(options.gpus[0]?.id ?? '');
  const [presetId, setPresetId] = useState(options.presets[0]?.id ?? '');
  const [precision, setPrecision] = useState(options.precisions[0] ?? '');

  const result = useMemo(() => {
    if (!modelId || !gpuId || !presetId || !precision) return null;
    try {
      return calculate(modelId, gpuId, presetId, precision);
    } catch (err) {
      return { error: err.message };
    }
  }, [modelId, gpuId, presetId, precision]);

  return {
    options,
    modelId, setModelId,
    gpuId, setGpuId,
    presetId, setPresetId,
    precision, setPrecision,
    result,
  };
}
