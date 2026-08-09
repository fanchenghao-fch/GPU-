import { useCalculator } from '../hooks/useCalculator';
import Selector from './Selector';
import ResultCard from './ResultCard';
import MemoryBreakdown from './MemoryBreakdown';
import FormulaSteps from './FormulaSteps';

export default function Calculator() {
  const {
    options,
    modelId, setModelId,
    gpuId, setGpuId,
    presetId, setPresetId,
    precision, setPrecision,
    result,
  } = useCalculator();

  return (
    <div className="
      w-full max-w-lg
      backdrop-blur-[40px] bg-white/40
      border border-white/30
      rounded-3xl shadow-xl
      p-6 sm:p-8
      space-y-6
    ">
      <h1 className="text-2xl font-semibold tracking-tight text-slate-800 text-center">
        GPU 卡数计算器
      </h1>

      <div className="space-y-4">
        <Selector
          label="模型"
          value={modelId}
          onChange={setModelId}
          items={options.models}
        />
        <Selector
          label="GPU"
          value={gpuId}
          onChange={setGpuId}
          items={options.gpus}
        />
        <Selector
          label="SLA 预设"
          value={presetId}
          onChange={setPresetId}
          items={options.presets}
          subtitleKey="description"
        />
        <Selector
          label="精度"
          value={precision}
          onChange={setPrecision}
          items={options.precisions.map((p) => ({ id: p, displayName: p }))}
        />
      </div>

      {result && !result.error && (
        <>
          <ResultCard result={result} />
          <MemoryBreakdown memory={result.modelMemoryGB} />
          <FormulaSteps formula={result.formula} />
        </>
      )}

      {result?.error && (
        <div className="text-red-500 text-sm text-center">
          计算错误：{result.error}
        </div>
      )}
    </div>
  );
}
