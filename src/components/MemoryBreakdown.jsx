export default function MemoryBreakdown({ memory }) {
  const { weightGB, kvCacheGB, otherGB, totalGB } = memory;

  const bars = [
    { label: '权重', value: weightGB, color: 'bg-sky-400' },
    { label: 'KV Cache', value: kvCacheGB, color: 'bg-indigo-400' },
    { label: '其他', value: otherGB, color: 'bg-violet-300' },
  ];
  const maxVal = Math.max(weightGB, kvCacheGB, otherGB, 1);

  return (
    <div className="
      bg-white/25 backdrop-blur
      border border-white/30
      rounded-2xl p-5 space-y-3
    ">
      <h3 className="text-sm font-semibold text-slate-700">模型显存明细</h3>

      {bars.map(({ label, value, color }) => (
        <div key={label} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span className="text-slate-500">{label}</span>
            <span className="font-mono tabular-nums text-slate-700">
              {value.toFixed(1)} GB
            </span>
          </div>
          <div className="h-2 bg-white/50 rounded-full overflow-hidden">
            <div
              className={`h-full ${color} rounded-full transition-all duration-500`}
              style={{ width: `${(value / maxVal) * 100}%` }}
            />
          </div>
        </div>
      ))}

      <div className="flex justify-between pt-2 border-t border-white/30">
        <span className="text-sm font-semibold text-slate-700">合计</span>
        <span className="font-mono font-semibold tabular-nums text-slate-800">
          {totalGB.toFixed(1)} GB
        </span>
      </div>
    </div>
  );
}
