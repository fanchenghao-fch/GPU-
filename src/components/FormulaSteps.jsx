export default function FormulaSteps({ formula }) {
  const steps = formula.split('\n').filter(Boolean);

  return (
    <div className="
      bg-white/25 backdrop-blur
      border border-white/30
      rounded-2xl p-5 space-y-2
    ">
      <h3 className="text-sm font-semibold text-slate-700 mb-2">计算步骤</h3>
      <ol className="space-y-1.5">
        {steps.map((step, i) => (
          <li key={i} className="flex gap-2 text-xs text-slate-600">
            <span className="font-mono text-slate-400 flex-shrink-0">
              {i + 1}.
            </span>
            <span className="break-words">{step}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
