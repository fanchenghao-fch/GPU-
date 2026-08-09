export default function ResultCard({ result }) {
  const { cards, servers, cardsPerServer, utilization } = result;

  return (
    <div className="
      bg-white/25 backdrop-blur
      border border-white/30
      rounded-2xl p-5 space-y-4
    ">
      <div className="text-center">
        <p className="text-6xl font-bold text-slate-800 tabular-nums tracking-tight">
          {cards}
        </p>
        <p className="text-sm text-slate-500 mt-1">张 GPU 卡</p>
      </div>

      <div className="flex justify-between items-center pt-2 border-t border-white/30">
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider">服务器</p>
          <p className="text-lg font-semibold tabular-nums text-slate-700">
            {servers} <span className="text-xs font-normal text-slate-400">台</span>
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-slate-500 uppercase tracking-wider">每台卡数</p>
          <p className="text-lg font-semibold tabular-nums text-slate-700">
            {cardsPerServer}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-500 uppercase tracking-wider">显存利用率</p>
          <p className="text-lg font-semibold tabular-nums text-slate-700">
            {(utilization * 100).toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  );
}
