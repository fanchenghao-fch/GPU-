export default function Selector({
  label,
  value,
  onChange,
  items,
  idKey = 'id',
  displayKey = 'displayName',
  subtitleKey = null,
}) {
  const selected = items.find((i) => i[idKey] === value);

  return (
    <div className="space-y-1.5">
      <label className="block text-xs font-medium uppercase tracking-wider text-slate-500">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="
          w-full px-4 py-2.5
          bg-white/25 backdrop-blur
          border border-white/30
          rounded-xl
          text-sm text-slate-800
          appearance-none
          bg-[url('data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2212%22 height=%2212%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke=%22%23475569%22 stroke-width=%222%22><path d=%22M6 9l6 6 6-6%22/></svg>')]
          bg-[length:12px]
          bg-[right_12px_center]
          bg-no-repeat
          focus:outline-none focus:ring-2 focus:ring-sky-400/50
          transition-all duration-200
        "
      >
        {items.map((item) => (
          <option key={item[idKey]} value={item[idKey]}>
            {item[displayKey]}
            {subtitleKey && item[subtitleKey] ? ` — ${item[subtitleKey]}` : ''}
          </option>
        ))}
      </select>
      {subtitleKey && selected?.[subtitleKey] && (
        <p className="text-xs text-slate-400 pl-1">{selected[subtitleKey]}</p>
      )}
    </div>
  );
}
