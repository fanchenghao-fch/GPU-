# GPU 卡数计算器 — UI 层设计方案

> 状态：待人工确认  
> 公式模块：已完成（104 tests passing），不动

---

## 一、技术栈

| 层 | 选型 | 理由 |
|----|------|------|
| 构建 | Vite 5 | ESM 原生、Vercel 零配置、启动快 |
| 框架 | React 18 | 稳定、生态完善 |
| 样式 | Tailwind CSS v3 | 成熟的 `backdrop-blur` 支持、Utility-first |
| 下拉 | 原生 `<select>` | 4 个下拉开 Headless UI 过度设计 |
| 状态 | `useState` + `useMemo` | 单页工具无需 Redux/Zustand |
| 字体 | 系统字体栈 | macOS → SF Pro / Windows → Segoe UI |
| 部署 | Vercel | 预览环境，已配置 `vercel.json` |

---

## 二、文件结构

```
GPU卡数计算器/
├── calculator/              # 不变
├── package.json             # 添加 dev/build 脚本 + 依赖
├── index.html               # Vite 入口
├── vite.config.js           # @calculator 路径别名
├── tailwind.config.js       # 毛玻璃色板
├── postcss.config.js
├── vercel.json              # Vercel 部署
├── .gitignore               # 加 dist/
├── ui-design.md             # 本文档
└── src/
    ├── main.jsx
    ├── index.css            # Tailwind directives
    ├── App.jsx              # 居中布局
    ├── hooks/
    │   └── useCalculator.js # 唯一状态管理
    └── components/
        ├── Calculator.jsx        # 主毛玻璃卡片
        ├── Selector.jsx          # 定制 <select>
        ├── ResultCard.jsx        # 卡数/服务器/利用率
        ├── MemoryBreakdown.jsx   # 权重/KV/其他 柱状图
        └── FormulaSteps.jsx      # 计算步骤
```

**共新增 13 个文件 + 修改 2 个文件（package.json, .gitignore）**

---

## 三、数据流

```
             getSelectOptions()
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      模型列表   GPU列表   预设/精度列表
          │         │         │
          └────┬────┴────┬────┘
               ▼         ▼            用户选择
           useCalculator hook
               │
               ▼
    calculate(modelId, gpuId, presetId, precision)
               │                     纯客户端同步，无 API
               ▼
    { cards, servers, utilization, modelMemoryGB, formula }
               │
     ┌─────────┼──────────┐
     ▼         ▼           ▼
  ResultCard  Memory    FormulaSteps
             Breakdown
```

---

## 四、毛玻璃样式体系

### 4.1 背景层
```css
body {
  background: linear-gradient(135deg, #f1f5f9, #f0f9ff, #eef2ff);
}
/* Tailwind: bg-gradient-to-br from-slate-100 via-sky-50 to-indigo-100 */
```

### 4.2 玻璃面板（三类）

| 层级 | 元素 | 背景 | 模糊 | 圆角 |
|------|------|------|------|------|
| 主卡片 | Calculator | `bg-white/40` | `backdrop-blur-[40px]` | `rounded-3xl` |
| 结果面板 | ResultCard, MemoryBreakdown, FormulaSteps | `bg-white/25` | `backdrop-blur` | `rounded-2xl` |
| 下拉控件 | `<select>` | `bg-white/25` | `backdrop-blur` | `rounded-xl` |

所有面板统一 `border border-white/30 shadow-xl`

### 4.3 效果原理

渐变背景（青→蓝→靛）透过半透明面板产生柔和色调——不是纯灰玻璃，而是有颜色倾向的毛玻璃。这是 Apple 近年 HIG 的核心手法。

---

## 五、组件设计

### 5.1 App.jsx

```jsx
<div className="min-h-screen flex items-start justify-center px-4 py-8 sm:py-16">
  <Calculator />
</div>
```

最外层布局，垂直居中（顶部对齐 + `items-start` 防止键盘弹出时挤压），移动端 `px-4`。

### 5.2 Calculator.jsx — 主卡片

```
┌──────────────────────────────────────┐
│         GPU 卡数计算器                │  ← h1, text-2xl, font-semibold
│                                      │
│  模型    [ Llama 3.1 70B      ▼ ]   │  ← Selector ×4
│  GPU     [ N300               ▼ ]   │
│  SLA预设 [ 标准推理           ▼ ]   │
│  精度    [ FP16               ▼ ]   │
│                                      │
│  ┌────────────────────────────────┐  │
│  │            4                   │  │  ← ResultCard
│  │         张 GPU 卡              │  │     大号数字 text-6xl
│  │  服务器: 1 │ 每台: 16 │ 94.6% │  │
│  └────────────────────────────────┘  │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ 模型显存明细                    │  │  ← MemoryBreakdown
│  │ 权重    ████████████████ 132GB │  │     三项横向柱状图
│  │ KV Cache ██               3GB  │  │
│  │ 其他    ████             13GB  │  │
│  │ 合计                   147GB   │  │
│  └────────────────────────────────┘  │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ 计算步骤                        │  │  ← FormulaSteps
│  │ 1. 模型显存=权重+KV+其他=147GB │  │     编号列表
│  │ 2. 单卡可用=48×0.97×0.9=42GB  │  │
│  │ 3. 卡数=ceil(147/42)=4        │  │
│  │ 4. 服务器=ceil(4/16)=1        │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

宽度 `max-w-lg` (512px)，高度自适应。

### 5.3 Selector.jsx

```jsx
// Props: label, value, onChange, items[], idKey, displayKey, subtitleKey?
```

- `<label>` — 小号大写跟踪字距（`text-xs uppercase tracking-wider`）
- `<select>` — 去掉原生箭头（`appearance-none`），用内联 SVG 画自定义箭头
- 可选副标题 — 选中 SLA 预设后在下方显示灰色描述文字
- 聚焦环 — `focus:ring-2 focus:ring-sky-400/50`

### 5.4 ResultCard.jsx

三个信息层级：
1. **卡数** — 60px 粗体（`text-6xl font-bold tabular-nums`），居中
2. **分隔线** — `border-t border-white/30`
3. **三列** — 服务器数 | 每台卡数 | 显存利用率（`flex justify-between`）

每列：小号标签（`text-xs text-slate-500`）+ 数值（`text-lg font-semibold`）

### 5.5 MemoryBreakdown.jsx

三项横向柱状图：

```
权重     ████████████████████████  131.5 GB   (sky-400)
KV Cache ██                         2.5 GB   (indigo-400)
其他     ████                      13.8 GB   (violet-300)
────────────────────────────────────────────
合计                               147.8 GB
```

- 柱宽 = `itemGB / max(三项) × 100%`（相对最大项，不是总和）
- 过渡动画 `transition-all duration-500`
- `font-mono tabular-nums` 数值等宽

### 5.6 FormulaSteps.jsx

将 `formula`（`\n` 分隔的字符串）渲染为 `<ol>` 编号列表。
纯展示组件，无交互。

---

## 六、useCalculator hook

```js
function useCalculator() {
  const options = useMemo(() => getSelectOptions(), []);

  const [modelId, setModelId]   = useState(options.models[0]?.id);
  const [gpuId, setGpuId]       = useState(options.gpus[0]?.id);
  const [presetId, setPresetId] = useState(options.presets[0]?.id);
  const [precision, setPrecision] = useState(options.precisions[0]);

  const result = useMemo(() => {
    if (!modelId || !gpuId || !presetId || !precision) return null;
    try { return calculate(modelId, gpuId, presetId, precision); }
    catch (err) { return { error: err.message }; }
  }, [modelId, gpuId, presetId, precision]);

  return { options, modelId, setModelId, gpuId, setGpuId,
           presetId, setPresetId, precision, setPrecision, result };
}
```

- 默认选中第一项 → 进来就有结果，无需"请选择"空白态
- `try/catch` → 异常不崩溃，展示为 error 文本
- `useMemo` 依赖精确 → 仅四值变化时重算

---

## 七、响应式策略

| 视口 | 卡片宽度 | 间距 | 字号 |
|------|----------|------|------|
| < 640px | `w-full` | `p-6` `space-y-4` | 卡数 `text-5xl` |
| ≥ 640px | `max-w-lg` | `p-8` `space-y-6` | 卡数 `text-6xl` |

移动端关键适配：
- `items-start` 而非 `items-center` — 避免键盘弹出时遮挡
- 下拉 `py-2.5` → 44px+ 触控区域
- 柱状图标签不换行（`whitespace-nowrap`）

---

## 八、依赖清单

```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.1",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.41",
    "tailwindcss": "^3.4.10",
    "vite": "^5.4.0"
  }
}
```

零运行时依赖，仅 React + Vite 工具链。

---

## 九、验证清单

| # | 验证项 | 方法 |
|---|--------|------|
| 1 | 公式模块不受影响 | `npm test`（104 tests） |
| 2 | 开发服务器启动 | `npm run dev` |
| 3 | 4 个下拉正确填充 | 检查模型 24 个、GPU 2 个、预设 7 个、精度 7 个 |
| 4 | 下拉切换实时更新结果 | 切换选项，观察卡数/明细变化 |
| 5 | 生产构建通过 | `npm run build` |
| 6 | 移动端响应式 | 375px 视口宽度检查 |
| 7 | 品牌名不出现 | 搜索页面源码无"沐曦"/"MetaX" |
