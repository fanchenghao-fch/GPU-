# CLAUDE.md

本文件为 Claude 在此仓库工作时的项目指引。

## 项目定位

GPU 卡数计算器：纯前端 SPA，无后端/登录/数据库/分享。面向销售人员快速估算「跑某个 AI 模型需要多少张 GPU 卡」。

核心公式：`卡数 = ceil(模型所需显存 / 单卡可用显存)`

## 技术栈

React 18 · Vite 5 · Tailwind CSS 3 · Node.js 原生 test runner（ESM）。UI 风格为 Apple 毛玻璃（frosted glass）。

## 目录与关键文件

```
calculator/  计算引擎（纯 JS，零运行时依赖）
  index.js   主入口：calculate() / getSelectOptions()
  constants/model-data.js   29 款模型（6 种注意力架构）
  constants/gpu-data.js     2 款 GPU（N300 / C600）
  constants/precision.js    7 种精度（FP32→INT4）
  presets/sla-presets.js    SLA 预设（5 推理 + 2 训练）
src/         UI 层（React + Tailwind）
  App.jsx    根组件：居中布局 + 备案号页脚
  components/ hooks/useCalculator.js
scripts/     部署脚本
  deploy-cos.mjs   构建并上传到腾讯云 COS
  cos.env          COS 密钥（已被 .gitignore 忽略，勿提交）
markdown/    文档
  VERSION.md  版本说明（架构/公式/文件清单/部署/历史，详细资料以它为准）
```

## 常用命令

- `npm test` —— 跑 103 项自动化测试
- `npm run dev` —— 本地开发服务器
- `npm run build` —— 构建到 `dist/`
- `npm run deploy:cos` —— 构建并部署到腾讯云 COS（可选 `--delete` 清理远端旧文件）

## 重要约束

1. **UI 中不得出现品牌名「沐曦 / MetaX」**。
2. `scripts/cos.env` 含真实密钥，已被 `.gitignore` 忽略，**切勿提交或外泄**。
3. 模型数据改动需同步：`model-data.js` 的模型总数、`index.test.js` 中 `models.length` 断言、`VERSION.md` 的模型库/架构计数。

## 部署

- 国内：https://gpu-calculator.online （腾讯云 COS 静态托管，自定义域名，已完成 ICP + 公安备案）
- 国际：https://gpu-card-calculator.vercel.app （Vercel）
- 备案号已悬挂在 `src/App.jsx` 页脚：ICP `沪ICP备2026042468号-1` + 公安 `沪公网安备31012102000212号`
