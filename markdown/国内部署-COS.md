# 国内部署 — 腾讯云 COS 静态网站托管

> 目的：GPU 卡数计算器在国内可访问（**无需备案、无需服务器**，成本 <1 元/月）
> 定位：Vercel 国际部署的国内镜像方案

---

## 一、方案说明

- 纯静态 SPA，构建产物为 `dist/`
- 上传到腾讯云 COS 对象存储，开启静态网站托管
- 使用 COS 默认域名访问（**免备案**）

---

## 二、一次性准备（腾讯云控制台，约 5 分钟）

### 1. 创建存储桶（Bucket）

控制台：https://console.cloud.tencent.com/cos/bucket

| 配置项 | 选择 |
|--------|------|
| 名称 | `gpu-card-calculator`（会自动拼接 APPID，最终形如 `gpu-card-calculator-1234567890`） |
| 地域 | 上海 `ap-shanghai` 或 广州 `ap-guangzhou` |
| 访问权限 | **公有读私有写** |
| 存储类型 | 标准存储 |
| 多 AZ | 不开启 |

### 2. 获取 API 密钥

控制台：https://console.cloud.tencent.com/cam/capi

- 复制 `SecretId` 和 `SecretKey`
- ⚠️ 密钥相当于账号密码，**切勿提交到 git 或发给他人**

### 3. 开启静态网站托管

存储桶 → 基础配置 → 静态网站：

- 开启
- 索引文档（默认首页）：`index.html`

### 4. 确认访问权限

若访问返回 403，检查「权限管理 → 存储桶访问权限」是否为「公有读私有写」。

---

## 三、本地配置

```bash
cp scripts/cos.env.example scripts/cos.env
```

编辑 `scripts/cos.env`，填入：

```
COS_SECRET_ID=你的SecretId
COS_SECRET_KEY=你的SecretKey
COS_BUCKET=gpu-card-calculator-1234567890
COS_REGION=ap-shanghai
```

---

## 四、部署

```bash
npm install          # 首次：安装 cos-nodejs-sdk-v5
npm run deploy:cos   # 构建 + 上传
```

清理远端多余文件（推荐，避免历史 hash 文件堆积）：

```bash
npm run deploy:cos -- --delete
```

---

## 五、访问地址

部署完成后脚本会打印两个地址：

| 类型 | 地址 | 说明 |
|------|------|------|
| 静态网站（推荐） | `https://<bucket>.cos-website.<region>.myqcloud.com/` | 自动加载 index.html |
| 对象访问（备用） | `https://<bucket>.cos.<region>.myqcloud.com/index.html` | 支持 HTTPS |

> 若 `cos-website` 域名 HTTPS 报错，用对象访问域名即可（默认支持 HTTPS）。

---

## 六、安全提示

- `scripts/cos.env` 已被 `.gitignore` 忽略，**切勿提交**
- SecretKey 泄露会被人盗用 COS，发现泄露立即到 [CAM 控制台](https://console.cloud.tencent.com/cam/capi) 禁用并重置

---

## 七、后续升级（可选）

当前用 COS 默认域名免备案。日后若要绑定自定义域名，需先完成 ICP 备案（见《国内部署-COS.md》备案相关说明），再在 COS/CDN 绑定域名。
