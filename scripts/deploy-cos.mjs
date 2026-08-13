#!/usr/bin/env node
/**
 * 部署到腾讯云 COS（静态网站托管）
 *
 * 用法：
 *   1. cp scripts/cos.env.example scripts/cos.env
 *   2. 填写 scripts/cos.env 中的密钥与 Bucket 信息
 *   3. npm install          # 首次安装 cos-nodejs-sdk-v5
 *   4. npm run deploy:cos   # 构建 + 上传
 *
 * 可选参数：
 *   --delete  上传后删除远端多余文件（清理历史 hash 文件）
 */
import { readdirSync, statSync, createReadStream, readFileSync, existsSync } from 'node:fs';
import { join, relative, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execSync } from 'node:child_process';
import COS from 'cos-nodejs-sdk-v5';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const DIST = join(ROOT, 'dist');

// ---------- 1. 读取配置 ----------
function loadConfig() {
  const cfg = { ...process.env };
  const envPath = join(__dirname, 'cos.env');
  if (existsSync(envPath)) {
    for (const line of readFileSync(envPath, 'utf8').split(/\r?\n/)) {
      const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^#]*?)\s*$/);
      if (m) cfg[m[1]] = m[2];
    }
  }
  return cfg;
}

const cfg = loadConfig();
const required = ['COS_SECRET_ID', 'COS_SECRET_KEY', 'COS_BUCKET', 'COS_REGION'];
const missing = required.filter((k) => !cfg[k]);
if (missing.length) {
  console.error('❌ 缺少配置项：' + missing.join(', '));
  console.error('   请复制 scripts/cos.env.example 为 scripts/cos.env 并填写');
  process.exit(1);
}

// ---------- 2. 构建 ----------
console.log('🔨 构建中（npm run build）...\n');
execSync('npm run build', { cwd: ROOT, stdio: 'inherit' });

// ---------- 3. 收集本地文件 ----------
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.txt': 'text/plain; charset=utf-8',
  '.map': 'application/json',
};

function walk(dir) {
  const result = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) result.push(...walk(full));
    else result.push(full);
  }
  return result;
}

function toKey(fullPath) {
  return relative(DIST, fullPath).split(/[\\/]/).join('/');
}

function cacheControl(key) {
  if (key === 'index.html') return 'no-cache';
  if (key.startsWith('assets/')) return 'max-age=31536000, immutable';
  return 'max-age=3600';
}

const localFiles = walk(DIST).map((f) => toKey(f));
console.log(`📦 本地文件 ${localFiles.length} 个\n`);

// ---------- 4. 初始化 COS ----------
const cos = new COS({
  SecretId: cfg.COS_SECRET_ID,
  SecretKey: cfg.COS_SECRET_KEY,
});

const bucket = cfg.COS_BUCKET;
const region = cfg.COS_REGION;

function putObject(key) {
  return new Promise((resolve, reject) => {
    const full = join(DIST, key);
    cos.putObject(
      {
        Bucket: bucket,
        Region: region,
        Key: key,
        Body: createReadStream(full),
        ContentType: MIME[extname(key)] || 'application/octet-stream',
        CacheControl: cacheControl(key),
      },
      (err, data) => (err ? reject(err) : resolve(data))
    );
  });
}

function listAllObjects() {
  return new Promise((resolve, reject) => {
    const keys = [];
    let marker;
    const next = () => {
      cos.getBucket(
        { Bucket: bucket, Region: region, Marker: marker, MaxKeys: 1000 },
        (err, data) => {
          if (err) return reject(err);
          for (const obj of data.Contents || []) keys.push(obj.Key);
          if (data.IsTruncated === 'true') {
            marker = data.NextMarker || data.Contents[data.Contents.length - 1].Key;
            next();
          } else {
            resolve(keys);
          }
        }
      );
    };
    next();
  });
}

function deleteObject(key) {
  return new Promise((resolve, reject) => {
    cos.deleteObject({ Bucket: bucket, Region: region, Key: key }, (err, data) =>
      err ? reject(err) : resolve(data)
    );
  });
}

// ---------- 5. 上传 ----------
console.log('📤 上传到 COS...');
for (const key of localFiles) {
  try {
    await putObject(key);
    console.log(`   ✓ ${key}`);
  } catch (err) {
    console.error(`   ✗ ${key}: ${err.message || err}`);
    process.exit(1);
  }
}

// ---------- 6. 清理远端多余文件（--delete） ----------
if (process.argv.includes('--delete')) {
  console.log('\n🧹 清理远端多余文件...');
  const remoteKeys = await listAllObjects();
  const stale = remoteKeys.filter((k) => !localFiles.includes(k));
  for (const key of stale) {
    await deleteObject(key);
    console.log(`   ✗ 删除 ${key}`);
  }
  if (!stale.length) console.log('   （无多余文件）');
}

// ---------- 7. 完成 ----------
const website = `https://${bucket}.cos-website.${region}.myqcloud.com/`;
const objectUrl = `https://${bucket}.cos.${region}.myqcloud.com/index.html`;
console.log('\n✅ 部署完成！');
console.log(`   静态网站地址（推荐）: ${website}`);
console.log(`   对象访问地址（备用）: ${objectUrl}`);
console.log('\n   ⚠️ 首次使用前需在 COS 控制台开启「静态网站」并设为公有读，见 markdown/国内部署-COS.md');
