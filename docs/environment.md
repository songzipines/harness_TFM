# 环境与迁移

## 已配置本机

当前 Windows 环境来自步骤 1/2：Python 3.12.15、PyTorch 2.9.1+cu128、tabpfn 9.2.0、tabicl 2.2.0、官方 LimiX 2.0.0。代码不重复创建环境，统一使用 `D:\TFM_proposal\tfm_runtime\envs\tfm\Scripts\python.exe`。

`requirements/windows.lock.txt` 记录本机完整包版本；其中 torch 下载源和 LimiX 源码安装需按下文单独准备。原始本机 freeze 保存在 runtime 内。不要把 Windows venv 复制到 Linux 后直接执行。

`configs/assets.lock.json` 保存四份 checkpoint 的官方 revision/SHA256、精确数据身份及索引哈希、LimiX commit/源码内容哈希和推理配置。源代码哈希忽略 CRLF/LF 差异；权重、parquet 和行索引不放宽校验。

## 新机器：推荐迁移什么

迁移 Git 仓库，以及 runtime 的 `weights/`、`data/` 和 `sources/LimiX/`（含缓存路径补丁）。不要迁移 Windows venv、Python 二进制和编译缓存。

Linux 服务器可以采用与本机相同的相邻目录布局：

```text
/your-disk/TFM_proposal/
├── harness_mvp/      # git clone
└── tfm_runtime/
    ├── weights/
    ├── data/
    ├── sources/LimiX/
    └── envs/tfm/     # 在服务器新建
```

默认配置使用相对 runtime 路径，布局相同时不用改配置。若目录不同，复制 pilot.json 改 runtime_root；输出身份将相应变化。Linux 没有 D 盘约束，但运行目录应放在有容量的数据盘。

## Linux 配置路径（尚未在服务器实测）

先准备 Python **3.12.15** 和 CUDA 驱动支持 12.8 的服务器。依赖安装时也把临时/缓存路径设到数据盘，避免占系统盘；例如：

```bash
cd /your-disk/TFM_proposal/harness_mvp
export TFM_RUNTIME_ROOT=/your-disk/TFM_proposal/tfm_runtime
export TMPDIR="$TFM_RUNTIME_ROOT/tmp"
export PIP_CACHE_DIR="$TFM_RUNTIME_ROOT/cache/pip"
export UV_CACHE_DIR="$TFM_RUNTIME_ROOT/cache/uv"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
python3.12 -m venv "$TFM_RUNTIME_ROOT/envs/tfm"
source "$TFM_RUNTIME_ROOT/envs/tfm/bin/activate"
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/models.txt
```

LimiX 源码必须来自官方仓库，不能安装同名遗传学包。如果没有迁移源码：

```bash
mkdir -p "$TFM_RUNTIME_ROOT/sources"
git clone https://github.com/limix-ldm-ai/LimiX.git "$TFM_RUNTIME_ROOT/sources/LimiX"
git -C "$TFM_RUNTIME_ROOT/sources/LimiX" checkout 516bf396333feb3198cf7aff8a6c10421f218e24
git -C "$TFM_RUNTIME_ROOT/sources/LimiX" apply "$PWD/patches/limix-d-drive-cache.patch"
python -m pip install -e "$TFM_RUNTIME_ROOT/sources/LimiX"
python -m pip check
```

迁移过且已打补丁的源码不要再次 apply。补丁只把原硬编码缓存路径改为 `LIMIX_CACHE_DIR`，不改模型预测逻辑。Linux 使用 Triton 3.5.1；Windows 使用 triton-windows 3.5.1.post24。

优先直接迁移步骤 2 已冻结的 `data/`，因为不同依赖生成的 parquet 字节可能不同，doctor 会拒绝变化。确需重新准备时，使用仓库中的 `scripts/download_weights.py`、`scripts/prepare_data.py`、`scripts/verify_data.py`，运行前设置 TFM_RUNTIME_ROOT：

```bash
python scripts/download_weights.py
python scripts/prepare_data.py
python scripts/verify_data.py
python -m harness doctor
```

准备脚本会联网下载；正式 run 离线，只读本地权重与固定数据。prepare_data 会重建 parquet/索引，请勿对已有实验输入盲目重复运行；若 doctor 不通过，诊断字节差异，不自动重写仓库资产锁。

doctor 通过后先做一次合成 smoke，再由用户开始 P1。新机器上的环境兼容性、4090 显存与完整数据集运行时间仍需实测；本地合成检查不能替代该验收。
