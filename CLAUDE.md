# CLAUDE.md — LiveTranslate-NG

> 本文档是给 Claude Code（以及后续接手者）的**仓库工作指南**。它比 README 更偏"怎么开发/打包/避坑"。核心事实以仓库当前源码 + `.github/workflows/*.yml` + `pyappify.yml` 为准；"分发模型/镜像/踩坑"一节融合了多次长会话定稿的决策，**改动这些一定要先看本文件对应小节**。

## 一句话

LiveTranslate-NG（`D:\biancheng\LiveTranslate-NG`，github `fjqz177/LiveTranslate-NG`）是**Windows 实时音频字幕/翻译悬浮层**：WASAPI 回路采集 → VAD → ASR（Whisper / FunASR / SenseVoice-ONNX）→ 多语翻译 → 悬浮字幕 + OBS 字幕窗。技术栈 **PyQt6**（纯 QWidget，无 qfluentwidgets——见 §5）。通过 **pyappify** 做"源代码/依赖分离"分发。

重写自旧仓库 `LiveTranslate`（`D:\biancheng\LiveTranslate`），2026-09 定稿方向见记忆 `lt-ng-pyappify-full-model`（**权威**）——它**推翻**了旧定稿 `lt-ng-rewrite-direction` 里的"core 壳 + App 自驱引擎 pack"，别再把 pack 方案当现状。

---

## 1. 目录结构与分层规则

```
src/livetranslate/
  __main__.py   # 唯一 import-order owner；启动/烟测/多进程引导都在这里
  app.py        # 组合根：装配三个顶层 QWidget + 托盘
  core/         # 纯逻辑，禁止 import audio/asr/ui；含 pipeline/translator/theme/settings/paths/privacy/i18n
  asr/          # 引擎注册+worker 子进程；只有这里允许 import 引擎后端
    engines/    # 具体引擎加载（whisper/funasr/sensevoice-onnx/anime-whisper）
    vad/        # VAD（segmenter 状态机 + scorer: silero/energy/disabled）
    worker.py client.py controller.py registry.py availability.py fbank.py
  modeling/     # hub_downloader(纯httpx)/manager/cache/registry —— 模型下载
  audio/        # backend(wasapi)/backends/resample/vad
  platform/     # 系统(权限/热键/窗口穿透)；window.py 是唯一允 import Qt 的 platform 模块
  ui/           # overlay/subtitle/panel/app_shell/settings_bridge
scripts/        # build_full_requirements.py / export_sensevoice_onnx.py / generate_icons.py
tests/          # 按层分 asr/audio/core/modeling/platform/ui/server + 架构约束测试
pyappify.yml    # 分发配置（cpu/gpu 变体）
requirements-full-{cpu,gpu}.txt  # 装机依赖（含 hash）
```

**硬分层**（`tests/test_architecture.py` 用 AST 扫描强制，改分层必被 CI 拦）：
- `core/`、`modeling/`、`audio/`、`asr/`（部分）**禁止 import PyQt6**（`test_no_qt_below_the_ui_layer`）。
- **GUI 进程永不 import 引擎后端**：只有 ASR worker 子进程能 `import asr/engines`（`test_gui_process_never_imports_engine_backends`）。引擎后端白名单仅 `asr/core.i18n/core.paths/core.privacy/modeling`。
- `core/` 不能 `import audio/asr/ui`（管线靠 Protocol 注入，`pipeline.py:42-115`）。
- `importlib.import_module` 被禁，唯一豁免是 `asr/worker.py` 的测试缝 `LIVETRANSLATE_TEST_ENGINE_FACTORY`。
- `engine_type`（如 `whisper`/`funasr`/`sensevoice-onnx`/`anime-whisper`/`remote-whisper`）是 M-MATRIX **单一真源**（`asr/registry.py`）：同时是 worker `_ENGINE_FACTORIES` 的 key、持久化 `settings['asr_engine']`、`worker_config['engine_type']`；GUI 顺序由 `GUI_ENGINE_ORDER` 派生，**不得复制/硬编码**。

---

## 2. 常用命令（开发 / 质量 / 打包）

```bash
uv run livetranslate              # 启动 App（真正入口 __main__.py）
uv run livetranslate --smoke      # 无头烟测：exit 0 + 输出含 "Smoke OK"
uv run livetranslate-pr           # = devtools.gate_main() 权威质量门（见下）
uv run livetranslate-check        # 别名（同 gate_main）
uv run pytest tests/              # 只跑测试
uv run python scripts/build_full_requirements.py  # 重新生成 requirements-full-{cpu,gpu}.txt
```

`livetranslate-pr`（fail-fast 按序）：
`uv lock --check` → `uv run ruff check .` → `uv run ruff format --check .` → `uv run mypy src` → `uv run pytest tests/`；带 `--smoke` 追加 `uv run livetranslate --smoke`；带 `--git-audit` 追加 stage 一致性检查。

> **本地一律用镜像**（记忆 `lt-ng-local-mirrors`，用户强要求）：`uv sync`/`pip`/导出用**清华 pypi** default + **南京大学 cu126** explicit（见 §7）。别用官方 `download.pytorch.org`（境内极慢）。

---

## 3. 启动 / 运行时关键约束（`__main__.py`，改这里必读）

`src/livetranslate/__main__.py` 是唯一 import-order owner，顺序**极其严格**：

1. `freeze_support()` 必须在**任何 livetranslate import 之前**（win32 冻结 worker 以 `--multiprocessing-fork` 重进本文件；缺它 → 单实例门吞真实 ASR 错误、表现为裸 `ASRWorkerExited`）。
2. `apply_cache_env()`（`modeling.manager`）必须在 **import torch 之前**，否则 `TORCH_HOME` 不生效。
3. **win32 上 torch 必须先于 PyQt6 import**（c10.dll 冲突，pytorch#166628）；平台门控（`__main__.py:40-46`）；torch import 失败置 None 做优雅降级。
4. `--smoke` 分支必须在 import 任何 app 模块前设 `QT_QPA_PLATFORM=offscreen` 并把 `LIVETRANSLATE_PORTABLE_DIR` 指向临时目录（`paths.py` 在 import 期算 `CONFIG_DIR`，须早于它）。

**数据区（`core/paths.py`）** 三个维度，优先级：
`LIVETRANSLATE_PORTABLE_DIR`（显式覆盖） > 冻结 `<install root>\data` > dev 仓库根；`LIVETRANSLATE_PLATFORM_DIRS=1` 时退出到 platformdirs（旧仓库根的 `models/` 仍优先，避免旧 checkout 被迫重下模型）。
- 冻结时机数据区 = `sys.executable` 的 parent.parent/data；sidecar 更新只替换 `<install>/app`，**数据跨更新存活**。
- dev 默认数据留在仓库根（与冻结行为同步）。

**管线数据链** `capture→VAD→ASR→翻译→输出`（`core/pipeline.py`）：`Pipeline.start()` 建 `ThreadPoolExecutor(max_workers=max(8, 副标题目标语言+1))` + `queue(16)` + `_capture_thread`/`_asr_thread` 两个 daemon。`_capture_loop`：读 chunk → RMS → overlay.update_monitor → `vad.process_chunk` → 产 `speech_segment` 则入队 `vad_flush`，无段且 incremental 开则周期性入 `interim`。`_asr_loop` 消费：`vad_flush`→`_process_segment`（噪声/语言滤）、`interim`→`_do_interim_asr`（只提交完整句 + trim_front + 回声去重）。翻译走 `_translate_async`（`translate_iter` 流式 partial → overlay）、副标题多语言 `extra_langs` 并行。**样本率硬编码 16000**（不走 config）。管线无 Qt 依赖（全部 Protocol 注入）。

**单实例**（`single_instance.py`）：第一实例拿平台锁 + 建 `QLocalServer("livetranslate")`；第二实例连不上锁 → QLocalSocket 发 `"wake"` 后退出；`WAKE_FAILED` 哨兵让上层弹窗解释。唤醒 socket 钉在全局 `_WAKE_SOCKETS` 防 qDrop。

---

## 4. 引擎与模型

**引擎注册表**（`asr/registry.py` L45-90，5 项，全 `platforms=("win32",)`，**无 cpu/gpu 字段**）：

| engine_type | 引擎 | tier | extras | download_gb |
|---|---|---|---|---|
| `whisper` | faster-whisper | recommended | `engine-whisper` | 1.5 |
| `sensevoice-onnx` | SenseVoice-ONNX | normal | `engine-sensevoice-onnx`(空) | 0.9 |
| `funasr` | sensevoice-funasr | legacy | `engine-funasr` | 1.0 |
| `anime-whisper` | anime-whisper | normal | 复用 `engine-whisper` | 3.0 |
| `remote-whisper` | remote | normal | 无 | 0.0 |

- `recommend_engine(_accel)` **恒返回 `faster-whisper`**（CPU+CUDA 双支持，缺省`model_size=medium`，CPU `compute_type` float16→int8）。
- **cpu/gpu 是 pyappify 安装期变体（default=cpu / gpu=torch cu126），不是引擎选择**。装满后所有引擎都可用，App 永不管理引擎（`pyappify.yml`）。
- `remote-whisper` 不在 `_ENGINE_FACTORIES`：`controller.load_engine_client()` 对 `remote-whisper` 直接建 in-process `RemoteASREngine`（httpx `/transcribe` + `/health`，`X-ASR-Token`），无子进程。

**模型下载**（`modeling/hub_downloader.py`，**纯 httpx + stdlib，无 modelscope/hf SDK**）：`Hub=Literal['ms','hf']`。ModelScope REST `/api/v1/models/{org}/{name}/repo/files`(tree) + `.../repo?FilePath=...`；HF `/api/models/{id}/tree/{rev}?recursive=true` + `/resolve`（302→CDN）。**Range 断点续传**写入 `<file>.incomplete` 成功 `tmp.replace`；416=已完整；sha256 有则 size+sha256 双校验。SEC-3：`_safe_dest` 拒绝绝对路径/`..`/Windows 保留设备名（防恶意 repo 逃出缓存根）。缓存布局与 SDK 字节兼容：ms→`<cache_dir>/<org>/<name>/`，hf→`models--<org>--<name>/snapshots/<rev>/`。
- 缺省 revision：ms→`master`，hf→`main`。

**SenseVoice-ONNX 是唯一例外**：无 hub 产物/无下载链接——它从 `scripts/export_sensevoice_onnx.py` 用 funasr `iic/SenseVoiceSmall`(CPU,ms) torch 导出（opset=17）；引擎直接加载 `models_dir()/sensevoice/sensevoice-small.onnx`，缺失抛 `FileNotFoundError`（提示先跑导出脚本）。前端 fbank+LFR 是 `asr/fbank.py` 纯 numpy 移植（对 torchaudio 1e-4 parity，LFR 7/6，FEAT_DIM=80），神经部分 onnxruntime，tokenizer sentencepiece——**torch-free**。base 里 `engine-sensevoice-onnx` extras 故意为空。

**VAD**（`audio/vad/`）：`segmenter.py` 纯状态机（无 torch/无 I/O），`scorer.py` 提供 `SileroConfidenceScorer`（onnxruntime，512 窗口 + LSTM h/c 状态，需 `reset()`）/ `EnergyConfidenceScorer`(RMS) / `AlwaysSpeechScorer`(disabled)；`VADProcessor` 按 `mode=silero/energy/disabled` 切换。Silero ONNX 路径 `default_onnx_path()`：优先 silero-vad pip 包内置 → frozen `models/vad/silero_vad.onnx` → `models_dir()/vad/`；`allow_missing=True` 静态降级 disabled（CI smoke 不崩）。

**Worker 子进程**（`asr/worker.py`）：`_ENGINE_FACTORIES`（engine_type→loader）；`worker_main(conn,config)` 循环 `recv` 多进程 Pipe，`dispatch handle_message`（transcribe/set_language/set_input_padding/ping/shutdown）；`_parse_device` 拆 `cuda:N`。`ASRClient`（spawn + Pipe，daemon）主进程代理（ready_timeout=180 / request 120 / shutdown=5）；`AsrController` 串行化单活跃 client（detach_current/activate + generation 计数、recover_worker(_restart_max=3)、recycle_worker(baseline+2048MB)、maybe_ping_worker(5s)）。

**引擎特有坑**：
- `funasr` 加载前必须 `neutralize_funasr_requirements()`（把模型目录 `requirements.txt` 改名 `.bundled`），否则 FunASR `trust_remote_code` 会在子进程偷偷 pip install（慢网挂死/拉 gradio）。
- `funasr-nano/mlt-nano` 内嵌 Qwen3-0.6B 权重 ~1.5GB 单独下载（`ensure_qwen_weights` 只在非 worker 启动路径调用，避免 180s ready 超时）。
- `anime-whisper` 是 HF-only，忽略 hub 设置。
- whisper 缺省走 `get_whisper_local_path` 从本地 cache（ModelScope/HF）加载，避免 faster-whisper 自己触发 HF 下载。

---

## 5. UI（重要：现状 ≠ 计划）

**当前源码里没有任何 qfluentwidgets / FluentWindow / NavigationInterface / MainWindow**（`pyproject` 仅 `PyQt6>=6.5,<7`）。它们只是 `docs/development/重写方案.md` 里的**未来 P3 「GUI Fluent」计划**，别当现状。

实际组合根（`app.py`）创建三个顶层 `QWidget` + 一个托盘：
- **SubtitleOverlay**（`ui/overlay_window.py`）—— **实际承担『主窗口』角色**，聊天式悬浮字幕窗（`FramelessWindowHint|WindowStaysOnTopHint|Tool` + `WA_TranslucentBackground` + `WA_ShowWithoutActivating`；自绘 QSS `rgba(15,15,25,200)` + `border-radius:8px`），托盘左键拉起。
- **SubtitleWindow**（`ui/subtitle_window.py`）—— 纯文本 OBS 捕获窗（自绘 `_SubtitleTextWidget` QPainterPath 描边 + 换行 + 出入场动画），固定宽度、高度自适应。
- **ControlPanel**（`ui/panel/panel.py`）—— 设置面板（左 `QListWidget` 导航 + 右 `QStackedWidget`），7 页：General/Translation/识别页(VadTab+HotkeysGroup)/字幕页(StyleTab+SubtitleTab)/CacheTab/DiagnosticsTab/AboutTab。
- `QSystemTrayIcon` 托盘（`app_shell.py` 的 `build_tray_shell`）。

**穿透/透明**统一走 `platform/window.py`（唯一允许 import Qt 的 platform 模块）：ctypes `GetWindowLongW/SetWindowLongW` 切 `_WS_EX_TRANSPARENT(0x20)`，`set_click_through(win,'all'/'interactive')`。Overlay 按区域分"头部交互区/机体穿透区"（只开着 50ms 轮询）；SubtitleWindow 全窗穿透 + showEvent 重断言。

**主题**（`core/theme.py` 单一权威）：`DEFAULT_STYLE` + `STYLE_PRESETS` 共 14 套（default/transparent/compact/light/dracula/nord/monokai/solarized/gruvbox/tokyo_night/catppuccin/one_dark/everforest/kanagawa）。**WCAG 对比度守卫**：字幕 ≥7:1、元数据 ≥3:1、半透明 <200 必须 ≥2px 描边（`validate_style_contrast`）。`tests/core/test_theme.py` 锁死每套预设——**新增/改预设颜色必须过守卫，否则 CI 红**。

**双层设置模型**（`panel/panel.py:74-174`）：`SettingsStore`（`ui/settings_bridge.py`，纯持久层包装）为唯一真值源，`_current_settings` 是面板草稿副本（各 tab 就地改，保存时 `_apply_settings` 原子提交 + `settings_changed.emit`）。Store 内部 dict 绝不外泄。

**面板外观**（`ui/panel/_chrome.py`）：DARK/LIGHT 两套【QSS 字符串 + QPalette 兜底】，主题必须 **app 级** `apply_app_theme`（widget 级 stylesheet 到不了 QComboBox 下拉/QColorDialog/QMenu 等顶层弹窗）。panel 根需 `WA_StyledBackground`。**不要给 ScrollPage/scroll viewport 加无选择器内联 stylesheet**——会匹配子树所有 widget 并画成透明/纯黑。

---

## 6. 分发模型（pyappify）—— 本项目最核心、最易踩坑

**目标（记忆 `lt-ng-pyappify-full-model`，2026-09-02 定稿，推翻旧 pack 方案）**：pyappify 负责三件事——
1. **源码**：git-tag 秒级更新（pyappify 只换 working/，依赖不动）。
2. **依赖**：pyappify profile（default=cpu / gpu）各 pip 装满**全部引擎依赖**（`requirements-full-{cpu,gpu}.txt`），用户首开选变体即装齐。
3. **变体**：cpu/gpu 两个 profile（换 profile = 重装 torch 变体）。
**App 只下模型**（`modeling/hub_downloader.py`）；App 本体**无任何运行时动态装 pip 包的代码**（旧 updater 已删）。

**`pyappify.yml` 关键**：
- `profiles[0]` **必须命名 `default`**（pyappify 更新时硬编码读它）。default=cpu：`requirements=requirements-full-cpu.txt`、`main_script=main.py`、`requires_python=3.12`、`use_pythonw=true`、`show_add_defender=true`，**不写 pip_args**（主 pypi 镜像交给用户在 pyappify 设置里选）。
- `profiles[1]` **`gpu`**：`requirements=requirements-full-gpu.txt`，`pip_args="--extra-index-url {PIP_TORCH_INDEX_URL}"`（**占位符**，由 pyappify runtime 展开成用户所选 torch 源）。

**用户首开流程**：装 `*-online-setup.exe` → 打开 pyappify → **选硬件变体（CPU/GPU）** → **选主 pip 镜像**（内置 PyPI官方/清华/阿里/USTC/华为/腾讯）→ 对 GPU 还**选 torch cu126 源**（官方 download.pytorch.org / 南大 mirror.nju.edu.cn）→ 一次性装满该变体全部依赖。之后源码 git-tag 更新秒级（requirements 不变 → 不重 pip）；「Change Profile」切换变体（重装 torch）。

**本地开发 vs CI 的镜像分工**：本地/复现用清华 pypi + 南大 cu126；CI（wsl 境外 runner）才用官方 cu126。

---

## 7. 依赖 / uv / requirements-full-*（改依赖必读）

**uv 镜像配置**（`pyproject.toml`）：
```toml
[[tool.uv.index]]  name="pypi"    url="https://pypi.tuna.tsinghua.edu.cn/simple"  default=true
[[tool.uv.index]]  name="pytorch" url="https://mirror.nju.edu.cn/pytorch/whl/cu126" explicit=true
[tool.uv.sources]  torch=[{index="pytorch"}]  torchaudio=[{index="pytorch"}]
```
**为何 `explicit=true`**：把 cu126 索引**只限 torch/torchaudio**，其它包仍从 pypi default 解析 → `uv export` 每个包都有完整 `--hash=`。否则某包（如 jinja2/markupsafe）缺 hash，`pip install -r` 在隐式 `--require-hashes` 下**整文件失败**（当年 gpu 装机 exit-1 真因）。

**两条红线**：
- 导出生成 requirements 时**绝不能传 `--index pytorch=<url>` 覆盖**——那把 cu126 提升为全局候选，重引入 `idna==3.4` 冲突，并让 jinja2/markupsafe 从 cu126 索引解决（uv 记不到 hash）。
- **CPU profile 的 `pip_args` 绝不能写 `--index-url`/`-i`**——`python_env.rs` 一旦在这类参数就 `use_config_index_url=false`，**绕过用户所选镜像**。GPU 用 `--extra-index-url`（不触发该分支），用户仍能选主镜像。

**`requirements-full-*.txt` 生成**（`scripts/build_full_requirements.py`，跑 `uv run python scripts/build_full_requirements.py`）：用 `uv export --no-dev --no-emit-project --default-index <pypi>` 输哈希锁定。
- **CPU**（`engine-funasr`+`engine-whisper`）额外 `--index pytorch=<pypi>` 强制 torch 从 PyPI → `torch==2.11.0`（CPU wheel）。
- **GPU**（`+engine-cuda`）**不传** index override → torch 走 explicit 南大 cu126 → `torch==2.11.0+cu126`。
- pypi_index 来自 env `LT_REQUIREMENTS_PYPI_INDEX`（本地默认清华、CI 设官方）。

**base 刻意 torch-free**：numpy/PyYAML/httpx/openai/psutil/pysbd/sentencepiece/pydantic/platformdirs/cryptography/packaging/onnxruntime/PyAudioWPatch(win32)/PyQt6。**modelscope + huggingface-hub SDK 不进 base**（用 hub_downloader 替代省 ~130MB）。**silero-vad 不进 dev 组**（≥5.0 硬依赖 torch+torchaudio 拖 ~2.4GiB，仅导出 Silero ONNX 时按需装）。`PyAudioWPatch` 只有 Windows wheel → CI 只跑 windows-latest。

---

## 8. CI / 打包链路

**`.github/workflows/release.yml`** 三层 + **`.github/workflows/ci.yml`**（单 check job）：
- **validate**（windows）：`uv sync --locked` + `uv run livetranslate-pr` + smoke gate（`uv run livetranslate --smoke`，env `LIVETRANSLATE_PORTABLE_DIR`，`exit 0` 且输出 `Smoke OK`；先清空 secrets env）。
- **package**（needs validate）：drift gate（重跑 export 并 diff 提交的 requirements-full-*.txt，`continue-on-error:true` 软门，grep 去 `^#` 行对比 pin+hash；CPU 传 `--index pytorch=$PYPI`、GPU 不传）+ `fjqz177/pyappify-action@master` 构建 + （`SIGN_BUILD=true` 时）SignPath。
- **publish**（Ubuntu）：`softprops/action-gh-release`；`files` 只在 `*-online-setup.exe` + `*.zip` + `*_sha256.txt`。

**你 fork 的 action 用法（release.yml）**：
```yaml
uses: fjqz177/pyappify-action@master
with:
  version: v0.0.25     # pyappify runtime tag（硬编码，需手动 bump，容易漏）
  build_exe_only: false
  online_only: true    # 只产 online-setup + zip + sha256，跳过所有 profile 离线包
```
- **`online_only`** 是你在 `fjqz177/pyappify-action` 加的 input（上游 `ok-oldking/pyappify-action` 没有）：`index.js` 里 `if (!onlineOnly) { for(profile...){...} }` 跳过 `-c setup` 离线构建。
- **action 内部 `git clone https://github.com/fjqz177/pyappify.git`**（你的 runtime fork，非官方）——你 fork 的 runtime 改动（torch 源选项卡等）由此进 installer。
- `version` 用于 `git checkout tags/<version>`（pyappify runtime 的 tag）。

**`SIGN_BUILD`** 默认 `false`（先出未签名安装器）；SignPath 项目 + `SIGNPATH_*` secrets 就绪后才置 `true`（首次需人工项目/审批）。

**CI 两个 workflow 都把 setup-uv 钉死 `0.12.7`**；`[tool.uv] required-version >=0.9.0`。`requires-python = >=3.10,<3.13`，但 pyappify `requires_python=3.12`、mypy `python_version=3.12`——**实际只在 3.12 打包**。

---

## 9. 打包/分发踩坑清单（本会话实测积累，改前必读）

1. **`version` 必须是合法 semver**：`prepareTauriConfig`（pyappify-action 的 build-config.js）把 tauri.conf.json 的 `"0.0.1"` 替换成 `version.replace(/^v/,'')`。**runtime tag 用 `v0.0.25`（无横杠）**；`v-0.0.25` 会得 `-0.0.25` 导致 tauri `version must be semver` 崩溃。官方 tag `v-0.0.24` 不崩是因为**官方 tag 处** tauri.conf.json version 已是合法值（1.2.3），而你的 tag 指向 master（version=0.0.1）就会命中。
2. **前端 TS：MUI `MenuItem` 的 `key`/`value` 不能是 `boolean`**。torch 下拉选项是 `string|number|boolean`，必须 `key={String(o)} value={String(o)}`（`App.tsx` torch 下拉）。本地没 `node_modules` 时 tsc 不会暴露——CI `pnpm tauri build` 的 `beforeBuildCommand`（tsc）才抓。
3. **Rust `current_profile` 是 `String` 非 `Option`**（`app_service.rs` 更新逻辑）：`None` 分支用 `String::new()`；取用用 `.as_str()`（**不是** Option 的 `.as_deref()`）。修 default-profile 更新 bug 时踩过。
4. **`cargo check` 可能假象**：并行执行时 Bash 会**抢在 Edit 落盘前读到旧文件** → 报 exit 0 实则没验证新代码。所以"改 Rust → 顺序跑 cargo check"，别并行，否则 CI 的 `tauri build` 才暴露真错。
5. **NSIS 离线包 `mmap` 崩溃**：gpu 离线包把 ~4.5GB torch+torchaudio 压 NSIS，触发宏 `error mmapping file ... is out of range`。**解法 = `online_only: true`（跳过离线构建）**，而非 continue-on-error（那只是容忍崩、每次仍白下载 4.5GB）。
6. **requirements 缺 hash → `pip --require-hashes` 整文件失败**：见 §7（`explicit cu126` + 别传 `--index pytorch=<url>`）。
7. **drift gate 是软门**：CI 在官方 PyPI 重导出（镜像表差/头注释 URL 不同），只 filter `^#` 对比 pin+hash；真正硬约束靠质量门 `uv lock --check`。别把它当阻断。
8. **`version: v0.0.25` 是硬编码漂移点**：不随 git tag 自动增长，升级 pyappify runtime 时手动同步。
9. **runtime fork / action fork 的改动须 push 到 GitHub 才进 CI**：`@master` 是浮动分支，`uses:` 拉的是 GitHub 上的 commit。记得 `node --check index.js && dist/index.js`（action 实际跑 `dist/`，改源码必须同步 dist，否则 `pnpm build` 覆盖回退）。

---

## 10. 仓库间关系（三个 fork，都是用户的）

| 仓库 | 角色 | 当前关键 |
|---|---|---|
| `fjqz177/LiveTranslate-NG` | 本仓库（被打包的 App） | `pyappify.yml` + `release.yml` + `requirements-full-*.txt`；`main` 分支 |
| `fjqz177/pyappify` (D:\biancheng\pyappify) | **运行时 launcher** | tag `v0.0.25`；加 torch 源选项卡（`config_manager.rs` "Pip Torch Index URL" + `python_env.rs` `{PIP_TORCH_INDEX_URL}` 占位符展开 + 前端 App/SettingsPage 仅 gpu 显示） + 修 default-profile 更新 bug |
| `fjqz177/pyappify-action` (D:\biancheng\pyappify-action) | **CI 打包 action** | 源码 `index.js` + 产物 `dist/index.js` 都加 `online_only`；clone URL 指向用户 runtime fork；`@master` |

> runtime fork 与本仓库是**两条链路**：本仓库 `release.yml` 中 `uses: fjqz177/pyappify-action` 只是"构建器"；它内部 clone `fjqz177/pyappify`（runtime）编译 launcher；装机后 launcher 里跑的正是那个 runtime（含 torch 选项卡）。改 runtime 功能 → 需在 `fjqz177/pyappify` 改、push、**更新 `version` tag**（见 §9-8）。

---

## 11. 相关记忆文件（跨会话决策权威）

这些在 `~/.claude/projects/D--biancheng-LiveTranslate-NG/memory/`（不在仓库内，但你值得知道它们存在）：
- `lt-ng-pyappify-full-model`：**当前权威分发定稿**（pyappify 装满 + 在线 setup + 用户选变体/镜像 + App 只下模型）。
- `lt-ng-local-mirrors`：本地 dev 用清华+南大。
- `lt-ng-engine-model-maps`：引擎↔模型 hub id 映射 + hub_downloader + SenseVoice-ONNX 导出例外。
- `lt-ng-rewrite-direction`：**旧定稿**（已被 full-model 推翻，仅作背景）。
