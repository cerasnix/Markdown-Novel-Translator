# Markdown / HTML / EPUB Novel Translator

## 项目文件

- `translate.py`：主脚本（翻译、断点续跑、可选打包）
- `prompt.example.txt`：Prompt 模板（复制为 `prompt.txt` 后可自定义）
- `config.example.json`：配置模板（复制为 `config.json`）

## 快速开始

1) 安装依赖

```bash
pip install openai
```

2) 初始化配置与 Prompt

```bash
cp config.example.json config.json
cp prompt.example.txt prompt.txt
```

3) 在 `config.json` 填入你的 `api_key` / `base_url` / `model_name`

4) 运行

```bash
python3 translate.py "/path/to/book.md"
python3 translate.py "/path/to/book.html"
python3 translate.py "/path/to/book.epub"
python3 translate.py "/path/to/book.epub" --post-package both
python3 translate.py
```

说明：`python3 translate.py`（无参数）会进入交互式模式，支持工作模式选择与路径校验。

## 输入与输出行为

### 输入类型

- 单文件：`.md` / `.markdown` / `.html` / `.htm` / `.epub`
- 目录：递归扫描上述文件类型

### 默认输出类型

- Markdown 输入 -> 输出 Markdown（同后缀）
- HTML 输入 -> 输出 HTML（HTMLZ 友好）
- EPUB 输入 -> 输出 HTML（统一进入 HTML 工作流，便于校对）

### 可选后处理打包

- `--post-package htmlz`：在 HTML 输出后生成 `.htmlz`
- `--post-package epubv3`：在 HTML 输出后生成 `.epub`
- `--post-package both`：同时生成 `.htmlz` 和 `.epub`

## EPUB 输入处理流程（新增能力说明）

当输入为 `.epub` 时，脚本会执行：

1. 读取 `META-INF/container.xml`，定位 OPF  
2. 解析 OPF 的 `manifest + spine`，按 spine 顺序抽取章节  
3. 提取章节 HTML/XHTML 的 `<body>` 内容并拼接为统一 HTML  
4. 内联 EPUB 内部图片（`img src` 转 `data:`）避免资源丢失  
5. 将拼接后的 HTML 按与 HTML 输入相同的分段翻译逻辑处理  
6. 输出为 `.html`，可继续 `--post-package` 生成 `.htmlz/.epub`

这意味着 EPUB 与 HTML 的下游校对、样式和打包流程完全一致。

## HTML 资源与 HTMLZ 兼容优化

- 输出文件名自动清洗为归档友好格式（减少特殊字符问题）
- 普通 HTML 输出会自动复制本地资源到 `assets/<book_ascii_slug>/...`
- 复制后的资源名与引用路径转为 ASCII 安全格式（含哈希后缀）
- 打包时会对本地 `img src` 再做内联，降低转换后图片失效风险
- 输出 HTML 会注入 `<title>` 与 `dc.*` 元数据，便于 Calibre 识别

## 翻译执行流程（运行时）

1. 启动 API 预检（可通过 `--skip-api-check` 关闭）  
2. 读取输入并分词/分段  
3. 按 chunk 策略调用模型翻译  
4. 实时进度输出（Prepare / Chunk Start / Chunk API / Chunk OK）  
5. 失败时重试，必要时自动降档 chunk  
6. 按需写入断点文件并支持续跑  
7. 完成后输出 HTML/Markdown，并可选后打包

## 常用参数

- `--config`：配置文件路径（默认 `config.json`）
- `--prompt`：Prompt 文件路径（默认 `prompt.txt`）
- `--suffix`：输出后缀（默认 `_CN`）
- `--skip-existing`：跳过已存在输出
- `--output-style bilingual|translated`：双语/仅译文
- `--html-translation-style blockquote|paragraph|details`：HTML 双语样式
- `--post-package none|htmlz|epubv3|both`：HTML 后处理打包
- `--no-resume`：关闭断点续跑
- `--no-realtime-write`：关闭实时写入
- `--skip-api-check`：跳过启动预检
- `--api-check-only`：仅执行预检并退出

## 稳定性与续跑

- 默认开启断点续跑，状态写入输出旁的 `*.resume.json`
- 续跑只恢复进度和上下文，不恢复旧分块参数（使用当前配置）
- 若服务端不支持 `reasoning` 参数，会自动回退继续运行

## 推荐参数（`config.example.json`）

- `chunk_size: 4`
- `max_chunk_segments: 80`
- `target_chunk_chars: 2600`
- `max_segment_chars: 1200`
- `max_chunk_chars: 5200`
- `context_tail_segments: 5`
- `request_timeout_seconds: 300`
- `api_test_timeout_seconds: 90`
- `summary_interval_batches: 10`
- `summary_interval_chars: 16000`
- `reasoning.effort: low`

## 致谢

- GPT-5.3-Codex
