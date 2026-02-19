# Markdown / HTML / EPUB 小说翻译器

[English README](README.md)

## 项目文件

- `translate.py`：主脚本（翻译、断点续跑、可选打包）
- `prompt.example.txt`：Prompt 模板（复制为 `prompt.txt` 后可自定义）
- `config.example.json`：配置模板（复制为 `config.json`）

## 快速开始

1) 安装依赖  
建议先使用 Conda、venv 等虚拟环境隔离运行环境。

```bash
pip install openai
```

2) 初始化配置和 Prompt

```bash
cp config.example.json config.json
cp prompt.example.txt prompt.txt
```

3) 在 `config.json` 填入 `api_key` / `base_url` / `model_name`

4) 运行

```bash
python3 translate.py "/path/to/book.md"
python3 translate.py "/path/to/book.html"
python3 translate.py "/path/to/book.epub"
python3 translate.py "/path/to/book.epub" --post-package both
python3 translate.py
```

仅执行 `python3 translate.py`（无参数）会进入交互式模式，可选择工作模式并校验输入路径。

## 输入与输出行为

### 支持输入

- 单文件：`.md` / `.markdown` / `.html` / `.htm` / `.epub`
- 目录：递归扫描以上格式

### 默认输出

- Markdown 输入 -> Markdown 输出
- HTML 输入 -> HTML 输出（HTMLZ 友好）
- EPUB 输入 -> HTML 输出（与 HTML 走同一后续流程）

### 可选后处理打包

- `--post-package htmlz`：HTML 输出后生成 `.htmlz`
- `--post-package epubv3`：HTML 输出后生成 `.epub`
- `--post-package both`：同时生成 `.htmlz` 和 `.epub`

## EPUB 输入处理流程

输入为 `.epub` 时，脚本会：

1. 读取 `META-INF/container.xml`，定位 OPF
2. 解析 OPF 的 `manifest + spine`，按 spine 顺序抽取章节
3. 提取章节 `<body>` HTML/XHTML，并拼接为统一 HTML 流
4. 内联 EPUB 内部图片（`img src` -> `data:`）降低资源丢失风险
5. 使用与 HTML 相同的分块翻译流程执行翻译
6. 输出 `.html`，并可选继续打包 `.htmlz/.epub`

这样 EPUB 与 HTML 的校对、样式、打包流程保持一致。

## HTML 资源与 HTMLZ 兼容优化

- 输出文件名会自动清洗，减少归档与转换时的特殊字符问题
- 普通 HTML 输出会自动复制本地资源到 `assets/<book_ascii_slug>/...`
- 资源文件名与引用路径会重写为 ASCII 安全格式（含哈希后缀）
- 打包阶段会再次处理本地 `img src` 内联，降低图片失效概率
- 输出 HTML 会注入 `<title>` 与 `dc.*` 元数据，便于 Calibre 识别

## 运行时处理流程

1. API 预检（可通过 `--skip-api-check` 关闭）
2. 读取输入并分词/分段
3. 按 chunk 策略调用模型翻译
4. 输出运行进度日志（`Prepare` / `Chunk Start` / `Chunk API` / `Chunk OK`）
5. 失败时重试，必要时自动降档 chunk
6. 写入并使用断点状态进行续跑
7. 输出结果并可选后处理打包

## 常用参数

- `--config`：配置路径（默认 `config.json`）
- `--prompt`：Prompt 路径（默认 `prompt.txt`）
- `--suffix`：输出后缀（默认 `_CN`）
- `--skip-existing`：跳过已存在输出
- `--output-style bilingual|translated`
- `--html-translation-style blockquote|paragraph|details`
- `--post-package none|htmlz|epubv3|both`
- `--no-resume`
- `--no-realtime-write`
- `--skip-api-check`
- `--api-check-only`

## 稳定性与续跑

- 默认开启断点续跑，状态文件位于输出文件旁（`*.resume.json`）
- 续跑恢复进度与上下文，不恢复旧分块参数（始终使用当前配置）
- 若服务端不支持 `reasoning` 参数，脚本会自动回退继续运行
- 会对返回结果做“疑似未翻译”检测：默认使用 `exact_only`（忽略空白/标点后文本一致即判定）；
  可选启用高相似兜底，且对“短名称/称谓样式”片段做免检以减少误触发
- 启用 chunk 级回声检测：若单次 API 批次中大多数片段被原样回传，
  会直接判定该 chunk 失败并进入重试/降档

## 推荐配置（`config.example.json`）

- `chunk_size: 4`
- `max_chunk_segments: 80`
- `target_chunk_chars: 2600`
- `max_segment_chars: 1200`
- `max_chunk_chars: 5200`
- `context_tail_segments: 5`
- `request_timeout_seconds: 300`
- `api_test_timeout_seconds: 90`
- `translation_similarity_check: true`
- `translation_similarity_threshold: 0.96`
- `translation_similarity_min_chars: 18`
- `translation_similarity_exact_only: true`
- `translation_similarity_name_guard: true`
- `translation_similarity_name_like_max_chars: 24`
- `translation_chunk_echo_check: true`
- `translation_chunk_echo_match_ratio: 0.98`
- `translation_chunk_echo_min_segments: 3`
- `translation_chunk_echo_min_total_chars: 80`
- `translation_chunk_echo_min_japanese_segments: 2`
- `summary_interval_batches: 10`
- `summary_interval_chars: 16000`
- `reasoning.effort: low`

## 致谢

- GPT-5.3-Codex
