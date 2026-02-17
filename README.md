# Markdown / HTML Novel Translator Starter

本项目包含：

- `translate_markdown.py`：Markdown / HTML 小说分段翻译脚本
- `prompt_markdown.txt`：默认通用 prompt（无角色定制引导）
- `prompt.example.json`：prompt JSON 示例（无角色定制引导）
- `config.example.json`：配置模板（不含敏感信息）

## Quick Start

1. 创建并激活环境（示例）：

```bash
conda activate <your_env_name>
pip install openai
```

2. 复制配置模板：

```bash
cp config.example.json config.json
```

3. 在 `config.json` 填入自己的 `api_key` / `base_url` / `model_name`。

4. 运行：

```bash
python3 translate_markdown.py "/path/to/novel.md"
python3 translate_markdown.py "/path/to/novel.html"
```

## 常用参数

- `--suffix`：输出后缀（默认 `_CN`）
- `--skip-existing`：跳过已存在输出
- `--config`：指定配置文件路径
- `--prompt`：指定 prompt 文件路径（默认 `prompt_markdown.txt`）
- `--reasoning-effort`：覆盖 `low/medium/high`
- `--output-style bilingual|translated`：双语或仅译文输出
- `--html-translation-style blockquote|paragraph|details`：HTML 双语模式下译文块样式
- `--no-realtime-write`：关闭实时写入

## HTML 双语输出说明

- 支持输入：`.html` / `.htm`（目录模式会同时扫描 `.md/.markdown/.html/.htm`）
- 双语模式下保留原始 HTML 行，并在可翻译段落下方插入 `<blockquote>` 译文块
- 译文块使用轻量内联样式（缩进 + 左边线），便于 Calibre 手动转换 EPUB 时保留对照层次
- 可通过 `--html-translation-style` 切换样式：
  - `blockquote`（推荐，EPUB 兼容性最好）
  - `paragraph`（普通段落样式，最朴素）
  - `details`（折叠块，部分阅读器可能不支持）

## 推荐默认参数（已在 example 配置中给出）

- `chunk_size: 3`
- `max_chunk_segments: 20`
- `target_chunk_chars: 1000`
- `max_segment_chars: 900`
- `max_chunk_chars: 2600`
- `summary_interval_batches: 8`
- `summary_interval_chars: 8000`
- `reasoning.effort: low`
