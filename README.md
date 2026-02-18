# Markdown / HTML Novel Translator Starter

本项目包含：

- `translate.py`：Markdown / HTML 小说分段翻译脚本
- `prompt.example.txt`：prompt 文本模板（无角色定制引导）
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

4. 初始化 prompt（首次一次即可）：

```bash
cp prompt.example.txt prompt.txt
```

5. 运行：

```bash
python3 translate.py "/path/to/novel.md"
python3 translate.py "/path/to/novel.html"
python3 translate.py "/path/to/novel.epub"
python3 translate.py "/path/to/novel.html" --htmlz-title "Book Title" --htmlz-author "Author Name"
python3 translate.py "/path/to/novel.epub" --post-package both
python3 translate.py
```

## 常用参数

- `--suffix`：输出后缀（默认 `_CN`）
- `--skip-existing`：跳过已存在输出
- `--config`：指定配置文件路径
- `--prompt`：指定 prompt 文件路径（默认 `prompt.txt`）
- `--reasoning-effort`：覆盖 `low/medium/high`
- `--output-style bilingual|translated`：双语或仅译文输出
- `--html-translation-style blockquote|paragraph|details`：HTML 双语模式下译文块样式
- `--post-package none|htmlz|epubv3|both`：输出 HTML 后可选继续打包
- `--htmlz-title`：覆盖 HTML 元数据标题
- `--htmlz-author`：设置作者元数据
- `--htmlz-language`：设置语言元数据（默认 `zh-CN`）
- `--htmlz-identifier`：设置标识符元数据
- `--htmlz-publisher`：设置出版社元数据
- `--no-resume`：关闭断点续跑（默认开启）
- `--no-realtime-write`：关闭实时写入
- `--skip-api-check`：跳过启动前 API 预检
- `--api-check-only`：仅执行 API 预检并退出

仅运行 `python3 translate.py`（不带参数）时会进入交互式模式，可选择工作模式并填写路径。
交互式路径输入支持 shell 转义形式（如 `\ `），并会在启动前校验路径是否存在。
若 `prompt.txt` 不存在，程序会尝试从 `prompt.example.txt` 自动创建。

## HTML / EPUB -> HTMLZ 优化输出说明

- 支持输入：`.html` / `.htm` / `.epub`（目录模式会同时扫描 `.md/.markdown/.html/.htm/.epub`）
- HTML 输入默认输出为“HTMLZ 友好”HTML（不自动打包，便于先校对再手动打包）
- EPUB 输入会先抽取 spine 顺序的章节 HTML，再按同样规则输出为 `.html`
- 输出文件名会自动清洗为归档友好格式（例如去除书名号、空白归一）
- 普通 HTML 输出会自动复制本地引用资源到输出目录下 `assets/<book_ascii_slug>/`
- 复制后的资源名与引用路径会转为 ASCII 安全格式（含哈希后缀），降低 HTMLZ/EPUB 转换时的路径编码兼容问题
- 输出文件会注入 `<title>` 与 `dc.*` 元数据标签，便于后续 Calibre 识别
- 双语模式下保留原始 HTML 行，并在可翻译段落下方插入译文块（默认 `blockquote`）
- 可选 `--post-package` 在 HTML 生成后继续输出 `.htmlz` 或 `.epub`（EPUB3）文件
- 默认启用断点续跑：异常中断后再次运行会从输出旁边的 `.resume.json` 继续
- 可通过 `--html-translation-style` 切换样式：
  - `blockquote`（推荐，EPUB 兼容性最好）
  - `paragraph`（普通段落样式，最朴素）
  - `details`（折叠块，部分阅读器可能不支持）

## 稳定性说明

- 默认会在正式翻译前执行一次 API 预检（连通性、模型返回结构、基础解析），预检通过后才进入翻译
- 断点文件：默认在输出文件旁写入 `*.resume.json`，任务成功完成后自动删除
- 续跑条件：输入内容与输出模式匹配时会自动恢复；不匹配会自动忽略旧断点并重新开始
- API 兼容：若服务端不支持 `reasoning` 参数，脚本会自动回退并继续翻译
- 参数变更兼容：续跑只恢复进度与上下文，分块策略始终使用当前配置

## 推荐默认参数（已在 example 配置中给出）

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
