# Markdown / HTML / EPUB Novel Translator

[中文说明 (README.zh.md)](README.zh.md)

## What this project includes

- `translate.py`: main script (translation, resume, optional packaging)
- `prompt.example.txt`: prompt template (copy to `prompt.txt` and customize)
- `config.example.json`: config template (copy to `config.json`)

## Quick start

1) Install dependencies  
It is recommended to use an isolated environment first (Conda / venv).

```bash
pip install openai
```

2) Initialize config and prompt

```bash
cp config.example.json config.json
cp prompt.example.txt prompt.txt
```

3) Fill `config.json` (`api_key`, `base_url`, `model_name`)

4) Run

```bash
python3 translate.py "/path/to/book.md"
python3 translate.py "/path/to/book.html"
python3 translate.py "/path/to/book.epub"
python3 translate.py "/path/to/book.epub" --post-package both
python3 translate.py
```

`python3 translate.py` (without args) starts interactive mode with workflow selection and path validation.

## Input / output behavior

### Supported input

- Single file: `.md` / `.markdown` / `.html` / `.htm` / `.epub`
- Directory: recursively scans the same file types

### Default output

- Markdown input -> Markdown output
- HTML input -> HTML output (HTMLZ-friendly)
- EPUB input -> HTML output (same downstream workflow as HTML)

### Optional post-package

- `--post-package htmlz`: generate `.htmlz` after HTML output
- `--post-package epubv3`: generate `.epub` after HTML output
- `--post-package both`: generate both

## EPUB input processing flow

When input is `.epub`, the script:

1. Reads `META-INF/container.xml` and locates OPF
2. Parses OPF `manifest + spine` and extracts chapters in spine order
3. Collects chapter `<body>` HTML/XHTML into one normalized HTML stream
4. Inlines EPUB internal images (`img src` -> `data:`) to reduce asset loss
5. Runs the same chunked translation pipeline as HTML input
6. Outputs `.html`, then optionally packages to `.htmlz` / `.epub`

This keeps EPUB and HTML review/packaging workflows consistent.

## HTML asset and HTMLZ compatibility optimizations

- Output file names are sanitized for archive compatibility
- Normal HTML output auto-copies local assets to `assets/<book_ascii_slug>/...`
- Copied asset names and refs are rewritten to ASCII-safe paths (with hash suffix)
- During packaging, local `img src` can be inlined again to reduce broken images
- Output HTML injects `<title>` and `dc.*` metadata for Calibre recognition

## Runtime pipeline

1. API preflight check (disable with `--skip-api-check`)
2. Input read + tokenize + segment preparation
3. Chunked model translation calls
4. Runtime progress logging (`Prepare`, `Chunk Start`, `Chunk API`, `Chunk OK`)
5. Retry on failure, with adaptive chunk downgrade if needed
6. Checkpoint save/resume handling
7. Final output write + optional post-package

## Common flags

- `--config`: config path (default `config.json`)
- `--prompt`: prompt path (default `prompt.txt`)
- `--suffix`: output suffix (default `_CN`)
- `--skip-existing`: skip existing outputs
- `--output-style bilingual|translated`
- `--html-translation-style blockquote|paragraph|details`
- `--post-package none|htmlz|epubv3|both`
- `--no-resume`
- `--no-realtime-write`
- `--skip-api-check`
- `--api-check-only`

## Resume and stability notes

- Resume is enabled by default via `*.resume.json` near output file
- Resume restores progress/context only (chunk policy uses current config)
- If `reasoning` is unsupported by provider, the script auto-falls back
- API responses are checked for suspected untranslated segments:
  default mode is `exact_only` (same text after whitespace/punctuation normalization);
  optional similarity fallback can be enabled, and name-like short segments are exempted
- Chunk-level echo detection is enabled: if most items in one API batch are returned as original text,
  the whole chunk is retried/downgraded immediately

## Recommended config baseline (`config.example.json`)

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

## Acknowledgements

- GPT-5.3-Codex
