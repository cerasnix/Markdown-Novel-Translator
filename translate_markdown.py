import argparse
import html
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment dependent
    OpenAI = None


HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
HORIZONTAL_RULE_RE = re.compile(r"^\s{0,3}(?:([-*_])\s*){3,}$")
LIST_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+")
BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?")
REFERENCE_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s+\S+")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
HTML_TAG_ONLY_RE = re.compile(r"^\s*</?[a-zA-Z][^>]*>\s*$")
HTML_INLINE_TEXT_RE = re.compile(
    r"^(?P<indent>\s*)(?P<open><(?P<tag>[a-zA-Z][\w:-]*)(?:\s+[^>]*)?>)"
    r"(?P<inner>.*?)(?P<close></(?P=tag)>\s*)$",
    re.DOTALL,
)
HTML_LINE_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPLIT_POINT_RE = re.compile(r"(?:[。！？!?；;](?:[」』”’\"\)\]]*\s*))|(?:\n+)")
HTML_TRANSLATABLE_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "figcaption"}


@dataclass
class Token:
    content: str
    translatable: bool
    source_format: str = "markdown"
    original_content: Optional[str] = None
    html_indent: str = ""


@dataclass
class PreparedSegment:
    token_index: int
    text: str


class MarkdownNovelTranslator:
    def __init__(
        self,
        config_path: str = "config.json",
        prompt_path: str = "prompt_markdown.txt",
        reasoning_effort: Optional[str] = None,
    ):
        if OpenAI is None:
            raise RuntimeError("Missing dependency: openai. Install with `pip install openai`.")

        self.config = self._load_json(config_path)
        self.system_prompt = self._load_text(prompt_path)
        self.reasoning = self._resolve_reasoning(reasoning_effort)
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"],
        )

    def _load_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _print(self, title: str, content: str, color: str = "cyan") -> None:
        colors = {
            "cyan": "\033[96m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "end": "\033[0m",
        }
        c = colors.get(color, colors["end"])
        print(f"\n{c}>>> {title} <<<{colors['end']}")
        print(content)
        print(f"{c}{'-' * 50}{colors['end']}")

    def _resolve_reasoning(self, override_effort: Optional[str]) -> dict:
        effort = None
        if override_effort:
            effort = override_effort
        else:
            cfg_reasoning = self.config.get("reasoning")
            if isinstance(cfg_reasoning, dict):
                effort = cfg_reasoning.get("effort")
            if effort is None:
                effort = self.config.get("reasoning_effort")
        effort = self._normalize_effort(effort)
        return {"effort": effort}

    def _normalize_effort(self, effort) -> str:
        if not isinstance(effort, str):
            return "low"
        value = effort.strip().lower()
        if value not in {"low", "medium", "high"}:
            return "low"
        return value

    def _print_progress(self, done: int, total: int, started_at: float) -> None:
        if total <= 0:
            return
        ratio = min(1.0, max(0.0, done / total))
        width = 30
        filled = int(width * ratio)
        bar = ("#" * filled) + ("-" * (width - filled))
        elapsed = max(0.001, time.time() - started_at)
        speed = done / elapsed
        eta = (total - done) / speed if speed > 0 else 0.0
        print(
            f"\rProgress [{bar}] {done}/{total} ({ratio * 100:5.1f}%) | "
            f"{speed:4.2f} seg/s | ETA {eta:6.1f}s",
            end="",
            flush=True,
        )
        if done >= total:
            print()

    def _match_fence(self, line: str):
        return re.match(r"^[ \t]*(`{3,}|~{3,})", line)

    def _is_heading_line(self, line: str) -> bool:
        return bool(HEADING_RE.match(line))

    def _is_horizontal_rule(self, line: str) -> bool:
        return bool(HORIZONTAL_RULE_RE.match(line))

    def _is_list_or_quote_line(self, line: str) -> bool:
        return bool(LIST_RE.match(line) or BLOCKQUOTE_RE.match(line))

    def _is_list_continuation(self, line: str) -> bool:
        if line.strip() == "":
            return False
        return bool(re.match(r"^\s{2,}\S", line))

    def _is_reference_definition(self, line: str) -> bool:
        return bool(REFERENCE_DEF_RE.match(line))

    def _is_table_separator(self, line: str) -> bool:
        return bool(TABLE_SEPARATOR_RE.match(line))

    def _is_html_tag_only(self, line: str) -> bool:
        return bool(HTML_TAG_ONLY_RE.match(line))

    def _collect_list_or_quote_block(self, lines: List[str], start_idx: int):
        block_lines = []
        idx = start_idx
        while idx < len(lines):
            line = lines[idx]
            if line.strip() == "":
                break
            if self._is_list_or_quote_line(line) or self._is_list_continuation(line):
                block_lines.append(line)
                idx += 1
                continue
            break
        return "".join(block_lines), idx

    def tokenize_markdown(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        lines = text.splitlines(keepends=True)
        if not lines:
            return [Token(content="", translatable=False)]

        idx = 0

        # Preserve YAML front matter as-is.
        if lines[0].strip() == "---":
            fm_end = None
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    fm_end = i
                    break
            if fm_end is not None:
                tokens.append(Token(content="".join(lines[: fm_end + 1]), translatable=False))
                idx = fm_end + 1

        in_fence = False
        fence_char = ""
        fence_len = 0
        paragraph_buffer: List[str] = []
        fence_buffer: List[str] = []

        def flush_paragraph():
            nonlocal paragraph_buffer
            if paragraph_buffer:
                tokens.append(Token(content="".join(paragraph_buffer), translatable=True))
                paragraph_buffer = []

        def flush_fence():
            nonlocal fence_buffer
            if fence_buffer:
                tokens.append(Token(content="".join(fence_buffer), translatable=False))
                fence_buffer = []

        while idx < len(lines):
            line = lines[idx]
            stripped = line.strip()

            if in_fence:
                fence_buffer.append(line)
                candidate = line.lstrip()
                if candidate.startswith(fence_char * fence_len):
                    in_fence = False
                    fence_char = ""
                    fence_len = 0
                    flush_fence()
                idx += 1
                continue

            fence_match = self._match_fence(line)
            if fence_match:
                flush_paragraph()
                fence_token = fence_match.group(1)
                fence_char = fence_token[0]
                fence_len = len(fence_token)
                in_fence = True
                fence_buffer.append(line)
                idx += 1
                continue

            if stripped == "":
                flush_paragraph()
                tokens.append(Token(content=line, translatable=False))
                idx += 1
                continue

            if self._is_horizontal_rule(line) or self._is_table_separator(line):
                flush_paragraph()
                tokens.append(Token(content=line, translatable=False))
                idx += 1
                continue

            if self._is_reference_definition(line) or self._is_html_tag_only(line):
                flush_paragraph()
                tokens.append(Token(content=line, translatable=False))
                idx += 1
                continue

            if self._is_heading_line(line):
                flush_paragraph()
                tokens.append(Token(content=line, translatable=True))
                idx += 1
                continue

            if self._is_list_or_quote_line(line):
                flush_paragraph()
                block_text, next_idx = self._collect_list_or_quote_block(lines, idx)
                if block_text:
                    tokens.append(Token(content=block_text, translatable=True))
                idx = next_idx
                continue

            paragraph_buffer.append(line)
            idx += 1

        flush_paragraph()
        flush_fence()
        return tokens

    def _extract_plain_text_from_html(self, html_fragment: str) -> str:
        normalized = HTML_LINE_BREAK_RE.sub("\n", html_fragment)
        normalized = HTML_TAG_RE.sub("", normalized)
        return html.unescape(normalized).strip()

    def _tokenize_html_line(self, line: str) -> Token:
        line_without_newline = line.rstrip("\n")
        match = HTML_INLINE_TEXT_RE.match(line_without_newline)
        if not match:
            return Token(content=line, translatable=False, source_format="html")

        tag = match.group("tag").lower()
        inner_html = match.group("inner")
        plain_text = self._extract_plain_text_from_html(inner_html)
        if tag not in HTML_TRANSLATABLE_TAGS or not plain_text:
            return Token(content=line, translatable=False, source_format="html")

        return Token(
            content=plain_text,
            translatable=True,
            source_format="html",
            original_content=line,
            html_indent=match.group("indent"),
        )

    def tokenize_html(self, text: str) -> List[Token]:
        lines = text.splitlines(keepends=True)
        if not lines:
            return [Token(content="", translatable=False, source_format="html")]
        return [self._tokenize_html_line(line) for line in lines]

    def _split_oversized_segment(self, text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]

        split_points = {match.end() for match in SPLIT_POINT_RE.finditer(text)}
        split_points.add(len(text))
        ordered_points = sorted(split_points)

        parts: List[str] = []
        start = 0
        min_size = max(220, max_chars // 3)

        while start < len(text):
            hard_limit = min(len(text), start + max_chars)
            chosen = None

            for point in reversed(ordered_points):
                if start < point <= hard_limit and (point - start) >= min_size:
                    chosen = point
                    break

            if chosen is None:
                extend_limit = min(len(text), hard_limit + max_chars // 4)
                for point in ordered_points:
                    if hard_limit < point <= extend_limit:
                        chosen = point
                        break

            if chosen is None or chosen <= start:
                chosen = hard_limit

            parts.append(text[start:chosen])
            start = chosen

        return parts

    def prepare_segments(self, tokens: List[Token]) -> List[PreparedSegment]:
        max_segment_chars = max(300, int(self.config.get("max_segment_chars", 900)))
        prepared: List[PreparedSegment] = []

        for token_index, token in enumerate(tokens):
            if not token.translatable:
                continue
            parts = self._split_oversized_segment(token.content, max_segment_chars)
            for piece in parts:
                prepared.append(PreparedSegment(token_index=token_index, text=piece))
        return prepared

    def _next_batch_end(
        self,
        start_idx: int,
        segments: List[PreparedSegment],
        min_segments: int,
        target_chars: int,
        max_chars: int,
        max_segments: int,
    ) -> int:
        end = start_idx
        char_count = 0
        while end < len(segments) and (end - start_idx) < max_segments:
            seg_len = len(segments[end].text)
            projected = char_count + seg_len
            if end > start_idx and projected > max_chars:
                break
            char_count = projected
            end += 1
            seg_count = end - start_idx
            if seg_count >= min_segments and char_count >= target_chars:
                break
            if char_count >= max_chars:
                break
        return max(start_idx + 1, end)

    def _normalize_pages_to_segments(self, pages, expected_count: int) -> Optional[List[str]]:
        if not isinstance(pages, list):
            return None

        normalized: List[str] = []
        for item in pages:
            if isinstance(item, str):
                normalized.append(item)
                continue
            if isinstance(item, list):
                parts = [str(x) for x in item]
                normalized.append("\n".join(parts))
                continue
            return None

        if len(normalized) != expected_count:
            return None
        return normalized

    def _extract_translated_segments(self, data: dict, expected_count: int) -> Optional[List[str]]:
        segments = data.get("segments")
        if isinstance(segments, list) and len(segments) == expected_count:
            return [str(x) for x in segments]

        pages = data.get("pages")
        normalized = self._normalize_pages_to_segments(pages, expected_count)
        if normalized is not None:
            return normalized
        return None

    def call_api(
        self,
        segment_texts: List[str],
        last_summary: str,
        recent_tail: List[str],
        need_summary_update: bool,
    ):
        target_language = self.config.get("target_language", "Simplified Chinese")
        # Keep compatibility with both markdown-novel prompts (segments) and
        # legacy mokuro prompts (pages + previous_context).
        input_pages = [[seg] for seg in segment_texts]
        user_payload = {
            "target_language": target_language,
            "previous_context": last_summary,
            "previous_context_summary": last_summary,
            "recent_translated_tail": recent_tail,
            "input_pages": input_pages,
            "input_segments": segment_texts,
            "must_keep_segment_count": len(segment_texts),
            "novel_mode": True,
            "need_summary_update": need_summary_update,
        }

        extra_body = {}
        cfg_extra_body = self.config.get("extra_body")
        if isinstance(cfg_extra_body, dict):
            extra_body.update(cfg_extra_body)

        # Default and configurable reasoning effort.
        # Sent in request body as: "reasoning": { "effort": "low|medium|high" }.
        extra_body["reasoning"] = self.reasoning

        try:
            request_timeout = float(self.config.get("request_timeout_seconds", 60))
            response = self.client.chat.completions.create(
                model=self.config.get("model_name", "gemini-3-flash-preview"),
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=float(self.config.get("temperature", 0.3)),
                extra_body=extra_body,
                timeout=request_timeout,
            )

            if not response.choices:
                return None, ""

            content = response.choices[0].message.content
            if not content:
                return None, last_summary

            data = json.loads(content)
            translated = self._extract_translated_segments(data, len(segment_texts))
            if translated is None:
                self._print(
                    "ParseFail",
                    f"unexpected response keys={list(data.keys())[:8]}",
                    "yellow",
                )
                return None, ""
            new_summary = str(data.get("new_summary", last_summary))
            return translated, new_summary
        except Exception as e:
            self._print("APIErr", str(e), "yellow")
            return None, ""

    def translate_chunk_with_retry(
        self,
        segment_texts: List[str],
        last_summary: str,
        recent_tail: List[str],
        need_summary_update: bool,
        retries: int = 2,
    ):
        for attempt in range(1, retries + 1):
            if attempt > 1:
                self._print(
                    "Retry",
                    f"chunk-size={len(segment_texts)} attempt={attempt}/{retries}",
                    "yellow",
                )
            translated, new_summary = self.call_api(
                segment_texts,
                last_summary,
                recent_tail,
                need_summary_update,
            )
            if translated is not None:
                return translated, new_summary
            if attempt < retries:
                time.sleep(1.2 * attempt)
        return None, ""

    def merge_translations(
        self,
        tokens: List[Token],
        prepared_segments: List[PreparedSegment],
        translated_segments: List[str],
    ) -> Dict[int, str]:
        if len(prepared_segments) != len(translated_segments):
            raise RuntimeError("Prepared segment count mismatch")

        grouped: Dict[int, List[str]] = {}
        for seg, text in zip(prepared_segments, translated_segments):
            grouped.setdefault(seg.token_index, []).append(str(text))

        token_translation_map: Dict[int, str] = {}
        for idx, token in enumerate(tokens):
            if not token.translatable:
                continue
            parts = grouped.get(idx)
            if not parts:
                raise RuntimeError(f"Missing translated parts for token index {idx}")
            token_translation_map[idx] = "".join(parts)
        return token_translation_map

    def reconstruct(self, tokens: List[Token], token_translation_map: Dict[int, str]) -> str:
        output = []
        for idx, token in enumerate(tokens):
            if token.translatable:
                if idx not in token_translation_map:
                    raise RuntimeError(f"Missing translation for token index {idx}")
                output.append(token_translation_map[idx])
            else:
                output.append(token.content)
        return "".join(output)

    def _format_markdown_bilingual_pair(
        self,
        original_text: str,
        translated_text: str,
    ) -> str:
        def to_markdown_quote(text: str) -> str:
            lines = text.rstrip("\n").splitlines()
            if not lines:
                return "> "
            return "\n".join(f"> {line}" if line.strip() else ">" for line in lines)

        original = original_text.rstrip("\n")
        translated_quote = to_markdown_quote(translated_text)
        return (
            f"{original}\n\n"
            f"{translated_quote}\n\n"
        )

    def _format_html_translation_block(
        self,
        translated_text: str,
        indent: str,
        html_translation_style: str,
    ) -> str:
        plain_text = self._extract_plain_text_from_html(translated_text)
        lines = [line.strip() for line in plain_text.splitlines() if line.strip()]
        if not lines:
            fallback = translated_text.strip()
            lines = [fallback] if fallback else [""]

        if html_translation_style == "paragraph":
            return "".join(
                f'{indent}<p class="cn-translation" lang="zh-CN" '
                f'style="margin:0.2em 0 0.85em 1.2em;color:#555;">{html.escape(line, quote=False)}</p>\n'
                for line in lines
            )

        if html_translation_style == "details":
            detail_lines = [
                (
                    f'{indent}<details class="cn-translation" '
                    'style="margin:0.3em 0 0.9em 1.2em;color:#555;">\n'
                ),
                (
                    f'{indent}  <summary style="cursor:default;list-style:none;">'
                    "译文</summary>\n"
                ),
            ]
            detail_lines.extend(
                f"{indent}  <p lang=\"zh-CN\">{html.escape(line, quote=False)}</p>\n"
                for line in lines
            )
            detail_lines.append(f"{indent}</details>\n")
            return "".join(detail_lines)

        quote_lines = [
            (
                f'{indent}<blockquote class="cn-translation" lang="zh-CN" '
                'style="margin:0.35em 0 0.85em 1.2em;padding-left:0.75em;'
                'border-left:0.18em solid #9aa0a6;color:#555;">\n'
            )
        ]
        quote_lines.extend(f"{indent}  <p>{html.escape(line, quote=False)}</p>\n" for line in lines)
        quote_lines.append(f"{indent}</blockquote>\n")
        return "".join(quote_lines)

    def _format_html_bilingual_pair(
        self,
        original_text: str,
        translated_text: str,
        indent: str,
        html_translation_style: str,
    ) -> str:
        original = original_text if original_text.endswith("\n") else (original_text + "\n")
        return original + self._format_html_translation_block(
            translated_text=translated_text,
            indent=indent,
            html_translation_style=html_translation_style,
        )

    def _format_html_translated_line(self, original_text: str, translated_text: str) -> str:
        line_without_newline = original_text.rstrip("\n")
        newline = "\n" if original_text.endswith("\n") else ""
        match = HTML_INLINE_TEXT_RE.match(line_without_newline)
        if not match:
            translated = self._extract_plain_text_from_html(translated_text) or translated_text.strip()
            return (translated + newline) if translated else newline

        translated_plain = self._extract_plain_text_from_html(translated_text)
        escaped = html.escape(translated_plain, quote=False)
        return (
            f"{match.group('indent')}{match.group('open')}"
            f"{escaped}"
            f"{match.group('close')}{newline}"
        )

    def _render_translatable_token(
        self,
        token: Token,
        translated_text: str,
        output_style: str,
        html_translation_style: str,
    ) -> str:
        if token.source_format == "html":
            original = token.original_content or token.content
            if output_style == "bilingual":
                return self._format_html_bilingual_pair(
                    original_text=original,
                    translated_text=translated_text,
                    indent=token.html_indent,
                    html_translation_style=html_translation_style,
                )
            return self._format_html_translated_line(original, translated_text)

        if output_style == "bilingual":
            return self._format_markdown_bilingual_pair(token.content, translated_text)
        return translated_text

    def _format_nontranslatable_bilingual(self, content: str) -> str:
        # Keep meaningful non-translatable blocks (e.g. front matter/code fences) once.
        if content.strip() == "":
            return ""
        return content if content.endswith("\n") else (content + "\n")

    def _render_nontranslatable_token(self, token: Token, output_style: str) -> str:
        if output_style != "bilingual":
            return token.content
        if token.source_format == "html":
            return token.content
        return self._format_nontranslatable_bilingual(token.content)

    def _flush_ready_output(
        self,
        tokens: List[Token],
        token_translation_map: Dict[int, str],
        next_token_index: int,
        output_path: Path,
        output_style: str,
        html_translation_style: str,
    ) -> int:
        pieces: List[str] = []
        while next_token_index < len(tokens):
            token = tokens[next_token_index]
            if token.translatable:
                translated = token_translation_map.get(next_token_index)
                if translated is None:
                    break
                pieces.append(
                    self._render_translatable_token(
                        token=token,
                        translated_text=translated,
                        output_style=output_style,
                        html_translation_style=html_translation_style,
                    )
                )
                next_token_index += 1
                continue

            pieces.append(self._render_nontranslatable_token(token, output_style))
            next_token_index += 1

        if pieces:
            with output_path.open("a", encoding="utf-8") as fp:
                fp.write("".join(pieces))
        return next_token_index

    def process_file(
        self,
        input_file: Path,
        output_file: Path,
        source_format: str,
        output_style: str = "bilingual",
        html_translation_style: str = "blockquote",
        realtime_write: bool = True,
    ):
        text = input_file.read_text(encoding="utf-8")
        if source_format == "html":
            tokens = self.tokenize_html(text)
        else:
            tokens = self.tokenize_markdown(text)
        prepared_segments = self.prepare_segments(tokens)
        total = len(prepared_segments)

        if total == 0:
            output_file.write_text(text, encoding="utf-8")
            self._print("Skip", f"{input_file.name}: no translatable segments found", "yellow")
            return True

        default_min_chunk_segments = max(1, int(self.config.get("chunk_size", 3)))
        default_max_chunk_segments = max(
            default_min_chunk_segments,
            int(self.config.get("max_chunk_segments", 20)),
        )
        default_target_chunk_chars = max(200, int(self.config.get("target_chunk_chars", 1000)))
        default_char_limit = max(1200, int(self.config.get("max_chunk_chars", 2600)))
        context_tail_segments = max(0, int(self.config.get("context_tail_segments", 3)))
        summary_interval_batches = max(1, int(self.config.get("summary_interval_batches", 8)))
        summary_interval_chars = max(0, int(self.config.get("summary_interval_chars", 8000)))

        summary = "Context initialized."
        i = 0
        done_segments = 0
        started_at = time.time()
        translated_all: List[str] = []
        current_min_chunk_segments = default_min_chunk_segments
        current_max_chunk_segments = default_max_chunk_segments
        current_target_chunk_chars = default_target_chunk_chars
        current_char_limit = default_char_limit
        batches_since_summary = 0
        chars_since_summary = 0

        token_part_total = Counter(seg.token_index for seg in prepared_segments)
        token_part_translated: Dict[int, List[str]] = {}
        completed_token_map: Dict[int, str] = {}
        next_token_to_write = 0

        if realtime_write:
            if output_style == "bilingual":
                if source_format == "html":
                    output_file.write_text("", encoding="utf-8")
                else:
                    output_file.write_text(
                        f"# 双语对照预览：{input_file.stem}\n\n"
                        "> 原文在上，译文在下；脚本会按分段实时写入。\n\n",
                        encoding="utf-8",
                    )
            else:
                output_file.write_text("", encoding="utf-8")

        while i < total:
            end = self._next_batch_end(
                start_idx=i,
                segments=prepared_segments,
                min_segments=current_min_chunk_segments,
                target_chars=current_target_chunk_chars,
                max_chars=current_char_limit,
                max_segments=current_max_chunk_segments,
            )
            chunk_texts = [seg.text for seg in prepared_segments[i:end]]
            recent_tail = translated_all[-context_tail_segments:] if context_tail_segments > 0 else []
            batch_chars = sum(len(x) for x in chunk_texts)
            need_summary_update = (
                i == 0
                or batches_since_summary >= summary_interval_batches
                or (summary_interval_chars > 0 and chars_since_summary >= summary_interval_chars)
            )

            translated, new_summary = self.translate_chunk_with_retry(
                segment_texts=chunk_texts,
                last_summary=summary,
                recent_tail=recent_tail,
                need_summary_update=need_summary_update,
                retries=int(self.config.get("retry_count", 2)),
            )

            if translated is not None:
                translated_all.extend(translated)
                for local_idx, translated_text in enumerate(translated):
                    prepared_idx = i + local_idx
                    token_idx = prepared_segments[prepared_idx].token_index
                    token_part_translated.setdefault(token_idx, []).append(str(translated_text))
                    if len(token_part_translated[token_idx]) == token_part_total[token_idx]:
                        completed_token_map[token_idx] = "".join(token_part_translated[token_idx])
                        del token_part_translated[token_idx]
                    done_segments += 1
                    self._print_progress(done_segments, total, started_at)

                if realtime_write:
                    next_token_to_write = self._flush_ready_output(
                        tokens=tokens,
                        token_translation_map=completed_token_map,
                        next_token_index=next_token_to_write,
                        output_path=output_file,
                        output_style=output_style,
                        html_translation_style=html_translation_style,
                    )

                self._print(
                    "Chunk OK",
                    (
                        f"{input_file.name}: segments {i + 1}-{end}/{total} | "
                        f"batch={end - i}, chars~{batch_chars}, "
                        f"summary={'Y' if need_summary_update else 'N'}"
                    ),
                    "green",
                )
                batches_since_summary += 1
                chars_since_summary += batch_chars
                if need_summary_update and new_summary.strip():
                    summary = new_summary.strip()
                    batches_since_summary = 0
                    chars_since_summary = 0
                i = end
                current_min_chunk_segments = default_min_chunk_segments
                current_max_chunk_segments = default_max_chunk_segments
                current_target_chunk_chars = default_target_chunk_chars
                current_char_limit = default_char_limit
                continue

            if current_min_chunk_segments == 1 and end == i + 1:
                self._print(
                    "Abort",
                    f"{input_file.name}: failed at segment {i + 1} after retries",
                    "red",
                )
                return False

            new_min_chunk_segments = max(1, current_min_chunk_segments // 2)
            new_max_chunk_segments = max(new_min_chunk_segments, current_max_chunk_segments // 2)
            new_target_chunk_chars = max(200, current_target_chunk_chars // 2)
            new_char_limit = max(1200, current_char_limit // 2)
            self._print(
                "Downgrade",
                (
                    f"{input_file.name}: min_chunk {current_min_chunk_segments}->{new_min_chunk_segments}, "
                    f"max_chunk {current_max_chunk_segments}->{new_max_chunk_segments}, "
                    f"target_chars {current_target_chunk_chars}->{new_target_chunk_chars}, "
                    f"char_limit {current_char_limit}->{new_char_limit}"
                ),
                "yellow",
            )
            current_min_chunk_segments = new_min_chunk_segments
            current_max_chunk_segments = new_max_chunk_segments
            current_target_chunk_chars = new_target_chunk_chars
            current_char_limit = new_char_limit

        if realtime_write:
            next_token_to_write = self._flush_ready_output(
                tokens=tokens,
                token_translation_map=completed_token_map,
                next_token_index=next_token_to_write,
                output_path=output_file,
                output_style=output_style,
                html_translation_style=html_translation_style,
            )
            if next_token_to_write != len(tokens):
                raise RuntimeError("Realtime writer ended with unresolved tokens")
        else:
            token_translation_map = self.merge_translations(tokens, prepared_segments, translated_all)
            if source_format == "html":
                rebuilt = "".join(
                    self._render_translatable_token(
                        token=tokens[idx],
                        translated_text=token_translation_map[idx],
                        output_style=output_style,
                        html_translation_style=html_translation_style,
                    )
                    if tokens[idx].translatable
                    else self._render_nontranslatable_token(tokens[idx], output_style)
                    for idx in range(len(tokens))
                )
            else:
                rebuilt = self.reconstruct(tokens, token_translation_map)
            output_file.write_text(rebuilt, encoding="utf-8")
        self._print("Done", f"{input_file.name} -> {output_file.name}", "cyan")
        return True

    def run(
        self,
        input_path: str,
        suffix: str = "_CN",
        skip_existing: bool = False,
        output_style: str = "bilingual",
        html_translation_style: str = "blockquote",
        realtime_write: bool = True,
    ):
        def detect_source_format(file_path: Path) -> Optional[str]:
            ext = file_path.suffix.lower()
            if ext in {".md", ".markdown"}:
                return "markdown"
            if ext in {".html", ".htm"}:
                return "html"
            return None

        path = Path(input_path)
        if path.is_file():
            if detect_source_format(path) is None:
                raise ValueError(f"Unsupported file type: {path.suffix}")
            files = [path]
        elif path.is_dir():
            files = sorted(
                p for p in path.rglob("*")
                if p.is_file() and detect_source_format(p) is not None
            )
        else:
            raise FileNotFoundError(f"Path not found: {input_path}")

        if not files:
            print("No markdown/html files found.")
            return

        for input_file in files:
            source_format = detect_source_format(input_file)
            if source_format is None:
                continue
            if input_file.stem.endswith(suffix):
                continue
            output_path = input_file.with_name(f"{input_file.stem}{suffix}{input_file.suffix}")
            if skip_existing and output_path.exists():
                self._print("Skip", f"{output_path.name} already exists", "yellow")
                continue
            try:
                self.process_file(
                    input_file,
                    output_path,
                    source_format=source_format,
                    output_style=output_style,
                    html_translation_style=html_translation_style,
                    realtime_write=realtime_write,
                )
            except Exception as e:
                self._print("Error", f"{input_file.name}: {e}", "red")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate markdown/html novel files with context-aware chunking."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to .md/.markdown/.html file, or folder containing these files",
    )
    parser.add_argument("--suffix", default="_CN", help="Output filename suffix (default: _CN)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files when output already exists")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--prompt", default="prompt_markdown.txt", help="Path to prompt text file")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Override reasoning.effort for requests (default from config or low)",
    )
    parser.add_argument(
        "--output-style",
        choices=["bilingual", "translated"],
        default="bilingual",
        help="Output format: bilingual up/down blocks or translated-only output",
    )
    parser.add_argument(
        "--html-translation-style",
        choices=["blockquote", "paragraph", "details"],
        default="blockquote",
        help="HTML bilingual translation block style (default: blockquote)",
    )
    parser.add_argument(
        "--no-realtime-write",
        action="store_true",
        help="Disable realtime segment writing and write file only after all segments finish",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    if not input_path:
        input_path = (
            input("Drag a markdown/html file/folder here: ")
            .strip()
            .replace("\\ ", " ")
            .strip("'")
            .strip('"')
        )

    translator = MarkdownNovelTranslator(
        config_path=args.config,
        prompt_path=args.prompt,
        reasoning_effort=args.reasoning_effort,
    )
    translator.run(
        input_path=input_path,
        suffix=args.suffix,
        skip_existing=args.skip_existing,
        output_style=args.output_style,
        html_translation_style=args.html_translation_style,
        realtime_write=not args.no_realtime_write,
    )
