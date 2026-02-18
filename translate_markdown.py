import argparse
import base64
import hashlib
import html
import json
import mimetypes
import posixpath
import re
import shutil
import time
import unicodedata
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

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
HTML_BODY_RE = re.compile(r"<body\b[^>]*>(?P<body>[\s\S]*?)</body>", re.IGNORECASE)
HTML_IMG_SRC_RE = re.compile(r"(<img\b[^>]*?\bsrc\s*=\s*)([\"'])([^\"']+)\2", re.IGNORECASE)
HTML_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
HTML_TITLE_RE = re.compile(r"<title\b[^>]*>(?P<title>[\s\S]*?)</title>", re.IGNORECASE)
HTML_ATTR_RE = re.compile(r"([:\w-]+)\s*=\s*([\"'])(.*?)\2")
VOID_TAG_RE = re.compile(
    r"<(?P<tag>area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr)\b(?P<attrs>[^<>]*)>",
    re.IGNORECASE,
)
BARE_AMP_RE = re.compile(r"&(?!(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9._:-]*);)")
SPLIT_POINT_RE = re.compile(r"(?:[。！？!?；;](?:[」』”’\"\)\]]*\s*))|(?:\n+)")
HTML_TRANSLATABLE_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "figcaption"}
BAD_FILENAME_CHARS_RE = re.compile(r"[\\/:*?\"<>|]+")
HTML_ASSET_URL_ATTR_RE = re.compile(
    r'(<(?P<tag>[a-zA-Z][\w:-]*)\b[^>]*?\b(?P<attr>src|href|poster|data)\s*=\s*)'
    r'(?P<quote>["\'])(?P<url>[^"\']+)(?P=quote)',
    re.IGNORECASE,
)
HTML_ASSET_SRCSET_RE = re.compile(
    r'(<(?P<tag>img|source)\b[^>]*?\bsrcset\s*=\s*)(?P<quote>["\'])(?P<srcset>[^"\']+)(?P=quote)',
    re.IGNORECASE,
)
HTML_ASSET_TAG_ATTRS = {
    "img": {"src"},
    "source": {"src"},
    "video": {"src", "poster"},
    "audio": {"src"},
    "script": {"src"},
    "link": {"href"},
    "object": {"data"},
    "embed": {"src"},
}


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
        self.reasoning_supported = True
        self.last_failure_reason = ""
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

    def _resume_state_path(self, output_file: Path) -> Path:
        return output_file.with_name(f"{output_file.name}.resume.json")

    def _resume_fingerprint(
        self,
        source_text: str,
        source_format: str,
        output_style: str,
        html_translation_style: str,
        use_htmlz_wrapper: bool,
    ) -> str:
        payload = {
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "source_format": source_format,
            "output_style": output_style,
            "html_translation_style": html_translation_style,
            "htmlz_wrapper": use_htmlz_wrapper,
        }
        packed = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(packed.encode("utf-8")).hexdigest()

    def _save_resume_state(self, resume_path: Path, state: dict) -> None:
        temp_path = resume_path.with_name(f"{resume_path.name}.tmp")
        with temp_path.open("w", encoding="utf-8") as fp:
            json.dump(state, fp, ensure_ascii=False, indent=2)
        temp_path.replace(resume_path)

    def _load_resume_state(self, resume_path: Path) -> Optional[dict]:
        if not resume_path.exists():
            return None
        try:
            with resume_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            if not isinstance(data, dict):
                return None
            return data
        except Exception:
            return None

    def _sanitize_archive_filename_stem(self, stem: str) -> str:
        cleaned = BAD_FILENAME_CHARS_RE.sub("_", stem.strip())
        cleaned = unicodedata.normalize("NFC", cleaned)
        cleaned = re.sub(r"[【】\[\]\(\)（）「」『』]", "", cleaned)
        cleaned = re.sub(r"[^\w.-]+", "_", cleaned, flags=re.UNICODE)
        cleaned = re.sub(r"_+", "_", cleaned).strip("._ ")
        return cleaned or "translated_novel"

    def _sanitize_ascii_component(
        self,
        value: str,
        fallback: str,
        max_length: int = 56,
    ) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
        ascii_value = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_value)
        ascii_value = re.sub(r"_+", "_", ascii_value).strip("._-")
        if not ascii_value:
            ascii_value = fallback
        ascii_value = ascii_value[:max_length].rstrip("._-")
        return ascii_value or fallback

    def _asset_subdir_by_extension(self, extension: str) -> str:
        ext = extension.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".avif", ".ico"}:
            return "images"
        if ext in {".css"}:
            return "styles"
        if ext in {".js", ".mjs"}:
            return "scripts"
        if ext in {".woff", ".woff2", ".ttf", ".otf"}:
            return "fonts"
        if ext in {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".mp4", ".webm"}:
            return "media"
        return "files"

    def _is_local_asset_url(self, url: str) -> bool:
        raw = (url or "").strip()
        if not raw:
            return False
        lowered = raw.lower()
        if lowered.startswith(
            ("data:", "http:", "https:", "//", "mailto:", "javascript:", "tel:", "#")
        ):
            return False
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", raw):
            return False
        if raw.startswith("/"):
            return False
        return True

    def _rewrite_html_assets_for_output(
        self,
        html_text: str,
        source_base_dir: Path,
        output_html_path: Path,
    ) -> Tuple[str, int, int]:
        output_dir = output_html_path.parent
        book_slug = self._sanitize_ascii_component(output_html_path.stem, "book")
        assets_root_rel = Path("assets") / book_slug
        source_to_dest: Dict[str, Path] = {}
        copied_files = 0
        rewritten_refs = 0

        def build_local_asset_url(raw_url: str) -> str:
            nonlocal copied_files, rewritten_refs
            if not self._is_local_asset_url(raw_url):
                return raw_url

            parsed = urlsplit(raw_url)
            source_rel_path = unquote(parsed.path)
            if not source_rel_path:
                return raw_url

            source_path = (source_base_dir / source_rel_path).resolve()
            if not source_path.exists() or not source_path.is_file():
                return raw_url

            source_key = source_path.as_posix()
            dest_rel = source_to_dest.get(source_key)
            if dest_rel is None:
                suffix_ascii = self._sanitize_ascii_component(source_path.suffix.lstrip("."), "", 12)
                suffix = f".{suffix_ascii.lower()}" if suffix_ascii else ""
                stem = self._sanitize_ascii_component(source_path.stem, "asset")
                digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:10]
                subdir = self._asset_subdir_by_extension(suffix)
                filename = f"{stem}_{digest}{suffix}"
                dest_rel = assets_root_rel / subdir / filename
                dest_path = output_dir / dest_rel
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path != dest_path:
                    shutil.copy2(source_path, dest_path)
                source_to_dest[source_key] = dest_rel
                copied_files += 1

            new_url = dest_rel.as_posix()
            if parsed.query:
                new_url = f"{new_url}?{parsed.query}"
            if parsed.fragment:
                new_url = f"{new_url}#{parsed.fragment}"
            if new_url != raw_url:
                rewritten_refs += 1
            return new_url

        def replace_attr(match: re.Match) -> str:
            tag_name = (match.group("tag") or "").lower()
            attr_name = (match.group("attr") or "").lower()
            allowed_attrs = HTML_ASSET_TAG_ATTRS.get(tag_name)
            if not allowed_attrs or attr_name not in allowed_attrs:
                return match.group(0)
            rewritten = build_local_asset_url(match.group("url"))
            return f"{match.group(1)}{match.group('quote')}{rewritten}{match.group('quote')}"

        def replace_srcset(match: re.Match) -> str:
            srcset_value = match.group("srcset")
            entries = [item.strip() for item in srcset_value.split(",")]
            rewritten_entries: List[str] = []
            for entry in entries:
                if not entry:
                    continue
                parts = entry.split()
                first_url = parts[0]
                parts[0] = build_local_asset_url(first_url)
                rewritten_entries.append(" ".join(parts))
            rewritten_srcset = ", ".join(rewritten_entries)
            return f"{match.group(1)}{match.group('quote')}{rewritten_srcset}{match.group('quote')}"

        rewritten_html = HTML_ASSET_URL_ATTR_RE.sub(replace_attr, html_text)
        rewritten_html = HTML_ASSET_SRCSET_RE.sub(replace_srcset, rewritten_html)
        return rewritten_html, copied_files, rewritten_refs

    def _build_htmlz_document_header(
        self,
        title: str,
        author: Optional[str],
        language: str,
        identifier: Optional[str],
        publisher: Optional[str],
    ) -> str:
        safe_title = html.escape(title, quote=True)
        safe_author = html.escape(author or "", quote=True)
        safe_language = html.escape(language or "zh-CN", quote=True)
        safe_identifier = html.escape(identifier or "", quote=True)
        safe_publisher = html.escape(publisher or "", quote=True)

        lines = [
            "<!DOCTYPE html>\n",
            f'<html lang="{safe_language}">\n',
            "<head>\n",
            '  <meta charset="utf-8"/>\n',
            f"  <title>{safe_title}</title>\n",
            f'  <meta name="title" content="{safe_title}"/>\n',
            f'  <meta name="language" content="{safe_language}"/>\n',
            '  <meta name="generator" content="Markdown Novel Translator Starter"/>\n',
            f'  <meta name="dc.title" content="{safe_title}"/>\n',
            f'  <meta name="dc.language" content="{safe_language}"/>\n',
            (
                "  <style>"
                "body{max-width:46em;margin:0 auto;padding:1.25em;line-height:1.65;"
                "font-family:serif;}img{max-width:100%;height:auto;}blockquote.cn-translation,"
                "p.cn-translation,details.cn-translation{font-size:0.96em;}"
                "</style>\n"
            ),
        ]
        if safe_author:
            lines.append(f'  <meta name="author" content="{safe_author}"/>\n')
            lines.append(f'  <meta name="dc.creator" content="{safe_author}"/>\n')
        if safe_identifier:
            lines.append(f'  <meta name="identifier" content="{safe_identifier}"/>\n')
            lines.append(f'  <meta name="dc.identifier" content="{safe_identifier}"/>\n')
        if safe_publisher:
            lines.append(f'  <meta name="publisher" content="{safe_publisher}"/>\n')
            lines.append(f'  <meta name="dc.publisher" content="{safe_publisher}"/>\n')

        lines.extend(["</head>\n", "<body>\n"])
        return "".join(lines)

    def _build_htmlz_document_footer(self) -> str:
        return "</body>\n</html>\n"

    def _parse_html_meta_tags(self, html_text: str) -> Dict[str, str]:
        meta_values: Dict[str, str] = {}
        for meta_tag in HTML_META_TAG_RE.findall(html_text):
            attrs = {key.lower(): value for key, _, value in HTML_ATTR_RE.findall(meta_tag)}
            name = attrs.get("name", "").strip().lower()
            content = attrs.get("content", "").strip()
            if name and content and name not in meta_values:
                meta_values[name] = html.unescape(content)
        return meta_values

    def _extract_html_metadata(self, html_text: str, fallback_title: str) -> Dict[str, str]:
        title_match = HTML_TITLE_RE.search(html_text)
        if title_match:
            raw_title = HTML_TAG_RE.sub("", title_match.group("title")).strip()
            title = html.unescape(raw_title) if raw_title else fallback_title
        else:
            title = fallback_title

        meta_values = self._parse_html_meta_tags(html_text)
        author = meta_values.get("author") or meta_values.get("dc.creator") or ""
        language = meta_values.get("language") or meta_values.get("dc.language") or "zh-CN"
        identifier = meta_values.get("identifier") or meta_values.get("dc.identifier") or ""
        publisher = meta_values.get("publisher") or meta_values.get("dc.publisher") or ""
        return {
            "title": title or fallback_title,
            "author": author,
            "language": language or "zh-CN",
            "identifier": identifier,
            "publisher": publisher,
        }

    def _inline_local_image_sources(self, html_text: str, base_dir: Path) -> str:
        def replace_image(match: re.Match) -> str:
            prefix, quote, src_value = match.groups()
            lower_src = src_value.lower()
            if lower_src.startswith(("data:", "http:", "https:", "//", "#")):
                return match.group(0)

            src_path = src_value.split("#", 1)[0]
            file_path = (base_dir / src_path).resolve()
            if not file_path.exists() or not file_path.is_file():
                return match.group(0)

            mime_type = mimetypes.guess_type(file_path.as_posix())[0] or "application/octet-stream"
            payload = file_path.read_bytes()
            data_uri = f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"
            return f"{prefix}{quote}{data_uri}{quote}"

        return HTML_IMG_SRC_RE.sub(replace_image, html_text)

    def _normalize_html_to_xhtml_fragment(self, fragment: str) -> str:
        def close_void_tag(match: re.Match) -> str:
            tag = match.group("tag")
            attrs = match.group("attrs")
            if attrs.strip().endswith("/"):
                return match.group(0)
            return f"<{tag}{attrs} />"

        normalized = VOID_TAG_RE.sub(close_void_tag, fragment)
        normalized = BARE_AMP_RE.sub("&amp;", normalized)
        return normalized

    def _build_htmlz_metadata_opf(self, metadata: Dict[str, str]) -> str:
        title = html.escape(metadata.get("title") or "Untitled", quote=False)
        language = html.escape(metadata.get("language") or "zh-CN", quote=False)
        author = html.escape(metadata.get("author") or "", quote=False)
        identifier = html.escape(metadata.get("identifier") or f"urn:uuid:{uuid.uuid4()}", quote=False)
        publisher = html.escape(metadata.get("publisher") or "", quote=False)
        creator_line = f"    <dc:creator>{author}</dc:creator>\n" if author else ""
        publisher_line = f"    <dc:publisher>{publisher}</dc:publisher>\n" if publisher else ""
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<package version="2.0" unique-identifier="bookid" xmlns="http://www.idpf.org/2007/opf">\n'
            "  <metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n"
            f"    <dc:title>{title}</dc:title>\n"
            f"    <dc:language>{language}</dc:language>\n"
            f"    <dc:identifier id=\"bookid\">{identifier}</dc:identifier>\n"
            f"{creator_line}"
            f"{publisher_line}"
            "  </metadata>\n"
            "  <manifest>\n"
            '    <item id="index" href="index.html" media-type="text/html"/>\n'
            "  </manifest>\n"
            "  <spine toc=\"ncx\">\n"
            '    <itemref idref="index"/>\n'
            "  </spine>\n"
            "</package>\n"
        )

    def _package_htmlz(self, html_output_path: Path, metadata: Dict[str, str], html_text: str) -> Path:
        htmlz_path = html_output_path.with_suffix(".htmlz")
        opf_content = self._build_htmlz_metadata_opf(metadata)
        with zipfile.ZipFile(htmlz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("index.html", html_text)
            zf.writestr("metadata.opf", opf_content)
        return htmlz_path

    def _build_epub_nav_xhtml(self, title: str) -> str:
        safe_title = html.escape(title, quote=False)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="zh-CN">\n'
            "<head>\n"
            f"  <title>{safe_title} - Navigation</title>\n"
            '  <meta charset="utf-8"/>\n'
            "</head>\n"
            "<body>\n"
            '  <nav epub:type="toc" id="toc">\n'
            f"    <h1>{safe_title}</h1>\n"
            "    <ol>\n"
            '      <li><a href="text/chapter.xhtml">正文</a></li>\n'
            "    </ol>\n"
            "  </nav>\n"
            "</body>\n"
            "</html>\n"
        )

    def _build_epub_chapter_xhtml(self, title: str, language: str, html_text: str) -> str:
        body_match = HTML_BODY_RE.search(html_text)
        body_content = body_match.group("body").strip() if body_match else html_text.strip()
        body_content = self._normalize_html_to_xhtml_fragment(body_content)
        safe_title = html.escape(title, quote=False)
        safe_language = html.escape(language or "zh-CN", quote=True)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<html xmlns="http://www.w3.org/1999/xhtml" lang="{safe_language}">\n'
            "<head>\n"
            '  <meta charset="utf-8"/>\n'
            f"  <title>{safe_title}</title>\n"
            "</head>\n"
            "<body>\n"
            f"{body_content}\n"
            "</body>\n"
            "</html>\n"
        )

    def _build_epub_package_opf(self, metadata: Dict[str, str]) -> str:
        title = html.escape(metadata.get("title") or "Untitled", quote=False)
        language = html.escape(metadata.get("language") or "zh-CN", quote=False)
        author = html.escape(metadata.get("author") or "", quote=False)
        identifier = html.escape(metadata.get("identifier") or f"urn:uuid:{uuid.uuid4()}", quote=False)
        publisher = html.escape(metadata.get("publisher") or "", quote=False)
        creator_line = f"    <dc:creator>{author}</dc:creator>\n" if author else ""
        publisher_line = f"    <dc:publisher>{publisher}</dc:publisher>\n" if publisher else ""
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">\n'
            "  <metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n"
            f"    <dc:identifier id=\"bookid\">{identifier}</dc:identifier>\n"
            f"    <dc:title>{title}</dc:title>\n"
            f"    <dc:language>{language}</dc:language>\n"
            f"{creator_line}"
            f"{publisher_line}"
            '    <meta property="dcterms:modified">2000-01-01T00:00:00Z</meta>\n'
            "  </metadata>\n"
            "  <manifest>\n"
            '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>\n'
            '    <item id="chapter" href="text/chapter.xhtml" media-type="application/xhtml+xml"/>\n'
            "  </manifest>\n"
            "  <spine>\n"
            '    <itemref idref="chapter"/>\n'
            "  </spine>\n"
            "</package>\n"
        )

    def _package_epubv3(self, html_output_path: Path, metadata: Dict[str, str], html_text: str) -> Path:
        epub_path = html_output_path.with_suffix(".epub")
        chapter_xhtml = self._build_epub_chapter_xhtml(
            title=metadata.get("title") or html_output_path.stem,
            language=metadata.get("language") or "zh-CN",
            html_text=html_text,
        )
        nav_xhtml = self._build_epub_nav_xhtml(metadata.get("title") or html_output_path.stem)
        package_opf = self._build_epub_package_opf(metadata)
        container_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
            "  <rootfiles>\n"
            '    <rootfile full-path="OEBPS/package.opf" media-type="application/oebps-package+xml"/>\n'
            "  </rootfiles>\n"
            "</container>\n"
        )

        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
            zf.writestr("META-INF/container.xml", container_xml, compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr("OEBPS/nav.xhtml", nav_xhtml, compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr("OEBPS/text/chapter.xhtml", chapter_xhtml, compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr("OEBPS/package.opf", package_opf, compress_type=zipfile.ZIP_DEFLATED)
        return epub_path

    def _package_after_html_output(self, output_html_path: Path, package_mode: str) -> List[Path]:
        if package_mode == "none":
            return []
        html_text = output_html_path.read_text(encoding="utf-8")
        html_text = self._inline_local_image_sources(html_text, output_html_path.parent)
        metadata = self._extract_html_metadata(html_text, fallback_title=output_html_path.stem)
        generated: List[Path] = []
        if package_mode in {"htmlz", "both"}:
            generated.append(self._package_htmlz(output_html_path, metadata, html_text))
        if package_mode in {"epubv3", "both"}:
            generated.append(self._package_epubv3(output_html_path, metadata, html_text))
        return generated

    def _resolve_epub_path(self, base_dir: str, href: str) -> str:
        normalized = posixpath.normpath(posixpath.join(base_dir, href))
        return normalized.lstrip("/")

    def _epub_local_name(self, tag_name: str) -> str:
        if "}" in tag_name:
            return tag_name.split("}", 1)[1]
        return tag_name

    def _extract_epub_metadata(self, opf_root: ET.Element) -> Dict[str, Optional[str]]:
        metadata_node = opf_root.find(".//{*}metadata")
        if metadata_node is None:
            return {
                "title": None,
                "author": None,
                "language": None,
                "identifier": None,
                "publisher": None,
            }

        values: Dict[str, Optional[str]] = {
            "title": None,
            "author": None,
            "language": None,
            "identifier": None,
            "publisher": None,
        }
        key_map = {
            "title": "title",
            "creator": "author",
            "language": "language",
            "identifier": "identifier",
            "publisher": "publisher",
        }

        for child in list(metadata_node):
            local_name = self._epub_local_name(child.tag).lower()
            mapped_key = key_map.get(local_name)
            if not mapped_key or values[mapped_key] is not None:
                continue
            text = (child.text or "").strip()
            if text:
                values[mapped_key] = text
        return values

    def _decode_epub_text(self, raw_bytes: bytes) -> str:
        for encoding in ("utf-8", "utf-8-sig", "utf-16", "cp932"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="ignore")

    def _inline_epub_image_src(
        self,
        html_fragment: str,
        chapter_dir: str,
        epub_file: zipfile.ZipFile,
        media_type_map: Dict[str, str],
    ) -> str:
        def replace_image(match: re.Match) -> str:
            prefix, quote, src_value = match.groups()
            lower_src = src_value.lower()
            if lower_src.startswith(("data:", "http:", "https:", "//", "#")):
                return match.group(0)

            source_path = src_value.split("#", 1)[0]
            resolved_path = self._resolve_epub_path(chapter_dir, source_path)
            try:
                payload = epub_file.read(resolved_path)
            except KeyError:
                return match.group(0)

            mime_type = media_type_map.get(resolved_path) or mimetypes.guess_type(resolved_path)[0]
            if not mime_type:
                mime_type = "application/octet-stream"
            data_uri = f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"
            return f"{prefix}{quote}{data_uri}{quote}"

        return HTML_IMG_SRC_RE.sub(replace_image, html_fragment)

    def _extract_epub_html_payload(self, input_file: Path) -> Dict[str, object]:
        with zipfile.ZipFile(input_file, "r") as epub_file:
            try:
                container_xml = epub_file.read("META-INF/container.xml")
            except KeyError as exc:
                raise RuntimeError(f"Invalid EPUB (missing container.xml): {input_file.name}") from exc

            container_root = ET.fromstring(container_xml)
            rootfiles = container_root.findall(".//{*}rootfile")
            if not rootfiles:
                raise RuntimeError(f"Invalid EPUB (no rootfile): {input_file.name}")

            opf_path = rootfiles[0].attrib.get("full-path")
            if not opf_path:
                raise RuntimeError(f"Invalid EPUB (rootfile path missing): {input_file.name}")

            try:
                opf_xml = epub_file.read(opf_path)
            except KeyError as exc:
                raise RuntimeError(f"Invalid EPUB (cannot read OPF): {opf_path}") from exc

            opf_root = ET.fromstring(opf_xml)
            metadata = self._extract_epub_metadata(opf_root)
            opf_dir = posixpath.dirname(opf_path)

            manifest_by_id: Dict[str, ET.Element] = {}
            media_type_map: Dict[str, str] = {}
            for item in opf_root.findall(".//{*}manifest/{*}item"):
                item_id = item.attrib.get("id")
                href = item.attrib.get("href")
                if item_id:
                    manifest_by_id[item_id] = item
                if href:
                    resolved = self._resolve_epub_path(opf_dir, href)
                    media_type_map[resolved] = (item.attrib.get("media-type") or "").strip()

            spine_items = [
                itemref.attrib.get("idref")
                for itemref in opf_root.findall(".//{*}spine/{*}itemref")
                if itemref.attrib.get("idref")
            ]

            chapter_fragments: List[str] = []
            for idref in spine_items:
                manifest_item = manifest_by_id.get(idref)
                if manifest_item is None:
                    continue

                href = manifest_item.attrib.get("href")
                media_type = (manifest_item.attrib.get("media-type") or "").lower()
                if not href:
                    continue
                if "html" not in media_type and not href.lower().endswith((".xhtml", ".html", ".htm")):
                    continue

                chapter_path = self._resolve_epub_path(opf_dir, href)
                try:
                    chapter_raw = epub_file.read(chapter_path)
                except KeyError:
                    continue

                chapter_text = self._decode_epub_text(chapter_raw)
                body_match = HTML_BODY_RE.search(chapter_text)
                body_html = body_match.group("body") if body_match else chapter_text
                body_html = body_html.strip()
                if not body_html:
                    continue

                body_html = self._inline_epub_image_src(
                    html_fragment=body_html,
                    chapter_dir=posixpath.dirname(chapter_path),
                    epub_file=epub_file,
                    media_type_map=media_type_map,
                )
                normalized = re.sub(r">\s*<", ">\n<", body_html)
                chapter_name = html.escape(posixpath.basename(chapter_path), quote=True)
                chapter_fragments.append(
                    f'<div class="epub-chapter" data-source="{chapter_name}">\n{normalized}\n</div>\n'
                )

            if not chapter_fragments:
                raise RuntimeError(f"No HTML chapters found in EPUB: {input_file.name}")

            return {
                "html": "\n".join(chapter_fragments),
                "meta": metadata,
            }

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

    def _mask_secret(self, value: str) -> str:
        if not value:
            return "(empty)"
        if len(value) <= 10:
            return f"{value[:2]}***{value[-2:]}"
        return f"{value[:6]}...{value[-4:]}"

    def run_api_preflight_check(self) -> bool:
        model_name = self.config.get("model_name", "gpt-5-mini")
        base_url = str(self.config.get("base_url", "")).strip()
        target_language = self.config.get("target_language", "Simplified Chinese")
        request_timeout = float(self.config.get("request_timeout_seconds", 300))
        check_timeout = float(self.config.get("api_test_timeout_seconds", min(request_timeout, 90)))
        check_timeout = max(10.0, check_timeout)
        api_key = str(self.config.get("api_key", ""))

        self._print(
            "API Check",
            (
                f"model={model_name}\n"
                f"base_url={base_url}\n"
                f"api_key={self._mask_secret(api_key)}\n"
                f"target_language={target_language}\n"
                f"request_timeout_seconds={request_timeout}\n"
                f"api_test_timeout_seconds={check_timeout}\n"
                f"reasoning={self.reasoning if self.reasoning_supported else 'disabled'}"
            ),
            "cyan",
        )

        test_segments = ["API connectivity test paragraph."]
        user_payload = {
            "target_language": target_language,
            "previous_context": "",
            "previous_context_summary": "",
            "recent_translated_tail": [],
            "input_pages": [[test_segments[0]]],
            "input_segments": test_segments,
            "must_keep_segment_count": 1,
            "novel_mode": True,
            "need_summary_update": False,
        }

        extra_body = {}
        cfg_extra_body = self.config.get("extra_body")
        if isinstance(cfg_extra_body, dict):
            extra_body.update(cfg_extra_body)

        attempts = 0
        while attempts < 2:
            attempts += 1
            local_extra_body = dict(extra_body)
            if self.reasoning_supported:
                local_extra_body["reasoning"] = self.reasoning

            check_started = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=float(self.config.get("temperature", 0.3)),
                    extra_body=local_extra_body,
                    timeout=check_timeout,
                )
                elapsed = time.time() - check_started

                if not response.choices:
                    self._print("API Check Fail", f"no choices returned in {elapsed:.2f}s", "red")
                    return False

                choice = response.choices[0]
                content = choice.message.content or ""
                if not content:
                    self._print("API Check Fail", f"empty content in {elapsed:.2f}s", "red")
                    return False

                data = json.loads(content)
                translated, parse_reason = self._extract_translated_segments(data, expected_count=1)
                if translated is None:
                    self._print(
                        "API Check Fail",
                        (
                            f"parse_error={parse_reason}\n"
                            f"response_keys={list(data.keys())[:12]}\n"
                            f"elapsed={elapsed:.2f}s"
                        ),
                        "red",
                    )
                    return False

                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
                total_tokens = getattr(usage, "total_tokens", None) if usage else None
                finish_reason = getattr(choice, "finish_reason", None)

                self._print(
                    "API Check OK",
                    (
                        f"elapsed={elapsed:.2f}s\n"
                        f"response_model={getattr(response, 'model', '(unknown)')}\n"
                        f"finish_reason={finish_reason}\n"
                        f"usage.prompt_tokens={prompt_tokens}\n"
                        f"usage.completion_tokens={completion_tokens}\n"
                        f"usage.total_tokens={total_tokens}\n"
                        f"response_keys={list(data.keys())[:12]}\n"
                        f"segments_count={len(translated)}\n"
                        f"new_summary_len={len(str(data.get('new_summary', '')))}"
                    ),
                    "green",
                )
                return True
            except Exception as e:
                message = str(e)
                if self.reasoning_supported and "Unknown parameter: 'reasoning'" in message:
                    self.reasoning_supported = False
                    self._print(
                        "API Check Compat",
                        "reasoning parameter unsupported; retrying check without reasoning",
                        "yellow",
                    )
                    continue
                self._print(
                    "API Check Fail",
                    (
                        f"exception_type={type(e).__name__}\n"
                        f"exception={message}\n"
                        f"attempt={attempts}/2"
                    ),
                    "red",
                )
                return False

        self._print("API Check Fail", "reasoning compatibility retry exhausted", "red")
        return False

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
        max_segment_chars = max(300, int(self.config.get("max_segment_chars", 1200)))
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

    def _extract_translated_segments(
        self,
        data: dict,
        expected_count: int,
    ) -> Tuple[Optional[List[str]], str]:
        segments = data.get("segments")
        if isinstance(segments, list):
            if len(segments) == expected_count:
                return [str(x) for x in segments], ""
            return None, f"segments_count_mismatch expected={expected_count} actual={len(segments)}"
        if segments is not None:
            return None, f"segments_type_invalid type={type(segments).__name__}"

        pages = data.get("pages")
        normalized = self._normalize_pages_to_segments(pages, expected_count)
        if normalized is not None:
            return normalized, ""
        if isinstance(pages, list):
            return None, f"pages_shape_mismatch expected={expected_count} actual={len(pages)}"
        if pages is not None:
            return None, f"pages_type_invalid type={type(pages).__name__}"
        return None, f"missing_segments_pages expected={expected_count}"

    def call_api(
        self,
        segment_texts: List[str],
        last_summary: str,
        recent_tail: List[str],
        need_summary_update: bool,
    ):
        target_language = self.config.get("target_language", "Simplified Chinese")
        self.last_failure_reason = ""
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
        if self.reasoning_supported:
            extra_body["reasoning"] = self.reasoning

        try:
            request_timeout = float(self.config.get("request_timeout_seconds", 300))
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
                self.last_failure_reason = "empty_response"
                return None, ""

            content = response.choices[0].message.content
            if not content:
                self.last_failure_reason = "empty_response"
                return None, last_summary

            data = json.loads(content)
            translated, parse_reason = self._extract_translated_segments(data, len(segment_texts))
            if translated is None:
                self.last_failure_reason = "parse_fail"
                self._print(
                    "ParseFail",
                    f"{parse_reason} | keys={list(data.keys())[:8]}",
                    "yellow",
                )
                return None, ""
            new_summary = str(data.get("new_summary", last_summary))
            return translated, new_summary
        except Exception as e:
            message = str(e)
            if self.reasoning_supported and "Unknown parameter: 'reasoning'" in message:
                self.reasoning_supported = False
                self._print(
                    "Compat",
                    "API rejected reasoning parameter; continuing without reasoning.",
                    "yellow",
                )
                return self.call_api(
                    segment_texts=segment_texts,
                    last_summary=last_summary,
                    recent_tail=recent_tail,
                    need_summary_update=need_summary_update,
                )
            self.last_failure_reason = "api_error"
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
            attempt_started = time.time()
            translated, new_summary = self.call_api(
                segment_texts,
                last_summary,
                recent_tail,
                need_summary_update,
            )
            attempt_elapsed = time.time() - attempt_started
            if translated is not None:
                self._print(
                    "Chunk API",
                    (
                        f"chunk-size={len(segment_texts)} attempt={attempt}/{retries} "
                        f"ok in {attempt_elapsed:.1f}s"
                    ),
                    "cyan",
                )
                return translated, new_summary
            self._print(
                "Chunk API",
                (
                    f"chunk-size={len(segment_texts)} attempt={attempt}/{retries} "
                    f"failed in {attempt_elapsed:.1f}s "
                    f"(reason={self.last_failure_reason or 'unknown'})"
                ),
                "yellow",
            )
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
        htmlz_title: Optional[str] = None,
        htmlz_author: Optional[str] = None,
        htmlz_language: str = "zh-CN",
        htmlz_identifier: Optional[str] = None,
        htmlz_publisher: Optional[str] = None,
        resume: bool = True,
        realtime_write: bool = True,
    ):
        prep_started_at = time.time()
        epub_meta: Dict[str, Optional[str]] = {}
        if source_format == "epub":
            epub_payload = self._extract_epub_html_payload(input_file)
            text = str(epub_payload["html"])
            raw_meta = epub_payload.get("meta")
            if isinstance(raw_meta, dict):
                epub_meta = {
                    "title": raw_meta.get("title"),
                    "author": raw_meta.get("author"),
                    "language": raw_meta.get("language"),
                    "identifier": raw_meta.get("identifier"),
                    "publisher": raw_meta.get("publisher"),
                }
            tokens = self.tokenize_html(text)
        else:
            text = input_file.read_text(encoding="utf-8")
            if source_format == "html":
                tokens = self.tokenize_html(text)
            else:
                tokens = self.tokenize_markdown(text)
        prepared_segments = self.prepare_segments(tokens)
        total = len(prepared_segments)
        translatable_tokens = sum(1 for token in tokens if token.translatable)
        prep_elapsed = time.time() - prep_started_at
        self._print(
            "Prepare",
            (
                f"{input_file.name}: format={source_format}, tokens={len(tokens)}, "
                f"translatable_tokens={translatable_tokens}, segments={total}, "
                f"elapsed={prep_elapsed:.1f}s"
            ),
            "cyan",
        )
        use_htmlz_wrapper = source_format in {"html", "epub"}
        resolved_html_title = (
            (htmlz_title or epub_meta.get("title") or input_file.stem).strip() or input_file.stem
        )
        resolved_html_author = htmlz_author if htmlz_author is not None else epub_meta.get("author")
        resolved_html_identifier = (
            htmlz_identifier if htmlz_identifier is not None else epub_meta.get("identifier")
        )
        resolved_html_publisher = (
            htmlz_publisher if htmlz_publisher is not None else epub_meta.get("publisher")
        )

        if total == 0:
            if use_htmlz_wrapper:
                wrapped = (
                    self._build_htmlz_document_header(
                        title=resolved_html_title,
                        author=resolved_html_author,
                        language=htmlz_language,
                        identifier=resolved_html_identifier,
                        publisher=resolved_html_publisher,
                    )
                    + text
                    + self._build_htmlz_document_footer()
                )
                output_file.write_text(wrapped, encoding="utf-8")
            else:
                output_file.write_text(text, encoding="utf-8")
            self._print("Skip", f"{input_file.name}: no translatable segments found", "yellow")
            return True

        default_min_chunk_segments = max(1, int(self.config.get("chunk_size", 4)))
        default_max_chunk_segments = max(
            default_min_chunk_segments,
            int(self.config.get("max_chunk_segments", 80)),
        )
        default_target_chunk_chars = max(200, int(self.config.get("target_chunk_chars", 2600)))
        default_char_limit = max(1200, int(self.config.get("max_chunk_chars", 5200)))
        request_timeout = float(self.config.get("request_timeout_seconds", 300))
        context_tail_segments = max(0, int(self.config.get("context_tail_segments", 5)))
        summary_interval_batches = max(1, int(self.config.get("summary_interval_batches", 10)))
        summary_interval_chars = max(0, int(self.config.get("summary_interval_chars", 16000)))
        resume_path = self._resume_state_path(output_file)
        resume_fingerprint = self._resume_fingerprint(
            source_text=text,
            source_format=source_format,
            output_style=output_style,
            html_translation_style=html_translation_style,
            use_htmlz_wrapper=use_htmlz_wrapper,
        )

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
        output_bytes_written = 0
        resumed = False

        token_part_total = Counter(seg.token_index for seg in prepared_segments)
        token_part_translated: Dict[int, List[str]] = {}
        completed_token_map: Dict[int, str] = {}
        next_token_to_write = 0

        if resume:
            previous_state = self._load_resume_state(resume_path)
            if previous_state and previous_state.get("version") == 1:
                if (
                    previous_state.get("fingerprint") == resume_fingerprint
                    and int(previous_state.get("total", -1)) == total
                ):
                    i = max(0, min(total, int(previous_state.get("i", 0))))
                    done_segments = max(0, min(total, int(previous_state.get("done_segments", i))))
                    summary = str(previous_state.get("summary", summary))
                    batches_since_summary = max(0, int(previous_state.get("batches_since_summary", 0)))
                    chars_since_summary = max(0, int(previous_state.get("chars_since_summary", 0)))
                    translated_tail = previous_state.get("translated_tail")
                    if isinstance(translated_tail, list):
                        translated_all = [str(x) for x in translated_tail]
                        if context_tail_segments > 0 and len(translated_all) > context_tail_segments:
                            translated_all = translated_all[-context_tail_segments:]
                    raw_partial = previous_state.get("token_part_translated")
                    if isinstance(raw_partial, dict):
                        token_part_translated = {
                            int(k): [str(v) for v in values]
                            for k, values in raw_partial.items()
                            if isinstance(values, list)
                        }
                    raw_completed = previous_state.get("completed_token_map")
                    if isinstance(raw_completed, dict):
                        completed_token_map = {int(k): str(v) for k, v in raw_completed.items()}
                    next_token_to_write = max(
                        0,
                        min(len(tokens), int(previous_state.get("next_token_to_write", 0))),
                    )
                    output_bytes_written = max(0, int(previous_state.get("output_bytes_written", 0)))
                    resumed = i > 0 or next_token_to_write > 0
                    if resumed:
                        self._print(
                            "Resume",
                            (
                                f"{input_file.name}: restored segment={i}/{total}, "
                                f"written_token={next_token_to_write}, "
                                "chunk_policy=current_config"
                            ),
                            "yellow",
                        )
                else:
                    self._print(
                        "Resume Ignore",
                        f"{input_file.name}: checkpoint mismatch, restart from beginning",
                        "yellow",
                    )

        def initialize_realtime_output() -> None:
            nonlocal output_bytes_written
            if output_style == "bilingual":
                if source_format in {"html", "epub"}:
                    if use_htmlz_wrapper:
                        output_file.write_text(
                            self._build_htmlz_document_header(
                                title=resolved_html_title,
                                author=resolved_html_author,
                                language=htmlz_language,
                                identifier=resolved_html_identifier,
                                publisher=resolved_html_publisher,
                            ),
                            encoding="utf-8",
                        )
                    else:
                        output_file.write_text("", encoding="utf-8")
                else:
                    output_file.write_text(
                        f"# 双语对照预览：{input_file.stem}\n\n"
                        "> 原文在上，译文在下；脚本会按分段实时写入。\n\n",
                        encoding="utf-8",
                    )
            else:
                if source_format in {"html", "epub"} and use_htmlz_wrapper:
                    output_file.write_text(
                        self._build_htmlz_document_header(
                            title=resolved_html_title,
                            author=resolved_html_author,
                            language=htmlz_language,
                            identifier=resolved_html_identifier,
                            publisher=resolved_html_publisher,
                        ),
                        encoding="utf-8",
                    )
                else:
                    output_file.write_text("", encoding="utf-8")
            output_bytes_written = output_file.stat().st_size if output_file.exists() else 0

        if realtime_write:
            if resumed:
                if output_file.exists():
                    current_size = output_file.stat().st_size
                    if current_size != output_bytes_written:
                        with output_file.open("rb+") as fp:
                            fp.truncate(output_bytes_written)
                else:
                    initialize_realtime_output()
                    rebuilt_next = self._flush_ready_output(
                        tokens=tokens,
                        token_translation_map=completed_token_map,
                        next_token_index=0,
                        output_path=output_file,
                        output_style=output_style,
                        html_translation_style=html_translation_style,
                    )
                    if rebuilt_next != next_token_to_write:
                        raise RuntimeError("Checkpoint rebuild mismatch")
                    output_bytes_written = output_file.stat().st_size if output_file.exists() else 0
            else:
                initialize_realtime_output()

        def persist_resume_state() -> None:
            if not resume:
                return
            tail = translated_all[-context_tail_segments:] if context_tail_segments > 0 else []
            state = {
                "version": 1,
                "fingerprint": resume_fingerprint,
                "total": total,
                "i": i,
                "done_segments": done_segments,
                "summary": summary,
                "batches_since_summary": batches_since_summary,
                "chars_since_summary": chars_since_summary,
                "translated_tail": tail,
                "token_part_translated": {str(k): v for k, v in token_part_translated.items()},
                "completed_token_map": {str(k): v for k, v in completed_token_map.items()},
                "next_token_to_write": next_token_to_write,
                "output_bytes_written": output_bytes_written,
            }
            self._save_resume_state(resume_path, state)

        def shrink_value(current: int, min_value: int, ratio: float) -> int:
            safe_ratio = max(0.1, min(0.95, ratio))
            candidate = max(min_value, int(current * safe_ratio))
            if candidate >= current and current > min_value:
                candidate = current - 1
            return max(min_value, candidate)

        self._print_progress(done_segments, total, started_at)
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
            elapsed = time.time() - started_at
            self._print(
                "Chunk Start",
                (
                    f"{input_file.name}: segments {i + 1}-{end}/{total} | "
                    f"batch={end - i}, chars~{batch_chars}, summary={'Y' if need_summary_update else 'N'}, "
                    f"chunk[min/max]={current_min_chunk_segments}/{current_max_chunk_segments}, "
                    f"target={current_target_chunk_chars}, limit={current_char_limit}, "
                    f"timeout={request_timeout:.0f}s, elapsed={elapsed:.1f}s"
                ),
                "cyan",
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
                    output_bytes_written = output_file.stat().st_size if output_file.exists() else 0

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
                persist_resume_state()
                continue

            if current_min_chunk_segments == 1 and end == i + 1:
                self._print(
                    "Abort",
                    f"{input_file.name}: failed at segment {i + 1} after retries",
                    "red",
                )
                return False

            failure_reason = self.last_failure_reason or "unknown"
            shrink_ratio = 0.8 if failure_reason == "parse_fail" else 0.5
            new_min_chunk_segments = shrink_value(
                current=current_min_chunk_segments,
                min_value=1,
                ratio=shrink_ratio,
            )
            new_max_chunk_segments = shrink_value(
                current=current_max_chunk_segments,
                min_value=new_min_chunk_segments,
                ratio=shrink_ratio,
            )
            new_target_chunk_chars = shrink_value(
                current=current_target_chunk_chars,
                min_value=200,
                ratio=shrink_ratio,
            )
            new_char_limit = shrink_value(
                current=current_char_limit,
                min_value=1200,
                ratio=shrink_ratio,
            )
            self._print(
                "Downgrade",
                (
                    f"{input_file.name}: reason={failure_reason}, ratio={shrink_ratio:.2f}, "
                    f"min_chunk {current_min_chunk_segments}->{new_min_chunk_segments}, "
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
            output_bytes_written = output_file.stat().st_size if output_file.exists() else 0
            if next_token_to_write != len(tokens):
                raise RuntimeError("Realtime writer ended with unresolved tokens")
            if use_htmlz_wrapper:
                with output_file.open("a", encoding="utf-8") as fp:
                    fp.write(self._build_htmlz_document_footer())
                output_bytes_written = output_file.stat().st_size if output_file.exists() else 0
        else:
            token_translation_map = self.merge_translations(tokens, prepared_segments, translated_all)
            if source_format in {"html", "epub"}:
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
                if use_htmlz_wrapper:
                    rebuilt = (
                        self._build_htmlz_document_header(
                            title=resolved_html_title,
                            author=resolved_html_author,
                            language=htmlz_language,
                            identifier=resolved_html_identifier,
                            publisher=resolved_html_publisher,
                        )
                        + rebuilt
                        + self._build_htmlz_document_footer()
                    )
            else:
                rebuilt = self.reconstruct(tokens, token_translation_map)
            output_file.write_text(rebuilt, encoding="utf-8")
            output_bytes_written = output_file.stat().st_size if output_file.exists() else 0

        if source_format == "html":
            try:
                rendered_html = output_file.read_text(encoding="utf-8")
                rewritten_html, copied_files, rewritten_refs = self._rewrite_html_assets_for_output(
                    html_text=rendered_html,
                    source_base_dir=input_file.parent,
                    output_html_path=output_file,
                )
                if rewritten_html != rendered_html:
                    output_file.write_text(rewritten_html, encoding="utf-8")
                    output_bytes_written = output_file.stat().st_size if output_file.exists() else 0
                if copied_files > 0 or rewritten_refs > 0:
                    self._print(
                        "Assets",
                        (
                            f"{output_file.name}: copied={copied_files}, "
                            f"rewritten={rewritten_refs}, root=assets/"
                        ),
                        "cyan",
                    )
            except Exception as e:
                self._print("Assets Warn", f"{output_file.name}: {e}", "yellow")
        if resume_path.exists():
            resume_path.unlink()
        self._print("Done", f"{input_file.name} -> {output_file.name}", "cyan")
        return True

    def run(
        self,
        input_path: str,
        suffix: str = "_CN",
        skip_existing: bool = False,
        output_style: str = "bilingual",
        html_translation_style: str = "blockquote",
        post_package: str = "none",
        htmlz_title: Optional[str] = None,
        htmlz_author: Optional[str] = None,
        htmlz_language: str = "zh-CN",
        htmlz_identifier: Optional[str] = None,
        htmlz_publisher: Optional[str] = None,
        resume: bool = True,
        realtime_write: bool = True,
    ):
        def detect_source_format(file_path: Path) -> Optional[str]:
            ext = file_path.suffix.lower()
            if ext in {".md", ".markdown"}:
                return "markdown"
            if ext in {".html", ".htm"}:
                return "html"
            if ext == ".epub":
                return "epub"
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
            print("No markdown/html/epub files found.")
            return

        for input_file in files:
            source_format = detect_source_format(input_file)
            if source_format is None:
                continue
            if input_file.stem.endswith(suffix):
                continue
            output_stem = f"{input_file.stem}{suffix}"
            if source_format in {"html", "epub"}:
                output_stem = self._sanitize_archive_filename_stem(output_stem)
            output_suffix = ".html" if source_format == "epub" else input_file.suffix
            output_path = input_file.with_name(f"{output_stem}{output_suffix}")
            if skip_existing and output_path.exists():
                self._print("Skip", f"{output_path.name} already exists", "yellow")
                continue
            try:
                success = self.process_file(
                    input_file,
                    output_path,
                    source_format=source_format,
                    output_style=output_style,
                    html_translation_style=html_translation_style,
                    htmlz_title=htmlz_title,
                    htmlz_author=htmlz_author,
                    htmlz_language=htmlz_language,
                    htmlz_identifier=htmlz_identifier,
                    htmlz_publisher=htmlz_publisher,
                    resume=resume,
                    realtime_write=realtime_write,
                )
                if not success:
                    continue
                if post_package != "none":
                    if output_path.suffix.lower() != ".html":
                        self._print(
                            "Pack Skip",
                            f"{output_path.name}: packaging requires .html output",
                            "yellow",
                        )
                    else:
                        packaged_files = self._package_after_html_output(
                            output_html_path=output_path,
                            package_mode=post_package,
                        )
                        if packaged_files:
                            self._print(
                                "Pack Done",
                                ", ".join(path.name for path in packaged_files),
                                "cyan",
                            )
            except Exception as e:
                self._print("Error", f"{input_file.name}: {e}", "red")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate markdown/html/epub novel files with context-aware chunking."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to .md/.markdown/.html/.epub file, or folder containing these files",
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
        "--post-package",
        choices=["none", "htmlz", "epubv3", "both"],
        default="none",
        help="Optional packaging after HTML output: htmlz, epubv3, both, or none",
    )
    parser.add_argument(
        "--htmlz-title",
        default=None,
        help="Override HTML <title> / metadata title for HTML outputs",
    )
    parser.add_argument(
        "--htmlz-author",
        default=None,
        help="Set HTML metadata author (dc.creator) for HTML outputs",
    )
    parser.add_argument(
        "--htmlz-language",
        default="zh-CN",
        help="Set HTML metadata language for HTML outputs (default: zh-CN)",
    )
    parser.add_argument(
        "--htmlz-identifier",
        default=None,
        help="Set HTML metadata identifier (dc.identifier) for HTML outputs",
    )
    parser.add_argument(
        "--htmlz-publisher",
        default=None,
        help="Set HTML metadata publisher (dc.publisher) for HTML outputs",
    )
    parser.add_argument(
        "--no-realtime-write",
        action="store_true",
        help="Disable realtime segment writing and write file only after all segments finish",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume and always restart from beginning",
    )
    parser.add_argument(
        "--skip-api-check",
        action="store_true",
        help="Skip startup API preflight check",
    )
    parser.add_argument(
        "--api-check-only",
        action="store_true",
        help="Run API preflight check only and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    translator = MarkdownNovelTranslator(
        config_path=args.config,
        prompt_path=args.prompt,
        reasoning_effort=args.reasoning_effort,
    )
    if not args.skip_api_check:
        passed = translator.run_api_preflight_check()
        if not passed:
            raise SystemExit(2)
    if args.api_check_only:
        raise SystemExit(0)

    input_path = args.input_path
    if not input_path:
        input_path = (
            input("Drag a markdown/html/epub file/folder here: ")
            .strip()
            .replace("\\ ", " ")
            .strip("'")
            .strip('"')
        )

    translator.run(
        input_path=input_path,
        suffix=args.suffix,
        skip_existing=args.skip_existing,
        output_style=args.output_style,
        html_translation_style=args.html_translation_style,
        post_package=args.post_package,
        htmlz_title=args.htmlz_title,
        htmlz_author=args.htmlz_author,
        htmlz_language=args.htmlz_language,
        htmlz_identifier=args.htmlz_identifier,
        htmlz_publisher=args.htmlz_publisher,
        resume=not args.no_resume,
        realtime_write=not args.no_realtime_write,
    )
