import os
import sys
import time
import base64
import re
import argparse
import tempfile
from pathlib import Path

from mistralai import Mistral
from mistralai.models.sdkerror import SDKError


def _is_retryable_api_error(exc: Exception) -> bool:
    msg = str(exc)
    return "Status 5" in msg or "internal_server_error" in msg or "Service unavailable" in msg


def _run_with_retry(action_name: str, func, attempts: int = 3, delay_seconds: float = 2.0):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except SDKError as exc:
            last_exc = exc
            if not _is_retryable_api_error(exc) or attempt == attempts:
                break
            print(
                f"Warning: {action_name} failed with a temporary API error. "
                f"Retrying ({attempt}/{attempts})...",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)

    assert last_exc is not None
    raise last_exc


def _parse_pages_arg(pages_arg: str) -> list[int]:
    pages: list[int] = []
    seen: set[int] = set()

    for raw_part in pages_arg.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s.strip())
                end = int(end_s.strip())
            except ValueError:
                raise SystemExit(
                    f"Error: invalid page range '{part}'. Use formats like '1-3' or '5'."
                ) from None
            if start <= 0 or end <= 0:
                raise SystemExit(
                    f"Error: invalid page range '{part}'. Page numbers must start at 1."
                )
            if end < start:
                raise SystemExit(
                    f"Error: invalid page range '{part}'. Range end must be >= start."
                )
            for page_1_based in range(start, end + 1):
                page_0_based = page_1_based - 1
                if page_0_based not in seen:
                    seen.add(page_0_based)
                    pages.append(page_0_based)
        else:
            try:
                page_1_based = int(part)
            except ValueError:
                raise SystemExit(
                    f"Error: invalid page value '{part}'. Use formats like '1-3' or '5'."
                ) from None
            if page_1_based <= 0:
                raise SystemExit(
                    f"Error: invalid page value '{part}'. Page numbers must start at 1."
                )
            page_0_based = page_1_based - 1
            if page_0_based not in seen:
                seen.add(page_0_based)
                pages.append(page_0_based)

    if not pages:
        raise SystemExit("Error: --pages did not contain any valid page entries.")

    return pages


def _guess_ext_from_data_url_prefix(prefix: str) -> str:
    if "image/png" in prefix:
        return ".png"
    if "image/jpeg" in prefix or "image/jpg" in prefix:
        return ".jpg"
    if "image/webp" in prefix:
        return ".webp"
    if "image/gif" in prefix:
        return ".gif"
    return ".bin"


def _safe_image_id(raw_id: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw_id)
    return cleaned or "image"


def _safe_file_name(raw_name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in raw_name)
    cleaned = cleaned.strip(".")
    return cleaned or "image.bin"


def _pages_to_ranges_zero_based(pages: list[int]) -> list[tuple[int, int]]:
    if not pages:
        return []

    ranges: list[tuple[int, int]] = []
    start = pages[0]
    end = pages[0]
    for idx in pages[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append((start, end))
            start = idx
            end = idx
    ranges.append((start, end))
    return ranges


def _format_pages_for_log(pages: list[int], max_ranges: int = 12) -> str:
    ranges = _pages_to_ranges_zero_based(pages)
    parts: list[str] = []
    for start, end in ranges:
        if start == end:
            parts.append(str(start + 1))
        else:
            parts.append(f"{start + 1}-{end + 1}")

    if len(parts) <= max_ranges:
        return ",".join(parts)

    shown = ",".join(parts[:max_ranges])
    return (
        f"{shown},... "
        f"({len(parts) - max_ranges} more ranges, {len(pages)} pages total)"
    )


def _resolve_output_format(args, out_path: Path) -> str:
    if args.output_format:
        return args.output_format

    if out_path.suffix.lower() in (".html", ".htm"):
        return "html"

    return "markdown"


def _embed_html_tables_in_markdown(md_text: str, ocr_response) -> tuple[str, int]:
    table_by_name: dict[str, str] = {}
    exported_tables = 0

    for page in ocr_response.pages:
        for table in page.tables or []:
            table_id = str(getattr(table, "id", "") or "").strip()
            table_content = str(getattr(table, "content", "") or "")
            if not table_id or not table_content:
                continue
            table_by_name[Path(table_id).name] = table_content
            exported_tables += 1

    if not table_by_name:
        return md_text, 0

    def _replace_link(match: re.Match[str]) -> str:
        link_text = match.group(1)
        target = match.group(2).strip()
        replacement = table_by_name.get(Path(target).name)
        if replacement is None:
            return match.group(0)
        return replacement

    updated = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _replace_link, md_text)
    return updated, exported_tables


def _stabilize_fenced_code_blocks(md_text: str, max_fence_lines: int = 120) -> str:
    lines = md_text.splitlines()
    out: list[str] = []
    in_fence = False
    fence_lines = 0

    for line in lines:
        stripped = line.lstrip()
        is_fence = stripped.startswith("```")

        if is_fence:
            in_fence = not in_fence
            fence_lines = 0
            out.append(line)
            continue

        # OCR occasionally opens a fenced block and forgets to close it for many pages.
        # If a new heading starts while still in a fence, close the fence first.
        if in_fence:
            normalized = stripped.replace("\u00a0", " ")
            if re.match(r"^#{1,6}\s+", normalized):
                out.append("```")
                in_fence = False
                fence_lines = 0
            elif fence_lines >= max_fence_lines and normalized:
                out.append("```")
                in_fence = False
                fence_lines = 0

        out.append(line)
        if in_fence:
            fence_lines += 1

    if in_fence:
        out.append("```")

    return "\n".join(out) + "\n"


def _markdown_to_html(md_text: str, title: str) -> str:
    try:
        import markdown as mdlib
    except ModuleNotFoundError:
        raise SystemExit(
            "Error: HTML output requires the 'markdown' package. "
            "Install dependencies via: pip install -r requirements.txt"
        ) from None

    body = mdlib.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        f"  <title>{title}</title>\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "  <style>\n"
        "    :root { color-scheme: light; }\n"
        "    body {\n"
        "      margin: 0;\n"
        "      padding: 2rem;\n"
        "      font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Helvetica, Arial, sans-serif;\n"
        "      line-height: 1.5;\n"
        "      color: #1f2937;\n"
        "      background: #f8fafc;\n"
        "    }\n"
        "    h1, h2, h3, h4, h5, h6 {\n"
        "      color: #0f172a;\n"
        "      margin-top: 1.4em;\n"
        "      margin-bottom: 0.5em;\n"
        "    }\n"
        "    p, ul, ol { margin: 0.6em 0; }\n"
        "    code { background: #eef2ff; padding: 0.1em 0.3em; border-radius: 4px; }\n"
        "    pre {\n"
        "      background: #f8fafc;\n"
        "      color: #1f2937;\n"
        "      border: 1px solid #cbd5e1;\n"
        "      padding: 0.9rem 1rem;\n"
        "      border-radius: 8px;\n"
        "      overflow-x: auto;\n"
        "      line-height: 1.45;\n"
        "      font-size: 0.92rem;\n"
        "    }\n"
        "    pre code {\n"
        "      background: transparent;\n"
        "      padding: 0;\n"
        "      border-radius: 0;\n"
        "      color: inherit;\n"
        "      white-space: pre;\n"
        "    }\n"
        "    table {\n"
        "      width: auto;\n"
        "      max-width: 100%;\n"
        "      border-collapse: collapse;\n"
        "      margin: 1rem 0 1.2rem;\n"
        "      background: #ffffff;\n"
        "      border: 1px solid #000000;\n"
        "      border-radius: 8px;\n"
        "      overflow: hidden;\n"
        "      font-size: 0.95rem;\n"
        "    }\n"
        "    th, td {\n"
        "      border: none;\n"
        "      padding: 0.5rem 0.65rem;\n"
        "      vertical-align: top;\n"
        "      text-align: left;\n"
        "    }\n"
        "    th {\n"
        "      background: #e2e8f0;\n"
        "      color: #0f172a;\n"
        "      font-weight: 700;\n"
        "    }\n"
        "    tr:nth-child(even) td { background: #f8fafc; }\n"
        "    img { max-width: 100%; height: auto; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def _extract_pdf_chapter_index(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        raise SystemExit(
            "Error: chapter index extraction requires the 'pypdf' package. "
            "Install dependencies via: pip install -r requirements.txt"
        ) from None

    reader = PdfReader(str(pdf_path))
    raw_outline = getattr(reader, "outline", None) or []
    chapters: list[tuple[int, str]] = []

    def _walk(items) -> None:
        for item in items:
            if isinstance(item, list):
                _walk(item)
                continue

            title = str(getattr(item, "title", "") or "").strip()
            if not title:
                continue

            page_no: int | None = None
            try:
                page_no = reader.get_destination_page_number(item) + 1
            except Exception:
                page_no = None

            if page_no is not None and page_no > 0:
                chapters.append((page_no, title))

    _walk(raw_outline if isinstance(raw_outline, list) else [raw_outline])

    # Stable order and dedup.
    dedup: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for page_no, title in sorted(chapters, key=lambda x: (x[0], x[1].lower())):
        key = (page_no, title.lower())
        if key not in seen:
            seen.add(key)
            dedup.append((page_no, title))
    return dedup


def _extract_local_heading_index(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        raise SystemExit(
            "Error: local index analysis requires the 'pypdf' package. "
            "Install dependencies via: pip install -r requirements.txt"
        ) from None

    reader = PdfReader(str(pdf_path))
    headings: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()

    for page_idx, page in enumerate(reader.pages):
        page_no = page_idx + 1
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if len(line) > 140:
                continue

            is_numbered = bool(re.match(r"^\d+(\.\d+)*[.)]?\s+[A-Z].+", line))
            is_chapter_word = bool(re.match(r"^(Chapter|Section|Appendix)\b", line, re.IGNORECASE))
            is_caps = bool(re.match(r"^[A-Z0-9][A-Z0-9 .,:;/()&+\-]{5,}$", line))

            if is_numbered or is_chapter_word or is_caps:
                key = (page_no, line.lower())
                if key not in seen:
                    seen.add(key)
                    headings.append((page_no, line))

    return headings


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF -> Markdown/HTML via Mistral OCR")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--out", help="Output file (default: <pdf>.md, or <pdf>.html with --output-format html)")
    parser.add_argument(
        "--output-format",
        choices=("markdown", "html"),
        help="Output format (default: markdown, auto html if --out ends with .html/.htm)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    parser.add_argument(
        "--pages",
        help="Process specific pages (1-based), e.g. '1-12' or '2,5,10-12'",
    )
    parser.add_argument(
        "--slice-pdf",
        action="store_true",
        help="With --pages, locally create a temporary PDF with only selected pages before upload",
    )
    parser.add_argument(
        "--images-dir",
        help="Directory to export extracted images (default: <out_stem>_images)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable image extraction and image export",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        help="Maximum number of images to extract",
    )
    parser.add_argument(
        "--image-min-size",
        type=int,
        help="Minimum image width/height in pixels to extract",
    )
    parser.add_argument(
        "--no-header-footer",
        action="store_true",
        help="Disable header/footer extraction (enabled by default)",
    )
    parser.add_argument(
        "--chapter-index-out",
        help="Optional output file for chapter/page index extracted from PDF bookmarks (before OCR upload)",
    )
    parser.add_argument(
        "--analyze-index",
        action="store_true",
        help="Analyze chapter/page index locally and exit (bookmarks first, then text heuristics)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"Error: input file does not exist: {pdf_path}")
    if args.image_limit is not None and args.image_limit <= 0:
        raise SystemExit("Error: --image-limit must be greater than 0.")
    if args.image_min_size is not None and args.image_min_size <= 0:
        raise SystemExit("Error: --image-min-size must be greater than 0.")

    if args.out:
        out_path = Path(args.out)
    elif args.output_format == "html":
        out_path = pdf_path.with_suffix(".html")
    else:
        out_path = pdf_path.with_suffix(".md")

    output_format = _resolve_output_format(args, out_path)

    if args.chapter_index_out or args.analyze_index:
        print("Analyzing chapter index locally (before OCR upload)...")
        chapters = _extract_pdf_chapter_index(pdf_path)
        source = "PDF bookmarks"
        if not chapters and args.analyze_index:
            print("No bookmarks found, falling back to text-based heading analysis...")
            chapters = _extract_local_heading_index(pdf_path)
            source = "text heuristics"

        if chapters:
            chapter_lines = [f"{page}\t{title}" for page, title in chapters]
            print(f"Chapters: found={len(chapter_lines)} (source: {source})")
            for line in chapter_lines:
                print(f"  {line}")
        else:
            chapter_lines = []
            print("Chapters: none found in PDF bookmarks.")

        if args.chapter_index_out:
            chapter_path = Path(args.chapter_index_out)
            if chapter_path.exists() and not args.force:
                raise SystemExit(
                    f"Error: chapter index output already exists: {chapter_path}. "
                    "Use --force to overwrite."
                )
            chapter_path.write_text("\n".join(chapter_lines) + ("\n" if chapter_lines else ""), encoding="utf-8")
            print(f"Chapter index: {chapter_path}")

        if args.analyze_index:
            return

    if out_path.exists() and not args.force:
        raise SystemExit(
            f"Error: output file already exists: {out_path}. "
            "Use --force to overwrite."
        )

    images_enabled = not args.no_images
    images_dir = None
    if images_enabled:
        images_dir = Path(args.images_dir) if args.images_dir else out_path.with_name(f"{out_path.stem}_images")

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("Please set MISTRAL_API_KEY as an environment variable.")

    client = Mistral(api_key=api_key)
    selected_pages = _parse_pages_arg(args.pages) if args.pages is not None else None
    upload_path = pdf_path
    temp_pdf_path: Path | None = None

    if args.slice_pdf and selected_pages is None:
        raise SystemExit("Error: --slice-pdf requires --pages.")

    if args.slice_pdf and selected_pages is not None:
        try:
            from pypdf import PdfReader, PdfWriter
        except ModuleNotFoundError:
            raise SystemExit(
                "Error: --slice-pdf requires the 'pypdf' package. "
                "Install dependencies via: pip install -r requirements.txt"
            ) from None

        t_slice_start = time.time()
        print("Reading source PDF for slicing...")
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        pages_human = _format_pages_for_log(selected_pages)
        print(
            f"Slicing PDF before upload: selected pages={pages_human} "
            f"(source pages={total_pages})"
        )
        for idx in selected_pages:
            if idx < 0 or idx >= total_pages:
                raise SystemExit(
                    f"Error: requested page {idx + 1} is out of range for a PDF with {total_pages} pages."
                )

        writer = PdfWriter()
        page_ranges = _pages_to_ranges_zero_based(selected_pages)
        try:
            # Faster path for large PDFs: append contiguous ranges directly.
            for start, end in page_ranges:
                writer.append(str(pdf_path), pages=(start, end + 1))
            print(f"Slicing strategy: range append ({len(page_ranges)} range(s))")
        except Exception:
            print("Slicing strategy fallback: per-page copy")
            for idx in selected_pages:
                writer.add_page(reader.pages[idx])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            temp_pdf_path = Path(tmp_file.name)
            with temp_pdf_path.open("wb") as out_f:
                writer.write(out_f)
        print(
            f"Sliced PDF created: {temp_pdf_path} ({len(selected_pages)} pages, "
            f"{time.time() - t_slice_start:.1f}s)"
        )
        upload_path = temp_pdf_path

    try:
        with upload_path.open("rb") as f:
            uploaded = _run_with_retry(
                "File upload",
                lambda: client.files.upload(
                    file={"file_name": upload_path.name, "content": f},
                    purpose="ocr",
                ),
            )

        ocr_kwargs = {
            "model": "mistral-ocr-latest",
            "document": {"file_id": uploaded.id},
            "table_format": "html" if output_format == "html" else "markdown",
            "extract_header": not args.no_header_footer,
            "extract_footer": not args.no_header_footer,
            "include_image_base64": images_enabled,
        }
        if args.image_limit is not None:
            ocr_kwargs["image_limit"] = args.image_limit
        if args.image_min_size is not None:
            ocr_kwargs["image_min_size"] = args.image_min_size
        if selected_pages is not None and not args.slice_pdf:
            ocr_kwargs["pages"] = selected_pages

        ocr_response = _run_with_retry(
            "OCR processing",
            lambda: client.ocr.process(**ocr_kwargs),
        )
    except SDKError as exc:
        raise SystemExit(
            "Error: Mistral API request failed. "
            f"Details: {exc}. Please retry in a moment."
        ) from None
    finally:
        if temp_pdf_path is not None:
            try:
                temp_pdf_path.unlink(missing_ok=True)
            except Exception:
                pass

    md = "\n\n".join(page.markdown for page in ocr_response.pages)

    exported_images = 0
    link_map: dict[str, str] = {}
    link_map_by_stem: dict[str, str] = {}
    used_names: set[str] = set()
    images_dir_ready = False
    if images_enabled and images_dir is not None:
        for page in ocr_response.pages:
            for image in page.images:
                image_base64 = getattr(image, "image_base64", None)
                if not image_base64:
                    continue

                ext = ".bin"
                b64_payload = image_base64
                if image_base64.startswith("data:") and "," in image_base64:
                    prefix, b64_payload = image_base64.split(",", 1)
                    ext = _guess_ext_from_data_url_prefix(prefix)

                try:
                    image_bytes = base64.b64decode(b64_payload)
                except Exception:
                    print(
                        f"Warning: failed to decode image '{image.id}' on page {page.index + 1}.",
                        file=sys.stderr,
                    )
                    continue

                if not images_dir_ready:
                    if images_dir.exists() and not args.force:
                        raise SystemExit(
                            f"Error: image output directory already exists: {images_dir}. "
                            "Use --force to overwrite."
                        )
                    images_dir.mkdir(parents=True, exist_ok=True)
                    images_dir_ready = True

                source_ref = str(getattr(image, "id", "") or "").strip()
                source_name = Path(source_ref).name if source_ref else ""
                if source_name:
                    file_name = _safe_file_name(source_name)
                    if Path(file_name).suffix == "":
                        file_name = f"{file_name}{ext}"
                else:
                    file_name = f"page_{page.index + 1:04d}_{_safe_image_id(image.id)}{ext}"

                base_name = Path(file_name).stem
                suffix = Path(file_name).suffix
                counter = 1
                unique_name = file_name
                while unique_name.lower() in used_names:
                    unique_name = f"{base_name}_{counter}{suffix}"
                    counter += 1
                file_name = unique_name
                used_names.add(file_name.lower())

                image_path = images_dir / file_name
                if image_path.exists() and not args.force:
                    raise SystemExit(
                        f"Error: image output already exists: {image_path}. "
                        "Use --force to overwrite."
                    )
                image_path.write_bytes(image_bytes)
                exported_images += 1

                rel_path = os.path.relpath(image_path, out_path.parent).replace("\\", "/")
                if source_ref:
                    link_map[source_ref] = rel_path
                    link_map[Path(source_ref).name] = rel_path
                    source_stem = Path(source_ref).stem.lower()
                    if source_stem and source_stem not in link_map_by_stem:
                        link_map_by_stem[source_stem] = rel_path

        for src, dst in link_map.items():
            md = md.replace(f"]({src})", f"]({dst})")

        def _replace_md_image_link(match: re.Match[str]) -> str:
            alt_text = match.group(1)
            target = match.group(2).strip()
            if "://" in target or target.startswith("/"):
                return match.group(0)

            exact = link_map.get(target)
            if exact:
                return f"![{alt_text}]({exact})"

            stem = Path(target).stem.lower()
            by_stem = link_map_by_stem.get(stem)
            if by_stem:
                return f"![{alt_text}]({by_stem})"

            return match.group(0)

        md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replace_md_image_link, md)

    exported_tables = 0
    if output_format == "html":
        md, exported_tables = _embed_html_tables_in_markdown(md, ocr_response)
        md = _stabilize_fenced_code_blocks(md)
        rendered = _markdown_to_html(md, out_path.stem)
        out_path.write_text(rendered, encoding="utf-8")
    else:
        out_path.write_text(md, encoding="utf-8")

    print(f"OK: {out_path}")
    print(f"Output format: {output_format}")
    if output_format == "html":
        print(f"Tables: embedded={exported_tables}, format=html")
    if images_enabled and images_dir is not None:
        print(f"Images: exported={exported_images}, dir={images_dir}")
    else:
        print("Images: disabled")

    usage_info = getattr(ocr_response, "usage_info", None)
    if usage_info is not None:
        pages_processed = getattr(usage_info, "pages_processed", None)
        doc_size_bytes = getattr(usage_info, "doc_size_bytes", None)
        usage_parts = []
        if pages_processed is not None:
            usage_parts.append(f"pages_processed={pages_processed}")
        if doc_size_bytes is not None:
            usage_parts.append(f"doc_size_bytes={doc_size_bytes}")
        if usage_parts:
            print(f"Usage: {', '.join(usage_parts)}")



if __name__ == "__main__":
    main()
