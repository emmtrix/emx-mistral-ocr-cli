# emx-mistral-ocr-cli

CLI tool for converting PDF documents to Markdown or HTML using Mistral OCR.

## Features

- PDF -> Markdown (default) or HTML output
- Automatic output format detection from `--out` extension (`.html`/`.htm` -> HTML)
- Optional page selection via `--pages` (`1-12`, `2,5,10-12`, ...)
- Optional local PDF slicing before upload (`--slice-pdf`) to help with very large PDFs (e.g. >1000 pages)
- Optional extracted image export
- HTML mode with embedded HTML tables and built-in CSS styling
- Local chapter index analysis before OCR (`--analyze-index`)
- Retry handling for temporary Mistral API errors
- Safe output behavior (no overwrite without `--force`)

## Requirements

- Python 3.10+
- A valid Mistral API key in environment variable `MISTRAL_API_KEY`

## Installation

Install via pip:

```bash
pip install emx-mistral-ocr-cli
```

Install from source (repo checkout):

```bash
pip install -r requirements.txt
```

Optional (editable install with console script):

```bash
pip install -e .
```

## Development / Run from Source

If you want to run directly from a git checkout (without installing the package from PyPI), install dependencies and execute the script:

```bash
pip install -r requirements.txt
python mistral_ocr_cli.py <input.pdf> [options]
```

## Setup

Set your API key:

Linux/macOS (bash/zsh):

```bash
export MISTRAL_API_KEY="your_key_here"
```

Windows PowerShell / PowerShell:

```powershell
$env:MISTRAL_API_KEY="your_key_here"
```

Windows cmd.exe:

```cmd
set MISTRAL_API_KEY=your_key_here
```

## Usage

```bash
emx-mistral-ocr-cli <input.pdf> [options]
```

Show help:

```bash
emx-mistral-ocr-cli -h
```

## Common Examples

Default Markdown output:

```bash
emx-mistral-ocr-cli doc.pdf
```

Write Markdown to a specific file:

```bash
emx-mistral-ocr-cli doc.pdf --out result.md
```

HTML output (auto-selected by extension):

```bash
emx-mistral-ocr-cli doc.pdf --out result.html
```

Explicit HTML output:

```bash
emx-mistral-ocr-cli doc.pdf --output-format html --out result.html
```

Process only selected pages:

```bash
emx-mistral-ocr-cli doc.pdf --pages "1-20"
```

Slice selected pages locally before upload:

```bash
emx-mistral-ocr-cli doc.pdf --pages "1150-1200" --slice-pdf --out result.html --force
```

Disable images entirely:

```bash
emx-mistral-ocr-cli doc.pdf --no-images
```

Export images to custom directory:

```bash
emx-mistral-ocr-cli doc.pdf --images-dir extracted_images
```

Analyze chapter index locally (no OCR call):

```bash
emx-mistral-ocr-cli doc.pdf --analyze-index
```

Analyze chapter index and write it to file:

```bash
emx-mistral-ocr-cli doc.pdf --analyze-index --chapter-index-out index.tsv --force
```

## Options

- `--out <path>`: Output file path
- `--output-format {markdown,html}`: Output format (default: `markdown`)
- `--force`: Overwrite existing outputs
- `--pages "<spec>"`: 1-based page selection, e.g. `1-12`, `2,5,10-12`
- `--slice-pdf`: Build temporary sliced PDF locally before upload (requires `--pages`). Useful when Mistral rejects very large PDFs (e.g. >1000 pages) and you want to process it in chunks.
- `--images-dir <dir>`: Directory for extracted images (default: `<out_stem>_images`)
- `--no-images`: Disable image extraction/export
- `--image-limit <n>`: Maximum number of images to extract
- `--image-min-size <px>`: Minimum image width/height
- `--no-header-footer`: Disable header/footer extraction
- `--chapter-index-out <file>`: Write local chapter index output
- `--analyze-index`: Local chapter index analysis and exit

## Notes

- In HTML mode, OCR tables are requested as HTML and embedded into the final HTML document. HTML is generally more expressive than Markdown for complex layouts (e.g. tables with `colspan`/`rowspan`, which standard Markdown tables do not support).
- For large PDFs, `--slice-pdf` can still take time (PDF parsing/writing), but it reduces upload size and processed content and can avoid API errors for extremely large documents (e.g. >1000 pages).
- `--analyze-index` is useful to discover chapter boundaries and page numbers so you can select specific chapters via `--pages`.
