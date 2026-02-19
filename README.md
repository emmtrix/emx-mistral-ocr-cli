# emx-mistral-ocr-cli

CLI tool for converting PDF documents to Markdown or HTML using Mistral OCR.

## Features

- PDF -> Markdown (default) or HTML output
- Automatic output format detection from `--out` extension (`.html`/`.htm` -> HTML)
- Optional page selection via `--pages` (`1-12`, `2,5,10-12`, ...)
- Optional local PDF slicing before upload (`--slice-pdf`)
- Optional extracted image export
- HTML mode with embedded HTML tables and built-in CSS styling
- Local chapter index analysis before OCR (`--analyze-index`)
- Retry handling for temporary Mistral API errors
- Safe output behavior (no overwrite without `--force`)

## Requirements

- Python 3.10+
- A valid Mistral API key in environment variable `MISTRAL_API_KEY`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

Set your API key:

```bash
export MISTRAL_API_KEY="your_key_here"   # Linux/macOS
```

```powershell
$env:MISTRAL_API_KEY="your_key_here"     # PowerShell
```

## Usage

```bash
python mistral_ocr_cli.py <input.pdf> [options]
```

Show help:

```bash
python mistral_ocr_cli.py -h
```

## Common Examples

Default Markdown output:

```bash
python mistral_ocr_cli.py doc.pdf
```

Write Markdown to a specific file:

```bash
python mistral_ocr_cli.py doc.pdf --out result.md
```

HTML output (auto-selected by extension):

```bash
python mistral_ocr_cli.py doc.pdf --out result.html
```

Explicit HTML output:

```bash
python mistral_ocr_cli.py doc.pdf --output-format html --out result.html
```

Process only selected pages:

```bash
python mistral_ocr_cli.py doc.pdf --pages "1-20"
```

Slice selected pages locally before upload:

```bash
python mistral_ocr_cli.py doc.pdf --pages "1150-1200" --slice-pdf --out result.html --force
```

Disable images entirely:

```bash
python mistral_ocr_cli.py doc.pdf --no-images
```

Export images to custom directory:

```bash
python mistral_ocr_cli.py doc.pdf --images-dir extracted_images
```

Analyze chapter index locally (no OCR call):

```bash
python mistral_ocr_cli.py doc.pdf --analyze-index
```

Analyze chapter index and write it to file:

```bash
python mistral_ocr_cli.py doc.pdf --analyze-index --chapter-index-out index.tsv --force
```

## Options

- `--out <path>`: Output file path
- `--output-format {markdown,html}`: Output format (default: `markdown`)
- `--force`: Overwrite existing outputs
- `--pages "<spec>"`: 1-based page selection, e.g. `1-12`, `2,5,10-12`
- `--slice-pdf`: Build temporary sliced PDF locally before upload (requires `--pages`)
- `--images-dir <dir>`: Directory for extracted images (default: `<out_stem>_images`)
- `--no-images`: Disable image extraction/export
- `--image-limit <n>`: Maximum number of images to extract
- `--image-min-size <px>`: Minimum image width/height
- `--no-header-footer`: Disable header/footer extraction
- `--chapter-index-out <file>`: Write local chapter index output
- `--analyze-index`: Local chapter index analysis and exit

## Notes

- Use `--pages` (plural).  
  `--page` may appear to work in some shells due to argument abbreviation behavior, but `--pages` is the supported option.
- In HTML mode, OCR tables are requested as HTML and embedded into the final HTML document.
- For large PDFs, `--slice-pdf` can still take time (PDF parsing/writing), but it reduces upload size and processed content.
