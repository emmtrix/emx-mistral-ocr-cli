# AGENTS.md

## Project Purpose
This repository is for developing CLI tools around **Mistral OCR**.
The goal is robust, scriptable processing of documents (especially PDF files) into structured Markdown/text.

## Working Principles
- Prefer small, clearly scoped changes.
- Keep CLI behavior stable (arguments, exit codes, stdout/stderr).
- Write fault-tolerant code with clear error messages.
- Avoid unnecessary dependencies and hidden side effects.
- Only create/amend commits when the user explicitly asks to "commit" (or "amend"). This permission applies to that single action only.

## Code Guidelines
- Use Python 3 and keep style consistent with existing code.
- Prefer clear, descriptive function and variable names.
- Separate IO (files/network/CLI) from core logic where practical.
- Add comments only where logic is not immediately obvious.

## CLI Guidelines
- Clearly document inputs via arguments/flags.
- Do not use exit code `0` for failure cases.
- Send normal output to `stdout`, errors to `stderr`.
- Do not silently overwrite existing files on output operations.

## Tests and Verification
- Validate changes with targeted tests or reproducible example commands whenever possible.
- New features should cover at least one clear happy path and one error case.
- Do not refactor unrelated, unaffected areas without need.

## Documentation
- Update `README` and usage examples when adding new flags or workflows.
- Keep examples realistic and directly executable.
