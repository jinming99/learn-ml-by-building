# Learn ML by Building lecture-note template

A self-contained LuaLaTeX template for concise, professional machine-learning lecture notes. Copy this entire folder for each lecture, then replace the metadata, section text, figures, and tables.

The visual foundation is the [`kaohandt` class from the kaobook project](https://github.com/fmarotta/kaobook/tree/master). The class, its license, the upstream manifest, and a compatible `tcolorbox` release are included so copied lecture folders remain buildable. Local adaptations are recorded in `THIRD-PARTY-NOTICES.md`.

## Quick start

1. Copy this folder and rename the copy for the lecture.
2. Edit `metadata.tex` and set a stable PDF filename in `Makefile`.
3. Replace the template guidance in `sections/` with the lecture's concise narrative.
4. Create every figure in `figures/*.tex` with TikZ or PGFPlots; create every table directly in LaTeX.
5. Run `make`, then render and inspect every page with `make render`.

The compiled PDF is written to `output/pdf/`. The retained `lecture-notes-template-preview.pdf` demonstrates the page hierarchy, figure format, table format, callouts, captions, and compact ending.

## Folder map

- `metadata.tex` - lecture-specific title, PDF metadata, and acknowledgment text
- `main.tex` - stable document shell and learning-goal block
- `lecture-notes.sty` - shared visual system
- `sections/` - authored lecture narrative
- `figures/` - figures included in the lecture
- `examples/figures/` - current native-LaTeX figure patterns for reference only
- `output/pdf/` - upload-ready PDF

The template adds a short acknowledgment to every standalone PDF. Adjust `\LectureAcknowledgmentText` in `metadata.tex` only when the assistance or source material differs.
