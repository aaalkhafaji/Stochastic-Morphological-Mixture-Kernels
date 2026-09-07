# Editable LaTeX supplementary guide

`JVCIR_Supplementary_Guide.tex` is the standalone, editable LaTeX version of the
four-page Supplementary Material S1 guide. All five tables are embedded directly
in the file. It uses standard TeX Live/MiKTeX packages and needs no external data,
figures, bibliography, Python, or additional project files to compile.

Compile from the package root:

```bash
pdflatex -interaction=nonstopmode -halt-on-error JVCIR_Supplementary_Guide.tex
pdflatex -interaction=nonstopmode -halt-on-error JVCIR_Supplementary_Guide.tex
```

The result is `JVCIR_Supplementary_Guide.pdf`. In Overleaf, upload the `.tex` file
and set it as the main document, using the pdfLaTeX compiler.

The archive supplies this compiled guide under its existing manuscript-linked
name, `Supplementary_Guide.pdf`. To compile directly to that filename, use:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Supplementary_Guide JVCIR_Supplementary_Guide.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Supplementary_Guide JVCIR_Supplementary_Guide.tex
```

This LaTeX source preserves the original guide's text and the reported numerical
tables, with mathematical symbols expressed as native LaTeX. It adds no new
experiment or scientific claim. The explicit page breaks preserve the guide's
four sections as four pages; longer edits may require adjusting those breaks.

The embedded tables are a snapshot of the frozen results. The existing
`code/build_supplement_guide.py` continues to regenerate the guide from result
files in its original ReportLab layout when the Python reproduction workflow is
run. It does not edit the new LaTeX source. If results are intentionally changed,
update the embedded LaTeX tables to match; if only the LaTeX wording or layout is
edited, compile the `.tex` directly to preserve those edits.

After intentional package changes, remove temporary LaTeX build files from the
release folder and refresh the file manifest with
`python code/check_package.py --write-manifest`, then verify it with
`python code/check_package.py`. The provided ZIP already has current checksums.
