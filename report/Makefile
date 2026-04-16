PDF = Team_7_ArchX_paper
LATEX = pdflatex
BIBTEX = bibtex

all: $(PDF).pdf

$(PDF).pdf: $(PDF).tex references.bib
	$(LATEX) $(PDF).tex
	$(BIBTEX) $(PDF)
	$(LATEX) $(PDF).tex
	$(LATEX) $(PDF).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

open: $(PDF).pdf
	open $(PDF).pdf

.PHONY: all clean open
