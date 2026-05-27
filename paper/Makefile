PAPER=paper

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(PAPER).bib
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER) || true
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf

view: $(PAPER).pdf
	xdg-open $(PAPER).pdf
