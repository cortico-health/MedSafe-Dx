.PHONY: paper clean

paper: paper.pdf

paper.pdf: paper.tex paper.bib
	pdflatex -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.pdf
