MAIN = root
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean

all: $(MAIN).pdf clean

$(MAIN).pdf: $(MAIN).tex sections/*.tex root.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls \
	      $(MAIN).fdb_latexmk $(MAIN).synctex.gz $(MAIN).nav $(MAIN).snm \
	      $(MAIN).vrb $(MAIN).run.xml $(MAIN)-blx.bib \
	      sections/*.aux

cleanall: clean
	rm -f $(MAIN).pdf
