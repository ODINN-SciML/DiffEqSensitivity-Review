#.ONESHELL:
#SHELL = /bin/bash

DOCNAME=main

all: tex

## tex               : Compiles latex into pdf in main repository and deletes all aux files
tex: compile-tex clean-tex
	mv tex/$(DOCNAME).pdf $(DOCNAME).pdf

## compile-tex       : Compile latex file with bibliografy
.PHONY: compile-tex
compile-tex:
	cd tex; pdflatex $(DOCNAME).tex
	# Depending the format for bib, use biber or bibtex
	cd tex; biber $(DOCNAME).bcf
	# bibtex $(DOCNAME).aux
	cd tex; pdflatex $(DOCNAME).tex
	cd tex; pdflatex $(DOCNAME).tex

## clean-tex         : Remove auxiliary files created during latex compilation
.PHONY: clean-tex
clean-tex:
	cd tex; rm *.blg *.bbl *.aux *.log *.toc *.bcf *.out *.xml



.PHONY : help
help : Makefile
	@echo ------------------
	@echo Make Help Commands
	@echo ------------------
	@sed -n 's/^##//p' $<
	@echo ------------------