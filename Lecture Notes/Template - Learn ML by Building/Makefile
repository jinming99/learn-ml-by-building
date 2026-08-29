SHELL := /bin/sh
LUALATEX ?= lualatex
JOB ?= lecture-notes-template-preview
BUILD := output/build
FINAL := output/pdf/$(JOB).pdf
RENDER := tmp/pdfs/render

.PHONY: all render clean

all: $(FINAL)

$(FINAL): main.tex metadata.tex lecture-notes.sty $(wildcard sections/*.tex) $(wildcard figures/*.tex) $(wildcard vendor/tcolorbox/*) kaohandt.cls kao.sty
	mkdir -p $(BUILD) output/pdf
	TEXINPUTS=.:vendor/tcolorbox//: $(LUALATEX) -interaction=nonstopmode -halt-on-error -file-line-error -jobname=$(JOB) -output-directory=$(BUILD) main.tex
	TEXINPUTS=.:vendor/tcolorbox//: $(LUALATEX) -interaction=nonstopmode -halt-on-error -file-line-error -jobname=$(JOB) -output-directory=$(BUILD) main.tex
	cp $(BUILD)/$(JOB).pdf $(FINAL)

render: $(FINAL)
	mkdir -p $(RENDER)
	pdftoppm -png -r 160 $(FINAL) $(RENDER)/page

clean:
	test ! -d $(BUILD) || find $(BUILD) -type f -delete
	test ! -d tmp || find tmp -type f -delete
