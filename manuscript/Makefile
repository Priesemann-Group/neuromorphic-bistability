# Always run latexmk
.PHONY: all plots clean

all: build/main.pdf build/si.pdf

build/si.pdf: plots si.tex
	mkdir -p build
	latexmk -shell-escape -jobname=build/si -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make si.tex

build/main.pdf: plots main.tex
	mkdir -p build
	latexmk -shell-escape -jobname=build/main -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make main.tex
clean:
	latexmk -CA
	make clean -C plotting
plots:
	make -C plotting
