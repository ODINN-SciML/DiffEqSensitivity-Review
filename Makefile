.ONESHELL:
SHELL = /bin/bash


## tex               : Compiles latex file into pdf version


## clean             : Remove output files
.PHONY : clean
clean : 
	rm -f results/*


.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<