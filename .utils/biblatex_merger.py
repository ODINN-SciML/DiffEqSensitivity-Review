#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python script to find all calls to cites in a series of .tex files and combine the respective
biblatex entries into one single formated biblatex file.

How to use
> python biblatex_merger.py <working directory> <output ff
"""

__version__ = "0.0.1"
__author__  = "Facundo Sapienza"


import sys

import re
import glob, os
import warnings

from pybtex.database.input import bibtex
from pybtex.database import BibliographyData

# This is so we can ignore duplicated entries
import pybtex.errors
pybtex.errors.set_strict_mode(False)

rx = re.compile(r'''(?<!\\)%.+|(\\(?:no)?citep?\{((?!\*)[^{}]+)\})''')

def find_extension(extention, path):
    """
    Find all files with required extension in path system
    """

    res = []   

    for root, dirs, files in os.walk(path):
        for file in files:
            if(file.endswith(extention)):
                res.append(os.path.join(root,file))
                
    return res

def merge_bib(path):

    # List of latex abd bib files to scan for references
    tex_files = find_extension(".tex", ".")
    bib_files = find_extension(".bib", ".")

    # Attach all the text in files to a single string
    latex = ""
    authors = []
    bib_data = []
    # New bibliography 
    filtered_bib_data = BibliographyData()
    
    for file in tex_files:
        # with open("tex/sections/adjoint-state.tex", 'r') as f:
        with open(file, 'r') as f:
            latex += f.read()
    
    # Format author entries for cases like \cite{author1, author2}
    authors_unformated = [m.group(2) for m in rx.finditer(latex) if m.group(2)]

    # Find all the authors entries
    for author_ref in authors_unformated:
        new_authors = author_ref.split(',')
        new_authors = [x.strip() for x in new_authors]
        authors += new_authors   
    
    # Collect all bib data 
    for file in bib_files:
        # print(file)
        # if file == "./bib_test.bib":
        #     continue
        parser = bibtex.Parser()
        bib_data.append(parser.parse_file(file))

    for entry in authors: 
        ref_founded = False
        for bib_source in bib_data:
            try:
                if not ref_founded:
                    filtered_bib_data.add_entry(entry, bib_source.entries[entry])
                    ref_founded = True
                    # print("Reference found {}".format(entry))
            except:
                pass
            if not ref_founded:
                warnings.warn("Reference not found: {}".format(entry))

    return filtered_bib_data


def save_bib(bib_data, file):
    bib_data.to_file(file, bib_format="bibtex")


if __name__ == "__main__":
    working_directory = sys.argv[1]
    output_file = sys.argv[2]
    filtered_bib_data = merge_bib(working_directory)
    save_bib(filtered_bib_data, output_file)
