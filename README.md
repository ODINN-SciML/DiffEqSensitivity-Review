# Differentiable programming for Differential equations: a review

[![CI](https://github.com/ODINN-SciML/DiffEqSensitivity-Review/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ODINN-SciML/DiffEqSensitivity-Review/actions/workflows/CI.yml)
![example workflow](https://github.com/ODINN-SciML/DiffEqSensitivity-Review/actions/workflows/latex.yml/badge.svg)
![example workflow](https://github.com/ODINN-SciML/DiffEqSensitivity-Review/actions/workflows/biblatex.yml/badge.svg)

[![All Contributors](https://img.shields.io/github/all-contributors/ODINN-SciML/DiffEqSensitivity-Review?color=ee8449&style=flat-square)](#contributors)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

### ⚠️ New preprint available! 📖 ⚠️

The review paper is now available as a preprint on arXiv: https://arxiv.org/abs/2406.09699 

If you want to cite this work, please use this BibTex citation:

```bibtex
@misc{sapienza2024differentiable,
      title={Differentiable Programming for Differential Equations: A Review}, 
      author={Facundo Sapienza and Jordi Bolibar and Frank Schäfer and Brian Groenke and Avik Pal and Victor Boussange and Patrick Heimbach and Giles Hooker and Fernando Pérez and Per-Olof Persson and Christopher Rackauckas},
      year={2024},
      eprint={2406.09699},
      archivePrefix={arXiv},
      primaryClass={id='math.NA' full_name='Numerical Analysis' is_active=True alt_name='cs.NA' in_archive='math' is_general=False description='Numerical algorithms for problems in analysis and algebra, scientific computation'}
}
```

---

This respository contains all the text, code and figures used for the review paper about sentitivity methods for differential equations. This topic received different names in different communities, but the core problem is quite simple. Given a system of differential equations
```math
\frac{du}{dt} = f(u, \theta, t),
```
with $u \in \mathbb R^n$ the unknow solution and $\theta \in \mathbb R^p$ a vector of parameters, how do we compute the gradient of a loss function
```math
\mathcal L (\theta) = L ( u(\cdot, \theta) )
```
with respect to the parameters $\theta$ of the dynamical model?

There are different methods that try to solve this problem, and we had roughly classify them in the following scheeme.

<p align="center">
	<img src="tex/figures/scheme-methods.png" width="50%">
</p>  
<p align="center">	
	<i>Schematic classification of different methods to compute gradients of functions involving solutions of differential equations.</i>
</p>

The goal of this review is to revisit all this methods and compare them.

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://facusapienza.org"><img src="https://avatars.githubusercontent.com/u/39526081?v=4?s=100" width="100px;" alt="Facundo Sapienza"/><br /><sub><b>Facundo Sapienza</b></sub></a><br /><a href="#code-facusapienza21" title="Code">💻</a> <a href="#doc-facusapienza21" title="Documentation">📖</a> <a href="#ideas-facusapienza21" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-facusapienza21" title="Maintenance">🚧</a> <a href="#projectManagement-facusapienza21" title="Project Management">📆</a> <a href="#research-facusapienza21" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://jordibolibar.wordpress.com"><img src="https://avatars.githubusercontent.com/u/2025815?v=4?s=100" width="100px;" alt="Jordi Bolibar"/><br /><sub><b>Jordi Bolibar</b></sub></a><br /><a href="#doc-JordiBolibar" title="Documentation">📖</a> <a href="#ideas-JordiBolibar" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-JordiBolibar" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://frankschae.github.io"><img src="https://avatars.githubusercontent.com/u/42201748?v=4?s=100" width="100px;" alt="frankschae"/><br /><sub><b>frankschae</b></sub></a><br /><a href="#code-frankschae" title="Code">💻</a> <a href="#doc-frankschae" title="Documentation">📖</a> <a href="#ideas-frankschae" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-frankschae" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://bgroenke.com"><img src="https://avatars.githubusercontent.com/u/841306?v=4?s=100" width="100px;" alt="Brian Groenke"/><br /><sub><b>Brian Groenke</b></sub></a><br /><a href="#code-bgroenks96" title="Code">💻</a> <a href="#doc-bgroenks96" title="Documentation">📖</a> <a href="#ideas-bgroenks96" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://avik-pal.github.io"><img src="https://avatars.githubusercontent.com/u/30564094?v=4?s=100" width="100px;" alt="Avik Pal"/><br /><sub><b>Avik Pal</b></sub></a><br /><a href="#code-avik-pal" title="Code">💻</a> <a href="#doc-avik-pal" title="Documentation">📖</a> <a href="#test-avik-pal" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://vboussange.github.io"><img src="https://avatars.githubusercontent.com/u/28376122?v=4?s=100" width="100px;" alt="vboussange"/><br /><sub><b>vboussange</b></sub></a><br /><a href="#doc-vboussange" title="Documentation">📖</a> <a href="#ideas-vboussange" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-vboussange" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/heimbach"><img src="https://avatars.githubusercontent.com/u/5150349?v=4?s=100" width="100px;" alt="Patrick Heimbach"/><br /><sub><b>Patrick Heimbach</b></sub></a><br /><a href="#doc-heimbach" title="Documentation">📖</a> <a href="#ideas-heimbach" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-heimbach" title="Mentoring">🧑‍🏫</a> <a href="#research-heimbach" title="Research">🔬</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gileshooker"><img src="https://avatars.githubusercontent.com/u/12737775?v=4?s=100" width="100px;" alt="gileshooker"/><br /><sub><b>gileshooker</b></sub></a><br /><a href="#doc-gileshooker" title="Documentation">📖</a> <a href="#ideas-gileshooker" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-gileshooker" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://statistics.berkeley.edu/people/fernando-perez"><img src="https://avatars.githubusercontent.com/u/57394?v=4?s=100" width="100px;" alt="Fernando Pérez"/><br /><sub><b>Fernando Pérez</b></sub></a><br /><a href="#doc-fperez" title="Documentation">📖</a> <a href="#financial-fperez" title="Financial">💵</a> <a href="#ideas-fperez" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-fperez" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.chrisrackauckas.com"><img src="https://avatars.githubusercontent.com/u/1814174?v=4?s=100" width="100px;" alt="Christopher Rackauckas"/><br /><sub><b>Christopher Rackauckas</b></sub></a><br /><a href="#code-ChrisRackauckas" title="Code">💻</a> <a href="#doc-ChrisRackauckas" title="Documentation">📖</a> <a href="#financial-ChrisRackauckas" title="Financial">💵</a> <a href="#ideas-ChrisRackauckas" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-ChrisRackauckas" title="Mentoring">🧑‍🏫</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


## Code :computer:

We provide code for the different simulations and examples we exhibit in the project. Most of this code is provided in the Julia programming language, given that many of the libraries to perform sensitivity analysis are supported there, plus Julia solvers for differential equations are currently the state of the art in scientific computing.

## How to use this repository :question:

This repository is organized in a way that contains most of the important elements of modern scientific workflow. We have included the following elements:
- `tex`: This is the folder where all the latex text belongs and what we use to compile `main.pdf`.
- `code`: Folder with both Jupyter notebooks and Julia scripts.
- `CI`: Continuous integration with GitHub actions to automatically compile and commit the manuscript. 
- `Makefile`: make file that automatizes all the commands that can be executed within this repository.

### Continuous integration

This repository has a [workflow](https://github.com/ODINN-SciML/DiffEqSensitivity-Review/blob/main/.github/workflows/latex.yml) implemented that automatically compiles the latex files into the file `main.pdf` and then commits this file directy to the repository. 
If you are working from your fork, this action should also work and you should be able to generate the pdf file automatically using GitHub actions. 
In order to trigger the GitHub action to compile and commit the Latex file, you just need to inclide the word `latex` in your commit message. 

This repository also includes an action to automatically merge all `*.bib` files into one single bibliography file `tex/bibliography.bib` that includes just the references that are cited in any of the `*.tex` files. 	
In order to trigger this action, just include the word `bib-merge` in your commit message. 

### Makefile

Make is an old but very useful technology that allows automation of computing processes. From the directory where you have this respository you can enter `make help` from a terminal to display the different functionalities currently supported in our `Makefile`. We currently support the following operations:
- `make tex`: Compiles the `main.tex` latex file inside the folder `tex` with its respective bibliografy, deletes auxiliaries files in the process and them move the generated pdf file to the home directory.

## Open Science from Scratch: contribute to the project! :wave:

This review started with some of the authors willing to understand this tools in a comprehensive way and gathering references from fields like statistics, applied mathematics and computer science. Unhappy with the lack of a general compendium of the different methods that exists to address this problem, we had decided to make a single document where all the methods can coexists under common ground and can be compareded under their different scopes and domains of applications.

We are driving by learning and undersranding and the original authors of this report decided to manage this as an open science project from the beginning. What does this entitles? Anyone interested in participating and contrubuting is welcome to join. We beleive that science will benefit for more open collaborations happening under the basis of people trying to understand a topic.

We encourage contributors to participate in this project! If you are interested in contributing, there are many ways in which you can help build this:
- :collision: **Report bugs in the code.** You can report problems with the code by oppening issues under the `Issues` tab in this repository. Please explain the problem you encounter and try to give a complete description of it so we can follow up on that.
- :books: **Suggest new bibliography.** If you are aware of references that may be useful to explore and expand this review, you can report it by creating an `Issue` in this repository, with the title of the issue being the title of the paper and adding the label `paper` to the issue.
- :bulb: **Request new features and explanations.** If there is an important topic or example that you feel falls under the scope of this review and you would like us to include it, please request it! We are looking for new insights into what the community wants to learn.


## Contact 

If you have any questions or want to reach out, feel free to send us an email to `fsapienza@berkeley.edu`.

## License 

The content of this project itself is licensed under the [Creative Commons Attribution 3.0 Unported license](https://creativecommons.org/licenses/by/3.0/), and the underlying source code used to format and display that content is licensed under the [MIT license](LICENSE).

<!-- ## Reference

If you find this article useful, you can cite it as follows 
```

``` -->


