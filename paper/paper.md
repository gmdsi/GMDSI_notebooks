---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - groundwater modelling
  - environemntal modelling
  - decision-support
  - uncertainty analysis
authors:
  - name: Rui T. Hugman
    orcid: 0000-0003-0891-3886
    affiliation: 1 
  - name: Jeremy T. White
    orcid: xxxx-xxxx-xxxx-xxxx
    affiliation: 1 
  - name: Mike Fienen
    orcid: xxxx-xxxx-xxxx-xxxx
    affiliation: "2,3,4" # (Multiple affiliations must be quoted)
  - name: Brioch Hemmings
    orcid: xxxx-xxxx-xxxx-xxxx
    affiliation: "3" # (Multiple affiliations must be quoted)
  - name: Katie Markovitch
    orcid: xxxx-xxxx-xxxx-xxxx
    affiliation: 1 

affiliations:
 - name: INTERA Geosciences
   index: 1
 - name: Institution 2 #TODO
   index: 2
 - name: Institution 3 #TODO
   index: 3
date: 20 September 2023
bibliography: paper.bib



List all authors and affiliations.
Describe the submission, and explain its eligibility for JOSE.
Include a “Statement of Need” section, explaining how the submitted artifacts contribute to computationally enabled teaching and learning, and describing how they might be adopted by others.
For learning modules, describe the learning objectives, content, instructional design, and experience of use in teaching and learning situations.
Tell us the “story” of the project: how did it come to be?
Cite key references, including a link to the open archive of the sofware or the learning module.


# # Citations
# 
# Citations to entries in paper.bib should be in
# [rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
# format.
# 
# For a quick reference, the following citation commands can be used:
# - `@author:2001`  ->  "Author et al. (2001)"
# - `[@author:2001]` -> "(Author et al., 2001)"
# - `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"
# 
# # Figures
# 
# Figures can be included like this: ![Example figure.](figure.png)

---

# Summary


The [GMDSI tutorial notebooks repository](https://github.com/gmdsi/GMDSI_notebooks/tree/main)



# Statement of Need
...geologists are dumb.


# Story of the project
GMDSI origin...

GMDSI "non-programatic" tutorials...

USGS origin...?

...fusion

# Resources
link to GMDSI webiste, documents and videos

link to pest manuals

link to pestpp repo + manuals + pyemu

...ref to flopy? meh

# Contents and Instructional Design
The tutorial notebooks are structured into three main parts:

## Part 0: Introductory Background

Part0 serves as the foundation, providing essential background material. Learners are encouraged to reference notebooks in Part0 to polish their understnading of concepts they encounter in Parts 1 and 2. Part0 is not intended to be a comprehensive resource for all background material, but rather to establish a solid understanding of the basics. 

Each notebook in Part 0 is standalone and covers a unique topic. These include:
 - Introduction to a synthetic model known as the "Freyberg" model. This model is used as a consistent example throughout the tutorial exercises, allowing learners to apply concepts in a practical context.
 - An introduction to the `pyemu` Python package that is used to complement and interface with PEST/PEST++.
 - Explanation of fundamental mathematical concepts that are relevant and will be encountered throughout the tutorial notebooks.


## Part 1: Introduction to PEST and the Gauss-Levenberg Marquardt Approach

Part 1 focuses on the Gauss-Levenberg Marquardt approach to parameter estimation and  associated uncertainty analysis in a groundwater modelling context. TThis was the foundation of the PEST software for multiple decades and the theory continues to resonate through newer techniques.

Part 1 is designed to be accessible without strict sequential dependencies. Learners have the flexibility to explore its contents in any order that suits their preferences or needs. These include:
 - Introduction to concepts such as non-uniquesness, identifiability, and equifinality.
 - Introduction to the PEST control file and the PEST/PEST++ interface.
 - Explorating the challenges of parameterization schemes on predictive ability, as well as how to mitigate them.
 - Introducing first-order second-moment (FOSM) and prior Monte Carlo uncertainty analysis approaches.


## Part 2: Python-based Decision-Support Modelling Workflows
Part 2 expands on the foundational knowledge gained in Part 1 and delves into advanced topics related to ensemble-based parameter estimation, uncertainty analysis and optimization methods. These advanced topics include management optimization and sequential data assimilation. Topics are laid out in manner that reflects real-world workflows, with a focus on practical application of concepts and problem solving.

Part 2 is structured with a specific order for learners to follow to ensure a logical progression of topics, inline with a real-world applied workflow. Learners have the option to explore various sequences covering advanced topics, such as:
 - Prior Monte Carlo
 - GLM and Data Worth, 
 - Ensemble-based history matching and uncertainty analysis with PEST++IES, 
 - Sequential data assimilation with PEST++DA, and 
 - optimization and multi-objective optimization under uncertainty with PEST++OPT and PEST++MOU.
 
Each of these sequences comprises multiple notebooks to be executed in a specified order. They demonstrate how to execute the workflow, interpret results, and apply the concepts to real-world problems.

In summary, the tutorial notebooks are organized to guide learners through a structured learning experience in the field of decision-support groundwater modelling. Part 0 provides foundational knowledge, while Parts 1 and 2 offer progressively advanced content.


# Experience of use in teaching and learning situations
...feedback from students and instructors...



...self-guided online series, recordings link

# Acknowledgements

The tutorials were originally developed with support from the U.S Geological Survey (USGS) and support from USGS continues. Continued development and support is funded by the Groundwater Modelling Decision Support Initiative (GMDSI). GMDSI is jointly funded by BHP and Rio Tinto. We thank users and stress-testers for their valuable feedback and continued community contributions to the repository.

# References