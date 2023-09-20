---
title: 'Self-Guided Decision Support Groundwater Modelling with Python'
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



# List all authors and affiliations.
# Describe the submission, and explain its eligibility for JOSE.
# Include a “Statement of Need” section, explaining how the submitted artifacts contribute to # computationally enabled teaching and learning, and describing how they might be adopted by # others.
# For learning modules, describe the learning objectives, content, instructional design, and # experience of use in teaching and learning situations.
# Tell us the “story” of the project: how did it come to be?
# Cite key references, including a link to the open archive of the sofware or the learning module.


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

pyEMU promotes the integration of uncertainty assessments into environmental modeling analyses, thereby improving the precision of decisions related to resource management based on model outputs. Additionally, it offers users an exploratory platform for enhancing their comprehension of uncertainty analysis principles. Although initially designed for groundwater modeling, pyEMU's methodologies are versatile and can be applied to diverse numerical environmental models, as long as they can be manipulated using text files and generate results that can be automatically extracted without manual interference.


# Story of the project

The Groundwater Modelling Decision Support Initiative ([GMDSI](https://gmdsi.org)) is an industry-backed and industry-aligned initiative. Established in mid-2019, its primary goal is to enhance the role of groundwater modeling in groundwater management, regulatory processes, and decision-making. At the core of GMDSI's mission lies the numerical simulation of groundwater movement and processes. Often, data related to groundwater are limited, leading to uncertainties in simulator predictions. However, despite this uncertainty, decisions must be made, and associated risks must be assessed. Modeling plays a central role in the evaluation of these risks.

GMDSI is dedicated to promoting, facilitating, and providing support for the improved utilization of modeling in decision support processes. Its activities endeavor to elevate the role of groundwater modeling in decision-making processes, recognizing the importance of model partner software and offering a range of activities aimed at industry engagement, education, practical examples, research, and software development.

A majority of groundwater modelers typically rely on Graphical User Interfaces (GUIs) for their modeling needs. However, it's important to note that each GUI has its unique characteristics and varying degrees of compatibility with external software like PEST. Creating educational materials for these GUIs would necessitate tailoring content to each GUI's specific features, potentially lagging behind the latest developments, and obtaining cooperation from the GUI developers themselves.

It's worth noting that decision-support modeling often demands capabilities that surpass what current GUIs can offer. For example, many of GMDSI's worked examples rely on custom-designed utilities or the integration of different software components. Currently, a significant portion of users may not have the expertise to independently implement such advanced approaches. Furthermore, the manual preparation of input files for implementing these complex workflows can be time-consuming. Programmatic workflows, such as those facilitated by ``pyEMU``, offer advantages by reducing the time and user input required for setup and execution. This approach is somewhat analogous to the role played by a GUI but offers added flexibility, allowing users to customize and design their own functions and utilities as needed. However, it comes with the drawback of increased potential for user-introduced errors.

Over time, more modelers are turning to Python packages like ``FloPy`` and ``pyEMU`` for model and PEST++ setup, moving away from GUIs. Unfortunately, the adoption of this approach is hindered by a steep learning curve primarily due to the scarcity of user-friendly training materials. The GMDSI tutorial notebooks aim to address this gap by providing a comprehensive, self-guided, and open-source resource for learning decision-support modeling workflows with Python.



USGS origin...? #TODO

...fusion


# Resources

The [GMDSI](https://gmdsi.org) hosts an extensive range of resources  and educational material on decision support modelling. These include numerous instructional video lectures, webinar recordings, non-programatic workflow tutorials, as well as  worked example reports desribing real-world applications. 

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

Part 1 focuses on the Gauss-Levenberg Marquardt (GLM) approach to parameter estimation and  associated uncertainty analysis in a groundwater modelling context. This was the foundation of the PEST software for multiple decades and the theory continues to resonate through newer techniques.

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