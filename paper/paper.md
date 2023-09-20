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

The [GMDSI tutorial notebooks repository](https://github.com/gmdsi/GMDSI_notebooks/tree/main) provide learners with a comprehensive set of tutorials for self-guided training on decision-support groundwater modelling using Python-based tools. Although targeted at groundwater modelling, they are readily transferable to other environmental modelling workflows. The tutorials are divided into three parts. The first covers fundamental theoretical concepts. These are intended as background reading for reference on an as-needed basis. Tutorials in the second part introduce learners to some of the core concepts parameter estimation in a groundwater modelling context, as well as providing a gentle introduction to the ``PEST``, ``PEST++`` and ``pyEMU`` software. Lasty, the third part demonstrates how to implement high-dimensional applied decision-support modelling workflows. Their aim is to provide examples of both “how to use” the software as well as “how to think” about using the software. 


# Statement of Need

Effective environmental management necessitates transparent acknowledgment of uncertainties in critical decision-making predictions, coupled with efforts to mitigate these uncertainties, especially when significant risks accompany management outcomes. The significance of uncertainty quantification (UQ) and parameter estimation (PE) in environmental modeling for decision support is widely acknowledged. UQ provides estimates of outcome uncertainty, while PE reduces this uncertainty through assimilating data. 

Implementing high-dimensional UQ and PE in real-world modeling can be challenging due to both theoretical complexity and practical logistics. Limited project time and funding also often hinder their application. Open-source software such as PEST `[@pest]` and PEST++ `[@whitepestpp]` provide tools for underaking UQ and PE analyses. However, the steep learning curve associated with their use and the lack of user-friendly training materials have been a barrier to entry.

There is a growing demand within the environemntal modelling community for transparent, reproducible, and accountable modeling processes, driven by the need for increased credibility and rigor in computational science and environmental simulation `[@white11rapid]`. While some script-based tools enhance the reproducibility of forward model construction `[@flopy]`, they often overlook UQ and PE analyses. In decision-support scenarios, these analyses are equally vital for robust model deployment as the forward model itself. 

The uptake of Python for environmental modeling has increased in recent years, due to its open-source nature, user-friendly syntax, and extensive scientific libraries. Python-based tools have been developed to facilitate UQ and PE analyses, such as ``pyEMU`` `[@White_A_python_framework_2016; @white2021towards]`. ``pyEMU`` is a Python package that provides a framework for implementing UQ and PE analyses with PEST and PEST++. It offers a range of capabilities, including parameter estimation, uncertainty analysis, and optimization. Although initially designed for groundwater modeling, ``pyEMU``'s methodologies are versatile and can be applied to diverse numerical environmental models, as long as they can be manipulated using text files and generate results that can be automatically extracted without manual interference.

The tutorial notebooks discussed herein provide a comprehensive, self-guided, and open-source resource for learning decision-support modeling workflows with Python. They are designed to be accessible to a broad audience, including students, researchers, and practitioners who aim to undertake applied environmental decision-support modelling. 


# Story of the Project

The Groundwater Modelling Decision Support Initiative ([GMDSI](https://gmdsi.org)) is an industry-backed and industry-aligned initiative. Established in mid-2019, its primary goal is to enhance the role of groundwater modeling in groundwater management, regulatory processes, and decision-making. At the core of GMDSI's mission lies the numerical simulation of groundwater movement and processes. Often, data related to groundwater are limited, leading to uncertainties in simulator predictions. However, despite this uncertainty, decisions must be made, and associated risks must be assessed. Modeling plays a central role in the evaluation of these risks.

GMDSI is dedicated to promoting, facilitating, and providing support for the improved utilization of modeling in decision support processes. Its activities endeavor to elevate the role of groundwater modeling in decision-making processes, recognizing the importance of model partner software and offering a range of activities aimed at industry engagement, education, practical examples, research, and software development.

A majority of groundwater modelers typically rely on Graphical User Interfaces (GUIs) for their modeling needs. However, it's important to note that each GUI has its unique characteristics and varying degrees of compatibility with external software like PEST. Creating educational materials for these GUIs would necessitate tailoring content to each GUI's specific features, potentially lagging behind the latest developments, and obtaining cooperation from the GUI developers themselves.

It's worth noting that decision-support modeling often demands capabilities that surpass what current GUIs can offer. For example, many of GMDSI's worked examples rely on custom-designed utilities or the integration of different software components. Currently, a significant portion of users may not have the expertise to independently implement such advanced approaches. Furthermore, the manual preparation of input files for implementing these complex workflows can be time-consuming. Programmatic workflows, such as those facilitated by ``pyEMU``, offer advantages by reducing the time and user input required for setup and execution. This approach is somewhat analogous to the role played by a GUI but offers added flexibility, allowing users to customize and design their own functions and utilities as needed. However, it comes with the drawback of increased potential for user-introduced errors.

Over time, more modelers are turning to Python packages like ``FloPy`` and ``pyEMU`` for model and PEST++ setup, moving away from GUIs. Unfortunately, the adoption of this approach is hindered by a steep learning curve primarily due to the scarcity of user-friendly training materials. The [GMDSI tutorial notebooks](https://github.com/gmdsi/GMDSI_notebooks/tree/main) aim to address this gap by providing a comprehensive, self-guided, and open-source resource for learning decision-support modeling workflows with Python.



USGS origin...? #TODO




# Resources

A webinar hosted by GMDSI introducing the tutorial notebooks can be viewed [here](https://vimeo.com/856752189). During the webinar the authors provide an overview of the notebooks, as well as a demonstration of how to use them and introduce an online self-guided course.

The [GMDSI](https://gmdsi.org) web-page also hosts an extensive range of resources  and educational material on decision support modelling. These include numerous instructional video lectures, webinar recordings, non-programatic workflow tutorials, as well as worked example reports desrcibing real-world applications. 

Software from the ``PEST`` suite can be downloaded from John Doherty's web-page [here](https://www.pesthomepage.org/). The [user manual](https://www.pesthomepage.org/Downloads/PEST%20Manuals/PEST%20Manual.pdf) contains lots of usefull information. The [PEST Book](https://pesthomepage.org/pest-book) is also a great resource for learning about the theory underpinning use of the software.

Software from the ``PEST++`` suite can be accessed from the Git-Hub [repository](https://github.com/usgs/pestpp/tree/master). The [user manual](https://github.com/usgs/pestpp/blob/develop/documentation/pestpp_users_manual.md) contains lots of usefull information, as well as  theoretical background to the software.

``pyEMU`` can be accessed from the Git-Hub [repository](https://github.com/pypest/pyemu/tree/master). The repo contains several example jupyter notebooks. The tutorial notebooks discussed herein provided a more exhaustive and structured learning experience.


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

The notebooks have been employed during the [Applied Decision Support Groundwater Modeling With Python: A Guided Self-Study Course](https://gmdsi.org/blog/guided-self-study-course/) hosted by GMDSI. This self-guided course comprised 4 to 5 online sessions, each lasting 1 to 2 hours. During each session the hosts go through a section of the tutorials and expand on some of the concepts. Learners were tasked with going through the notebooks in between sessions to stimulate discussion and questions. Sessions were recorded and can be accessed [on the GMDSI website](https://gmdsi.org/education/videos/). Beyond the live online sessions, learners were incetivized to make use of the Git-Hub [Discussions](https://github.com/gmdsi/GMDSI_notebooks/discussions) feature to retain a search-engine findable record of common questions.


# Acknowledgements

The tutorials were originally developed with support from the U.S Geological Survey (USGS) and support from USGS continues. Continued development and support is funded by the Groundwater Modelling Decision Support Initiative (GMDSI). GMDSI is jointly funded by BHP and Rio Tinto. We thank users and stress-testers for their valuable feedback and continued community contributions to the repository.

# References