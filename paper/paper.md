---
title: 'Self-Guided Decision Support Groundwater Modelling with Python'
tags:
  - Python
  - groundwater modelling
  - environmental modelling
  - decision-support
  - uncertainty analysis
authors:
  - name: Rui T. Hugman
    orcid: 0000-0003-0891-3886
    affiliation: 1 
  - name: Jeremy T. White
    orcid: 0000-0002-4950-1469
    affiliation: 1 
  - name: Michael N. Fienen
    orcid: 0000-0002-7756-4651
    affiliation: "2" # (Multiple affiliations must be quoted)
  - name: Brioch Hemmings
    orcid: 0000-0001-6311-8450
    affiliation: "3" # (Multiple affiliations must be quoted)
  - name: Katherine H. Markovich
    orcid: 0000-0002-4455-8255
    affiliation: 1 

affiliations:
 - name: INTERA Geosciences, Perth, Western Australia, Australia
   index: 1
 - name: U.S. Geological Survey, Upper Midwest Water Science Center, Madison, WI USA
   index: 2
 - name: Wairakei Research Centre, GNS Science, Taupō, New Zealand
   index: 3
date: 20 September 2023
bibliography: paper.bib



# List all authors and affiliations.
# Describe the submission, and explain its eligibility for JOSE.
# Include a “Statement of Need” section, explaining how the submitted artifacts contribute to # computationally enabled teaching and learning, and describing how they might be adopted by # others.
# For learning modules, describe the learning objectives, content, instructional design, and # experience of use in teaching and learning situations.
# Tell us the “story” of the project: how did it come to be?
# Cite key references, including a link to the open archive of the software or the learning module.


# # Citations
# 
# Citations to entries in paper.bib should be in
# [rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
# format.
# 
# For a quick reference, the following citation commands can be used:
# - `@author:2001. -. "Author et al. (2001)"
# - `[@author:2001]` -> "(Author et al., 2001)"
# - `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"
# 
# # Figures
# 
# Figures can be included like this: ![Example figure.](figure.png)

---

# Summary

The [GMDSI tutorial notebooks repository](https://github.com/gmdsi/GMDSI_notebooks/tree/main) provides learners with a comprehensive set of tutorials for self-guided training on decision-support groundwater modelling using Python-based tools. Although targeted at groundwater modelling, they are based around model-agnostic tools and readily transferable to other environmental modelling workflows. The tutorials are divided into three parts. The first covers fundamental theoretical concepts. These are intended as background reading for reference on an as-needed basis. Tutorials in the second part introduce learners to some of the core concepts parameter estimation in a groundwater modelling context, as well as providing a gentle introduction to the ``PEST``, ``PEST++`` and ``pyemu`` software. Lastly, the third part demonstrates how to implement highly parameterized applied decision-support modelling workflows. The tutorials aim to provide examples of both “how to use” the software as well as “how to think” about using the software. 
A key advantage to using notebooks in this context is that the workflows described run the same code as practitioners would run on a large-scale real-world application. Using a small synthetic model facilitates rapid progression through the workflow.

# Statement of Need

Managing the environment and natural resources is often faced with making decisions on actions that trade off between economic, social, and environmental outcomes. In a groundwater management context, they may include decisions on such things as how much to pump from a well without impacting a nearby stream or how to design a well-field to maximize yield while minimizing the risk of saltwater intrusion. These decisions are often complex and rely on the use of numerical models to simulate the natural system and evaluate the outcomes of different management actions [@kelly2013]. 

Effective environmental management is often aided by using models. Such models face the challenges of limited data and the inherent simplifications of the complex natural world necessary to make them. This translates to uncertainties in critical decision-making predictions. Transparency is critical in reporting and mitigating these uncertainties, especially when significant risks accompany management outcomes. The significance of uncertainty quantification (UQ) and parameter estimation (PE) in environmental modelling for decision support is widely acknowledged [@Anderson2015]. UQ provides estimates of outcome uncertainty, while PE reduces this uncertainty through assimilating data, using a formal comparison of model outputs with observation data to inform model parameter values. 

Implementing highly parameterized UQ and PE in real-world modelling can be challenging due to both theoretical complexity and practical logistics. Limited project time and funding also often hinder their application. Open-source software such as ``PEST`` [@pest] and ``PEST++`` [@whitepestpp] provide tools for undertaking UQ and PE analyses. They have been employed to tackle a diverse range of real-world decision-support problems, ranging from risk-based wellhead protection [@fienen2022risk], groundwater contaminant system remediation design [@FIENEN2024], evaluating the impact of groundwater use on river baseflow [@Foster2021],  managed aquifer recharge scheme design to protect against seawater intrusion [@Standen2022], optimizing vineyard irrigation [@KNOWLING2023108225], and many others. 

However, the steep learning curve associated with their use and the lack of user-friendly training materials have been a barrier to uptake. There is a growing demand within the environmental modelling community for transparent, reproducible, and accountable modelling processes, driven by the need for increased credibility and rigor in computational science and environmental simulation [@white11rapid; @fienen2016hess]. While some script-based tools enhance the reproducibility of forward model construction [@flopy], they often overlook UQ and PE analyses. In decision-support scenarios, these analyses are equally vital for robust model deployment as the forward model itself. 

The uptake of Python for environmental modelling has increased in recent years, due to its open-source nature, user-friendly syntax, and extensive scientific libraries. Python-based tools have been developed to facilitate UQ and PE analyses, such as ``pyemu`` [@White_A_python_framework_2016; @white2021towards]. ``pyemu`` is a Python package that provides a framework for implementing UQ and PE analyses with ``PEST`` and ``PEST++``. It offers a range of capabilities, including parameter estimation, uncertainty analysis, and management optimization. Although initially designed for groundwater modelling, ``pyemu``'s methodologies are versatile and can be applied to diverse numerical environmental models, as long as they use text files for input and generate machine-readable outputs that can be extracted without manual intervention.

The tutorial notebooks discussed herein provide a comprehensive, self-guided, and open-source resource for learning decision-support modelling workflows with Python. They are designed to be accessible to a broad audience, including students, researchers, and practitioners who aim to undertake applied environmental decision-support modelling. 


# Story of the Project

The Groundwater Modelling Decision Support Initiative ([GMDSI](https://gmdsi.org)) is an industry-backed and industry-aligned initiative. Established in mid-2019, its primary goal is to enhance the role of groundwater modelling in groundwater management, regulatory processes, and decision-making. At the core of GMDSI's mission lies the numerical simulation of groundwater movement and processes. Often, data related to groundwater are limited, leading to uncertainties in simulator predictions. However, despite this uncertainty, decisions must be made, and associated risks must be assessed. Modelling plays a central role in the evaluation of these risks.

GMDSI is dedicated to promoting, facilitating, and providing support for the improved utilization of modelling in decision support processes. Its activities endeavor to elevate the role of groundwater modelling in decision-making processes, recognizing the importance of model partner software for UQ and PE, and offering a range of activities aimed at industry engagement, education, practical examples, research, and software development.

Many groundwater modelers typically rely on Graphical User Interfaces (GUIs) for their modelling needs. However, each GUI has its unique characteristics and varying degrees of compatibility with external software like ``PEST`` and ``PEST++``. Creating educational materials for these GUIs would necessitate tailoring content to each GUI's specific features, obtaining cooperation from the GUI developers themselves and potentially lagging behind the latest developments. Many GUIs are commercial products as well which limits accessibility.

Decision-support modelling often demands capabilities that surpass what current GUIs can offer. For example, many of GMDSI's worked examples rely on custom-designed utilities or the integration of different software components. Currently, a significant portion of users may not have the expertise to independently implement such advanced approaches. Furthermore, the manual preparation of input files for implementing these complex workflows can be time-consuming. Programmatic workflows, such as those facilitated by ``pyemu``, offer advantages by reducing the time and user input required for setup and execution. This approach is somewhat analogous to the role played by a GUI but offers added flexibility, allowing users to customize and design their own functions and utilities as needed. However, it comes with the drawback of increased potential for user-introduced errors.

Anecdotally, we have seen that more modelers are turning to Python packages like ``FloPy`` [@flopy] and ``pyemu`` [@whitepyemu] for model and ``PEST++`` setup. Unfortunately, the adoption of this approach is hindered by a steep learning curve primarily due to the scarcity of user-friendly training materials. The [GMDSI tutorial notebooks](https://github.com/gmdsi/GMDSI_notebooks/tree/main) aim to address this gap by providing a comprehensive, self-guided, and open-source resource for learning decision-support modelling workflows with Python.

The roots of the materials making up the tutorial notebooks were from a traditional, week-long classroom course curriculum developed for internal training at the US Geological Survey (USGS) by a subset of the authors of this paper. For this course, the instructors leveraged the power of Jupyter Notebooks [@jupyter] as a mechanism to teach both the fundamental background and application of inverse theory. Jupyter Notebooks include the ability to provide interactive code and graphics along with detailed, formatted text making them excellent teaching and narrative tools. High-level mathematical libraries in Python (and other high-level languages with easy plotting utilities such as MATLAB and R) provide an opportunity for students to explore linear algebra and statistical modelling principles that underlie the PE and UQ techniques implemented in ``PEST`` and ``PEST++``. Furthermore, the combination of text, code, and graphics provides an interactive platform for mixing theory and applications and, potentially, providing a template for application on real-world applications. The native support for Python makes the connection between worked examples and notebooks seamless and has connections with other worked examples [@white2020r3; @White2020DR; @fienen2022risk; https://github.com/doi-usgs/neversink_workflow]

After three iterations of teaching the in-person class, the instructors concluded that the materials and approach were valuable, but came to question the level of retention by students in a 40-hour intensive setting. It is well-documented that without repetition and rapid adoption of new techniques, they can fade quickly from memory [@glaveski2019companies]. As a result, the authors, with support from the GMDSI, endeavored to build on the positive aspects of using Jupyter Notebooks and explore alternative teaching environments instead of week-long classes. The first major change was to add sufficient narration and explanation to the notebooks to improve possibilities for self-study. The initial design through in-person instruction was to have the notebooks serve as illustrations to assist in a narrative discussion, so bolstering of the explanatory text was necessary to help them stand alone. The next change was to refactor the organization from a strictly linear progression to the current three-part organization discussed below. This led to a hybrid model of self-study punctuated by discussion and background lectures online. 


# Experience of use in teaching and learning situations

The notebooks were employed during the [Applied Decision Support Groundwater modelling With Python: A Guided Self-Study Course](https://gmdsi.org/blog/guided-self-study-course/) hosted by GMDSI. This self-guided course comprised 5 online sessions, each lasting 1 to 2 hours and focused on the workflows of Part 2. During each session the instructors go through a section of the tutorials and expand on some of the concepts. Learners were tasked with going through the notebooks in between sessions to stimulate discussion and questions. Sessions were recorded and can be accessed [on the GMDSI website](https://gmdsi.org/education/videos/). Beyond the live online sessions, learners were incentivized to make use of the GitHub [Discussions](https://github.com/gmdsi/GMDSI_notebooks/discussions) feature to retain a search-engine findable record of common questions that persist beyond the time-frame of the course and can help address other questions that arise in the community. 

Feedback from the 65 students who participated in the course was anecdotal but informative. \autoref{fig:responses} summarizes the responses by 34 respondents to four questions, comprising 52%. The majority of respondents indicated a preference for this hybrid self-guided/online instruction approach over an in-person week-long intensive class with only one respondent indicating preference for self-guided study of the course materials only. Just under 60% of the respondents reported being able to keep up with most or all of the assigned self-study notebooks, while 41% reported falling behind. Given 5 categories of comfort level working with ``PEST++`` (1 being most comfortable, and 5 being least) before and after the class, there was a notable shift toward higher comfort level. Interestingly, when evaluating individual responses, the majority (56%) reported being more comfortable with ``PEST++`` after the course (defined as an increase of one level) and 15% reported being much more comfortable (an increase of two levels). However, 21% reported the same comfort level before and after while 24% reported being less or much less comfortable (a decrease or one or two levels, respectively). Without further questions, we cannot know whether these decreases reflect a humble realization that their mastery was less complete than they thought, _a priori_, or whether the material was confounding. 

Open-ended feedback from the participants was generally positive and also included some constructive criticism. Participants appreciated the opportunity to ask questions and several reported hearing the discussion around other peoples' questions as being valuable and clarifying aspects of the material. The main critical suggestions included incorporating more real-world examples rather than relying, as we 100% did in the notebook design, on the synthetic model. Participants also noted the twin challenges of a large amount of information coupled with trying to be accountable to keep up in the class as potentially limiting the value relative to a week-long course. We conclude from this experience that the hybrid approach has value but there may still be a better approach for future educational opportunities. 

![Summary of responses to post-course survey based on 34 responses. Panel A summarizes whether respondents would prefer and intensive in-person workshop or this hybrid option. Panel B summarizes how much of the notebooks respondents were able to complete throughout the course. Panel C summarizes respondent comfort level with ``PEST++`` before and after the course. Panel D highlights individual changes in comfort level reported due to the course.\label{fig:responses}](./responses.png)


# Acknowledgements

The tutorials were originally developed with support from the US Geological Survey (USGS) and support from USGS continues through the HyTest training project. Continued development and support is funded by the Groundwater Modelling Decision Support Initiative (GMDSI). GMDSI is jointly funded by BHP and Rio Tinto. We thank Dr. John Doherty for his tireless and pioneering efforts starting `PEST` and continuing to innovate and Dr. Randall Hunt for his leadership in `PEST` and `PEST++` applications and development and contributions to the initial curriculum for this material and the early version of the notebooks. We thank Kalle Jahn (USGS), [Ines Rodriguez](https://github.com/incsanchezro) and [codyalbertross](https://github.com/codyalbertross) who made reviews that improved this manuscript. Lastly, we thank users and stress-testers for their valuable feedback and continued community contributions to the repository.

# Disclaimer 
Any use of trade, firm, or product names is for descriptive purposes only and does not imply endorsement by the US Government.

# References
