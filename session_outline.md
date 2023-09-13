# General Session Outline

We are planning to have weekly sessions for four to five weeks.  These sessions are meant to provide you to opportunity to ask questions and build understanding; there will be short, informal presentations/discussions of new concepts for the upcoming week.  These sessions also have the side-effect of encouraging you to complete each weeks assignments.  

Below is an aspirational outline of what we hope cover for over the next four to five weeks

## Week 1: The Freyberg model, Bayes equation, and the Pest Interface

- standard introductions and goals
    + not just "shift+enter" - take the time to explore and understand
    + learning by struggling
    + using GH discussions/issues for help
- notebook structure
    + a quick aside on using git and github (the fork and PR approach)
        * live demo of fork-PR model
        * managing file sizes
    + crawl-walk-run: intro, part1, part2 notebooks
    + dependency management in the notebooks
- the Freyberg model
    + background
    + history and future together in one model
- intro to Bayes equation
    + live demo "intro to Bayes" notebook
- the pest interface 
    + the value of "non-intrusiveness"
    + components of model interaction
        * create inputs, run model, read outputs
    + components needed for analysis
        * parameter values, bounds, and prior info
        * observation values and weights
        * aux and diagnostic observations
- live coding demo of using pyemu.PstFrom
    + understanding the pest interface in action 
    + understanding multiplier parameters in action
- homework:
    + work through the part2 "PstFrom setup pest" notebook
        * referencing back to part 1 notebooks on pilot points and geostats as needed
        * referencing the pest and pest++ users guides as needed
        

## Week 2: The Likelihood and Prior

- review and discuss last weeks homework
- Where are we now
    + we have "wrapped" the Freyberg model
    + we have generated a prior parameter ensemble
    + we have not defined a likelihood/objective function
    + we have not evaluated the prior parameter ensemble
- next steps: define the objective function
    + set observation values
    + set weights
    + define noise
    + discussion about weights and noise
- next steps: prior monte carlo and rejection sampling
    + background on prior monte carlo
    + what can you learn from prior monte carlo
    + an aside on parallelization mechanics (and how it will crush your machine!)
- homework
    + work through the part2 "set obsvals and weights" notebook
    + work through the part2 "prior monte carlo" notebook
    

## Week 3: The Posterior

- review and discuss last weeks homework
- Where are we now?
    + we are ready to solve Bayes equation!
    + the problem with Bayes equation in groundwater modeling
- next steps: approximating the Posterior
    + understanding GLM, jacobian matrices, and response surface
    + intro to ensemble methods
    + implications and mechanics of using ensembles for scenario testing
- homework:
    + work through the part2 "ies" notebooks


## Week 4: Beyond Uncertainty Analysis: Management Optimization

- review and discuss last weeks homework
- Where are we now?
    + We have sought posterior forecast distributions!
    + "your results are uncertain, your welcome!"
    + Can we do better?
- Introduction to management optimization
    + terms and jargon
    + relation to forecasts
    + what about uncertainty?
    + linear programming
    + evolutionary multi-objective optimization
- homework:
    + work through the part2 "opt" notebooks
    + work through the part2 "mou" notebooks


## Week 5: wrapping up

- review and discuss last weeks homework
- Where are we now?
    + done!
- next steps
    + considerations for applying these approaches in your modeling
        * fast and stable underlying model
        * additional computing resources
        * using git(hub) for modeling projects
        * scripts vs notebooks

