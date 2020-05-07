Supervised Learning Progression
![Progress](https://github.com/whiterose-fsociety/machine-learning/blob/version_1/progress.jpg "progress")


Phases of Machine Learning Assignment
![Progress](https://github.com/whiterose-fsociety/machine-learning/blob/version_1/phase.png "Phases")


# machine-learning
Supervised Machine Learning Algorithms to classify datasets

## Requirements

## Install Dependencies and modules

  `pip3 install numpy`
 
  `pip3 install pandas`
  
  `pip3 install nested_lookup`
  
  `pip3 install statistics`
  
  `pip3 install seaborn`

# Random Forests


## Rationale
The combination of learning models increases the classification accuracy – to create a model of low variance.


## What are Random Forrests
Tree models are known to be high variance, low bias models – they are prone to overfit data. ID3 or CART are also relatively unstable (1 mistake can mess up the whole model).  Changing one row of the initial table can change the values for the information gain calculation.
Random forrest has proved to be one of the most useful ways to address the issues of overfitting and instabillity.

The Random Forrest approach is based on two concepts, called bagging and subspace sampling.

## Bagging
### Bootstrap aggregration.
Technique that combines the predictions from multiple machine learning algoritms together to make more accurate predictions than any individual model.
	How:
    1. Create datasets of the same length with replacement
    2. Train a model on each one
    3. Take majority prediction model for unseen query
    4.  We take the mean or median for regression tree models 
        *  Mode – Classification trees
        *  Mean – Regression Trees



# Credits : Soon to be updated...
> Please report any credits that I may have missed. Thanks :D
> 1858893@students.wits.ac.za
https://python-course.eu/Random_Forests.php



StatsQuest
https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
