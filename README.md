Supervised Learning Progression
![alt text](https://github.com/whiterose-fsociety/machine-learning/blob/version_1/progress.jpg "progress")


# machine-learning
Supervised Machine Learning Algorithms to classify datasets

## Requirements

## Install Dependencies and modules

 $ pip3 install numpy
 
 $ pip3 install pandas

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
    • Create datasets of the same length with replacement
    • Train a model on each one
    • Take majority prediction model for unseen query
    • We take the mean or median for regression tree models 
        ◦ Mode – Classification trees
        ◦ Mean – Regression Trees



# Credits
https://python-course.eu/Random_Forests.php



StatsQuest
https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
