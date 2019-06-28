# predictiveness-curve

[![Downloads](https://pepy.tech/badge/predictiveness-curve)](https://pepy.tech/project/predictiveness-curve) [![Downloads](https://pepy.tech/badge/predictiveness-curve/month)](https://pepy.tech/project/predictiveness-curve/month) [![Downloads](https://pepy.tech/badge/predictiveness-curve/week)](https://pepy.tech/project/predictiveness-curve/week)

## What's Predictiveness Curve?
Predictiveness curve is a method to display two graphs simultaneously. In both figures, the x-axis is risk percentile, the y-axis of one figure is the value of risk, and the y-axis of the other figure is true positive fractions. This makes it possible to visualize whether the model of risk fits in the medical field and which value of risk should be used as the basis for the model. See [Am. J. Epidemiol. 2008; 167:362–368](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2939738/) for details.

## Install

This module implements functions to plot `Predictiveness Curve`.  
Install with :

`pip install predictiveness-curve`

## Example

```python
from predictiveness_curve import plot_predictiveness_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
y = data.target
X = data.data

training_X, test_X, training_y, test_y = train_test_split(
    X, y, test_size=0.5, random_state=42)

clsf = RandomForestClassifier(n_estimators=100, random_state=42)
clsf.fit(training_X, training_y)
probabilities = clsf.predict_proba(test_X)[:, 1]

plot_predictiveness_curve(probabilities, test_y)
```

See [notebooks directory](https://github.com/yamasakih/predictiveness-curve/tree/master/notebooks) for details.
