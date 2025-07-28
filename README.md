# Sephora-Product-Analysis

## Project Overview

**Goal**: The project goal is to predict whether a Sepjora product should be recommended based on its features.
**Database**: [kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data)

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```python
