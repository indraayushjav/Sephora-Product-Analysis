# Sephora-Product-Analysis

## Project Overview

**Goal**: The project goal is to **predict** whether a Sephora product should be recommended based on its features using exploratory data analysis.

**Database**: [kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data)

1. Set up database
2. Exploratory Data Analysis (EDA)
3. Visualizations
4. Modeling


## 1. Database Setup

- Import necessary libraries
- Open and combine datasets
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

df_prod = pd.read_csv('product_info.csv')
df_rev1 = pd.read_csv('reviews_0-250.csv')
df_rev2 = pd.read_csv('reviews_250-500.csv')
df_rev3 = pd.read_csv('reviews_500-750.csv')
df_rev4 = pd.read_csv('reviews_750-1250.csv')
df_rev5 = pd.read_csv('reviews_1250-end.csv')

df_reviews = pd.concat([
    df_rev1,
    df_rev2,
    df_rev3,
    df_rev4,
    df_rev5
], ignore_index = True)

print('Shape of product data:', df_prod.shape)
print('Shape of review data:', df_reviews.shape)

# Shape of product data: (8494, 27)
# Shape of review data: (1094411, 19)
```


```python
#Finding not commom columns between product and review datasets

non_common_cols = df_reviews.columns.difference(df_prod.columns)
non_common_cols = list(non_common_cols)
non_common_cols.append('product_id')
```

```python
#Merge the product with review data with uncommon cols based on the product id and fill missing data with na

df = pd.merge(df_prod, df_reviews[non_common_cols], how = 'outer', on = 'product_id')
print('Shape of the new dataframe: ', df.shape)
df = df.replace([np.inf,-np.inf], np.nan)
df.head()

# Shape of the new dataframe:  (1100554, 41)
```

<img width="1338" height="347" alt="Screenshot 2025-07-28 at 10 48 32 AM" src="https://github.com/user-attachments/assets/1adccace-44d9-4813-a746-57cf4fbebefa" />

## 2. Exploratory Data Analysis (EDA)

- Handled missing values, duplicates
- Identified and visualized strong correlation between values

```python
#Looking at unique, missing datasets

col_name =[]
types = []
unique = []
count = [] 
missing = []

for col in df.columns:
    col_name.append(col)
    types.append(df[col].dtypes)
    unique.append(df[col].nunique())
    count.append(df[col].count())
    missing.append(df[col].isna().sum())

info_table = pd.DataFrame({
    'col_name' : col_name,
    'types' : types,
    'unique value' : unique,
    'value count' : count,
    'missing' : missing
})
print(info_table)
```
<img width="553" height="673" alt="Screenshot 2025-07-28 at 11 07 29 AM" src="https://github.com/user-attachments/assets/ac717e66-99e3-4e5a-adcb-f041802d53e6" />

### Handle missing values
```python
cols_to_drop = ['variation_desc', 'value_price_usd', 'sale_price_usd', 'highlights', 'tertiary_category', 
'variation_type', 'child_max_price', 'review_title', 'child_min_price', 'variation_value', 'child_count', 'Unnamed: 0']
df = df.drop(columns = cols_to_drop, axis = 1)
df.dropna(axis = 0, inplace = True)
print('Shape of data after cleaning', df.shape)

#Shape of data after cleaning (390521, 29)
```

```python
# 1. change data time data types
df['submission_time'] = pd.to_datetime(df['submission_time'], errors='coerce')

# 2. Change the data type of reviews column to int
df['reviews'] = df['reviews'].astype(int)
df['loves_count'] = df['loves_count'].astype(int)

# 3. Create new columns for month, year, date
df['submission_time'] = pd.to_datetime(df['submission_time'])
df['year']= df['submission_time'].dt.year
df['month']= df['submission_time'].dt.month
df['day']= df['submission_time'].dt.day
df['weekday']= df['submission_time'].dt.weekday
```

### Remove duplicates

```python
print("Sum of duplicates in dataset:", sephora_df.duplicated().sum())
sephora_df  = sephora_df.drop_duplicates()
print("Check after removing duplicates: ", sephora_df.duplicated().sum())

# Sum of duplicates in dataset: 19
# Check after removing duplicates:  0
```
### Clean data

```python
## change skin type to lower case
sephora_df['skin_type'] = sephora_df['skin_type'].str.lower().str.strip()

## change eye color to lower case and keep consistency.
sephora_df['eye_color'] = sephora_df['eye_color'].replace("gray", "grey")
sephora_df['eye_color']  = sephora_df['eye_color'].str.lower().str.strip()
```
### Correlation Heatmap

```python
plt.figure(figsize=(16, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```
<img width="934" height="589" alt="Screenshot 2025-07-28 at 11 25 55 AM" src="https://github.com/user-attachments/assets/2c32bfa2-fd0f-4232-a3ff-c1bc06d25382" />

>_From this heat map we can see that there is a strong *positive correlation* between *reviews* and *# of loves product received* meaning products with more reviews tend to be more loved by customers. Similarly, *total positive feedback* and *total feedback count* also showed a strong positive relationship with review volume, reinforcing the idea that popular products tend to attract more overall user interaction._

>_We can also learn that there is *negative correlation* between *helpfulness scores* and *negative feedback counts*, implying that reviews marked as more helpful tend to have fewer negative reactions. Additionally, *loves count* was found to have a negative correlation with *price* and *online-only availability*, indicating that higher-priced or online-exclusive products may not be as widely loved or accessible._
>_This helps us see the customer engagement trends and come up with product recomendation strategies._

## 3. Visualizations

### a. Average price and Rating by Product Category

```python
dfg = df.groupby(['secondary_category'])[['price_usd','rating']].mean().sort_values('price_usd',ascending =False).reset_index()
print(dfg.head())


fig, axes = plt.subplots(1,2,figsize=(18,10))
sns.barplot(data=dfg, x = 'secondary_category', y = 'price_usd', ax = axes[0])
axes[0].set_title('Average Price by Secondary Category')
axes[0].tick_params(axis='x', rotation=90)


sns.barplot(data=dfg, x = 'secondary_category', y = 'rating', ax = axes[1])
axes[1].set_title('Average Rating by Secondary Category')
axes[1].tick_params(axis='x', rotation=90)
```
<img width="329" height="141" alt="Screenshot 2025-07-28 at 12 13 03 PM" src="https://github.com/user-attachments/assets/bcb5aeac-b313-4456-bd80-b9749a06f632" />
<img width="851" height="581" alt="Screenshot 2025-07-28 at 12 13 30 PM" src="https://github.com/user-attachments/assets/943e5bc8-3290-4cb7-8170-fb1298e4ba70" />

>_From this plot we can see that *Value & Gift Sets* is the most expensive category although it is not the highest rated category. On the other hand we can see that *High-Tech Tools, Cleansers, Shop by Concern, Mini Size, Lip Balms* are highly rated categories and are on the cheaper side. This means that customers usually buy products they use daily or to treat specific concerns._

### b. Analysis on Top Brands



