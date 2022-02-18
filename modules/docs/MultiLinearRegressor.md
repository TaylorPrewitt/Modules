# MultiLinearRegressor

### Multi-linear regression with summary statistics and built in feature selection.
<sup>Quick links to function docstrings <br>
|| <a href="#mlr">MultiLinearRegressor</a> || <a href="#nan">Na_id</a> || <a href="#set">set_variables</a> || <a href="#enc">encode_data</a> || <a href="#sel">mlr_selections</a> ||</sup>


<br>


## Description 

1. Input is a DataFrame with both training and target parameters<br>
    * Ex: Pandas DataFrame with columns of the independent variables X<sub>1</sub> to X<sub>n</sub> and dependent variable y <br>
2. Has options for controlling: <br>
    *  Data preprocessing <br>
        * Parse for and handle NaN values<br>
        * Drop any user specifed parameters<br>
        * Sampling size<br>
    *  Feature selection metrics <br>
    *  Encoding methods for any categorical variables <br>
3. Selects features via mean statistics generated over repeated tests on random subsets <br>
4. Picks features meeting both p-value and f-stat criteria <br>


<br>

<b><u>Returns</u></b> a dictionary containing: 
* Model object
* Coefficients for selected features
* Fit standardizing and scaling object
* Explained variance and adjusted R<sup>2</sup>
* Metrics used during feature selection
    * These are the f-stat and p-value for each feature before selection

<br>


## Functions


<a name="mlr"></a>

### MultiLinearRegressor
<sup>Core function</sup>

```
MultiLinearRegressor Parameters
----------
df : DataFrame
    Data for regression including target variable.
target : string
    Name of the column in the df to act as the dependent variable.
nan : string, optional
    Rule for how to handle NaN values. 
    Possible values: 
        "drop" to drop any rows with NaN values.
        "ffill" to forward fill any NaN values.
        "bfill" to backward fill any NaN values.
    The default is "drop".
encoding : string, optional
    Rule for how to handle any categorical or nonnumeric data. 
    Possible values: 
        "drop" to drop any columns of non-numeric data. 
        "dummy" to create columns of dummy variables to represent categorical data.
        "ordinal" to label with an integer value. Range of integers is the number of unique values in column. 
    The default is "drop".
exclude : string/list, optional
    Column name or list of column names to exclude from regression.
    The default is "None".
p : float, optional
    Statistical p-value threshold. Any parameter scoring below the threshold is dropped.
    The default is 0.01.
f : float, optional
    Statistical f-score threshold. Any parameter scoring below the threshold is dropped.
    The default is 4.
ts : float, optional
    Proporatate size of testing data split. Must be in range (0,1). 
    The default is 0.3.
trials : int, optional
    Scaling factor for number of sampling trials to determine model. 
    The default is 5.
seed : int, optional
    Random seed for data sampling.
    The default is 42.
Returns
-------
mlr : dict
    Multi Linear Regression model with coefficients, model, and feature statistics.
```

<a name="nan"></a>

***

### Na_id 
<sup>Helper function</sup>

```
Rectifies missing data in a dataset. 
Parameters
----------
df : DataFrame
    All data including target variable..
nan : string, optional
    Rule for how to handle NaN values. 
    Possible values: 
        "drop" to drop any rows with NaN values.
        "ffill" to forward fill any NaN values.
        "bfill" to backward fill any NaN values.
    The default is "drop".
Returns
-------
df : DataFrame
    Data with filled or dropped NaN values.
```


<a name="set"></a>

***

### set_variables 
<sup>Helper function</sup>

```
Breaks up dataset into independent and dependent variable sets.
Parameters
----------
df : DataFrame
    Data for regression including target variable..
target : string
    Name of the column in the df to act as the dependent variable.
excl : string/list, optional
    Column name or list of column names to exclude from regression.
    The default is "None".
Returns
-------
X : DataFrame
    Independent variables.
y : Series
    Target/dependent variable.
```


<a name="enc"></a>

***

### encode_data 
<sup>Helper function</sup>
```
Parses for non-numeric data and encodes or drops any.
Parameters
----------
df : DataFrame
    Data to be parsed and encoded.
encoding : string, optional
    Rule for how to handle any categorical or nonnumeric data. 
    Possible values: 
        "drop" to drop any columns of non-numeric data. 
        "dummy" to create columns of dummy variables to represent categorical data.
        "ordinal" to label with an integer value. Range of integers is the number of unique values in column. 
    The default is "drop". 
Returns
-------
dict
    Data for regression.
    Keys:
        "df" is the all numeric dataframe of independent variables. 
        "labels": labels for categorical data. 
        "columns": names of the df columns with numeric data. 
        "drop_cols": names of columns dropped from df, 
        "scaler": scaler fit to data.
```

<a name="sel"></a>

***

### mlr_selections 
<sup>Helper function</sup>

```
Create a linear model with only significant features.
Parameters
----------
X_enc : dict
    Data for regression.
    Keys:
        "df" is the all numeric dataframe of independent variables. 
        "labels": labels for categorical data. 
        "columns": names of the df columns with numeric data. 
        "drop_cols": names of columns dropped from df, 
        "scaler": scaler fit to data.
y : Series
    Target/dependent variable for regression.
p : float, optional
    Statistical p-value threshold. Any parameter scoring below the threshold is dropped. 
    The default is 0.01.
f : TYPE, optional
    Statistical f-score threshold. Any parameter scoring below the threshold is dropped. 
    The default is 4.
ts : TYPE, optional
    Proporatate size of testing data split. Must be in range (0,1). 
    The default is 0.3.
resample_scale : TYPE, optional
    Scaling factor for number of sampling trials to determine model. 
    The default is 5.
seed : TYPE, optional
    Random seed for data sampling. 
    The default is 42.
Returns
-------
output : dict
    Multi Linear Regression model with coefficients, model, and feature statistics.
```


