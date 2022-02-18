import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas.api.types as ptypes
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import sklearn.metrics as metrics



def Na_id(df, nan="drop"):
    '''
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

    '''
    if df.isnull().values.any() == True:
        # Find index values with missing data. 
        nan_data = df.loc[df.isnull().any(axis=1)].index
        # Alert the dataset has missing data and give location.
        print("There are missing values in the data.")
        print(f"Index value(s) with missing data: {nan_data}.")
        
        # Continue based NaN rule input. 
        if nan == "drop":
            df = df.dropna()
            print("Droping any NaN present in dataset.")
        elif nan == "ffill":
            df = df.fillna(method="ffill", axis=1)
            print("Foward filling any NaN present in dataset.")
        elif nan == "bfill":
            df = df.fillna(method="bfill", axis=1)
            print("Backward filling any NaN present in dataset.")
    else:
        # Give message assuring there is no missing values. 
        print("There are NO missing values in the data.")

    return df



def set_variables(df, target, excl="None"):
    '''
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

    '''
    # Set dependent variable
    y = df[target]
    
    # Removing dependent variable to make independent set.
    X = df.drop(target, axis = 1)
    
    # Removing any unwanted columns from X. 
    exclude_list = [excl]
    # If any value is passed other than None, ensure it is not a column in X.
    if excl != "None":
        for n in exclude_list:
            X = X.drop(n, axis = 1)
            
    return X, y
    


def encode_data(df, encoding="drop"):
    '''
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
    '''
    categories = {}
    column_names = []
    drop_cols = []

    # Parsing each column of df to determine if dtype is numeric.
    # Handles any non-numeric data by dropping or encoding.
    for column in df.columns:
        
        if encoding == "dummy":
            
            if ptypes.is_numeric_dtype(df[column]) == False:
                encoded_features = pd.get_dummies(df[column], drop_first=False)
                df = pd.concat([df, encoded_features], axis=1)
                categories[f"Categories_{column}"] = df[column].unique()
                df = df.drop([column], axis=1)
                drop_cols.append(column)
            else:
                column_names.append(column)
                
            # If no non-numerical data, make note.
            try:
                len(categories)
            except UnboundLocalError:   
                categories = "No categorical variabels in set"       
                
        elif encoding == "ordinal":
            
            ordinal = preprocessing.OrdinalEncoder()
            
            if ptypes.is_numeric_dtype(df[column]) == False:
                enocoding = ordinal.fit(df[[column]])
                categories[f"Categories_{column}"] = enocoding.categories_
                df[f"data_code_{column}"] = enocoding.transform(df[[column]])
                df = df.drop([column], axis=1)
                drop_cols.append(column)

            else:
                column_names.append(column)
            
            # If no non-numerical data, make note.
            try:
                len(categories)
            except UnboundLocalError:   
                categories = "No categorical variabels in set"    
                
                
        elif encoding == "drop":
            
            if ptypes.is_numeric_dtype(df[column]) == False:
                df = df.drop(column, axis=1)
                drop_cols.append(column)
                categories = "All categorical variables dropped."
            else:
                column_names.append(column)
            
            # If no non-numerical data, make note.                
            try:
                len(categories)
            except UnboundLocalError:   
                categories = "No categorical variabels in set"
   
    # Scaling and standardizing the numeric data.     
    df = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df))
    
    # Printing some notes about actions performed on data.
    if encoding == "dummy":
        print(f"Created dummy variable columns for {len(drop_cols)} non-numeric column(s).")
    elif encoding == "ordinal":
        print(f"Created numeric labels for {len(drop_cols)} non-numeric data column(s).")
    elif encoding == "drop":
        print(f"Removed {len(drop_cols)} non-numeric data column(s).")

        
    return {"df":df, 
            "labels":categories, 
            "columns":column_names, 
            "drop_cols":drop_cols, 
            "scaler":preprocessing.StandardScaler().fit(df)}



def mlr_selections(X_enc, y, p=0.01, f=4, ts=0.3, resample_scale=5, seed=42 ):
    '''
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

    '''

    # Initialize Linear Regression from sklearn.
    lr = LinearRegression()
    
    # Scale the number of trials/iterations based on number of features & scalar.
    trials_count = resample_scale*len(X_enc['df'].columns)
    
    # Create test stats for features for selections metric
    results = []
    for j in range(seed, seed+trials_count):
        # Creating trials over differing seeds.
        X_train, X_test, y_train, y_test = train_test_split(X_enc['df'], y, 
                                                            test_size = ts, 
                                                            random_state = j)
        
        # List of f-stat and p-values for each feature in this trial.
        regressor = pd.DataFrame(f_regression(X_train, y_train)).T
        regressor.columns = ["Fstat", "pval"]
        results.append(regressor)
    # Build df with length=number of features and width is 2*number of trials    
    results_df = pd.concat(results, ignore_index=False, axis=1)
    Fstat = results_df['Fstat'].mean(axis=1)
    pval = results_df['pval'].mean(axis=1)
    model_feat_stats = pd.concat([Fstat, pval], axis=1)
    model_feat_stats.columns = ["Fstat", "pval"]
    
    # Selecting significant features using f-stat and p-value thresholds.
    feat_index = []
    for n in range(len(model_feat_stats)):
        if model_feat_stats['pval'][n] < p:
            if model_feat_stats['Fstat'][n] > f:
                feat_index.append(n)
    print(f"The selected model used {len(feat_index)} out of {len(model_feat_stats)} available features.")
    X = X_enc['df'].iloc[:,feat_index]
    
    # Fitting a linear model with the selected features.
    model = lr.fit(X, y)
    y_pred = model.predict(X)
    
    # Model evaluation. R^2 and explained variance.
    r2 = metrics.r2_score(y, y_pred, multioutput='variance_weighted')
    print(f"The selected model adj-R2 is {r2}.")
    explained_variance = metrics.explained_variance_score(y, y_pred)
    
    # Calling model coefficients and feature names.
    coef = pd.Series(model.coef_)
    coef_names = pd.Series(X.columns).T   
    names = pd.DataFrame(X_enc['columns'])
    
    # Looking at all names and dropped columns.
    for n in range(len(names)):
        for i in range(len(X_enc['drop_cols'])):
            # If feature name is in the drop_cols list, remove from names list.
            if names[0][n] == X_enc['drop_cols'][i]:
                names = names.drop(index=n)
    # Mapping feature names to coefficients.
    names = names.iloc[coef_names]
    names = names.reset_index()
    
    # Creating a df of features mapped to coefficients.
    model_coef_df = pd.concat([names, coef], axis=1)
    model_coef_df.columns = ["original_index", "feature", "coefficient"]
    
    # Setting function output package.
    output = {'model':model,
              'model_coef_df':model_coef_df,
              'r2':r2,
              'explained_variance':explained_variance,
              'all_feature_stats':model_feat_stats,
              "fit_scaler":X_enc['scaler']}
    
    return output



def MultiLinearModel(df, target, 
                     nan="drop", exclude="None", encoding="drop", 
                     p=0.01, f=4, ts=0.3, trials=5, seed=42 ):
    '''
    Create a multi linear model with regression and feature selection.  
    
    Parameters
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

    '''
    # Find and deal with any NaN values.
    data = Na_id(df=df, nan=nan)
    
    # Create datasets based on chosen target variable. 
    X, y = set_variables(df=data, excl=exclude, target=target)
    
    # Find and deal with any non-numeric variables.
    data_dict = encode_data(df=X, encoding=encoding) 
    
    # With datasets and metadata build a MultiLinearRegressor
    mlr = mlr_selections(X_enc=data_dict, y=y, p=p, f=f, 
                         ts=ts, resample_scale=trials, seed=seed )    
    
    return mlr
     












