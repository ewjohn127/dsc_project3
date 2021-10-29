import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

#Function to drop columns irrelevent to business problem
def df_dropped(df_predictors):
    
    #Columns to be dropped
    dropped_cols = ['respondent_id',
                    'h1n1_concern',
                    'h1n1_knowledge',
                    'opinion_h1n1_vacc_effective',
                    'opinion_h1n1_risk',
                    'opinion_h1n1_sick_from_vacc',
                    'doctor_recc_h1n1',
                    'hhs_geo_region']

    print(f'Features dropped from original set: {dropped_cols}')
    
    return df_predictors.drop(dropped_cols, axis=1)



#Function to drop columns with precentage of nan values greater than 10%
def df_dropped_nan(df_predictors):

    #Columns to be dropped due to nan
    dropped_cols = ['health_insurance','income_poverty','employment_industry','employment_occupation']

    print(f'Features dropped from set: {dropped_cols}')

    return df_predictors.drop(dropped_cols, axis=1)




#Function to create a pipeline to impute
def pipe_impute():
    
    #Columns to bbe imputed with most_frequent strategy
    frequent_columns = ['behavioral_antiviral_meds', 
                        'behavioral_avoidance',
                        'behavioral_face_mask', 
                        'behavioral_wash_hands',
                        'behavioral_large_gatherings', 
                        'behavioral_outside_home',
                        'behavioral_touch_face', 
                        'doctor_recc_seasonal',
                        'chronic_med_condition', 
                        'child_under_6_months', 
                        'health_worker',
                        'education', 
                        'rent_or_own', 
                        'marital_status', 
                        'employment_status',
                        'sex']

    #Columns to be imputed with median strategy                                  
    median_columns = ['opinion_seas_vacc_effective', 
                      'opinion_seas_risk',
                      'opinion_seas_sick_from_vacc',
                      'household_adults', 
                      'household_children']

    non_imputed_cols = ['age_group', 'race', 'census_msa']

    #Impute specific columns with ColumnTransformer
    col_imputer = ColumnTransformer(transformers=[
        ("sim", SimpleImputer(strategy='most_frequent'), frequent_columns),
        ("sib", SimpleImputer(strategy='median'), median_columns)
        ],
        remainder="passthrough")

    #Create a pipeline containing the impute ColumnTransformer
    impute_pipe = Pipeline(steps=[
        ('col_imputer', col_imputer)
        ])

    columns = frequent_columns + median_columns + non_imputed_cols
    
    return impute_pipe, columns




#Function to create a pipeline for encoding
def pipe_encode(X):

    #Columns to be OneHotEncoded
    ohe_cols = ['opinion_seas_vacc_effective', 
                'opinion_seas_risk',
                'opinion_seas_sick_from_vacc',
                'age_group','education',
                'race',
                'employment_status', 
                'census_msa']

    #Columns to be OrdinalEncoded
    oe_cols = ['sex','marital_status','rent_or_own']

    #OrdinalEncode and OneHotEncode specific columns with ColumnTransformer
    col_oe_ohe = ColumnTransformer(transformers=[
        ('oe', OrdinalEncoder(categories='auto'), oe_cols),
        ("ohe", OneHotEncoder(categories="auto", drop='first'), ohe_cols)
        ], 
        remainder='passthrough')

    #Create a pipeline containing the encoding ColumnTransformer and the StandardScaler
    encode_scale_pipe = Pipeline(steps=[
        ('col_oe_ohe', col_oe_ohe),
        ('ss', StandardScaler())
        ])
    
    #Fit the data into the pipeline

    transformed_array = encode_scale_pipe.fit_transform(X)

    #Retrieve and create column names for newly transformed data
    encoder = col_oe_ohe.named_transformers_['ohe']
    category_labels = encoder.get_feature_names(ohe_cols)

    #variable for columns to be dropped from original data
    impute_drop =  oe_cols + ohe_cols

    #variable for columns transformed
    oe_ohe_labels = oe_cols + list(category_labels)

    #Final dataframe of newly transformed data
    

    return encode_scale_pipe, col_oe_ohe, impute_drop, ohe_cols, oe_cols



#Function to grid search given a grid and a model
def grid_search(grid, model, X, y, cv=5):

    #Initializes GridSearch with given grid and given model
    gs = GridSearchCV(model, grid, cv=cv, return_train_score=True)

    #Fits X and y to grid search
    gs.fit(X, np.ravel(y))

    #Return the best parameters according to the GridSearch
    return gs.best_params_




#Function using recurssive feature elimination to find the best features of a given model and feature number
def rfe(X, y, n_features=5, model=LogisticRegression()):
    
    #Initializes cross val score list
    cv_rfe = []

    #Initializes a list of features to keep
    keep_list = []

    #loops through different features to find the most important ones
    for n in range(1,n_features+1):
        num_features_to_select = n

        select = RFE(model, n_features_to_select=num_features_to_select)
        select.fit(X=X, y=y)
        feature_list = [(k,v) for k,v in zip(X.columns,select.support_)]
        current_keep_list = []
        for k,v in feature_list:
            if v:
                current_keep_list.append(k)

        #Uses cross_val_score to determine the roc_auc score of the current model within the loop
        current_cv = cross_val_score(model , X[current_keep_list], y, cv=3, scoring='roc_auc').mean()

        cv_rfe.append(current_cv)
        keep_list.append(current_keep_list)
    
    print(f'Features selected: {keep_list[-1]}')

    #Returns the final cross_val list and final keep list
    return cv_rfe, keep_list