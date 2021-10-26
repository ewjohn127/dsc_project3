import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def df_dropped(df_predictors):
    
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

def show_percent_nan(df_predictors):

    percent_nan = df_predictors.isna().sum() / df_predictors.shape[0] * 100
    return percent_nan.map(round)[percent_nan > 10]

def df_dropped_nan(df_predictors):

    dropped_cols = ['health_insurance','income_poverty','employment_industry','employment_occupation']

    print(f'Features dropped from set: {dropped_cols}')

    return df_predictors.drop(dropped_cols, axis=1)

def preprocessing(X):
    
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
                                                     
    median_columns = ['opinion_seas_vacc_effective', 
                      'opinion_seas_risk',
                      'opinion_seas_sick_from_vacc',
                      'household_adults', 
                      'household_children']

    ohe_cols = ['opinion_seas_vacc_effective', 
                'opinion_seas_risk',
                'opinion_seas_sick_from_vacc',
                'age_group','education',
                'race',
                'employment_status', 
                'census_msa']

    oe_cols = ['sex','marital_status','rent_or_own']

    non_imputed_cols = ['age_group', 'race', 'census_msa']

    #Impute certain columns with ColumnTransformer
    col_imputer = ColumnTransformer(transformers=[
        ("sim", SimpleImputer(strategy='most_frequent'), frequent_columns),
        ("sib", SimpleImputer(strategy='median'), median_columns)
        ],
        remainder="passthrough")

    #OrdinalEncode and OneHotEncode certain columns with ColumnTransformer
    col_oe_ohe = ColumnTransformer(transformers=[
        ('oe', OrdinalEncoder(categories='auto'), oe_cols),
        ("ohe", OneHotEncoder(categories="auto", drop='first'), ohe_cols)
        ], 
        remainder='passthrough')

    # Create a pipeline containing the impute ColumnTransformer
    impute_pipe = Pipeline(steps=[
        ('col_imputer', col_imputer)
        ])

    #Fit and transform X_train through impute pipeline
    imputed = impute_pipe.fit_transform(X)

    #Create new dataframe with newly imputed data
    X_pipe_impute = pd.DataFrame(imputed, columns=frequent_columns + median_columns + non_imputed_cols)


    #Create a pipeline containing the encoding ColumnTransformer
    encode_scale_pipe = Pipeline(steps=[
        ('col_oe_ohe', col_oe_ohe),
        ('ss', StandardScaler())
        ])

    #Fit and transform imputed data through encode pipeline
    transformed_data = encode_scale_pipe.fit_transform(X_pipe_impute)

    #Isolate and create feature names of the OneHotEncoded features
    encoder = col_oe_ohe.named_transformers_['ohe']
    category_labels = encoder.get_feature_names(ohe_cols)

    # Make a dataframe with the transformed data
    return pd.DataFrame(transformed_data, columns=oe_cols + 
                                          list(category_labels) + 
                                          list(X_pipe_impute.drop(ohe_cols + oe_cols, axis=1).columns))

def grid_search(grid, model, X, y):

    gs = GridSearchCV(model, grid, cv=3, return_train_score=True)
    gs.fit(X, np.ravel(y))

    return gs.best_params_

def rfe(X, y, n_features=5, model=LogisticRegression()):
    cv_rfe = []
    keep_list = []
    for n in range(1,n_features+1):
        num_features_to_select = n
        select = RFE(model, n_features_to_select=num_features_to_select)
        select.fit(X=X, y=y)
        feature_list = [(k,v) for k,v in zip(X.columns,select.support_)]
        current_keep_list = []
        for k,v in feature_list:
            if v:
                current_keep_list.append(k)
    
        current_cv = cross_val_score(model , X[current_keep_list], y, cv=3, scoring='roc_auc').mean()

        cv_rfe.append(current_cv)
        keep_list.append(current_keep_list)

    return cv_rfe, keep_list