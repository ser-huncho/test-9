import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import joblib
from scipy.stats import f_oneway, chi2_contingency
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

app_train = pd.read_csv("application_train.csv")


# 1. Handling Missing Value ---------------

null = app_train.isna().sum().reset_index().rename(columns = {"index": "column_name", 0: "null_counts"}).sort_values(by = ["null_counts"])
null = null.set_index("column_name")
null["null_percentage"] = (null["null_counts"] / app_train.shape[0] * 100).round(2)

# Delete columns that have >50% missing values
del_column = list(null[null["null_percentage"] > 50].index)
app_train.drop(columns = del_column, inplace = True)

# Change the dtype
list_num = list(app_train.select_dtypes(include = "number").columns)
list_cat = list(app_train.select_dtypes(include = "object").columns)

num_to_cat = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 
              'FLAG_EMAIL', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 
              'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 
              'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
              'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 
              'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 
              'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 
              'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16','FLAG_DOCUMENT_17', 
              'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

app_train[num_to_cat] = app_train[num_to_cat].astype(object)

# Update numerical and categorical columns
list_num = list(app_train.select_dtypes(include = "number").columns)
list_cat = list(app_train.select_dtypes(include = "object").columns)

num_mean = app_train[list_num].mean()
num_mean.to_csv("save_files/num_mean.csv")

# Impute missing value in numerical data using "mean"
app_train[list_num] = app_train[list_num].fillna(num_mean)

dict_cat_mode = {}
for i in list_cat:
    mode_value = app_train[i].mode()[0]
    dict_cat_mode[i] = mode_value
# save
joblib.dump(dict_cat_mode, "save_files/dict_cat_mode.joblib")

# Impute missing value in categorical data using "mode"
app_train[list_cat] = app_train[list_cat].fillna(dict_cat_mode)

# Delete the document features that almost only have one value
delete_features = [i for i in list_cat if "DOCUMENT" in i]
not_remove = ["FLAG_DOCUMENT_8", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_3"] # the proporsion are better
delete_features = [i for i in delete_features if i not in not_remove]
delete_features.append("FLAG_MOBIL") # almost only have one value

app_train.drop(columns = delete_features, inplace = True)

# Update categorical columns
list_cat = [i for i in list_cat if i not in delete_features]

# "XNA" in CODE_GENDER replace by NAN
# "Unknown" in NAME_FAMILY_STATUS replace by NAN
# "XNA" in ORGANIZATION_TYPE replace by "Other"
app_train["CODE_GENDER"].replace("XNA", np.nan, inplace = True)
app_train["NAME_FAMILY_STATUS"].replace("Unknown", np.nan, inplace = True)
app_train["ORGANIZATION_TYPE"].replace("XNA", "Other", inplace = True)

# Impute missing value in categorical data using "mode"
app_train["CODE_GENDER"] = app_train["CODE_GENDER"].fillna(app_train["CODE_GENDER"].mode()[0])
app_train["NAME_FAMILY_STATUS"] = app_train["NAME_FAMILY_STATUS"].fillna(app_train["NAME_FAMILY_STATUS"].mode()[0])


# 2. Feature Selection --------------

# CORRELATION
# Numerical & Categorical -> ANOVA
# Categorical & Categorical -> Chi Square

# ANOVA
dict_anova = {}
for i in list_num:
    category_group = app_train.groupby("TARGET")[i].apply(list)
    pvalue_anova = f_oneway(*category_group)[1]
    dict_anova[i] = pvalue_anova

# Chi Square
dict_chisquare = {}
for i in list_cat:
    crosstab_result = pd.crosstab(index = app_train[i], 
                                  columns = app_train["TARGET"])
    pvalue_chisquare = chi2_contingency(crosstab_result)[1]
    dict_chisquare[i] = pvalue_chisquare

# Combine p_value from ANOVA and Chi Square
pvalue_all = dict(dict_anova)
pvalue_all.update(dict_chisquare)

# Just select variables that have correlation with "TARGET"
correlated_var = []
for var, pvalue in pvalue_all.items():
    if pvalue <= 0.05:
        correlated_var.append(var)

correlated_var.insert(0, "SK_ID_CURR")

# Update numerical and categorical columns
list_num = [i for i in list_num if i in correlated_var]
list_cat = [i for i in list_cat if i in correlated_var]
# save
joblib.dump(list_num, "save_files/list_num.joblib")
joblib.dump(list_cat, "save_files/list_cat.joblib")

app_train = app_train[correlated_var]
app_train.to_csv("save_files/data_train_with_correlated_var.csv", index = False)


# 3. Handling Categorical Data ---------------

app_train_ok = pd.read_csv("save_files/data_train_with_correlated_var.csv")

list_cat_ok = joblib.load("save_files/list_cat.joblib")
app_train_ok[list_cat_ok] = app_train_ok[list_cat_ok].astype(object)

# Use Frequency Encoding
encoder_dict = {}
for var in list_cat_ok:
    encoder_dict[var] = (app_train_ok[var].value_counts() / len(app_train_ok)).to_dict()

# Apply the encoder value in the new columns
for var in list_cat_ok:
    app_train_ok[var] = app_train_ok[var].map(encoder_dict[var])

# Save the "encoder_dict"
joblib.dump(encoder_dict, "save_files/encoder_dict.joblib")

app_train_ok = app_train_ok.drop("SK_ID_CURR", axis = 1)


# 4. Handling Imbalanced Data ---------------

X = app_train_ok.drop("TARGET", axis = 1)
y = app_train_ok["TARGET"]

# Undersampling the majority class
undersample = RandomUnderSampler(random_state = 42)
X_res, y_res = undersample.fit_resample(X, y)


# 5. Feature Scaling ---------------

list_num_ok = joblib.load("save_files/list_num.joblib")
list_num_ok.remove("SK_ID_CURR")
list_num_ok.remove("TARGET")

X_res_stand = X_res.copy()

# Apply standardization on numerical features
dict_standardization = {}
for i in list_num_ok:
    # fit on training data column
    scale = StandardScaler()
    scale.fit(X_res_stand[[i]])
    # save the scaler object per variable
    joblib.dump(scale, "save_files/joblib_standardization/" + i + ".joblib")
    dict_standardization[i] = "save_files/joblib_standardization/" + i + ".joblib"
    # transform the training data column
    X_res_stand[i] = scale.transform(X_res_stand[[i]])

# save
joblib.dump(dict_standardization, "save_files/dict_standardization.joblib")

# Combine "X_res_stand" and "y_res" to save the data
data_resample = pd.concat(objs = [X_res_stand, y_res], axis = 1)
data_resample.to_csv("save_files/data_final_resample_standardization.csv", index = False)


# 6. Modeling and Evaluation (Based on Features Importance) ---------------

# Select features importance using SelectFromModel
sel = SelectFromModel(RandomForestClassifier(random_state = 42, 
                                            max_features = "log2",
                                            n_estimators = 1000))
sel.fit(X, y)

selected_feat = X.columns[(sel.get_support())]

# Update the X data
X_sel = X[selected_feat]
y_sel = y

# Random Forest (only run the best model)
# Modeling using "best_params_" from GridSearchCV
# forest_classifier_sel = RandomForestClassifier(random_state = 42, 
#                                                max_features = "sqrt", 
#                                                n_estimators = 1000)
# forest_classifier_sel.fit(X_sel, y_sel)

# save the Random Forest classifier model (selected features)
# joblib.dump(forest_classifier_sel, "save_files/randomforest_classifier_model_sel.joblib")


# Logistic Regression (for the lighter model)
# Modeling using "best_params_" from GridSearchCV
logistic_classifier_sel = LogisticRegression(random_state = 42, 
                                             C = 100, 
                                             penalty = "l2", 
                                             solver = "newton-cg")
logistic_classifier_sel.fit(X_sel, y_sel)

# save and run the Logistic classifier model (selected features) for the lighter model
joblib.dump(logistic_classifier_sel, "save_files/logistic_classifier_model_sel.joblib")





