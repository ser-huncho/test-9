from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import time


# List for looping in html file
orgtype = joblib.load("save_files/list_orgtype.joblib")
orgtype.sort()
len_orgtype = len(orgtype)
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
len_days = len(days)

# Import model
encoder = joblib.load("save_files/encoder_dict.joblib")
dict_stand = joblib.load("save_files/dict_standardization.joblib")
model_logistic = joblib.load("save_files/logistic_classifier_model_sel.joblib")


# Features that used
selected_feat = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
                 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'HOUR_APPR_PROCESS_START',
                 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'YEARS_BEGINEXPLUATATION_MODE',
                 'TOTALAREA_MODE', 'DAYS_LAST_PHONE_CHANGE',
                 'AMT_REQ_CREDIT_BUREAU_YEAR', 'WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE']


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", 
                           len_orgtype = len_orgtype, 
                           orgtype = orgtype, 
                           len_days = len_days, 
                           days = days)


@app.route("/output", methods = ["POST"])
def output():
    # Access from the "name" attribute within the tag
    applicant_income = float(request.form["applicant_income"])
    credit_amount = float(request.form["credit_amount"])
    loan_annuity = float(request.form["loan_annuity"])
    goods_price = float(request.form["goods_price"])
    region_pop = float(request.form["region_pop"])
    applicant_age = int(request.form["applicant_age"])
    day = request.form["day"]
    hour = int(request.form["hour"])
    org_type = request.form["org_type"]
    credit_bureau = int(request.form["credit_bureau"])
    days_employed = int(request.form["days_employed"])
    days_reg = int(request.form["days_reg"])
    days_id = int(request.form["days_id"])
    days_phone = int(request.form["days_phone"])
    exp_mode = float(request.form["exp_mode"])
    area_mode = float(request.form["area_mode"])
    source_2 = float(request.form["source_2"])
    source_3 = float(request.form["source_3"])


    # Categorical Data ----------

    # 1. Day
    # dict of day and its value from freq encoder
    dict_day_encoder = {}
    for k,v in encoder["WEEKDAY_APPR_PROCESS_START"].items():
        dict_day_encoder[k.lower()] = round(v, 6)

    for k,v in dict_day_encoder.items():
        if day.lower() == k:
            day_encode = v


    # 2. Organization Type
    dict_orgtype_encoder = {}
    for k,v in encoder["ORGANIZATION_TYPE"].items():
        dict_orgtype_encoder[k.lower()] = round(v, 6)

    org_type_encode = None
    for k,v in dict_orgtype_encoder.items():
        if org_type.lower() == k:
            org_type_encode = v
    

    # Numerical Data ----------

    num_value = []
    for i in selected_feat:
        if i == 'AMT_INCOME_TOTAL':
            applicant_income1 = joblib.load(dict_stand[i]).transform([[applicant_income]])
            num_value.append(applicant_income1)
        if i == 'AMT_CREDIT':
            credit_amount1 = joblib.load(dict_stand[i]).transform([[credit_amount]])
            num_value.append(credit_amount1)
        if i == 'AMT_ANNUITY':
            loan_annuity1 = joblib.load(dict_stand[i]).transform([[loan_annuity]])
            num_value.append(loan_annuity1)
        if i == 'AMT_GOODS_PRICE':
            goods_price1 = joblib.load(dict_stand[i]).transform([[goods_price]])
            num_value.append(goods_price1)
        if i == 'REGION_POPULATION_RELATIVE':
            region_pop1 = joblib.load(dict_stand[i]).transform([[region_pop]])
            num_value.append(region_pop1)
        if i == 'DAYS_BIRTH':
            applicant_age1 = joblib.load(dict_stand[i]).transform([[applicant_age]])
            num_value.append(applicant_age1)
        if i == 'DAYS_EMPLOYED':
            days_employed1 = joblib.load(dict_stand[i]).transform([[days_employed]])
            num_value.append(days_employed1)
        if i == 'DAYS_REGISTRATION':
            days_reg1 = joblib.load(dict_stand[i]).transform([[days_reg]])
            num_value.append(days_reg1)
        if i == 'DAYS_ID_PUBLISH':
            days_id1 = joblib.load(dict_stand[i]).transform([[days_id]])
            num_value.append(days_id1)
        if i == 'HOUR_APPR_PROCESS_START':
            hour1 = joblib.load(dict_stand[i]).transform([[hour]])
            num_value.append(hour1)
        if i == 'EXT_SOURCE_2':
            source_21 = joblib.load(dict_stand[i]).transform([[source_2]])
            num_value.append(source_21)
        if i == 'EXT_SOURCE_3':
            source_31 = joblib.load(dict_stand[i]).transform([[source_3]])
            num_value.append(source_31)
        if i == 'YEARS_BEGINEXPLUATATION_MODE':
            exp_mode1 = joblib.load(dict_stand[i]).transform([[exp_mode]])
            num_value.append(exp_mode1)
        if i == 'TOTALAREA_MODE':
            area_mode1 = joblib.load(dict_stand[i]).transform([[area_mode]])
            num_value.append(area_mode1)
        if i == 'DAYS_LAST_PHONE_CHANGE':
            days_phone1 = joblib.load(dict_stand[i]).transform([[days_phone]])
            num_value.append(days_phone1)
        if i == 'AMT_REQ_CREDIT_BUREAU_YEAR':
            credit_bureau1 = joblib.load(dict_stand[i]).transform([[credit_bureau]])
            num_value.append(credit_bureau1)


    # Combine into DataFrame ----------
    df = pd.DataFrame(np.concatenate(num_value), index = selected_feat[:-2]).T
    df[selected_feat[-2]] = day_encode
    df[selected_feat[-1]] = org_type_encode


    # Predict the Data
    pred = int(model_logistic.predict(df))

    if pred == 1:
        msg1 = "The Applicant Has Payment Difficulties"
        msg2 = "Loan Rejected"
    elif pred == 0:
        msg1 = "The Applicant Doesn’t Have Payment Difficulties"
        msg2 = "Loan Accepted"
    
    return render_template("output.html",
                           msg1 = msg1,
                           msg2 = msg2,
                           applicant_income = applicant_income,
                           credit_amount = credit_amount,
                           loan_annuity = loan_annuity,
                           goods_price = goods_price,
                           region_pop = region_pop,
                           applicant_age = applicant_age,
                           day = day.title(),
                           hour = hour,
                           org_type = org_type.title(),
                           credit_bureau = credit_bureau,
                           days_employed = days_employed,
                           days_reg = days_reg,
                           days_id = days_id,
                           days_phone = days_phone,
                           exp_mode = exp_mode,
                           area_mode = area_mode,
                           source_2 = source_2,
                           source_3 = source_3)


if __name__ == "__main__":
	app.run(debug = True)