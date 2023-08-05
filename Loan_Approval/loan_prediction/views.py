from django.shortcuts import render
import pandas as pd
import json
from forms import MyForm
from src.utils import object_loader
from src.pipeline.prediction_pipeline import PredictionPipeline

# Create your views here.

def form(request):
    if request.method == "POST":
        form = MyForm(request.POST)
        if form.is_valid():
            form.save()
    else:
        form = MyForm()

    return render(request, 'form.html', {'form':form})




def prediction(request):

    gender = request.GET['Gender']
    Married = request.GET['Married']
    Dependents = request.GET['Dependents']
    Education = request.GET['Education']
    Self_Employed = request.GET['Self_Employed']
    ApplicantIncome = request.GET['ApplicantIncome']
    CoapplicantIncome = request.GET['CoapplicantIncome']
    LoanAmount = request.GET['LoanAmount']
    Loan_Amount_Term = request.GET['Loan_Amount_Term']
    Credit_History = request.GET['Credit_History']
    Property_Area = request.GET['Property_Area']
    

    schema_cols = json.load('schema_cols.json')
    
    schema_cols['ApplicantIncome'] = ApplicantIncome
    schema_cols['CoapplicantIncome'] = CoapplicantIncome
    schema_cols['LoanAmount'] = LoanAmount
    schema_cols['Education'] = Education
    schema_cols['Gender'] = gender
    schema_cols['Married'] = Married
    schema_cols['Self_Employed'] = Self_Employed
    schema_cols['Loan_Amount_Term'] = Loan_Amount_Term
    schema_cols['Credit_History'] = Credit_History
    
    
    rel_col = [x for x in schema_cols if x[:10] =='Dependents']
    col = 'Dependents' + str(Dependents)
    for i in range(len(rel_col)):
        if rel_col[i] == col:
            schema_cols[rel_col[i]] = 1
        else:
            schema_cols[rel_col[i]] = 0
    

    rel_col = [x for x in schema_cols if x[:13] =='Property_Area']
    col = 'Property_Area' + str(Property_Area)
    for i in range(len(rel_col)):
        if rel_col[i] == col:
            schema_cols[rel_col[i]] = 1
        else:
            schema_cols[rel_col[i]] = 0
    
    


    val = {k: [v] for k,v in schema_cols.items()}
    df = pd.DataFrame(val, dtype=float)

    pred_pipeline = PredictionPipeline()
    prediction = pred_pipeline.predict(df)


    if prediction==1:
        output = 'your loan application has been approved'
    if prediction==0:
        output = 'your loan application has been declined.'

    return render(request, prediction.html, {"output":output} )

