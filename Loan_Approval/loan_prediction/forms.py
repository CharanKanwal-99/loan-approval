from django import forms
from models import MyModel

class MyForm(forms.ModelForm):
    class Meta:
        model = MyModel()
        fields = ['Gender','Married','Dependents','Education','Self_Employed','Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Amount_Term', 'Credit_History','Property_Area']
        labels = {'Gender': 'gender', 'Married':'married', 'Dependents':'dependents','Education':'education','Self-Employed':'self-employed','Applicant_Income': 'applicant_income','Coapplicant_Income': 'coapplicant_income','Loan_Amount':'loan_amount','Loan_Amount_Term':'loan_amount_term','Credit_History':'credit_history','Property_Area':'property_area'}
