from django.db import models

# Create your models here.

class MyModel(models.Model):
    Gender = models.CharField()
    Married = models.CharField()
    Dependents = models.IntegerField()
    Education = models.CharField()
    Self_Employed = models.CharField()
    Applicant_Income = models.IntegerField()
    Coapplicant_Income = models.IntegerField()
    Loan_Amount = models.IntegerField()
    Loan_Amount_Term = models.IntegerField()
    Credit_History = models.CharField()
    Property_Area = models.CharField()
