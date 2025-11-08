# (D:\Udemy\Complete_DSMLDLNLP_Bootcamp\UPractice1\venv) 
# D:\Udemy\Complete_DSMLDLNLP_Bootcamp\UPractice2\Holiday_Package_Prediciton>python 2-Flask_App.py

from flask import Flask, request, render_template
import pickle
import pandas as pd

with open("best_model.pkl", "rb") as File:
    model = pickle.load(File)

with open("scaler.pkl", "rb") as File:
    scaler = pickle.load(File)

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def welcome():
    if request.method == 'POST':
        CustomerID = int(request.form['CustomerID'])
        Age = int(request.form['Age'])
        TypeofContact = request.form['TypeofContact']
        CityTier = int(request.form['CityTier'])
        DurationOfPitch = float(request.form['DurationOfPitch'])
        Occupation = request.form['Occupation']
        Gender = request.form['Gender']
        NumberOfPersonVisiting = int(request.form['NumberOfPersonVisiting'])
        NumberOfFollowups = int(request.form['NumberOfFollowups'])
        ProductPitched = request.form['ProductPitched']
        PreferredPropertyStar = int(request.form['PreferredPropertyStar'])
        MaritalStatus = request.form['MaritalStatus']
        NumberOfTrips = int(request.form['NumberOfTrips'])
        Passport = int(request.form['Passport'])
        PitchSatisfactionScore = int(request.form['PitchSatisfactionScore'])
        OwnCar = int(request.form['OwnCar'])
        NumberOfChildrenVisiting = int(request.form['NumberOfChildrenVisiting'])
        Designation = request.form['Designation']
        MonthlyIncome = float(request.form['MonthlyIncome'])

        new_df = pd.DataFrame({
        "CustomerID" : [CustomerID] ,
        "Age" : [Age] ,
        "TypeofContact" : [TypeofContact] ,
        "CityTier" : [CityTier] ,
        "DurationOfPitch" : [DurationOfPitch] ,
        "Occupation" : [Occupation] ,
        "Gender" : [Gender] ,
        "NumberOfPersonVisiting" : [NumberOfPersonVisiting] ,
        "NumberOfFollowups" : [NumberOfFollowups] ,
        "ProductPitched" : [ProductPitched] ,
        "PreferredPropertyStar" : [PreferredPropertyStar] ,
        "MaritalStatus" : [MaritalStatus] ,
        "NumberOfTrips" : [NumberOfTrips] ,
        "Passport" : [Passport] ,
        "PitchSatisfactionScore" : [PitchSatisfactionScore] ,
        "OwnCar" : [OwnCar] ,
        "NumberOfChildrenVisiting" : [NumberOfChildrenVisiting] ,
        "Designation" : [Designation] ,
        "MonthlyIncome" : [MonthlyIncome]})

        new_df.drop(['CustomerID','NumberOfChildrenVisiting'], inplace=True, axis=1)
        new_df['Occupation'] = new_df['Occupation'].map({'Free Lancer':0,'Small Business':2,'Salaried':3,'Large Business':1})
        new_df['ProductPitched'] = new_df['ProductPitched'].map({'Deluxe':3, 'Basic':4, 'Standard':2, 'Super Deluxe':1, 'King':0})
        new_df['MaritalStatus'] = new_df['MaritalStatus'].map({'Single':1, 'Divorced':2, 'Married':3, 'Unmarried':0})
        new_df['Designation'] = new_df['Designation'].map({'Manager':3, 'Executive':4, 'Senior Manager':2, 'AVP':1, 'VP':0})

        if new_df['TypeofContact'][0] == "Self Enquiry":
            new_df['TypeofContact_Self Enquiry'] = True
            new_df.drop("TypeofContact",inplace = True, axis = 1)
        elif new_df['TypeofContact'][0] == "Company Invited":
            new_df['TypeofContact_Self Enquiry'] = False
            new_df.drop("TypeofContact",inplace = True, axis = 1)

        if new_df['Gender'][0] == "Female":
            new_df['Gender_Male'] = False
            new_df.drop("Gender", inplace = True, axis = 1)

        elif new_df['Gender'][0] == "Male":
            new_df['Gender_Male'] = True
            new_df.drop("Gender", inplace = True, axis = 1)

        new_df = scaler.transform(new_df)
        prediction = model.predict(new_df)

        if prediction[0] == 1:
            prediction_text = f"The Predicted result is that Customer will purchace the package."
        elif prediction[0] == 0:
            prediction_text = f"The Predicted result is that Customer will not purchace the package."
        return f"<h1>{prediction_text}</h1>"

    return render_template("front_page1.html")

if __name__=="__main__":
    app.run(debug=True)