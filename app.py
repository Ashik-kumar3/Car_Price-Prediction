import streamlit as st
import pickle
from PIL import Image
import pandas as pd
import numpy as np

def main():
    st.title(":rainbow(CAR PRICE PREDICTION)")

    image = Image.open('DSC_5903.jpg.webp')

    st.image(image,width=800)

    Name = st.text_input(':red[Name]','')
    Location = st.text_input(':green[Location]','')
    Year = st.text_input(':red[Year]','')
    Kilometers_Driven = st.text_input(':yellow[Kilometers_Driven]','')
    Fuel_Type = st.text_input(':blue[Fuel_Type]','')
    Transmission = st.text_input(':orange[Transmission]','')
    Owner_Type = st.text_input(':red[Owner_Type]','')
    Mileage = st.text_input(':violet[Mileage]','')
    Engine = st.text_input(':green[Engine]','')
    Power = st.text_input(':blue[Power]','')
    Seats = st.text_input(':yellow[Seats]','')
    
    if Owner_Type=="First":
      Owner_Type=0
    elif Owner_Type=="Second":
        Owner_Type=1
    elif Owner_Type=="Third":
        Owner_Type=2
    else:
        Owner_Type=3

    model = pickle.load(open('model.sav','rb'))
    ohe_loc = pickle.load(open('ohe_loc.sav','rb'))
    ohe1_fuel = pickle.load(open('ohe1_fuel.sav','rb'))
    ohe2_Transmission = pickle.load(open('ohe2_Transmission.sav','rb'))
    te_name = pickle.load(open('te_name.sav','rb'))

    # features = [['Name','Location','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats']]
    pred = st.button("PREDICT")
    print(pred)

    if pred:

        Location = ohe_loc.transform([[Location]])
        Fuel_Type = ohe1_fuel.transform([[Fuel_Type]])
        Transmission = ohe2_Transmission.transform([[Transmission]])
        # Name = te_name.transform([['Name']])

        Location = pd.DataFrame(Location,columns=ohe_loc.get_feature_names_out())
        Fuel_Type = pd.DataFrame(Fuel_Type,columns=ohe1_fuel.get_feature_names_out())
        Transmission = pd.DataFrame(Transmission,columns=ohe2_Transmission.get_feature_names_out())


        data = pd.DataFrame([{
        'Name': Name,
        'Year':int(Year),
        'Kilometers_Driven': float(Kilometers_Driven),
        'Owner_Type' : Owner_Type,
        'Mileage': float(Mileage),
        'Engine': float(Engine),
        'Power': float(Power),
        'Seats': int(Seats),
        }])
        
        #data['Name'] = te_name.transform(data['Name'])
        #data['Name'] = te_name.transform(data['Name'].astype(str))
        features = pd.concat([data, Location, Fuel_Type, Transmission], axis=1)
        # colms = pd.concat([data,Location,Fuel_Type,Transmission])
        # features = pd.DataFrame([colms])
        data = te_name.transform(features)
        

        prediction = model.predict(data)
    

        st.write(f"The price of car is {prediction}")

main()

#Maruti Wagon R LXI CNG	Mumbai	2010	72000	CNG	Manual	First	26.6 km/kg	998 CC	58.16 bhp	5.0	NaN	1.75
