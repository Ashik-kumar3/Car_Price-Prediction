import streamlit as st
import pickle
from PIL import Image
import pandas as pd
import numpy as np

def main():
    st.title(":rainbow(Liver Disease Prediction)")

    image = Image.open('DSC_5903.jpg.webp')

    st.image(image,width=800)
    #Age	Gender	TB	DB	Alkphos	Sgpt	Sgot	TP	ALB	A/G Ratio	Selector	Gender_Female	Gender_Male
    Age = st.text_input(':red[Age]','')
    Gender = st.text_input(':green[Gender]','')
    TB = st.text_input(':red[TB]','')
    DB = st.text_input(':yellow[DB]','')
    Alkphos = st.text_input(':blue[Alkphos]','')
    Sgpt = st.text_input(':orange[Sgpt]','')
    Sgot = st.text_input(':red[Sgot]','')
    TP = st.text_input(':violet[TP]','')
    ALB = st.text_input(':green[ALB]','')
    A/G Ratio = st.text_input(':blue[A/G Ratio]]','')
    Selector = st.text_input(':yellow[Selector]','')
    


    knn = pickle.load(open('knn.sav','rb'))
    ohe_gender = pickle.load(open('ohe_gender.sav','rb'))

    # features = [['Name','Location','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats']]
    pred = st.button("PREDICT")
    print(pred)

    if pred:

        Gender = ohe_gender.transform([[Gender]])

        Gender = pd.DataFrame(Gender,columns=ohe_gender.get_feature_names_out())
        


        data = pd.DataFrame([{
        'Age': Age,
        'TB':int(TB),
        'DB': float(DB),
        'Alkphos' : float(Alkphos),
        'Sgpt': float(Sgpt),
        'Sgot': float(Sgot),
        'TB': float(TB),
        'ALB': int(ALB),
        'A/GRatio' : float('A/G Ratio')
        }])
        

        features = pd.concat([data, Gender,], axis=1)
        
        

        prediction = knn.predict(data)
    

        st.write(f"The result is {prediction}")

main()

#Maruti Wagon R LXI CNG	Mumbai	2010	72000	CNG	Manual	First	26.6 km/kg	998 CC	58.16 bhp	5.0	NaN	1.75
