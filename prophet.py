# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
import os


from streamlit import caching

import pystan
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import json
from fbprophet.serialize import model_to_json, model_from_json
import holidays
import altair as alt
import plotly as plt
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.figure_factory as ff
import base64
import itertools
import json


# POUR LANCER L'INTERFACE EN LOCAL:
#   streamlit run interface.py

# POUR LANCER L'INTERFACE SUR LE WEB, après avoir mis le code sur le dépot 
# https://github.com/Lee-RoyMannier/tourisme
# https://share.streamlit.io/lee-roymannier/tourisme/main/interface.py

# st.set_option('deprecation.showPyplotGlobalUse', False)

# TODO: 
# Se référer à 2019 pour faire les prévisions. 

### I - LECTURE DES DONNEES 
def lecture_donnees(data):
    # Formatage de l'index en date
    data = data.set_index(data.columns[0])
    data.index = data.index.map(lambda x: datetime.strptime(x, "%Y-%m-%d").date())

    # Formatage des nombres à vigule en flottant
    data = data.applymap(lambda x: float(x.replace(",", ".")))

    return data

def prep_data(df):
    df_input = df.rename({date_col:"ds",metric_col:"y"},errors='raise',axis=1)
    st.markdown("The selected date column is now labeled as **ds** and the values columns as **y**")
    df_input = df_input[['ds','y']]
    df_input =  df_input.sort_values(by='ds',ascending=True)
    return df_input

def ordre_alpha(categorie):
    """ Pour faciliter la navigation parmi les fichiers, ces derniers sont
    classés par ordre alphabétique. On réorganisera ainsi les paires de
    clé/valeur du dictionnaire 'categorie'."""
    ordonne = sorted(categorie.items(), key=lambda x: x[0])
    categorie = {}
    for donnee in ordonne:
        categorie[donnee[0]] = donnee[1]
    return categorie

def convertion_nom_pays(correspondances_pays, code_iso):
    """ Nom en Français d'un pays à partir de son code iso en 2 lettres.
    Retourne par exemple "France" pour "FR" """
    try:
        nom_converti = correspondances_pays.loc[code_iso]["nom_pays"]
        return nom_converti
    except: 
        return code_iso

@st.cache
def acquisition_donnees():
    # Code iso des pays traduits en noms français courts à partir d'un fichier
    pays = pd.read_csv("iso-pays.csv", header=None)
    pays = pays[[2,4]]
    pays.columns = ["iso", "nom_pays"]
    pays = pays.set_index("iso")
    
    # Lecture des fichiers des tables d'analyse et de leurs noms respectifs.
    # On parcoure pour cela les dossiers de données , organisés en trois
    # principales catégories: les destinations françaises, toutes les
    # destinations et une analyse génériques.  
    data = {}
    emplacement = os.path.join("data_tourisme")
    dossiers_source = ['generiques',
                       'destinations_francaises',
                       'Toutes_destinations']
    for dossier in dossiers_source:
        data[dossier] = {}
        source = os.path.join("data_tourisme/"+dossier)
        sous_dossier = os.listdir(source)[-1]

        data_dossier = "/".join([emplacement, dossier, sous_dossier])
        for donnee_tourisme in os.listdir(data_dossier):
            try:
                donnees_brut = data_dossier + "/" + donnee_tourisme
                analyse = pd.read_csv(donnees_brut, sep=";",
                                      encoding="ISO-8859-1",
                                      engine='python')
                
                # Le nom du fichier est décomposé pour former le nom qui sera affiché
                decompose = donnee_tourisme.split("_")
                type_analyse = decompose[1]
                type_analyse = type_analyse.split("-")
                nouv_type_analyse = type_analyse[1]
                
                # Les analyses générales
                if type_analyse[0] == "Generique":
                    data[dossier][nouv_type_analyse] = analyse
                    
                # Les analyses par pays
                else:
                    nom_pays = convertion_nom_pays(pays, decompose[0])
                    if not nom_pays in data[dossier].keys():
                        data[dossier][nom_pays] = {}
                    data[dossier][nom_pays][nouv_type_analyse] = analyse
                    
            except:
                pass
           
    # Réorganisation par ordre alphabétique des données
    for type_analyse in data:
        data[type_analyse] = ordre_alpha(data[type_analyse])
        if type_analyse != "generiques":
            for pays in data[type_analyse]:
                data[type_analyse][pays] = ordre_alpha(data[type_analyse][pays])
    return data, dossiers_source


### II - MISE EN FORME

nom_pays_modif = {
    'Guadeloupe': 'OM-Antilles',
    'Nouvelle' : 'OM-Pacifique',
    'Mayotte' : f'OM-Oc.Indien'
    }

def find_key(v): 
    result = v
    for k, val in nom_pays_modif.items(): 
        if v == val: 
            result = k 
    return result

def changement_nom(pays_nom):
    if (pays_nom in nom_pays_modif.keys()):
        return nom_pays_modif[pays_nom]
    else:
        return pays_nom
    
def changement_OM(col_df):
    i = 0
    list_col = col_df
    for nom in list_col:
        list_col[i] = changement_nom(nom)
        i = i+1
    return list_col


month_str = {
    1: "janvier" , 2: "février"  , 3: "mars", 
    4: "avril"   , 5: "mai"      , 6: "juin", 
    7: "juillet" , 8: "août"     , 9: "septembre",
    10:"octobre" , 11:"novembre" , 12:"décembre"}


def duree_str(date1, date2):
    """Ecrit l'interval entre deux dates d'une manière intuitive et sans 
    redondances. Les dates en entrée sont au format de la librairie datetime.
    
    Par exemple, si on en en entrée:
    >>> date1 = datetime(2020, 10, 3)
    >>> date2 = datetime(2020, 10, 10)
    
    Alors l'interval entre les deux date s'écrira: 'du 3 au 10 octobre 2020' à
    la place par exemple de l'écriture redondante: 'du 3 octobre 2020 au 10 
    octobre 2020'.
    
    Si cela est nécessaire, les années et les mois sont précisés pour chaque
    date. Par exemple, on écrira: 'du 3 octobre 2020 au 10 septembre 2021'."""
    
    d1 = min(date1, date2)
    d2 = max(date1, date2)
    
    def day_str(j):
        if j==1:
            return "1er"
        else:
            return str(j)
    
    a1, m1, j1 = str(d1.year), month_str[d1.month], day_str(d1.day)
    a2, m2, j2 = str(d2.year), month_str[d2.month], day_str(d2.day)
    
    if a1==a2 and m1==m2:
        return  j1+" au "+j2+" "+m2+" "+a2
    elif a1==a2 and m1!=m2:    
        return  j1+" "+m1+" au "+j2+" "+m2+" "+a2 
    else:
        return  j1+" "+m1+" "+a1+" au "+j2+" "+m2+" "+a2

### III - CALCULS





### VI - INTERFACES WEB


    

def interface():

    data, dossiers_source = acquisition_donnees()
    types_analyse = {"Mots clés génériques": data[dossiers_source[0]],
                     "Destinations Françaises": data[dossiers_source[1]],
                     "Destinations Françaises et Européennes": data[dossiers_source[2]]}
    txt = "Types d'analyses: " 
    noms_types = list(types_analyse.keys())
    mode = st.sidebar.selectbox(txt, noms_types)
    
        
    ### ANALYSE GENERIQUE
    if mode == noms_types[0]:
        # Récupération des noms de tables d'analyse et construction de la 
        # liste déroulante
        noms_analyses = list(types_analyse[mode].keys())
        fichier = st.sidebar.selectbox("Quelle analyse effectuer?", noms_analyses)
        data = lecture_donnees(types_analyse[mode][fichier])
        

    ### ANALYSE PAR PAYS
    else:
        tous_pays = list(types_analyse[mode].keys())
        pays_choisi = st.sidebar.selectbox("Quel pays?", tous_pays)
        detail_analyse = list(types_analyse[mode][pays_choisi].keys())
        detail_analyse = changement_OM(detail_analyse)
        analyse_pays = st.sidebar.selectbox("Quelle analyse effectuer?",
                                           detail_analyse)
        analyse_pays = find_key(analyse_pays)
        data = lecture_donnees(types_analyse[mode][pays_choisi][analyse_pays])
    
    st.sidebar.header('Effectuer une prévision')
    columns = list(data.columns)
    metric_col = st.sidebar.selectbox("Sur quelle catégorie obtenir des prévisions ?",index=0,options=columns,key="values")
    data_index = data[metric_col].index.name
    data_cast = data.reset_index()[[data_index,metric_col]]
    data_cast.columns = ['ds','y']
    
    print('-'*10)
    print(data_cast)
   
                    
    st.header('Visualiation des données brutes')
    col1, col2 = st.columns([3, 1])
    with col1:
        try:
            line_chart = alt.Chart(data_cast).mark_line().encode(
            x = 'ds:T',
            y = "y:Q",tooltip=['ds:T', 'y']).properties(title=" ").interactive()
            st.altair_chart(line_chart,use_container_width=True)
        except:
            st.line_chart(data_cast['y'],use_container_width =True,height = 300)
        
    with col2:
        st.write("Statistiques:")
        st.write(data_cast.describe())
    
    
    periods_input = st.sidebar.number_input("Horizon de prévision (en semaines) ",
            min_value = 1, max_value = 52,value=12)
    periods_input = periods_input
        
    st.sidebar.subheader("Configuration des paramètres (ne pas toucher pour le moment)")
    

    with st.sidebar.container():

        with st.expander("Saisonalité"):
            seasonality = st.radio(label=' ',options=['additive','multiplicative'])

        with st.expander("Composantes tendentielles"):
            st.write("Ajouter des tendances:")
            daily = st.checkbox("Daily")
            weekly= st.checkbox("Weekly")
            monthly = st.checkbox("Monthly")
            yearly = st.checkbox("Yearly")

        with st.expander("Modèle"):
         
            growth = st.radio(label=' ',options=['linear',"logistic"]) 

            if growth == 'linear':
                growth_settings= {
                            'cap':1,
                            'floor':0
                        }
                cap=1
                floor=0
                data_cast['cap']=1
                data_cast['floor']=0

            if growth == 'logistic':
                st.info('Configure saturation')

                cap = st.slider('Cap',min_value=0.0,max_value=1.0,step=0.05)
                floor = st.slider('Floor',min_value=0.0,max_value=1.0,step=0.05)
                if floor > cap:
                    st.error('Invalid settings. Cap doit être > floor.')
                    growth_settings={}

                if floor == cap:
                    st.warning('Cap doit être > floor')
                else:
                    growth_settings = {
                        'cap':cap,
                        'floor':floor
                        }
                    data_cast['cap']=cap
                    data_cast['floor']=floor
            
            
        with st.expander('Vacances'):
            
            countries = ['Nom du pays','Italy','Spain','United States','France','Germany','Ukraine']
            
            with st.container():
                years=[2021]
                selected_country = st.selectbox(label="Selectionner un pays",options=countries)

                if selected_country == 'Italy':
                    for date, name in sorted(holidays.IT(years=years).items()):
                        st.write(date,name) 
                            
                if selected_country == 'Spain':
                    
                    for date, name in sorted(holidays.ES(years=years).items()):
                            st.write(date,name)                      

                if selected_country == 'United States':
                    
                    for date, name in sorted(holidays.US(years=years).items()):
                            st.write(date,name)
                            
                if selected_country == 'France':
                    
                    for date, name in sorted(holidays.FR(years=years).items()):
                            st.write(date,name)
                            
                if selected_country == 'Germany':
                    
                    for date, name in sorted(holidays.DE(years=years).items()):
                            st.write(date,name)
                            
                if selected_country == 'Ukraine':
                    
                    for date, name in sorted(holidays.UKR(years=years).items()):
                            st.write(date,name)

                else:
                    holidays = False
                            
                holidays = st.checkbox('Ajouter un période de vacance')

        with st.expander('Hyperparameters'):            
            seasonality_scale_values= [0.1, 1.0,5.0,10.0]    
            changepoint_scale_values= [0.01, 0.1, 0.5,1.0]
            changepoint_scale= st.select_slider(label= 'Changepoint prior scale',options=changepoint_scale_values)
            seasonality_scale= st.select_slider(label= 'Seasonality prior scale',options=seasonality_scale_values)    


    with st.container():
       st.subheader("Prédiction 🔮")
       st.write("Entrainer le modèle pour faire une prédiction")
       
       if input:
           
           if st.checkbox("Entainement du modèle",key="fit"):
               if len(growth_settings)==2:
                   m = Prophet(seasonality_mode=seasonality,
                               daily_seasonality=daily,
                               weekly_seasonality=weekly,
                               yearly_seasonality=True,
                               growth=growth,
                               changepoint_prior_scale=1,
                               seasonality_prior_scale= seasonality_scale,
                               changepoint_range=0.85)
                   if holidays:
                       m.add_country_holidays(country_name=selected_country)
                   
                   m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)
                       
                   #if monthly:
                       #m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)
                       
    
                   with st.spinner('Fitting the model..'):
    
                       m = m.fit(data_cast)
                       future = m.make_future_dataframe(periods=periods_input,freq='W')
                       future['cap']=cap
                       future['floor'] = floor
                       st.write("Le modèle va faire une prédiction jusqu'en ", future['ds'].max())
                       st.success('Le modèle est opérationel')
    
               else:
                   st.warning('Configuration invalide')
    
           if st.checkbox("Générer la prédiction",key="predict"):
               try:
                   with st.spinner("Prédiction..."):
    
                       forecast = m.predict(future)
                       forecast_print = forecast[['ds', 'yhat','yhat_lower','yhat_upper']]
                       forecast_print.sort_values(by = 'ds', ascending = False, inplace=True ) 
                       st.dataframe(forecast_print)
                       fig1 = m.plot(forecast)
                       #st.write(fig1)
                       output = 1
    
                       if growth == 'linear':
                           fig2 = m.plot(forecast)
                           a = add_changepoints_to_plot(fig2.gca(), m, forecast)
                           st.write(fig2)
                           output = 1
               except:
                   st.warning("Il faut d'abord entrainer le modèle.. ")
                       
           
           if st.checkbox('Monter les composantes de tendances'):
               try:
                   with st.spinner("Loading.."):
                       fig3 = m.plot_components(forecast)
                       st.write(fig3)
               except: 
                   st.warning("Effectuer d'abord un prévision..") 
   
    
    

### VII - PROGRAMME PRINCIPAL
interface()
# export_ppt(generation_generique=False)
