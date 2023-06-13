import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image



# ------- CONFIGURACION DE LA PAGINA---------------------------------#
st.set_page_config(page_title="mi primera app",page_icon="⚓",layout= 'wide')

# """(para que no nos muestre (los waring) lo que cabia de streamlist y nos muestre solo lo que hagamos)"""
st.set_option('deprecation.showPyplotGlobalUse', False) 
# ------- CONFIGURACION DE LA PAGINA---------------------------------#


# ------- COSAS QUE VAMOS A USAR EN TODA LA APP----------------------#
df = pd.read_csv(r'C:/Users/fabia/OneDrive/Documents/Fabi/DataAnalisis/Bootcamp/samplerepo/Temario/Modulo1/12-Scripts y Streamlit/Streamlit/data/titanic.csv')



#opening the image
image = Image.open(r"C:/Users/fabia/OneDrive/Documents/Fabi/DataAnalisis/Bootcamp/samplerepo/Temario/Modulo1/12-Scripts y Streamlit/Streamlit/img/Titanic_logo.svg.png")

# ------- TITULO-----------------------------------------------------#
st.image(image, caption='Enter any caption here',width=250)
st.title("DATA SET TITANIC")
st.text('intro: titanic es un barquito que se hundió en el mar')
st.write('intro: titanic es un barquito que se hundió en el mar')


# ------- SIDE BAR-----------------------------------------------------#
st.sidebar.image(image, caption='Enter any caption here')

filtro_clase = st.sidebar.multiselect("Clase", df["Pclass"].unique())
if filtro_clase:
    df = df[df["Pclass"].isin(filtro_clase)]
filtro_genero = st.sidebar.selectbox('Sexo', df['Sex'].unique())
if filtro_genero:
    df = df.loc[df['Sex'] == filtro_genero]
st.dataframe(df)

# ------- COL-----------------------------------------------------#
col1, col2, col3 = st.columns(3)

with col1:
   st.image(image, caption='Enter any caption here',width=100)
   st.write ("INFORMACIÓN TITANIC1")

with col2:
    st.image(image, caption='Enter any caption here',width=100)
    st.write ("INFORMACIÓN TITANIC2")

with col3:
    st.image(image, caption='Enter any caption here',width=100)
    st.write ("INFORMACIÓN TITANIC3")


#--------------------pestañas----------------------------#
tab1, tab2 , tab3, tab4 = st.tabs(["Genero", "Pclass", "Edad","titanic"])
with tab1:
    st.write("hola")
with tab2:
    st.image(image, caption='Enter any caption here',width=100)
    st.write('INFORMACIÓN DE TITANIC')
with tab3:
    prop_sex = df["Sex"].value_counts()
    fig = go.Figure(
        data=[
            go.Pie(
                labels=(prop_sex / len(df) * 100).index,
                values=(prop_sex / len(df) * 100).values,
                text=prop_sex.index
            )
        ]
    )
    st.plotly_chart(fig)

with tab4:
    df['Survived2'] = df['Survived'].map({1: 'Sobrevivientes', 0: 'Fallecidos'})
    Sobrevivientes= df.groupby('Survived2')['PassengerId'].count().reset_index()
    fig2 = px.pie(Sobrevivientes, values='PassengerId', names='Survived2', template="plotly_dark", title="Cantidad de Sobrevivintes ",hole=0.3,color_discrete_sequence=['turquoise','violet']
    )
    st.plotly_chart(fig2)

#--------------------pestañas----------------------------#
