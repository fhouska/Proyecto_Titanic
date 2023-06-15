import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


# ------- CONFIGURACION DE LA PAGINA---------------------------------#
st.set_page_config(page_title="DataSetTitanic",page_icon="⚓",layout= 'wide')
st.set_option('deprecation.showPyplotGlobalUse', False) 

# ------- SUBIR EL DF----------------------#
df = pd.read_csv('data/titanic.csv')

# ------- SUBIR LOGO----------------------#
image = Image.open('img/logo.png')

st.image(image, caption='',width=400)

# ------- TITULO-----------------------------------------------------#
st.title("Análisis del Dataset: Titanic")
st.write('Este Dataset contiene información que describen a los pasajeros que iban a bordo del Titanic. ')

st.write("Contenido de las columnas:")
st.write( 
" * PassengerId: identificador único del pasajero.""\r\n"
" * Survived: si el pasajero sobrevivió al naufragio, codificada como 0 (no) y 1 (si).""\r\n"
" * Pclass: clase a la que pertenecía el pasajero: 1, 2 o 3.""\r\n"
" * Name: nombre del pasajero.""\r\n"
" * Sex: sexo del pasajero.""\r\n"
" * Age: edad del pasajero.""\r\n"
" * SibSp: número de hermanos, hermanas, hermanastros o hermanastras en el barco.""\r\n"
" * Parch: número de padres e hijos en el barco.""\r\n"
" * Ticket: identificador del billete.""\r\n"
" * Fare: precio pagado por el billete.""\r\n"
" * Cabin: identificador del camarote asignado al pasajero.""\r\n"
" * Embarked: puerto en el que embarcó el pasajero'""\r\n"
" * Survived_Indicator: Columna adicional que se generó para describir la columna: 'Survived'""\r\n"
" * Title: Columna adicional que se generó extrayendo el títlo de la columna: 'Name' ""\r\n"
" * With_Family: Columna adicional que se generó a partir de sumar la columna: 'SibSp'+'Parch' ""\r\n"
" * Alone: Columna adicional que se generó a partir de la columna: 'With_Family' que identifica si el pasajero viajó solo o no.""\r\n")
"\r\n"
"\r\n"
"\r\n"
"\r\n"

st.dataframe(data=df, width=None, height=200, 
    use_container_width=False, hide_index=True, column_order=None, 
    column_config=None)



# ------- SIDE BAR-----------------------------------------------------#
# st.sidebar.image(image, caption='',width=150)

"\r\n"
"\r\n"
"\r\n"
"\r\n"
st.subheader('Agrupamos nuestro análisis en 5 etapas:')


# Variables nuevas que necesito para que funcionen las gráficas.
colorx =[ '#8A307F', 'MEDIUMturquoise','wheat']
df['Survived_Indicator'] = df['Survived'].map({1: 'Sobrevivientes', 0: 'No Sobrevivientes'})

# Esta función extrae de la columna Name el título con el que el pasajero se identificaba
import re

def get_title(name): 

    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return 




#--------------------pestañas----------------------------#
tab1, tab2 , tab3, tab4, tab5= st.tabs(["Procesamiento de Datos", "Correlación de las Variables", "Pasajeros","Clases y Lugar de Embarque","Conclusiones"])
with tab1:
    st.subheader("Procesamiento de Datos:")
    """Se realizó una exploración inicial de los datos para comprender su estructura característica. \r
    Por lo cual tenemos una base de datos con una forma de 12 columnas y 891 filas.\r"""
    """Paraestacar durante el proceso transformación de los datos, revisamos datos nulos y Valores duplicados.
    """

   
    df_original = pd.read_csv('data/titanic_original.csv')
    st.subheader("Datos Nulos:")
   
    df_null = df_original.isnull().sum().reset_index()
    st.write(df_null,height=100)

    st.write("Se identificaron 3 columnas con datos Nulos. Se procedió a analizar cuanto es el porcentaje de datos faltantes respecto al total.")
    st.write("      • Total null columna: Embarked   0.22 %")
    st.write("      •  Total null columna: Age   19.87 %")
    st.write("      •  Total null columna: Cabin   77.1 %")
    st.write("Se procedió a eliminar la columna Cabin que consideramos que no contamos con información suficiente para poder completar.")
    st.write("En la columna Age, se optó por completar sus datos faltantes por su media.")
    st.write("Y en la columna Embarked lo completamos por su moda.")
    st.subheader("Valores Duplicados:",anchor=None, help=None)
    st.write("No Se identificaron valores Duplicados en el DF.")


with tab2:
# MAPA DE CALOR DE LAS VARIABLES 
    Variables_Cuant = ['PassengerId','Survived','Pclass','Age','SibSp','Parch','Fare']        
# Calculamos la correlación entre las columnas seleccionadas y obtenemos la matriz 
    df_corr = df[Variables_Cuant].corr().sort_values(by='Survived',ascending=False,axis=0).sort_values(by='Survived',ascending=False,axis=1)

    fig1 = go.Figure(data=go.Heatmap(x=df_corr.columns, y=df_corr.columns,z=df_corr.values,
                                colorscale='tropic'))
    fig1.update_layout(
    width=500, 
    height=500, 
    title='Mapa de calor de Correlación',
    template='plotly_dark',
    yaxis=dict(autorange='reversed'), # esto le agregamos de manera forzada ya que el mapa no estaba ordenando de manera descendente. 
    )
    st.plotly_chart(fig1)   

    st.write("Esta es una gráfica de Mapa de Calor en funcion de la variable: Survived. ")
    st.write("""Los colores más oscuros indican una correlación más fuerte, mientras que los colores más claros indican una correlación más débil o nula.
Se puede observar que no hay una correlación fuerte entre las variables esto significa que podrían no estar relacionadas con la probabilidad de supervivencia.
"""
    )
  

with tab3:
    st.write("""En esta gráfica tipo Pie muestra el porcentaje de los pasajeros que han sobrevivido y no sobrevivido al accidente del Titanic.
    De manera simple se puede ver que un 61,6% de los pasajeros no sobrevivieron a accidente.
    """)
    
    Sobrevivientes= df.groupby('Survived_Indicator')['PassengerId'].count().reset_index()
    fig2 = px.pie(Sobrevivientes, values='PassengerId', names='Survived_Indicator', template="plotly_dark", title="Total Pasajeros",hole=0.3,color_discrete_sequence = colorx
    )
    st.plotly_chart(fig2)
    "\r\n"
    "\r\n"
    "\r\n"
    "\r\n"
    

# ANALISIS DE SOBREVIVIENTES SEGUN EL EDAD

    """El histograma muestra cómo se distribuyen los pasajeros, que sobrevivieron y no, en diferentes rangos de edad.
    Como podemos ver la mayor concentración esta en el rango de edad de 25 y 30 años por lo que se puede ver que en su gran mayoría eran personas jóvenes. 
    Para ese rango tenemos un total de 94 sobrevivientes, mientras que el de no sobrevivientes es de 191.
    Otro rango que podemos destacar es el de 0 a 5 donde la cantidad de sobrevivientes es mayor, ya que la cantidad de sobrevivientes es de 31 niños mientras que los no sobrevivientes fueron 13 niños.
    """
    bins = pd.cut(df['Age'], np.arange(0, 100, 5)).astype(str)
    grouped = df.groupby([bins, 'Survived_Indicator']).size().reset_index(name='Count')

    fig3 = px.bar(grouped, x='Age', y='Count', color='Survived_Indicator',
             color_discrete_sequence=colorx, opacity=0.85,template='plotly_dark')

    fig3.update_layout(
    title='Distribución de Pasajeros en función de su edad',
    xaxis=dict(title='Edad'),
    yaxis=dict(title='Cantidad'),
    bargap=0.1,
    showlegend=True,
    barmode='group' 
    )
    st.plotly_chart(fig3)
    "\r\n"
    "\r\n"
    "\r\n"
    "\r\n"

# ANALISIS DE SOBREVIVIENTES SEGUN EL TITULO
    """Este histograma muestra la cantidad de pasajeros en cada categoría de título que tenía dentro de la columna “Name”. Observamos que la mayoría de los pasajeros tienen títulos comunes como “Miss”, “Mrs.” y “Miss”. Con estos títulos podemos ver el género y el estado civil de los pasajeros. 
    Podemos observar que la categoría más numerosa es “Mr.” Esto sugiere que la mayoría de los pasajeros eran hombres
    """

# Aplicamos la función para generar una nueva columna con el título
    df['Title'] = df['Name'].apply(get_title) 
# Agrupamos por la columnaSurvived y TiTle para tener asi las cantidades de titulos en los sobrevivientes.
    survived_by_title = df[['Survived_Indicator','Title']].value_counts().reset_index()

    fig4 = px.bar(survived_by_title, x='Title',y='count', color='Survived_Indicator',
                color_discrete_sequence=colorx, opacity=0.85,template='plotly_dark')

    fig4.update_layout(
        title='Distribución de pasajeros en función de su título',
        xaxis=dict(title='Títulos'),
        yaxis=dict(title='Cantidad'),
        bargap=0.1,
        showlegend=True,
        barmode='group' 
    )
    st.plotly_chart(fig4)
    "\r\n"
    "\r\n"
    "\r\n"
    "\r\n"

# ANALISIS DE SOBREVIVIENTES SEGUN EL GÉNERO
    """El histograma muestra la proporción de pasajeros masculinos y femeninos que sobrevivieron y no sobrevivieron.
    Cuando analizamos la proporción de sobrevivientes dentro de cada género vemos que, a pesar de que había más hombres a bordo, un mayor porcentaje de mujeres sobrevivió en comparación con los hombres. 
    Esto nos sugiere la idea de que el género fue un factor determinante en la supervivencia durante el accidente del Titanic.
    """


    fig5 = px.histogram(df, x='Sex', color='Survived_Indicator', barmode='stack', nbins=25,
                    color_discrete_sequence=colorx, opacity=0.85,template='plotly_dark')

    fig5.update_layout(
        title='Distribución de los pasajeros por distinción género',
        xaxis=dict(title='Pasajeros'),
        yaxis=dict(title='Cantidad')
        )
    st.plotly_chart(fig5)
    "\r\n"
    "\r\n"
    "\r\n"
    "\r\n"

# ANALISIS EN CUANTO A VIAJAR SOLO O ACOMPAÑADO

    """El histograma muestra la distribución de pasajeros según si viajaban con familia o solos y la proporción de sobrevivientes y no sobrevivientes.
    Podemos ver que la proporción de pasajeros que viajaban con familia y sobrevivieron es menor en comparación con los que no sobrevivieron. Por otro lado, la proporción de pasajeros que viajaban solos y sobrevivieron es mayor en comparación con los que no sobrevivieron.
    Estos hallazgos sugieren que viajar con familia podría haber sido negativo en términos de posibilidades de supervivencia durante el accidente del Titanic. Es posible que aquellos que estaban acompañados por familiares prioricen ayudar y salvar a sus familiares antes que a sí mismos. 
    """

# Creamos una nueva columna con la suma de las columnas SibSp y Parch para saber si viajaban solos con con Familares.
    df['With_Family'] = df['SibSp'] + df['Parch']

    # Creamos una nueva columna donde indica si el pasajero vajaba solo.
    df['Alone'] = 0
    df.loc[df.With_Family == 0, 'Alone'] = 1
    df['Alone'] = df['Alone'].map({1: 'With_Family', 0: 'Alone'})

    Compania = df[['Survived_Indicator','Alone']].value_counts().reset_index()
   
    fig6 = px.bar(Compania, x='Alone', y='count', color='Survived_Indicator',
                color_discrete_sequence=colorx, opacity=0.85,template='plotly_dark')
    fig6.update_layout(
        title='Distribución pasajeros segun si viajaban solos o con familiares',
        xaxis=dict(title='Pasajeros'),
        yaxis=dict(title='Cantidad'),
        )

    st.plotly_chart(fig6)
    "\r\n"
    "\r\n"
    "\r\n"
    "\r\n"


with tab4:
    # fig8 = px.box(df, y="Fare",x= 'Pclass', template="plotly_dark",color_discrete_sequence=[colorx[1]])
    # st.plotly_chart(fig8)

# ANALISIS por clase

    """Existe una notoria diferencia en la cantidad de sobrevivientes entre las diferentes clases. """
    """La clase 1 tiene un número considerablemente mayor de sobrevivientes en comparación con las clases 2 y 3. Esto puede ser atribuido a varios factores, como la ubicación 
    de las cabinas en el barco, el acceso a los botes salvavidas y posiblemente la priorización de los pasajeros de primera clase en la asignación de botes salvavidas y/o 
    medidas de salvación que se hayan considerado."""
    
    # Agrupar los datos por Survived - Pclass - Sex
    experience_job_ds = df.groupby(['Survived_Indicator', 'Pclass',]).size().reset_index(name='Count')
    fig9 = px.treemap(experience_job_ds, path=['Survived_Indicator', 'Pclass'], values='Count', 
                    color_discrete_sequence=colorx, template='plotly_dark')

    fig9.update_layout(
        title='Sobrevivientes por Clase',
        xaxis=dict(title='Edad'),
        yaxis=dict(title='Cantidad'),
        bargap=0.1,
        showlegend=True
    )
    st.plotly_chart(fig9)
    "\r\n"
    "\r\n"
    "\r\n"
    "\r\n"

# ANALISIS por lugar de Embarque

    """    Este mapa de calor representa a través de los colores la intensidad de la relación entre el lugar de embarque y los sobrevivientes.
    Se puede observar que existe una mayor tendencia de no sobrevivir si el lugar de embarque ha sido el S y una tendencia a sobrevivir si se ha embarcado en el puerto Q.

    Si juntamos este mapa de calor con el histograma de cómo están compuestas las clases tiene sentido ya que la mayoría de las personas se han embarcado en el puerto S.
    """
    fig11 = px.density_heatmap (df, x="Embarked", y="Survived_Indicator", marginal_x="histogram", marginal_y="rug",
                          template='plotly_dark',color_continuous_scale= colorx)

    fig11.update_layout(
        title='Mapa de calor por lugar de embarque',
        xaxis=dict(title='Lugar de Embarque'),
        yaxis=dict(title=' ')
        )
    st.plotly_chart(fig11)    

    "\r\n"
    "\r\n"

    st.write(
        " * 'S': Southampton, Inglaterra" "\r\n"
        " * 'C': Cherbourg, Francia" "\r\n"
        " * 'Q': Queenstown, Irlanda" "\r\n")

    fig10 = px.histogram(df, x='Embarked', color='Pclass', barmode='stack', nbins=25,
                   color_discrete_sequence=colorx, opacity=0.85,template='plotly_dark')

    fig10.update_layout(
        title='Distribución de pasajeros según donde se embarcaron y su categoría',
        xaxis=dict(title='Lugar de Embarque'),
        yaxis=dict(title='Cantidad'),
        )
    st.plotly_chart(fig10)


    with tab5:
        st.subheader("Conclusiones")
        st.write("Podemos ver que la proporción de pasajeros que sobrevivieron al desastre del Titanic fue inferior a la proporción de pasajeros que no sobrevivieron." 
             "Solo un 38% de los pasajeros sobrevivieron.""\r\n")
    
        st.write("En la Grafica de la distribución de edades de los pasajeros muestra que la mayoría de ellos se encontraban en el rango de 20 a 30 años. Sin embargo," 
             "cuando analizamos las correlaciones no encontramos una correlación entre estas dos variables (Age, Survived).")
        st.write("En cuanto al género parece estar asociado a la supervivencia, ya que pudimos ver en la gráfica de distribución por género, que la mayoría de mujeres" 
                 " ha sobrevivido en comparación con los hombres. Esto puede deberse a la prioridad dada en las mujeres y niños durante la asignación de botes salvavidas" 
                 "y durante el rescate.")
        st.write("Los pasajeros que viajaban solos tenían mayor probabilidad de sobrevivir en comparación con aquellos que viajaban con familiares a bordo. Esto puede "
                 "deberse al priorizar buscar y salvar a sus familiares en lugar de a si mismos, o en la toma de decisiones en momentos de emergencia.")
        st.write("Se puede ver en la gráfica de sobrevivientes que según el lugar de empaque los pasajeros que se embarcaron en Queenstown: “Q” tuvieron mayor probabilidad"
                 " de sobrevivir en comparación a los que se embarcaron en Cherboug: “C”  y Southampton: “S”. Siendo este último el que muestra menor probabilidad de sobrevivir.")
        st.write(" Y para finalizar podemos observar que los pasajeros que viajaban en la primera clase tienen una mayor proporción que el resto de las clases. Esto puede indicar "
                 "que se hayan priorizado al momento de asignar los botes o al lugar donde se encontraban las cabinas. Esto puede indicar que puede haber sido un factor determinante"
                 " en su probabilidad de sobrevivir.")



