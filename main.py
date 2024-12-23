#importing the libraries

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

#setting up Basic Streamlit Page
st.set_page_config(page_title="Simple Iris Flower Prediction APP",page_icon="🌿")
st.title(" 🌿 Iris Flower Prediction APP 🌿")
st.subheader(" **This App Predicts The Iris Flower Type !** ")


# --Sidebar--
st.sidebar.header("USER INPUT PARAMETERS 🔃",divider=True)

# --USer input conversion into Pandas Dataframe--
def user_input_feature():
    sepal_length = st.sidebar.slider('Sepal Length',4.3,7.9,5.4)
    sepal_width = st.sidebar.slider("Sepal Width",2.0,4.4,3.4)
    petal_length = st.sidebar.slider("Petal Length",1.0,6.9,1.3)
    petal_width = st.sidebar.slider("Petal Width",0.1,2.5,0.2)
    data = {
        'sepal_length':sepal_length,
        'sepal_width':sepal_width,
        'petal_length':petal_length,
        'petal_width':petal_width
    }
    feature = pd.DataFrame(data,index=[0])
    return feature

    
df = user_input_feature()
st.subheader(" ~ User Input Parameters ~")
st.write(df)
st.sidebar.text("Made By Sumit Kumar ✒️")
# st.divider()




# --Machine Learning Code From Scikit-learn library--
iris = datasets.load_iris()
x = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(x,y)

prediction = clf.predict(df)
prediction_probability = clf.predict_proba(df)


col1,col2,col3 = st.columns(3,gap="medium",border=True)
with col1:
    st.subheader("Class Label")
    st.write(iris.target_names)
with col2:
    st.subheader(" **Predicition Result** ")
    st.write(iris.target_names[prediction])
with col3:
    st.subheader("Prediction Probability")
    st.write(prediction_probability)
    
st.divider()
st.subheader("Check out my Contact link's 😇")
st.link_button(label="portfolio",url="https://sumit-portfolio-ds.netlify.app/",type="primary")
st.link_button(label="Linkedin",url="https://www.linkedin.com/in/beasumit/",type="primary")


hide_st_style = '''
<style>
#mainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}
</style>
'''

st.markdown(hide_st_style,unsafe_allow_html=True)