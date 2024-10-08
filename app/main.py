import streamlit as st
import numpy as np 
import pandas as pd 
import pickle
import plotly.graph_objects as go


#Getting clean data
def get_clean_data():

    data = pd.read_csv(r'data/diabetes.csv')

    return data


#Adding of sidebar
def add_sidebar():

    st.sidebar.header('Factor Measurements')

    data = get_clean_data()

    input_dict = {}
    
    sidebar_label = [
        ('Pregnancies', 'Pregnancies' ),
        ('Glucose', 'Glucose'),
        ('Blood_Pressure', 'BloodPressure'),
        ('Skin_Thickness', 'SkinThickness'),
        ('Insulin', 'Insulin'),
        ('BMI', 'BMI'),
        ('Diabetes_Pedigree_Function', 'DiabetesPedigreeFunction'),
        ('Age', 'Age'),
        
    ]

    for label, key in sidebar_label:
        input_dict[key] = st.sidebar.slider(label,
            min_value = float(0),
            max_value = float(data[key].max()),
            value = float(data[key].mean())
            
        )

    return input_dict

#Scaling dictionary for plot visualisation

def get_scaled_input_data(input_dict):
  data = get_clean_data()
  X = data.drop(['Outcome'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

#Getting chart for plot
def get_chart(input_data):

    input_data = get_scaled_input_data(input_data)

    categories = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin thickness', 
                'Insulin', 'BMI', 
                'Diabetes Pedigree function', 'Age'
                ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
          0.51, 0.68, 0.70, 0.45, 0.70, 0.77, 0.46, 0.45
        ],

        theta=categories,
        fill='toself',
        fillcolor= 'blue',
        opacity= 0.14,
        name='Potential diabetic spectrum'

    ))

    
    fig.add_trace(go.Scatterpolar(
        r=[
          0.29, 0.58, 0.53, 0.47, 0.58, 0.67, 0.50, 0.50
        ],

        theta=categories,
        fill='toself',
        fillcolor= 'green',
        opacity= 0.20,
        name='Potential diabetic spectrum'

    ))


    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['Pregnancies'], input_data['Glucose'], input_data['BloodPressure'],
          input_data['SkinThickness'], input_data['Insulin'], input_data['BMI'],
          input_data['DiabetesPedigreeFunction'], input_data['Age']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.update_layout(
        polar=dict(
         radialaxis=dict(
            visible=True,
             range=[0, 1]
      )),
    showlegend=True
  )
    
    return fig

def bar_chart(input_data):
    input_data = get_scaled_input_data(input_data)
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig = plt.figure(figsize=(10.8,4))
    a = sns.barplot(data=input_data, palette=sns.color_palette('rocket'))
    for i in range(8):
        a.bar_label(a.containers[i])
    plt.ylabel('Measurements', color ="#5B0888" )
    plt.title('Readings on a scale of 0 to 1', weight ='bold', fontdict={'fontsize': 20}, color ="#5B0888")
    st.pyplot(fig)



def plotly(input_data):
    input_data = get_scaled_input_data(input_data)
    inputlist_array = np.array(list(input_data.values())).reshape(1, -1)

    fig = go.Figure()

    fig.add_trace(go.Bar(
    x=inputlist_array,
    y=inputlist_array

    
    ))

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=inputlist_array,
        #y=input_data.values(),
        name='Diabetes readings scaled from 0 to 1'
        ))

    return fig



#Adding the model for prediction
def add_prediction(input_data):
    model = pickle.load(open(r'model/model.pkl',  'rb'))
    scaler = pickle.load(open(r'model/scaler.pkl','rb'))

    inputlist_array = np.array(list(input_data.values())).reshape(1, -1)
    scaled = scaler.transform(inputlist_array)

    prediction = model.predict(scaled)

    st.subheader('Diabetes Prediction')
    st.title(' ')
    st.write('Predicted as:')

    if prediction[0]==0:
        st.write("<span class='diagnose positive diabetes'>Diabetic</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnose negative diabetes'>Non Diabetic</span>", unsafe_allow_html=True)
    
    st.title('')
    st.write("While this software can help doctors diagnose patients, it shouldn't be used as a substitute for professional diagnosis.")



#User profile
username = 'Lamda'
contact1 = '''
LinkedIn: <i>Link</i>  <br> 
Email: tetteycollins@gmail.com'''

url = "www.linkedin.com/in/tettey-collins-kwabena-10351332a"
contact = "LinkedIn: [Link](https://www.linkedin.com/in/tettey-collins-kwabena-10351332a)"

imagepath = r'assets/images/croped.PNG'

def user_card(username, contact, imagepath):
    img_col, contact_col = st.columns([0.7,4])
    
    with img_col:
        st.image(imagepath, caption='Profile Picture')
    with contact_col:
        st.write('Username: ', username)
        st.write(contact, unsafe_allow_html=True)
    


def main():
    st.set_page_config(
        page_title='Diabetes Predictor',
        page_icon=r'assets/icons/icons8_glucometer.ico',
        layout='wide',
        initial_sidebar_state='expanded'
        
    )
    
    with open(r'assets/style/style.css') as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    user_card(username, contact, imagepath)
    
    input_data = add_sidebar()

    col1, col2 = st.columns([4,1])

    with col1:
        st.title('Diabetes Predictor')
        st.write("Based on data it receives from your lab, this app uses a machine learning model to determine if you have diabetes or not. Sliders in the sidebar allow you to manually update the measurements as well.", unsafe_allow_html=True)

        chart = get_chart(input_data)
        st.plotly_chart(chart)
    
    with col2:
        add_prediction(input_data)

    barcol, emptycol = st.columns([3,1])

    with barcol:
        #bar_chart(input_data)
        #plotly(input_data)
        st.bar_chart(get_scaled_input_data(input_data), color="#ecbebe")
        


    
if __name__=='__main__':
    main()