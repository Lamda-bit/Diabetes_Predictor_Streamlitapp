import numpy as np 
from sklearn.preprocessing._data import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import pickle 
import seaborn as sns


def get_clean_data():
    data = pd.read_csv('Data\diabetes.csv')
    return data

def create_model(data):
    X = data.drop(columns='Outcome', axis=1)
    Y = data['Outcome'] 

    scaler = StandardScaler()
    Xscale = scaler.fit_transform(X)
    #print(Xscale)

    x_train, x_test, y_train, y_test = train_test_split(Xscale, Y, test_size=0.2, stratify= Y, random_state= 2)
    
    #print(x_train.shape , y_train.shape)

    model = svm.SVC(kernel='linear')
    
    model.fit(x_train, y_train)
    x_prediction = model.predict(x_train)
    #print(x_prediction)
    accuracy = accuracy_score(x_prediction, y_train)
    class_report = classification_report(x_prediction, y_train)
    #print(accuracy)
    #print(class_report)

    test_data = (0,105,64,41,142,41.5,0.173,22)
    test_data = np.asarray(test_data).reshape(1,-1)
    test = scaler.transform(test_data)
    predict_test = model.predict(test)
    #print('---------------------------------')
    #print(predict_test)

    return model, scaler, X

    

def main():
    data = get_clean_data()

    model, scaler, X = create_model(data)


    with open(r'Model/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    with open(r'Model/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    
    with open(r'Model/cleanData.pkl', 'wb') as file:
        pickle.dump(data, file)

if __name__=='__main__':
    main()