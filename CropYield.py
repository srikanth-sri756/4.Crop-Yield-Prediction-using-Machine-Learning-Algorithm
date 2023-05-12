from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor



main = tkinter.Tk()
main.title("Crop Yield Prediction using Machine Learning Algorithm")
main.geometry("1300x1200")


global filename
global X_train, X_test, y_train, y_test
global X,Y
global dataset
global le
global model

def upload():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Tweets dataset loaded\n')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    dataset['Production'] = dataset['Production'].astype(np.int64)
    text.insert(END,str(dataset.head())+"\n")

def processDataset():
    global le
    global dataset
    global X_train, X_test, y_train, y_test
    global X,Y
    text.delete('1.0', END)
    le = LabelEncoder()

    dataset['State_Name'] = pd.Series(le.fit_transform(dataset['State_Name']))
    dataset['District_Name'] = pd.Series(le.fit_transform(dataset['District_Name']))
    dataset['Season'] = pd.Series(le.fit_transform(dataset['Season']))
    dataset['Crop'] = pd.Series(le.fit_transform(dataset['Crop']))
    text.insert(END,str(dataset.head())+"\n")
    datasets = dataset.values
    cols = datasets.shape[1]-1
    X = datasets[:,0:cols]
    Y = datasets[:,cols]
    Y = Y.astype('uint8')
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END,"\n\nTotal records found in dataset is : "+str(len(X))+"\n")
    text.insert(END,"80% records used to train machine learning algorithm : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% records used to train machine learning algorithm : "+str(X_test.shape[0])+"\n")


def trainModel():
    global model
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global X,Y

    model = DecisionTreeRegressor(max_depth=100,random_state=0,max_leaf_nodes=20,max_features=5,splitter="random")
    model.fit(X,Y)
    predict = model.predict(X_test)
    mse = mean_squared_error(predict,y_test)
    rmse = np.sqrt(mse)/ 1000
    text.insert(END,"Training process completed\n")
    text.insert(END,"Decision Tree Machine Learning Algorithm Training RMSE Error Rate : "+str(rmse)+"\n\n")


def cropYieldPredict():
    global model
    global le
    text.delete('1.0', END)
    testname = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(testname)
    test.fillna(0, inplace = True)
    test['State_Name'] = pd.Series(le.fit_transform(test['State_Name']))
    test['District_Name'] = pd.Series(le.fit_transform(test['District_Name']))
    test['Season'] = pd.Series(le.fit_transform(test['Season']))
    test['Crop'] = pd.Series(le.fit_transform(test['Crop']))
    test = test.values
    test = normalize(test)
    cols = test.shape[1]
    test = test[:,0:cols]
    predict = model.predict(test)
    for i in range(len(predict)):
        production = predict[i] * 100
        crop_yield = (production / 10000) / 10
        text.insert(END,"Test Record : "+str(test[i])+" Production would be : "+str(production)+" KGs\n")
        text.insert(END,"Yield would be : "+str(crop_yield)+" KGs/acre\n\n")
    

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Crop Yield Prediction using Machine Learning Algorithm')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Crop Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
#pathlabel.place(x=700,y=100)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=700,y=150)
processButton.config(font=font1) 

mlButton = Button(main, text="Train Machine Learning Algorithm", command=trainModel)
mlButton.place(x=700,y=200)
mlButton.config(font=font1) 

predictButton = Button(main, text="Upload Test Data & Predict Yield", command=cropYieldPredict)
predictButton.place(x=700,y=250)
predictButton.config(font=font1)

closeButton = Button(main, text="Close", command=close)
closeButton.place(x=700,y=300)
closeButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
