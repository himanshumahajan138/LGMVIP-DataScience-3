#>> Import Important Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,precision_score,f1_score,recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

#>> read the data file and covert it to dataframe and then use it as dataframe
df = pd.read_csv("Iris.csv")
df.drop(columns="Id",inplace=True)

#>> checking the shape of the dataframe
print(df.shape)

#>> checking information about variables or columns
print(df.info())

#>> checking descriptive analysis of numerical data 
print(df.describe())

#>> checking descriptive analysis of categorical data
print(df["Species"].describe())

#>> checking null values count and visualize them 
print(df.isnull().sum())
plt.subplots()
plt.subplot(1,2,1)
df.isnull().sum().plot(kind="bar")
plt.subplot(1,2,2)
sns.heatmap(data=df.isnull(),cmap="viridis")
plt.show()

#>> conclusion that no null value in dataset 

#>> handling target variable i.e Species

#>> checking count of unique values
print(df["Species"].value_counts())

#>> as values are uniformly distributed so no further checking needed 

#>> checking outliers
plt.subplots(figsize=(10,8))
for i,a in enumerate(df.drop(columns="Species").columns):
    plt.subplot(2,2,i+1)
    sns.boxplot(data=df[a],color="lightgreen")
    plt.title(a)
plt.show()

#>> treating outliers 
def outlier_treatment(data,how):
    low = data.quantile(.25)-1.5*(data.quantile(.75)-data.quantile(.25))
    up = data.quantile(.75)+1.5*(data.quantile(.75)-data.quantile(.25))
    if how == "mode":
        data[data>up],data[data<low]=data.mode(),data.mode()
    elif how == "mean":
        data[data>up],data[data<low]=data.mean(),data.mean()
    elif how == "lim":
        data[data>up],data[data<low]=up,low
        
    elif how == "median":
        data[data>up],data[data<low]=data.median(),data.median()
    else:
        None
    return data

#>> after checking box plot outlier was detected in sepal width in cm columns so treatment for that 
df["SepalWidthCm"] = outlier_treatment(data=df["SepalWidthCm"],how="lim")

#>> after treatment visuaize outlier
sns.boxplot(df["SepalWidthCm"],color="lightgreen")
plt.title("sepal width in cm after treatment")
plt.show()

#>> checking relation between target and independent variables 
x = df.drop(columns="Species")
y = df["Species"]
sns.pairplot(data=df,hue="Species")
plt.show()


#>> now splitiing data to test and train 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#>> feature scalling

#>> apply standard scaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#>> creating Decision Tree calsscification model for prediction
dc = DecisionTreeClassifier(criterion="gini",splitter="best",random_state=0)
dc.fit(x_train,y_train)

#>> predicting the test set
y_dc_pred = dc.predict(x_test)

#>> checking metrics

#>> metrics score
print("Accuracy Score : ",accuracy_score(y_test,y_dc_pred))
print("Precision Score : ",precision_score(y_test,y_dc_pred,average="weighted"))
print("F1 Score : ",f1_score(y_test,y_dc_pred,average="weighted"))
print("Recall Score : ",recall_score(y_test,y_dc_pred,average="weighted"))

#>> confusion matrix
print("Confusion Matrix \n",confusion_matrix(y_test,y_dc_pred))

#>> display confusion matrix 
conf_dis = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_dc_pred))
conf_dis.plot(cmap="viridis")
plt.show()

#>> Visualize Decision Tree
plt.figure(figsize=(15,12))
tree.plot_tree(dc,feature_names=df.columns,class_names=y.unique(),filled=True)
plt.title("Decision Tree Visualization")
plt.show()