# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Mohmaed Aathil M
RegisterNumber: 25008235
*/
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)

```

## Output:
Data:

<img width="1001" height="458" alt="Screenshot 2025-12-11 090218" src="https://github.com/user-attachments/assets/a4108ee9-ed71-4bc4-a64b-ec4a2bbb1c4a" />


DataShape:

<img width="235" height="55" alt="Screenshot 2025-12-11 090428" src="https://github.com/user-attachments/assets/14ce7dd0-078d-4537-87f2-1de5336b67ef" />

X.shape:


<img width="215" height="47" alt="Screenshot 2025-12-11 090528" src="https://github.com/user-attachments/assets/59c8e73b-0791-4e0b-8bb4-eb5dab45a5c5" />


Y.shape:

<img width="219" height="27" alt="Screenshot 2025-12-11 090654" src="https://github.com/user-attachments/assets/3ec0fe63-2c6d-44db-abe6-32a23261b63e" />

X_train:


<img width="870" height="326" alt="Screenshot 2025-12-11 090818" src="https://github.com/user-attachments/assets/824fab4a-8739-48a1-b808-a3dd8d5dee41" />

X_train.shape:


<img width="198" height="31" alt="Screenshot 2025-12-11 090904" src="https://github.com/user-attachments/assets/b5a2c410-6638-4898-9970-23cea62b6768" />

Y_pred:


<img width="820" height="51" alt="Screenshot 2025-12-11 090949" src="https://github.com/user-attachments/assets/9175f08c-29e3-4da5-ad12-f9f99ef380c6" />

Accuracy:


<img width="287" height="59" alt="Screenshot 2025-12-11 091034" src="https://github.com/user-attachments/assets/f381962c-735b-4504-817b-6fc8bd3a0433" />

Confusion matrix:

<img width="206" height="66" alt="Screenshot 2025-12-11 091133" src="https://github.com/user-attachments/assets/bbe0a014-c228-4651-becc-58c5c86deb21" />

Classification Report:


<img width="638" height="204" alt="Screenshot 2025-12-11 091245" src="https://github.com/user-attachments/assets/58b0518e-d6d5-4897-b0a5-e3a17c65422c" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
