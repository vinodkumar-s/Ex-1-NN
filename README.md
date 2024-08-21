<H3>ENTER YOUR NAME : VINOD KUMAR S</H3>
<H3>ENTER YOUR REGISTER NO. : 212222240116</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 21/08/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

### IMPORT LIBRARIES : 

```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```

### READ THE DATA: 
```py
df=pd.read_csv("Churn_Modelling.csv")
```

### CHECK DATA: 
```py
df.head()
df.tail()
df.columns
```

### CHECK THE MISSING DATA:
```py
df.isnull().sum()
```

### ASSIGNING X:
```py
X = df.iloc[:,:-1].values
X
```

### ASSIGNING Y:
```py
Y = df.iloc[:,-1].values
Y
```

### CHECK FOR OUTLIERS:
```py
df.describe()
```

### DROPPING STRING VALUES DATA FROM DATASET:
```py
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```

### CHECKING DATASETS AFTER DROPPING STRING VALUES DATA FROM DATASET:
```py
data.head()
```

### NORMALIE THE DATASET USING (MinMax Scaler):
```py
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```

### SPLIT THE DATASET:
```py
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print(X)
print(Y)
```

### TRAINING AND TESTING MODEL:
```py
X_train ,X_test ,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:

### DATA CHECKING:

![DC](https://github.com/SivaChandranR07/Ex-1-NN/assets/113497395/4a9861dd-c08c-4c08-9218-df9d4e89c495)


### MISSING DATA:

![MD](https://github.com/SivaChandranR07/Ex-1-NN/assets/113497395/6260c60c-1bfc-4806-b778-b63481df25b1)

### DUPLICATES IDENTIFICATION:
![DI](https://github.com/SivaChandranR07/Ex-1-NN/assets/113497395/1c6dadd3-fc7d-414e-8979-74363928147e)

### VALUE OF Y:
![VY](https://github.com/SivaChandranR07/Ex-1-NN/assets/113497395/89fe00c2-51f7-4fb9-944c-298329b291df)

### OUTLIERS:
![outliers](https://github.com/SivaChandranR07/Ex-1-NN/assets/113497395/bb71d1ce-305d-4389-8c55-4d372cf89c07)

### CHECKING DATASET AFTER DROPPING STRING VALUES DATA FROM DATASET:
![CD](https://github.com/SivaChandranR07/Ex-1-NN/assets/113497395/6978a892-4d84-45f7-a61f-d4da24af6537)

### NORMALIZE THE DATASET:
![ND](https://github.com/SivaChandranR07/Ex-1-NN/assets/113497395/dd09c337-092e-44c9-9c06-3a49a265ce3b)

### SPLIT THE DATASET:
![SD](https://github.com/SivaChandranR07/Ex-1-NN/assets/113497395/ef5e94d1-14f4-4841-9e05-94564d0a3c22)

### TRAINING AND TESTING MODEL:
![TAT](https://github.com/SivaChandranR07/Ex-1-NN/assets/113497395/2582cd4e-aaed-4610-97ed-91c3b8676e9f)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


