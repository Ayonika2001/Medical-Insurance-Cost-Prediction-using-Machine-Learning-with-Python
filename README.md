### Importing the Dependencies


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
```

### Data Collection & Analysis


```python
# loading the data from csv file to a Pandas DataFrame
insurance_dataset=pd.read_csv("insurance.csv")
```


```python
# first 5 rows of the dataframe
insurance_dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>female</td>
      <td>27.900</td>
      <td>0</td>
      <td>yes</td>
      <td>southwest</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>male</td>
      <td>33.770</td>
      <td>1</td>
      <td>no</td>
      <td>southeast</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>male</td>
      <td>33.000</td>
      <td>3</td>
      <td>no</td>
      <td>southeast</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>male</td>
      <td>22.705</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>male</td>
      <td>28.880</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>3866.85520</td>
    </tr>
  </tbody>
</table>
</div>




```python
# number of rows and columns
insurance_dataset.shape
```




    (1338, 7)




```python
# getting some information about the dataset
insurance_dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1338 entries, 0 to 1337
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       1338 non-null   int64  
     1   sex       1338 non-null   object 
     2   bmi       1338 non-null   float64
     3   children  1338 non-null   int64  
     4   smoker    1338 non-null   object 
     5   region    1338 non-null   object 
     6   charges   1338 non-null   float64
    dtypes: float64(2), int64(2), object(3)
    memory usage: 73.3+ KB
    

## Categorical Features:
- Sex
- Smoker
- Region


```python
# checking for missing values
insurance_dataset.isnull().sum()
```




    age         0
    sex         0
    bmi         0
    children    0
    smoker      0
    region      0
    charges     0
    dtype: int64




```python
# checking for missing values
insurance_dataset.isnull().sum()
```




    age         0
    sex         0
    bmi         0
    children    0
    smoker      0
    region      0
    charges     0
    dtype: int64



### Data Analysis


```python
# statistical Measures of the dataset
insurance_dataset.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.207025</td>
      <td>30.663397</td>
      <td>1.094918</td>
      <td>13270.422265</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.049960</td>
      <td>6.098187</td>
      <td>1.205493</td>
      <td>12110.011237</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>15.960000</td>
      <td>0.000000</td>
      <td>1121.873900</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>26.296250</td>
      <td>0.000000</td>
      <td>4740.287150</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>30.400000</td>
      <td>1.000000</td>
      <td>9382.033000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>51.000000</td>
      <td>34.693750</td>
      <td>2.000000</td>
      <td>16639.912515</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.000000</td>
      <td>53.130000</td>
      <td>5.000000</td>
      <td>63770.428010</td>
    </tr>
  </tbody>
</table>
</div>




```python
# distribution of age value
sns.set()
plt.figure(figsize=(5,5))
sns.histplot(insurance_dataset['age'],kde=True)
plt.title("Age Distribution")
plt.show()
```


    
![png](output_12_0.png)
    



```python
# Gender Column
plt.figure(figsize=(5,5))
sns.countplot(x='sex',data=insurance_dataset)
plt.title("Sex Distribution")
plt.show()
```


    
![png](output_13_0.png)
    



```python
insurance_dataset['sex'].value_counts()
```




    sex
    male      676
    female    662
    Name: count, dtype: int64




```python
# BMI Column
plt.figure(figsize=(5,5))
sns.histplot(insurance_dataset['bmi'],kde=True)
plt.title("BMI Distribution")
plt.show()
```


    
![png](output_15_0.png)
    


***Normal BMI Range--> 18.5-24.9***


```python
# children column
plt.figure(figsize=(5,5))
sns.countplot(x='children',data=insurance_dataset)
plt.title("Children")
plt.show()
```


    
![png](output_17_0.png)
    



```python
insurance_dataset['children'].value_counts()
```




    children
    0    574
    1    324
    2    240
    3    157
    4     25
    5     18
    Name: count, dtype: int64




```python
# smoker column
plt.figure(figsize=(5,5))
sns.countplot(x='smoker',data=insurance_dataset)
plt.title("Smoker")
plt.show()
```


    
![png](output_19_0.png)
    



```python
insurance_dataset['smoker'].value_counts()
```




    smoker
    no     1064
    yes     274
    Name: count, dtype: int64




```python
# region column
plt.figure(figsize=(5,5))
sns.countplot(x='region',data=insurance_dataset)
plt.title("Region")
plt.show()
```


    
![png](output_21_0.png)
    



```python
insurance_dataset['region'].value_counts()
```




    region
    southeast    364
    southwest    325
    northwest    325
    northeast    324
    Name: count, dtype: int64




```python
# distribution of charges value
plt.figure(figsize=(5,5))
sns.histplot(insurance_dataset['charges'],kde=True)
plt.title("charges Distribution")
plt.show()
```


    
![png](output_23_0.png)
    


### Data Pre-Processing


```python
pd.set_option('future.no_silent_downcasting', True)
# encoding smoker column
insurance_dataset.replace({'sex':{'male':0,'female':1}},inplace=True)

# encoding smoker column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}},inplace=True)

# encoding region column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)
```

### Splitting the Features and Target


```python
X=insurance_dataset.drop(columns='charges',axis=1)
Y=insurance_dataset['charges']
```


```python
print(X)
```

          age sex     bmi  children smoker region
    0      19   1  27.900         0      0      1
    1      18   0  33.770         1      1      0
    2      28   0  33.000         3      1      0
    3      33   0  22.705         0      1      3
    4      32   0  28.880         0      1      3
    ...   ...  ..     ...       ...    ...    ...
    1333   50   0  30.970         3      1      3
    1334   18   1  31.920         0      1      2
    1335   18   1  36.850         0      1      0
    1336   21   1  25.800         0      1      1
    1337   61   1  29.070         0      0      3
    
    [1338 rows x 6 columns]
    


```python
print(Y)
```

    0       16884.92400
    1        1725.55230
    2        4449.46200
    3       21984.47061
    4        3866.85520
               ...     
    1333    10600.54830
    1334     2205.98080
    1335     1629.83350
    1336     2007.94500
    1337    29141.36030
    Name: charges, Length: 1338, dtype: float64
    

### Splitting the data into Training data & Testing data


```python
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
```


```python
print(X.shape,X_train.shape,X_test.shape)
```

    (1338, 6) (1070, 6) (268, 6)
    

### Model Training

### Linear Regression


```python
# loading the linear Regression model
regressor=LinearRegression()
```


```python
regressor.fit(X_train,Y_train)
```





### Model Evaluation


```python
# prediction on training data
training_data_prediction=regressor.predict(X_train)
```


```python
# R squared value
r2_train=metrics.r2_score(Y_train,training_data_prediction)
print("R squared value: ",r2_train)
```

    R squared value:  0.751505643411174
    


```python
# prediction on test data
test_data_prediction=regressor.predict(X_test)
```


```python
# R squared value
r2_test=metrics.r2_score(Y_test,test_data_prediction)
print("R squared value: ",r2_test)
```

    R squared value:  0.7447273869684077
    

### Building a predictive system


```python
input_data=(31,1,25.74,0,1,0)

# Changing input_data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

# reshape the array
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=regressor.predict(input_data_reshaped)
print(prediction)

print("The insurance cost is USD ",prediction[0])
```

    [3760.0805765]
    The insurance cost is USD  3760.0805764960587
    

