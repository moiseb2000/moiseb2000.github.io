```python
import markdown
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#importing the data

df_train = pd.read_csv('C:/Users/moise/Downloads/House Prices Advanced Techniques/train.csv')
train = pd.DataFrame(df_train)

df_test = pd.read_csv('C:/Users/moise/Downloads/House Prices Advanced Techniques/test.csv') 
test = pd.DataFrame(df_test)

print(train.shape)
print(df_test.shape)
```

    (1460, 81)
    (1459, 80)
    

The House Prices Advanced Regression Techniques Dataset is available on Kaggle for dowloanding. The dataset is broken into two files, 1. training data, and 2. testing data. The training data has 1460 observations and 80 features. The testing data has 1459 observations with 79 features. The testing data does not contain the target column. The feature types are a combination of int64, float64, and object types. The target variable is a continious variable.


```python
#Data exploration 

train.head()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
test.head()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>120</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>60</td>
      <td>RL</td>
      <td>78.0</td>
      <td>9978</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>120</td>
      <td>RL</td>
      <td>43.0</td>
      <td>5005</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>144</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
# training Data Exploration 
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     1452 non-null   object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    


```python
#Testing Data Exploration 
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 80 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1459 non-null   int64  
     1   MSSubClass     1459 non-null   int64  
     2   MSZoning       1455 non-null   object 
     3   LotFrontage    1232 non-null   float64
     4   LotArea        1459 non-null   int64  
     5   Street         1459 non-null   object 
     6   Alley          107 non-null    object 
     7   LotShape       1459 non-null   object 
     8   LandContour    1459 non-null   object 
     9   Utilities      1457 non-null   object 
     10  LotConfig      1459 non-null   object 
     11  LandSlope      1459 non-null   object 
     12  Neighborhood   1459 non-null   object 
     13  Condition1     1459 non-null   object 
     14  Condition2     1459 non-null   object 
     15  BldgType       1459 non-null   object 
     16  HouseStyle     1459 non-null   object 
     17  OverallQual    1459 non-null   int64  
     18  OverallCond    1459 non-null   int64  
     19  YearBuilt      1459 non-null   int64  
     20  YearRemodAdd   1459 non-null   int64  
     21  RoofStyle      1459 non-null   object 
     22  RoofMatl       1459 non-null   object 
     23  Exterior1st    1458 non-null   object 
     24  Exterior2nd    1458 non-null   object 
     25  MasVnrType     1443 non-null   object 
     26  MasVnrArea     1444 non-null   float64
     27  ExterQual      1459 non-null   object 
     28  ExterCond      1459 non-null   object 
     29  Foundation     1459 non-null   object 
     30  BsmtQual       1415 non-null   object 
     31  BsmtCond       1414 non-null   object 
     32  BsmtExposure   1415 non-null   object 
     33  BsmtFinType1   1417 non-null   object 
     34  BsmtFinSF1     1458 non-null   float64
     35  BsmtFinType2   1417 non-null   object 
     36  BsmtFinSF2     1458 non-null   float64
     37  BsmtUnfSF      1458 non-null   float64
     38  TotalBsmtSF    1458 non-null   float64
     39  Heating        1459 non-null   object 
     40  HeatingQC      1459 non-null   object 
     41  CentralAir     1459 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1459 non-null   int64  
     44  2ndFlrSF       1459 non-null   int64  
     45  LowQualFinSF   1459 non-null   int64  
     46  GrLivArea      1459 non-null   int64  
     47  BsmtFullBath   1457 non-null   float64
     48  BsmtHalfBath   1457 non-null   float64
     49  FullBath       1459 non-null   int64  
     50  HalfBath       1459 non-null   int64  
     51  BedroomAbvGr   1459 non-null   int64  
     52  KitchenAbvGr   1459 non-null   int64  
     53  KitchenQual    1458 non-null   object 
     54  TotRmsAbvGrd   1459 non-null   int64  
     55  Functional     1457 non-null   object 
     56  Fireplaces     1459 non-null   int64  
     57  FireplaceQu    729 non-null    object 
     58  GarageType     1383 non-null   object 
     59  GarageYrBlt    1381 non-null   float64
     60  GarageFinish   1381 non-null   object 
     61  GarageCars     1458 non-null   float64
     62  GarageArea     1458 non-null   float64
     63  GarageQual     1381 non-null   object 
     64  GarageCond     1381 non-null   object 
     65  PavedDrive     1459 non-null   object 
     66  WoodDeckSF     1459 non-null   int64  
     67  OpenPorchSF    1459 non-null   int64  
     68  EnclosedPorch  1459 non-null   int64  
     69  3SsnPorch      1459 non-null   int64  
     70  ScreenPorch    1459 non-null   int64  
     71  PoolArea       1459 non-null   int64  
     72  PoolQC         3 non-null      object 
     73  Fence          290 non-null    object 
     74  MiscFeature    51 non-null     object 
     75  MiscVal        1459 non-null   int64  
     76  MoSold         1459 non-null   int64  
     77  YrSold         1459 non-null   int64  
     78  SaleType       1458 non-null   object 
     79  SaleCondition  1459 non-null   object 
    dtypes: float64(11), int64(26), object(43)
    memory usage: 912.0+ KB
    


```python
#Exploring the Target
train['SalePrice'].describe()
```




    count      1460.000000
    mean     180921.195890
    std       79442.502883
    min       34900.000000
    25%      129975.000000
    50%      163000.000000
    75%      214000.000000
    max      755000.000000
    Name: SalePrice, dtype: float64



The mean Sale Price is \\$180921. The min is about \\$35000. The max is \\$755000. 


```python
#Feature Selection using a correlation matrix
tcorr = train.corr()
tcorr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(tcorr.SalePrice > 0.5)
```

    SalePrice         True
    OverallQual       True
    GrLivArea         True
    GarageCars        True
    GarageArea        True
    TotalBsmtSF       True
    1stFlrSF          True
    FullBath          True
    TotRmsAbvGrd      True
    YearBuilt         True
    YearRemodAdd      True
    GarageYrBlt      False
    MasVnrArea       False
    Fireplaces       False
    BsmtFinSF1       False
    LotFrontage      False
    WoodDeckSF       False
    2ndFlrSF         False
    OpenPorchSF      False
    HalfBath         False
    LotArea          False
    BsmtFullBath     False
    BsmtUnfSF        False
    BedroomAbvGr     False
    ScreenPorch      False
    PoolArea         False
    MoSold           False
    3SsnPorch        False
    BsmtFinSF2       False
    BsmtHalfBath     False
    MiscVal          False
    Id               False
    LowQualFinSF     False
    YrSold           False
    OverallCond      False
    MSSubClass       False
    EnclosedPorch    False
    KitchenAbvGr     False
    Name: SalePrice, dtype: bool
    

    C:\Users\moise\AppData\Local\Temp\ipykernel_29680\4177789230.py:2: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      tcorr = train.corr()
    

According to the correlation Matrix the following features are mostly correlated to Sale Price. OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF. Our problem statement is create a regression model to model the features above 50% correlation threshold against Sale Price.  


```python
#relationship between features
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
cols = ['OverallQual', 'GrLivArea', 'GarageCars','GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
'YearRemodAdd']
scatterplotmatrix(train[cols].values, figsize=(40, 30), names=cols, alpha=0.5) 
plt.tight_layout()
plt.show()
```


    
![png](Brutus_Moise_MLHousePricesPrediction_files/Brutus_Moise_MLHousePricesPrediction_10_0.png)
    



```python
#The target does not have any NA values
train['SalePrice'].isnull().sum()
train.drop_duplicates(inplace=True)


```

Got rid of all the rows with missing values and duplicate values.


```python
#Selecting Data after feature selection
X = train[['OverallQual', 'GrLivArea', 'GarageCars','GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
'YearRemodAdd']]
y = train['SalePrice']
```


```python
#Checking the features for missing values
X.isnull().sum()
```




    OverallQual     0
    GrLivArea       0
    GarageCars      0
    GarageArea      0
    TotalBsmtSF     0
    1stFlrSF        0
    FullBath        0
    TotRmsAbvGrd    0
    YearBuilt       0
    YearRemodAdd    0
    dtype: int64




```python
#remove outliers

#replacing values > 0.75 quatertile with the mean
X['OverallQual'] = np.where(X['OverallQual'] > 6.099315, 5.747084, X['OverallQual'])
X['GrLivArea'] = np.where(X['GrLivArea'] > 1515.463699, 1342.785788, X['GrLivArea'])
X['GarageCars'] = np.where(X['GarageCars'] > 2, 1.767123, X['GarageCars'])
X['GarageArea'] = np.where(X['GarageArea'] > 576, 472.980137, X['GarageArea'])
X['TotalBsmtSF'] = np.where(X['TotalBsmtSF'] > 1298.250000, 1057.429452, X['TotalBsmtSF'])
X['1stFlrSF'] = np.where(X['1stFlrSF'] > 1391.250000, 1162.626712, X['1stFlrSF'])
X['FullBath'] = np.where(X['FullBath'] > 2, 1.565068, X['FullBath'])
X['TotRmsAbvGrd'] = np.where(X['TotRmsAbvGrd'] > 7, 6.517808, X['TotRmsAbvGrd'])
X['YearBuilt'] = np.where(X['YearBuilt'] > 2000, 1971.267808, X['YearBuilt']) 
X['YearRemodAdd'] = np.where(X['YearRemodAdd'] > 2004, 1984.865753, X['YearRemodAdd'])
```

    C:\Users\moise\AppData\Local\Temp\ipykernel_29680\3310879775.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X['OverallQual'] = np.where(X['OverallQual'] > 6.099315, 5.747084, X['OverallQual'])
    C:\Users\moise\AppData\Local\Temp\ipykernel_29680\3310879775.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X['GrLivArea'] = np.where(X['GrLivArea'] > 1515.463699, 1342.785788, X['GrLivArea'])
    C:\Users\moise\AppData\Local\Temp\ipykernel_29680\3310879775.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X['GarageCars'] = np.where(X['GarageCars'] > 2, 1.767123, X['GarageCars'])
    C:\Users\moise\AppData\Local\Temp\ipykernel_29680\3310879775.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X['GarageArea'] = np.where(X['GarageArea'] > 576, 472.980137, X['GarageArea'])
    C:\Users\moise\AppData\Local\Temp\ipykernel_29680\3310879775.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X['TotalBsmtSF'] = np.where(X['TotalBsmtSF'] > 1298.250000, 1057.429452, X['TotalBsmtSF'])
    C:\Users\moise\AppData\Local\Temp\ipykernel_29680\3310879775.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X['1stFlrSF'] = np.where(X['1stFlrSF'] > 1391.250000, 1162.626712, X['1stFlrSF'])
    C:\Users\moise\AppData\Local\Temp\ipykernel_29680\3310879775.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X['FullBath'] = np.where(X['FullBath'] > 2, 1.565068, X['FullBath'])
    C:\Users\moise\AppData\Local\Temp\ipykernel_29680\3310879775.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X['TotRmsAbvGrd'] = np.where(X['TotRmsAbvGrd'] > 7, 6.517808, X['TotRmsAbvGrd'])
    C:\Users\moise\AppData\Local\Temp\ipykernel_29680\3310879775.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X['YearBuilt'] = np.where(X['YearBuilt'] > 2000, 1971.267808, X['YearBuilt'])
    C:\Users\moise\AppData\Local\Temp\ipykernel_29680\3310879775.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X['YearRemodAdd'] = np.where(X['YearRemodAdd'] > 2004, 1984.865753, X['YearRemodAdd'])
    

This step essentially removes all the values above the 3rd quartile and replaces those values with the mean. My model before this step was not very accurate so I got rid of the outliers. 


```python
X.describe()

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
      <th>OverallQual</th>
      <th>GrLivArea</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>FullBath</th>
      <th>TotRmsAbvGrd</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.418084</td>
      <td>1239.569260</td>
      <td>1.606633</td>
      <td>407.136307</td>
      <td>916.149144</td>
      <td>1030.297089</td>
      <td>1.532635</td>
      <td>5.986669</td>
      <td>1962.845536</td>
      <td>1980.185419</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.682524</td>
      <td>194.171039</td>
      <td>0.571886</td>
      <td>137.219435</td>
      <td>241.477690</td>
      <td>196.544511</td>
      <td>0.505850</td>
      <td>0.926214</td>
      <td>23.568226</td>
      <td>17.388509</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.000000</td>
      <td>1129.500000</td>
      <td>1.000000</td>
      <td>334.500000</td>
      <td>795.750000</td>
      <td>882.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.747084</td>
      <td>1342.785788</td>
      <td>2.000000</td>
      <td>472.980137</td>
      <td>991.500000</td>
      <td>1087.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>1342.785788</td>
      <td>2.000000</td>
      <td>480.000000</td>
      <td>1057.429452</td>
      <td>1162.626712</td>
      <td>2.000000</td>
      <td>6.517808</td>
      <td>1973.000000</td>
      <td>1995.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000</td>
      <td>1513.000000</td>
      <td>2.000000</td>
      <td>576.000000</td>
      <td>1298.000000</td>
      <td>1391.000000</td>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
X.hist()
```




    array([[<AxesSubplot:title={'center':'OverallQual'}>,
            <AxesSubplot:title={'center':'GrLivArea'}>,
            <AxesSubplot:title={'center':'GarageCars'}>],
           [<AxesSubplot:title={'center':'GarageArea'}>,
            <AxesSubplot:title={'center':'TotalBsmtSF'}>,
            <AxesSubplot:title={'center':'1stFlrSF'}>],
           [<AxesSubplot:title={'center':'FullBath'}>,
            <AxesSubplot:title={'center':'TotRmsAbvGrd'}>,
            <AxesSubplot:title={'center':'YearBuilt'}>],
           [<AxesSubplot:title={'center':'YearRemodAdd'}>, <AxesSubplot:>,
            <AxesSubplot:>]], dtype=object)




    
![png](Brutus_Moise_MLHousePricesPrediction_files/Brutus_Moise_MLHousePricesPrediction_18_1.png)
    



```python
# Train Test Split Data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
```

I split the data into a 20/80 split because our data does not have a large number of observations. I wanted to capture a high amount of training data to combat underfitting. The 20/80 split ratio with 1,460 observations allows me to still have enough observations left over to test my model. I did not use the training data exclusively to train my data because I am also interested in the performance of the training set before applying it to the test set. 


```python
#StandardScaler implementation 
from sklearn.preprocessing import StandardScaler
mms = StandardScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)
```

I then implemented StandardScaler which effectively Standardize our features. I also experimented with MinMaxScaler but it yielded a lower R^2 score. 


```python
#Model 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Compute the mean squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

#residuals graph 
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='lower right')
plt.hlines(y=0, xmin=0, xmax=1000000, color='black', lw=2)
plt.show()

#Compute the r^2 term
from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f', r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred))

print('Intercept: %.3f' % regressor.intercept_)
```

    MSE: 4030520722.7063875
    


    
![png](Brutus_Moise_MLHousePricesPrediction_files/Brutus_Moise_MLHousePricesPrediction_23_1.png)
    


    R^2 train: %.3f, test: %.3f 0.4605084869102253 0.4348636115125337
    Intercept: 182208.342
    


```python
#Random forest model 
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1)
forest.fit(X_train, y_train)
y_train_predm = forest.predict(X_train)
y_test_predm = forest.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

plt.scatter(y_train_predm, y_train_predm - y_train, c='steelblue', edgecolor='white', marker='o', s=35, alpha=0.9, label='Training data')
plt.scatter(y_test_predm, y_test_predm - y_test, c='limegreen', edgecolor='white', marker='s', s=35, alpha=0.9, label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=0, xmax=1000000, color='black', lw=2)
plt.tight_layout()
plt.show()

#Compute the r^2 term
from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f', r2_score(y_train, y_train_predm), r2_score(y_test, y_test_predm))
```

    C:\Users\moise\anaconda3\lib\site-packages\sklearn\ensemble\_forest.py:396: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.
      warn(
    

    MSE train: 3286698918.028, test: 4030520722.706
    


    
![png](Brutus_Moise_MLHousePricesPrediction_files/Brutus_Moise_MLHousePricesPrediction_24_2.png)
    


    R^2 train: %.3f, test: %.3f 0.9084092889084732 0.8417574774710049
    

My first model modeled the data using linear reggression. That model resulted in an MSE of 4030520722 and a R^2 0.46 on the training data and 0.43 on the test data. This model does not do a good job at predicting the sale price based on the features we selected. For my second model, I ran the data through a random forrest model which yielded better results. This model accounted for 0.91% of the training data and 88% of the test data. The randon forrest model is the most robust model to use on this data set. 


```python

```
