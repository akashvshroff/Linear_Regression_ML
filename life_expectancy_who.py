import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv(r'C:\Users\akush\Desktop\Programming\Projects\Linear_Regression_ML\Life Expectancy Data.csv')
data = data[data.Year == 2015]

#clean the dataset and remove any rows that have these 3 null and then all columns that are null
data.dropna(subset = ['Population','GDP','Hepatitis B'], inplace = True)
data.dropna(axis = 1, inplace = True)

#data.isnull().values.any()
fin_columns = list(data.columns)
fin_columns.remove('Country')
fin_columns.remove('Year')
data = data.sample(frac=1).reset_index(drop=True)

countries = list(data['Country'])
data = data[fin_columns]
print(list(data.columns))
data["Status"] = np.where(data["Status"] == 'Developed',1,0) #converting our binary to integer format

predict = 'Life expectancy '
x = np.array(data.drop([predict],1))
y = np.array(data[predict])

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
linear = linear_model.LinearRegression()

linear.fit(x_train,y_train)
acc = linear.score(x_test,y_test)
print(acc)

predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print('Country: {}; Predicted Life Expectancy: {:02f}, Actual Life Expectancy: {}. '.format(countries[i],predictions[i],y_test[i]))
