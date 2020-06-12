import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv(r'C:\Users\akush\Desktop\Programming\Projects\Linear_Regression_ML\student-mat.csv', sep = ';')
data = data[["G1","G2","G3","studytime","failures","absences","famrel","health","schoolsup"]]
data["schoolsup"] = np.where(data["schoolsup"] == 'yes',1,0)

predict = "G3"

x = np.array(data.drop([predict], 1)) #all our attributes
y = np.array(data[predict]) #our label


x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train,y_train)
acc = linear.score(x_test,y_test)
print(acc)

# print("Co: \n",linear.coef_)
# print("Intercept: \n",linear.intercept_)

predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(f'{round(predictions[i])},{y_test[i]}')
