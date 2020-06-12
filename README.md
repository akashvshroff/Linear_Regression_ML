# Outline:

- A linear regression model is a tool for predicting the values of a dependent variable by constructing a line of best fit against its relationship with one or more dependent variables. By plotting such a line that attempts to predict the nature of the relationship, one can predict the dependent variable for some combination of independent variables.
- Using such a system, I tried to implement 2 such models.
    - The first was to predict the grades of a student given data about their previous grades, hours studied, personal lives etc, the dataset for which can be found [here.](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
    - The second was to use information from WHO which detailed various country-specific indicators such as Population, GDP, spread of diseases, facilities etc in order to predict the life expectancy. The dataset from this model can be obtained from Kaggle [here.](https://www.kaggle.com/kumarajarshi/life-expectancy-who)
- Overall, while the models I implemented were extremely basic, I could achieve an accuracy of about 85% in both scenarios after tinkering with the parameters I chose.
- A detailed description of how I went around using sklearn to complete the model is below.

# Purpose:

- I have always been extremely fascinated by the idea of Machine Learning and training a system to make certain predictions and so I wanted to try it out myself.
- Linear Regression is one of the most basic ML algorithms and implementing it gave me a clear idea as to the process of ML and how data is to be selected and cleaned. This facet was reinforced more in the Life Expectancy model as there I was dealing with missing/NaN values that had to be taken care of.
- Ultimately these projects introduced me to the power of ML and its usability as well as the ease of implementation through modules such as sklearn. The builds also reinforced my understanding and knowledge of the very useful numpy and pandas libraries.

# Description:

- In both programs, the first step was to read the csv data into a pandas dataframe which is rather self explanatory - in some countries the ';' (semi-colon) serves as a ',' (comma) and that can be easily accounted for while reading the csv file.
- Then comes cleaning the data, and this only applied to the life expectancy model as the students grades came as a very well packaged bundle with no missing models.
    - Since I am just starting out with ML, I chose to only analyse the year 2015 for all countries.
    - Upon visually inspecting the data and reading other user's responses on Kaggle, I found that most of the missing data comes in the Population, GDP and Hepatitis B columns for the smaller countries on the list and since no form of approximation would allow me to safely infer those values, I merely removed those countries from consideration.
    - I then removed all the columns where there were any NaN values and was left with 132 countries and 22 columns, enough for a simple linear regression model.
- Then the parameters that will contribute to the model were selected and I arrived at these after a lot of tinkering and intuition, (code for trial and error but that's the point, to have fun!).
- Separating the training and testing data followed and sklearn's abstraction eased the process immensely. For both models, I chose 1/10th of the entire dataset as the testing set in order to properly gauge the accuracy.
- Mathematically linear regression is calculated through the least-squares method coupled with a verbose formula but all of this is neatly abstracted in the sklearn and creating a linear regression is as simple as:

    ```python
    linear = linear_model.LinearRegression()
    ```

- The line of best fit and accuracy could be calculated afterwards:

    ```python
    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)
    ```

    - The accuracy here refers to the R^2 value, i.e the proportion of the variance of the dependent variable that can be explained by the independent variable in the model.
- To verify the mathematical nuances of the model, one can also view the slope and y-intercept through the following snippet:

    ```python
    print("Co: \n",linear.coef_)
    print("Intercept: \n",linear.intercept_)
    ```

- Finally, to gauge the predictions visually and see if the model was any good, I printed the model's predictions of the test cases coupled with the actual values of the cases. Needless to say, it worked!
