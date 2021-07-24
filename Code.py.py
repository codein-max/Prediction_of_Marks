
pip install pandas numpy matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_set=pd.read_csv('http://bit.ly/w-data')
print(data_set.head(24))

data_set.plot(x='Hours',y='Scores',style='o')
plt.title('hours vs percentage')
plt.xlabel('hours studied')
plt.ylabel('percentage obtained')
plt.show()

x=data_set[['Hours']].values
y=data_set[['Scores']].values
x,y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn .linear_model import LinearRegression
Regression = LinearRegression()
Regression.fit(x_train,y_train)

line=Regression.coef_*x+Regression.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.show()

print(x_test)
y_pred=Regression.predict(x_test)

df=pd.DataFrame({'Actual':y_test,'Predction':y_pred})
df

scores_pred=np.array([9.25])
scores_pred=scores_pred.reshape(-1,1)
predict=Regression.predict(scores_pred)
print('no.of hours={}'.format(9.5))
print('predicted={}'.format(predict[0]))

