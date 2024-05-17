import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
df = pd.read_excel("surgery.xlsx")

from sklearn.model_selection import train_test_split
X = df.drop(['target'], axis = 1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 5)

model= GaussianNB().fit(X_train, y_train)
prediction_test = model.predict(X_test)
print(prediction_test)

import pickle
pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))