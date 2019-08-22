from __future__ import division
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


churn_df = pd.read_csv('churn_csv.csv', sep=';')
col_names = churn_df.columns.tolist()

print ("Column names:")
print (col_names)

to_show = col_names[:6] + col_names[-6:]

print ("\nSample data:")
churn_df[to_show].head(6)

# сохраняем результат в вектор
churn_result = np.array(churn_df['Churn'])
y = np.where(churn_result,1,0)

# удаляем лишние столбцы
to_drop = ['state','Area Code','Phone','Churn']
churn_feat_space = churn_df.drop(to_drop,axis=1)
X = churn_feat_space.values.astype(np.float)

# подготовка матрицы
scaler = StandardScaler()
X = scaler.fit_transform(X)

print ("Feature space holds %d observations and %d features" % X.shape)
print ("Unique target labels:", np.unique(y))

# прогноз
forest=RF(n_estimators=50)
forest.fit(X,y)
churn_prob=forest.predict_proba(X)
response = churn_prob[:,1]

#сохранение в CSV
export=pd.DataFrame(response)
export.to_csv("churn_predict_50.csv")
