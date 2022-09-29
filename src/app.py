from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Importacion de datos
url='https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv'
df=pd.read_csv(url,sep=',')
# Ajuste de nombre columna
df.rename(columns={'19-Oct':'10-19'},inplace=True)
# Definicion de feature and target variable
X=df[['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80+','White-alone pop', 'Black-alone pop', 'Native American/American Indian-alone pop', 'Asian-alone pop', 'Hawaiian/Pacific Islander-alone pop', 'Two or more races pop','R_birth_2018', 'R_death_2018', 'R_INTERNATIONAL_MIG_2018', 'R_DOMESTIC_MIG_2018','Less than a high school diploma 2014-18', 'High school diploma only 2014-18', "Some college or associate's degree 2014-18", "Bachelor's degree or higher 2014-18", 'POVALL_2018', 'MEDHHINC_2018', 'Employed_2018', 'Active Physicians per 100000 Population 2018 (AAMC)', 'Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)','Total nurse practitioners (2019)', 'Total physician assistants (2019)', 'Total Hospitals (2019)', 'Internal Medicine Primary Care (2019)', 'Family Medicine/General Practice Primary Care (2019)', 'Total Specialist Physicians (2019)']]
y=df['Obesity_number']
# Muestra de entrenamiento y prueba
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=1807)


best_ppl=Pipeline(steps=[('scale', MinMaxScaler()),('model',Lasso(alpha=5))])
# Se entrena el modelo
best_ppl.fit(X_train,y_train)
# Prediccion
y_pred_best = best_ppl.predict(X_test)
# Se guarda el modelo
import pickle
filename = '/workspace/My-first-ml-algorithim-/models/finalized_model.sav'
pickle.dump(best_ppl, open(filename, 'wb'))