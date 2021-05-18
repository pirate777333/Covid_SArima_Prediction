import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import boxcox
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import itertools
import warnings
warnings.filterwarnings('ignore')

# URL
url_potvrdeni='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

# IMPORT DATA
try:
    df_potvrdeni=pd.read_csv(url_potvrdeni)
except:
    print('Something went wrong... URL problems')

# COLUMN Country/Region IS NOW INDEX

# CONFIRMED
df_grupirano_potvrdeni=df_potvrdeni.groupby('Country/Region').sum()
df_grupirano_potvrdeni_dates=df_grupirano_potvrdeni.iloc[:,2:]

drzava=input('Input Country:  ')
drzava=drzava.lower()
drzava=drzava.title()
print('Your Country:  ' + drzava)

try:
    df_drzavni_potvrdeni=df_grupirano_potvrdeni_dates.loc[drzava].to_frame()

except:
    print('Došlo je do greške, ime države je neispravno ili ne postoji')

# CREATE VARIABLES TO CALCULATE DIFFERENCES FOR DAILY NUMBERS OF CASES (CONFIRMED, DEATHS, RECOVERED)
v1=df_grupirano_potvrdeni_dates.shape[1]

v4=v1-1

##########################################################################################

# CONFIRMED

# X 
x_drzavni_potvrdeni=df_drzavni_potvrdeni.index[-v4:].tolist()

# Y 
y_drzavni_potvrdeni=[]
for i in df_drzavni_potvrdeni.iloc[-v4:].values:
    for j in i:
        y_drzavni_potvrdeni.append(j)

y_drzavni_potvrdeni2=[]
for i in df_drzavni_potvrdeni.iloc[-v1:-1].values:
    for j in i:
        y_drzavni_potvrdeni2.append(j)

# REVERSED 2 DEFINED LISTS        
yr=list(reversed(y_drzavni_potvrdeni))
yrc=list(reversed(y_drzavni_potvrdeni2))

# TO ARRAY
yra=np.array(yr)
yrca=np.array(yrc)

# DIFFERENCE 2 ARRAYAS
ny=yra-yrca

# TO LIST, REVERSED
ny=list(reversed(ny))

# X TO DATETIME
x_drzavni_potvrdeni_dates=[]
for s in x_drzavni_potvrdeni: #'8/29/20'
    x_drzavni_potvrdeni_dates.append(datetime.strptime(s, '%m/%d/%y'))

#print(ny)
#print(x_drzavni_potvrdeni_dates)

##fig=px.line(y=ny,x=x_drzavni_potvrdeni_dates,
##       template='plotly_dark',title='Covid Confirmed Cases',
##       labels=dict(x="Date", y="Covid Confirmed Cases"))
##fig.show()

df_new = pd.DataFrame({'Datum':x_drzavni_potvrdeni_dates,
                       'Brojke':ny})
df_new=df_new.set_index('Datum')
#print(df_new.iloc[153:,:])
df_new=df_new.iloc[153:,:]
#print(df_new.dtypes)

##fig=px.line(y=df_new.Brojke,x=df_new.index,
##       template='plotly_dark',title='Covid Confirmed Cases',
##       labels=dict(x="Date", y="Covid Confirmed Cases"))
##fig.show()

##decomposition=sm.tsa.seasonal_decompose(df_new['Brojke'],model='multiplicative')
##decomposition.plot()
##plt.show()

result=adfuller(df_new['Brojke'],autolag='AIC')
print(result[1])
result2=kpss(df_new['Brojke'])
print(result2[1])

##differ = df_new.diff()[1:]
###print(differ)
##
####fig=px.line(y=differ.Brojke,x=differ.index,
####       template='plotly_dark',title='Covid Confirmed Cases',
####       labels=dict(x="Date", y="Covid Confirmed Cases"))
####fig.show()
##
##result=adfuller(differ['Brojke'],autolag='AIC')
##print(result[1])
##result2=kpss(differ['Brojke'])
##print(result2[1])



#print(df_new.shape)

##length_train=48
##train=df_new.iloc[:length_train,:]
##test=df_new.iloc[length_train:,:]
###print(train.shape)
###print(test.shape)
###print(train)
##train.index = pd.to_datetime(train.index)
##train.iloc[:,0]=pd.to_numeric(train.iloc[:,0],downcast='float')
##train.columns=['total_pop']
###print(test)
##test.index = pd.to_datetime(test.index)
##test.iloc[:,0]=pd.to_numeric(test.iloc[:,0],downcast='float')
##test.columns=['total_pop']

plot_acf(df_new, ax=plt.gca(), lags=30)
plt.show()

plot_pacf(df_new, ax=plt.gca(), lags=30)
plt.show()

##stepwise_fit=auto_arima(df_new['Brojke'],trace=True,suppress_warnings=True,
##                        start_p=0, d=None, start_q=0, max_p=5, max_d=5, max_q=5)
##print(stepwise_fit.summary())


##p = d = q = range(0,2)
##pdq = list(itertools.product(p,d,q))
##seasonal_pdq = [(x[0],x[1],x[2],7) for x in pdq]
##
##metric_aic = dict()
##
##for pm in pdq:
##    for pm_seasonal in seasonal_pdq:
##        try:
##            model = SARIMAX(df_new, order=pm, seasonal_order=pm_seasonal,
##                            enforce_stationarity=False,
##                            enforce_invertibility=False)
##            model_aic=model.fit()
##            #print(pm,pm_seasonal,model_aic.aic)
##            metric_aic.update({(pm,pm_seasonal):model_aic.aic})
##        except:
##            continue
##
##for k,v in sorted(metric_aic.items(),key=lambda x: x[1]):
##    print(k,v)

#1,1,1
#0,1,1,7
model = SARIMAX(df_new, order=(5,1,2), seasonal_order=(5,1,2,7),
                            enforce_stationarity=False,
                            enforce_invertibility=False)

modelfit=model.fit()
print(modelfit.summary())

pred=modelfit.forecast(7)
pred_df=pred.to_frame()
print(pred)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_new.index,
    y=df_new['Brojke'],
    name = 'History',
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=pred_df.index,
    y=pred_df.iloc[:,0],
    name='Prediction',
    mode='lines'       
))

fig.update_layout(template='plotly_dark', title='Total Population')
fig.show()

model = SARIMAX(df_new, order=(1,1,1), seasonal_order=(0,1,1,7),
                            enforce_stationarity=False,
                            enforce_invertibility=False)

modelfit=model.fit()
print(modelfit.summary())

pred=modelfit.forecast(7)
pred_df=pred.to_frame()
print(pred)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_new.index,
    y=df_new['Brojke'],
    name = 'History',
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=pred_df.index,
    y=pred_df.iloc[:,0],
    name='Prediction',
    mode='lines'       
))

fig.update_layout(template='plotly_dark', title='Total Population')
fig.show()

##order_arima=(0,1,0)
##order_sarima=(1,0,1,12)
##
##model=SARIMAX(df_new, order=order_arima, seasonal_order=order_sarima)
##modelfit=model.fit()
###print(modelfit.summary())
##
##pred=modelfit.forecast(7)
##pred_df=pred.to_frame()
###print(pred)




##model2=ARIMA(df_new['Brojke'],order=(5,1,2))
##model2=model2.fit()
##
###0-4,0-1-2,0-4
##lista_predvidanja_dani= [7,14,30]
##
##for i in lista_predvidanja_dani:
##    pred=model2.predict(start=len(df_new),end=len(df_new)+i,typ='levels').rename('ARIMA Preds')
##    print(pred)
##    pred=pred.to_frame()
##
##    fig = go.Figure()
##
##    fig.add_trace(go.Scatter(
##        x=df_new.index,
##        y=df_new['Brojke'],
##        name = 'History',
##        mode='lines'
##    ))
##    fig.add_trace(go.Scatter(
##        x=pred.index,
##        y=pred.iloc[:,0],
##        name='Prediction',
##        mode='lines'       
##    ))
##
##    fig.update_layout(template='plotly_dark', title='Total Population')
##    fig.show()
