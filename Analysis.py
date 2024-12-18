# import librarii
from pandas import read_csv
import pandas as pd

# import baza de date
houses_df = read_csv('Housing.csv')


################## 2. PREZENTAREA BAZEI DE DATE ##################
##################################################################
######  OPERATII PRELIMINARE  ######
# verificam cate NA avem
houses_df.isnull().sum()

# Din baza initiala se va face o selectie care sa includa conditii pentru cel putin 2 variabile
# sa nu avem camera de oaspeti
houses_df1 = houses_df.loc[houses_df['guestroom'] == 'no']
# casa sa fie la drumul principal
houses_df2 = houses_df1.loc[houses_df['mainroad'] == 'yes']
# sa avem mai putin de 3 parcari
houses_df3 = houses_df2.loc[houses_df['parking'] < 3]
# dataset-ul final 
houses_df_final = houses_df3


######  TRANSFORMAREA VARIABILEI  ######
# definim noua variabila categoriala dintr-o variabila numerica
def function_num_to_cat(df):  
    if df['stories'] == 1:
        return 'one'
    elif df['stories'] == 2:
        return 'two'
    else:
        return 'more'

# aplicam functia de mai sus
houses_df_final['stories_cat'] = houses_df_final.apply(lambda df: function_num_to_cat(df), axis=1)

# eliminam variabilele
houses_df_final = houses_df_final.drop(['stories','mainroad','guestroom','hotwaterheating','prefarea','furnishingstatus'], axis=1)

######  EXPORT BAZA DE DATE  ######
houses_df_final.to_csv('Houses_FINAL.csv')

######  DESCRIEREA BAZEI DE DATE  ######
print(houses_df_final.info())
# analiza succinta pentru var numerice + nenumerice
describe_all = houses_df_final.describe(include='all') 
houses_df_final.nunique()

################## 3. ANALIZA GRAFICA SI NUMERICA A VARIABILELOR ANALIZATE ##################
##################################################################
######  ANALIZA DESCRIPTIVA A VARIABILELOR NUMERICE  ######
houses_df_num = houses_df_final[['price','area','bedrooms','bathrooms','parking']]
# masuri de tendinta centrala si de variabilitate
print(houses_df_num.describe())
# masuri de distributie: asimetria (skewness)
#from scipy.stats import skew
houses_df_num.skew(axis = 0, skipna = True)  # print(skew(houses_df_num))
# masuri de distributie: boltirea
#from scipy.stats import kurtosis
houses_df_num.kurtosis(axis = 0, skipna = True)  # print(kurtosis(houses_df_num))

######  ANALIZA DESCRIPTIVA A VARIABILELOR NENUMERICE  ######
houses_df_nenum= houses_df_final[['basement','airconditioning', 'stories_cat']]

for col in houses_df_nenum:
    print("******************")
    print("*** "+ col+ " ***")
    print("******************")
    print("Numar")
    print("******************")
    print(houses_df_nenum[col].value_counts())
    print("******************")
    print("%")
    print("******************")
    print(houses_df_nenum[col].value_counts(normalize =True))
    print("_____________________________________________________")

######  ANALIZA GRAFICA A VARIABILELOR NUMERICE  ######
#import matplotlib.pyplot as plt
print(houses_df_num.hist('price', bins= 10 , align='right', color='green', edgecolor='black'))
print(houses_df_num.hist('area', bins= 10 , align='right', color='pink', edgecolor='black'))
print(houses_df_num.hist('bedrooms', bins= 3 , align='right', color='red', edgecolor='black'))
print(houses_df_num.hist('bathrooms', bins= 5 , align='right', color='yellow', edgecolor='black'))
print(houses_df_num.hist('parking', bins= 3 , align='right', color='blue', edgecolor='black'))

######  ANALIZA GRAFICA A VARIABILELOR NENUMERICE  ######
print(houses_df_nenum.basement.hist(bins=3, color='red'))
print(houses_df_nenum.airconditioning.hist(bins=3, color='blue'))
print(houses_df_nenum.stories_cat.hist(bins=5, color='orange'))

######  IDENTIFICAREA OUTLIERILOR SI TRATAREA ACESTORA  ######
houses_df_num.boxplot('price')
houses_df_num.boxplot('area')
houses_df_num.boxplot('bedrooms')
houses_df_num.boxplot('bathrooms')
houses_df_num.boxplot('parking')


################## 4. ANALIZA STATISTICA A VARIABILELOR CATEGORIALE ##################
##################################################################
######  TABELAREA DATELOR  ######
cross_table_basement_airconditioning = pd.crosstab(houses_df_nenum.basement, houses_df_nenum.airconditioning, margins=True, margins_name="Total" )
print(cross_table_basement_airconditioning)

cross_table_basement_stories_cat = pd.crosstab(houses_df_nenum.basement, houses_df_nenum.stories_cat, margins=True, margins_name="Total" )
print(cross_table_basement_stories_cat)

cross_table_airconditioning_stories_cat =pd.crosstab(houses_df_nenum.airconditioning,houses_df_nenum.stories_cat,margins=True,margins_name="Total" )
print(cross_table_airconditioning_stories_cat)

######  ANALIZA DE ASOCIERE  ######
from scipy.stats import chi2_contingency
chi_test_basement_airconditioning=chi2_contingency(cross_table_basement_airconditioning)
chi_test_basement_airconditioning

chi_test_basement_stories_cat=chi2_contingency(cross_table_basement_stories_cat)
chi_test_basement_stories_cat

chi_test_airconditioning_stories_cat=chi2_contingency(cross_table_airconditioning_stories_cat)
chi_test_airconditioning_stories_cat

######  ANALIZA DE CONCORDANTA  ######
from scipy.stats import chisquare
chisquare(f_obs=houses_df_final['basement'].value_counts(),f_exp=None)
chisquare(f_obs=houses_df_final['airconditioning'].value_counts(),f_exp=None)
chisquare(f_obs=houses_df_final['stories_cat'].value_counts(),f_exp=None)


################## 5. ESTIMAREA SI TESTAREA MEDIILOR ##################
##################################################################
######  ESTIMAREA MEDIEI PRIN IC  ######
import numpy as np
import scipy as sp
import scipy.stats
def interval_de_incredere (data, incredere):
    a=1.0*np.array(data)
    n=len(a)
    media=np.mean(a)
    sem=scipy.stats.sem(a)
    h = sem*sp.stats.t.ppf((1+incredere)/2., n-1)
    return media-h, media+h

print('Interval de incredere al variabilei price este:', interval_de_incredere(houses_df_final['price'], 0.95))
print('Interval de incredere al variabilei area este:', interval_de_incredere(houses_df_final['area'], 0.95))
print('Interval de incredere al variabilei bedrooms este:', interval_de_incredere(houses_df_final['bedrooms'], 0.95))
print('Interval de incredere al variabilei bathrooms este:', interval_de_incredere(houses_df_final['bathrooms'], 0.95))
print('Interval de incredere al variabilei parking este:', interval_de_incredere(houses_df_final['parking'], 0.95))

######  TESTAREA MEDIILOR POPULATIEI  ######
## testarea unei medii cu o valoare fixa
from scipy import stats
print(stats.ttest_1samp(houses_df_final.price, 4600000))
print(stats.ttest_1samp(houses_df_final.area, 3000)) # am dat o valoare inafara iC
print(stats.ttest_1samp(houses_df_final.bedrooms, 2.90))
print(stats.ttest_1samp(houses_df_final.bathrooms, 2)) # am dat o valoare inafara iC
print(stats.ttest_1samp(houses_df_final.parking, 0.65))

## Testarea diferenței dintre 2 medii (esantioane independente sau esantioane perechi)
stories_one = houses_df_final.loc[houses_df_final['stories_cat']=='one']
stories_two = houses_df_final.loc[houses_df_final['stories_cat']=='two']
print(stats.ttest_ind(stories_one.area, stories_two.area))


## testarea diferenței dintre 3 sau mai multe medii
from statsmodels.formula.api import ols
model = ols('area~stories_cat', data=houses_df_final).fit()
import statsmodels.api as sms
print(sms.stats.anova_lm(model, typ=2))


################## 6. ANALIZA DE REGRESIE SI CORELATIE ##################
##################################################################
######  ANALIZA DE CORELATIE  ######
# Matrice coeficient de corelatie
import seaborn as sns
sns.heatmap(houses_df_num.corr(),square = True,annot = True, vmax=0.8)
# Testarea coeficientul de corelatie
from scipy.stats import pearsonr 
print(pearsonr(houses_df_final.price,houses_df_final.area))
print(pearsonr(houses_df_final.price,houses_df_final.bedrooms))
print(pearsonr(houses_df_final.price,houses_df_final.bathrooms))
print(pearsonr(houses_df_final.price,houses_df_final.parking))
print(pearsonr(houses_df_final.area,houses_df_final.bedrooms))
print(pearsonr(houses_df_final.area,houses_df_final.bathrooms))
print(pearsonr(houses_df_final.area,houses_df_final.parking))
print(pearsonr(houses_df_final.bedrooms,houses_df_final.bathrooms))
print(pearsonr(houses_df_final.bedrooms,houses_df_final.parking))
print(pearsonr(houses_df_final.bathrooms,houses_df_final.parking))

######  ANALIZA DE REGRESIE  ######
## regresie liniar simpla
import statsmodels.api as sm
Y = houses_df_final.price
X = houses_df_final.area
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())
print('Parametrii',results.params)
print('R2',results.rsquared)

## regresie liniar multipla
from pandas import DataFrame
Y = houses_df_final.price
X_multiple=DataFrame({
    'area': houses_df_final.area,
    'bedrooms': houses_df_final.bedrooms,
    'bathrooms': houses_df_final.bathrooms,
    'parking': houses_df_final.parking})

X_multiple= sm.add_constant(X_multiple)
model_multiple=sm.OLS(Y, X_multiple)
results_multiple=model_multiple.fit()
print(results_multiple.summary()) 
print('Parametrii',results_multiple.params)
print('R2',results_multiple.rsquared)

## regresie neliniara
X_nel = DataFrame({'area' : houses_df_final.area, 'area^2' : houses_df_final.area**2 })
X_nel = sm.add_constant(X_nel)
Y = houses_df_final.price
model_nel = sm.OLS(Y, X_nel)
results_nel = model_nel.fit()
print(results_nel.summary())
print('Parametrii',results_nel.params)
print('R2',results_nel.rsquared)

######  TESTARE IPOTEZE  ######
##--> Regresie liniar simpla
# erori
print('Parameters:', results.params)
print('R2:', results.rsquared)
print('Predicted values:', results.predict())
print('Erori de modelare:', results.resid)
# Salvarea rezidurilor
erori_rls = results.resid

# Testarea ipotezei privind media erorilor este nula
import scipy.stats as stats
print(stats.ttest_1samp(erori_rls, 0))

# Testarea ipotezei de normalitate a erorilor
from scipy.stats import normaltest
print(normaltest(erori_rls))

# Testarea ipotezei de homoscedasticitate a erorilor
import statsmodels.stats.api as sms
test_GQ=sms.het_goldfeldquandt(erori_rls, results.model.exog)
print(test_GQ)

# Testarea autocorelarii erorilor
import statsmodels.tsa.api as smt
acf=smt.graphics.plot_acf(erori_rls, lags=10, alpha=0.05)
acf.show()

##--> Regresie liniar multipla
# erori
print('Parameters:', results.params)
print('R2:', results.rsquared)
print('Predicted values:', results.predict())
print('Erori de modelare:', results.resid)
# Salvarea rezidurilor
erori_rlm = results_multiple.resid

# Testarea ipotezei privind media erorilor este nula
import scipy.stats as stats
print(stats.ttest_1samp(erori_rlm, 0))

# Testarea ipotezei de normalitate a erorilor
from scipy.stats import normaltest
print(normaltest(erori_rlm))

# Testarea ipotezei de homoscedasticitate a erorilor
import statsmodels.stats.api as sms
test_GQ=sms.het_goldfeldquandt(erori_rlm, results_multiple.model.exog)
print(test_GQ)

# Testarea autocorelarii erorilor
import statsmodels.tsa.api as smt
acf=smt.graphics.plot_acf(erori_rlm, lags=10, alpha=0.05)
acf.show()

