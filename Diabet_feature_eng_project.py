
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import missingno as msno
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x:'%.3f'%x)
pd.set_option('display.max_rows', None)


def load_set():
    data=pd.read_csv("datasets/diabetes.csv")
    return data

df=load_set()

def check_df(dataframe):
    print("########### Shape #############")
    print(dataframe.shape)
    print("########### Columns ###############")
    print(dataframe.columns)
    print("######### data type ##############")
    print(dataframe.dtypes)
    print("############## Na number #########:")
    print(dataframe.isnull().sum())
    print("######## QUANTILE ##############")
    print(dataframe.quantile([0.00, 0.05, 0.50, 0.95,0.99, 1.00]).T)

check_df(df)
df.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_col= [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != 'O' and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes =='O' and dataframe.nunique > car_th]
    cat_col= cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    #Num cols
    num_col= [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_col = [col for col in num_col if col not in num_but_cat]

    print(f'Observations : {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_col)}')
    print(f'num_cols: {len(num_col)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_col, num_col, cat_but_car
cat_col, num_col, cat_but_car= grab_col_names(df)

def cat_summary(dataframe, cat_col, plot=False):
    print(pd.DataFrame({"Value_Number": dataframe[cat_col].value_counts(),
                        "Ratio": dataframe[cat_col].value_counts()/len(dataframe)*100}))
    if plot:
        sns.countplot(x= dataframe[cat_col], data=dataframe)
        plt.show()

cat_summary(df,"Outcome", True)

def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_col].describe(quantiles).T)
    if plot:
        dataframe[num_col].hist()
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show()

for col in num_col:
    num_summary(df, col, True)

##### Analise to target ##########

def target_analyse(dataframe,target , col_name):
    print(dataframe.groupby(target).agg({col_name : ['mean', 'count']}))

for col in num_col:
    target_analyse(df,'Outcome', col)

############## KORELASYON ################


df.corr() #Korelasyonu 1  ve -1 e yakın olanların arasında bir doğrusallık vardır negatif veya pozitif bir doğrusallık

# Korelasyon matrisi

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block= True)


##################################
# BASE MODEL KURULUMU
##################################

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")


############ Plot İmportance


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)

### Feature Engineering ############
#Bir insanda outcome ve pregnancies değerleri dışında 0 olan bir değer olamaz. Bunları Nan olarak say.

zero_columns = [col for col in num_col if df[col].min()==0 and col not in ["Pregnancies, Outcome"]]
for col in zero_columns:
    df[col] = np.where(df[col]==0, np.nan, df[col])

#Eksik gözlem analizi

df.isnull().sum()

def missing_value_tables(dataframe, na_name= False):
    na_col= [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_col].isnull().sum().sort_values(ascending=False)
    ratio= dataframe[na_col].isnull().sum()/len(dataframe[na_col])*100
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n\n')
    if na_name:
        return na_col


na_columns= missing_value_tables(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_columns)

#Eksik değerleri doldur


for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()





##################################
# AYKIRI DEĞER ANALİZİ
##########################


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quantile1= dataframe[col_name].quantile(q1)
    quantile3= dataframe[col_name].quantile(q3)
    IQR= quantile3-quantile1
    lower_lim= quantile1-1.5*IQR
    upper_lim= quantile3+1.5*IQR
    return lower_lim,upper_lim

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))

################# Özellik Çıkarımı######################

df.loc[(df["Age"]>=21) & (df["Age"]<50), "NEW_AGE_CAT"]= 'Mature'
df.loc[df["Age"]>=50, "NEW_AGE_CAT"]= 'Senior'

# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
# Alternatif yöntem: df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])

df.loc[(df["BMI"]<=18.5), "NEW_BMI_CAT"]= 'Underweight'
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NEW_BMI_CAT"]= 'Healthy'
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NEW_BMI_CAT"]= 'Overweight'
df.loc[(df["BMI"] > 30), "NEW_BMI_CAT"]= 'Obez'

# Glukoz degerini kategorik değişkene çevirme

df["NEW_GLUCOSE"]= pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])


# # Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma 3 kırılım yakalandı
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"


# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"



# İnsulin Değeri ile Kategorik değişken türetmek
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Anormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

# sıfır olan değerler önemli
df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]


# Kolonların büyültülmesi
df.columns = [col.upper() for col in df.columns]

df.head()

cat_col, num_col, cat_but_car= grab_col_names(df)


##################################
# ENCODING
##################################


binary_cols= [col for col in df.columns if df[col].nunique() == 2 and df[col].dtypes == 'O']

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_col, drop_first=True)

############# STANDARTLAŞTIRMA #############

standart_scaler= StandardScaler()
df[num_col] = standart_scaler.fit_transform(df[num_col])
df.head()
df.shape




##################################
# MODELLEME
##################################

y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# nıhaı model
# Accuracy: 0.79
# Recall: 0.711
# Precision: 0.67
# F1: 0.69
# Auc: 0.77

# Base Model
# Accuracy: 0.77 (TP+TN)/(TP+TN+FP+FN)
# Recall: 0.706 # pozitif sınıfın ne kadar başarılı tahmin edildiği TP/(TP+FN)
# Precision: 0.59 # Pozitif sınıf olarak tahmin edilen değerlerin başarısı  TP/(TP+FP)
# F1: 0.64  2 * ( Precision*Recall ) / ( Precision+Recall )
# Auc: 0.75

##################################
# FEATURE IMPORTANCE
##################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)





























