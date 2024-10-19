#Python 3.11 ortamı için birçok yararlı analitik kütüphaneyi yüklüyoruz.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgbm

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,r2_score,roc_auc_score,precision_score,recall_score,f1_score
from sklearn.model_selection import KFold,GridSearchCV,RandomizedSearchCV

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

##############################################################################################################
#Veriler için sütun isim atamalarını yapıyoruz.
columns=["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
         "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
         ,"sensor20","sensor21","sensor22","sensor23"]

##############################################################################################################
#Verileri ilgili dizinlerden okutuyoruz.
train=pd.read_csv("train_FD001.txt",sep=" ",names=columns)
test=pd.read_csv("test_FD001.txt",sep=" ",names=columns)
test_results=pd.read_csv("RUL_FD001.txt",sep=" ",header=None)

##############################################################################################################
#Veriye ön bakış sağlıyoruz.
train.info()
test.info()
train.head()
print('Unique ID: ', train.id.unique())
train.shape
test.shape

##############################################################################################################
#Motor çevrim ömrü için id ve cycle kırılımında bir veri görselleştirmesi yapıyoruz.
unique_ids = train['id'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_ids)))
np.random.shuffle(colors)

plt.style.use('dark_background')
plt.figure(figsize=(8,35))
ax=train.groupby('id')['cycle'].max().plot(kind='barh',width=0.8, stacked=True,align='center',rot=0, color=colors)
plt.title('Engines LifeTime',fontweight='bold',size=30)
plt.xlabel('Cycle Time',fontweight='bold',size=20)
plt.xticks(size=10)
plt.ylabel('Engine ID',fontweight='bold',size=20)
plt.yticks(size=10)
plt.grid(True)
plt.tight_layout()
plt.show()

##############################################################################################################
#Null sütununu kaldırma işlemi
test_results.columns=["rul","null"]
test_results.head()

test_results.drop(["null"],axis=1,inplace=True)
test_results['id']=test_results.index+1
test_results.head()

##############################################################################################################
#Id - cycle kırılımında yeni bir RUL değişkeni oluşturuyoruz.
rul = pd.DataFrame(test.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']

rul.head()

##############################################################################################################
#Motor tükeniş ömrünü gösteren yeni sütunlar oluşturuyoruz.
test_results['rul_failed']=test_results['rul']+rul['max']
test_results.head()

test_results.drop(["rul"],axis=1,inplace=True)
test=test.merge(test_results,on=['id'],how='left')
test["remaining_cycle"]=test["rul_failed"]-test["cycle"]
test.head()

##############################################################################################################
#Eksik - boş veri gözlemi
test.isnull().sum()

#Eksik - boş veri sütunlarını veri setlerinden çıkarıyoruz.
df_train=train.drop(["sensor22","sensor23"],axis=1)
df_test=test.drop(["sensor22","sensor23"],axis=1)

df_test.drop(["rul_failed"],axis=1,inplace=True)
df_test.columns

##############################################################################################################
#Train setine "kalan döngü" sütunu ekleme
df_train['remaining_cycle'] = df_train.groupby(['id'])['cycle'].transform('max') - df_train['cycle']
df_train.head()

##############################################################################################################
#ID=1 olan motorların döngüsüne bakıyoruz.
cycle=30
df_train['label'] = df_train['remaining_cycle'].apply(lambda x: 1 if x <= cycle else 0)
df_test['label'] = df_test['remaining_cycle'].apply(lambda x: 1 if x <= cycle else 0)

op_set=["op"+str(i) for i in range(1,4)]
sensor=["sensor"+str(i) for i in range(1,22)]

test.id.unique()

##############################################################################################################
#ID=1 olan motorların sensör döngü dağılımları
plt.style.use('dark_background')
ax = sns.pairplot(test.query("cycle"), x_vars=op_set, y_vars=sensor, height=2, aspect=1.2)

##############################################################################################################
df_train.label.unique()
df_test.head()
df_test.columns

##############################################################################################################
#Kullanışsız değişkenleri temizleme
df_test.drop(["id","cycle","op3","sensor1","sensor5","sensor6","sensor10","sensor16","sensor18","sensor19"],axis=1,inplace=True)
df_test.label.unique()

x=df_train.drop(["id","cycle","op3","sensor1","sensor5","sensor6","sensor10","sensor16","sensor18","sensor19","remaining_cycle","label"],axis=1)
y=df_train.label
print('x shape : ',x.shape)
print('y shape : ',y.shape)

##############################################################################################################
#Model oluşturma
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)

print('X_train shape : ',X_train.shape)
print('X_test shape : ',X_test.shape)
print('y_train shape : ',y_train.shape)
print('y_test shape : ',y_test.shape)

import lightgbm as lgb
lgb1 = lgb.LGBMClassifier(learning_rate=0.01, n_estimators=5000, num_leaves=100, objective='binary', metrics='auc', random_state=50, n_jobs=-1)

lgb1.fit(X_train, y_train)
lgb1.score(X_test, y_test)
preds = lgb1.predict(X_test)
print('Acc Score: ',accuracy_score(y_test, preds))
print('Roc Auc Score: ',roc_auc_score(y_test, preds))
print('Precision Score: ',precision_score(y_test, preds))
print('Recall Score: ',recall_score(y_test, preds))
print('f1 score: ',f1_score(y_test, preds,average='binary'))

# Acc Score:  0.9546886358129392
# Roc Auc Score:  0.8966428878692285
# Precision Score:  0.8858603066439523
# Recall Score:  0.8125
# f1 score:  0.8475957620211899

##############################################################################################################
#Değişkenlerin önem sırasına göre sıralanmış grafiği
colors = [plt.cm.twilight_shifted(i/float(len(x.columns)-1)) for i in range(len(x.columns))]
columns_X_train=x.columns.tolist()
X_train=pd.DataFrame(X_train)
X_train.set_axis(columns_X_train, axis=1)
feat_importances = pd.Series(lgb1.feature_importances_, index=X_train.columns)
plt.figure(figsize=(15,10))
plt.rcParams.update({'font.size': 16})
plt.title('Önemli değişkenler(16)',color='black',fontweight='bold',size=25)
feat_importances.nlargest(16).plot(kind='bar', color=colors, width=0.8, align='center')
plt.ylabel('Özellikler',color='black',fontweight='bold',size=15)
plt.xlabel('Önemli skorlar',color='black',fontweight='bold',size=15)
plt.tight_layout()
plt.grid(True)
plt.show()

##############################################################################################################
#Alternatif grafik
colors = [plt.cm.cool(i/float(len(x.columns)-1)) for i in range(len(x.columns))]
ax = lgb.plot_importance(lgb1, max_num_features=16,figsize=(16,12),height=0.5,color=colors)
ax.set_title('Önemli özellikler',color='black',fontweight='bold',size=18)
ax.set_xlabel('Değişken noktaları',color='black',fontweight='bold',size=14)
ax.set_xticks(np.arange(0,32501,2500))
ax.set_ylabel('Değişkenler',color='black',fontweight='bold',size=14)

##############################################################################################################
#Model için en iyi parametre bulma işlemleri
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV
stf_kf=StratifiedKFold(n_splits=5)

import xgboost as xgb

xgb_classifier=xgb.XGBClassifier(n_estimators=725,n_jobs=-1)
params={
         'learning_rate': np.arange(0.01, 0.11, 0.025),
         'max_depth': np.arange(1, 10, 1),
         'min_child_weight': np.arange(1, 10, 1),
         'subsample': np.arange( 0.7, 1, 0.05),
         'gamma': np.arange(0.5, 1, 0.1),
         'colsample_bytree': np.arange( 0.1, 1, 0.05),
         'scale_pos_weight': np.arange( 20, 200, 10)}

rs_cv_classifier=RandomizedSearchCV(xgb_classifier,param_distributions=params,cv=stf_kf,n_jobs=-1)

rs_cv_classifier.fit(X_train,y_train)
y_pred=rs_cv_classifier.predict(X_test)

print("En iyi parametreler: \n",rs_cv_classifier.best_params_)
print("XGBClassifier Acc Skoru: ",accuracy_score(y_pred,y_test))

# En iyi parametreler:
#  {'subsample': 0.85,
#  'scale_pos_weight': 80,
#  'min_child_weight': 9,
#  'max_depth': 7,
#  'learning_rate': 0.085,
#  'gamma': 0.7999999999999999,
#  'colsample_bytree': 0.9500000000000001}

# XGBClassifier Acc Skoru: 0.9500848073661255

##############################################################################################################
#Tahmin verileri düzenlemesi ve karmaşıklık matrisi oluşturma
df_test_pred = rs_cv_classifier.predict(df_test.drop(['remaining_cycle','label'],axis=1))
cm=confusion_matrix(df_test.iloc[:,-1], df_test_pred, labels=None, sample_weight=None)

print(cm)
print("Test Accuracy Skoru: ", accuracy_score(df_test.iloc[:,-1],df_test_pred))

# Test Accuracy Skoru:  0.9829718998167379

plt.figure(figsize=(16,12))
sns.heatmap(cm, annot=True, cmap='Pastel1', fmt='d',annot_kws={"size": 25},linewidths=0.7)
##############################################################################################################
