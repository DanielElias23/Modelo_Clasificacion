import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

#Es una data que ya esta codificada, no escalada, tiene una columna categorica con 3 clases, 1, 2 ,3
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/wine.data",names= ['Class', 'Alcohol', 'Malic acid', 'Ash',
         'Alcalinity of ash' ,'Magnesium', 'Total phenols',
         'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',     'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
         'Proline'])
         
#Para descargar la data para verla en otra app
#df.to_csv("wine_data.csv", index=False)

print(df.shape)
print(df.columns.tolist())
print(df.describe())

      
print(df.info())
print(df['Class'].unique())

X = df.loc[:, df.columns != 'Class']
y = df[['Class']]

#Hay un desequilibrio de clases, pero no estan pronunciado es aceptable
df['Class'].value_counts().plot.bar(color=['green', 'red', "blue"])
plt.show()

print(df['Class'].value_counts())

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

X = mm.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.3, random_state=42)

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, f1_score

from sklearn.tree import DecisionTreeClassifier

#params_grid = {
#    'criterion': ['gini', 'entropy'],
#    'max_depth': [5, 10, 15, 20],
#    'min_samples_leaf': [1, 2, 5]
#}

#dt = DecisionTreeClassifier(random_state=123)

#grid_search = GridSearchCV(estimator = dt, 
#                        param_grid = params_grid, 
#                        scoring='f1',
#                        cv = 5, verbose = False)
                        
#grid_search.fit(X_train, y_train)
#best_params = grid_search.best_params_

#print(best_params)

#Modelo con los mejores parametros
dt2 = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=1,random_state=123)

dt2.fit(X_train, y_train)

y_pred_dt = dt2.predict(X_test)


#cr = classification_report(y_test, y_pred_dt)
#print(cr)

precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_dt, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_dt,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_dt)
print("Modelo Arbol de decision")
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")
print(" ")


#Con regression logistica no sirve GridSearch hay que hacer los cambios de parametros manual

from sklearn.linear_model import LogisticRegression

logR2 = LogisticRegression(random_state=123, solver="liblinear", penalty="l2", C=0.00001)

logR2.fit(X_train, y_train)

y_pred_logR2 = logR2.predict(X_test)

precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_logR2, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_logR2,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_logR2)
print("Regression Logistica")
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")
print(" ")


from sklearn.neighbors import KNeighborsClassifier


#Existen muchos k que pueden ser los mejores, pero no siempre son coincidentes con el error mejor, es mejor verlo graficamente
#Elegimos el mejor k
#max_k = 40
#f1_scores = list()
#error_rates = list() #1-accuary

#for k in range(1, max_k):
    
#    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
#    knn = knn.fit(X_train, y_train)
    
#    y_pred_knn = knn.predict(X_test)
#    f1 = f1_score(y_pred_knn, y_test, average='weighted')
#    f1_scores.append((k, round(f1_score(y_test, y_pred_knn, average='weighted'), 4)))
#    error = 1-round(accuracy_score(y_test, y_pred_knn), 4)
#    error_rates.append((k, error))
    
#f1_results = pd.DataFrame(f1_scores, columns=['K', 'F1 Score'])
#error_results = pd.DataFrame(error_rates, columns=['K', 'Error Rate'])

#sns.set_context('talk')
#sns.set_style('ticks')
#plt.figure(dpi=300)
#ax = f1_results.set_index('K').plot(figsize=(12, 12), linewidth=6)
#ax.set(xlabel='K', ylabel='F1 Score')
#ax.set_xticks(range(1, max_k, 2));
#plt.title('KNN F1 Score')
#plt.savefig('knn_f1.png')
#plt.show()

#sns.set_context('talk')
#sns.set_style('ticks')
#plt.figure(dpi=300)
#ax = error_results.set_index('K').plot(figsize=(12, 12), linewidth=6)
#ax.set(xlabel='K', ylabel='Error Rate')
#ax.set_xticks(range(1, max_k, 2))
#plt.title('KNN Elbow Curve')
#plt.savefig('knn_elbow.png')
#plt.show()

knn = KNeighborsClassifier(n_neighbors=17, weights='distance')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_knn, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_knn,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_knn)
print("KNN")
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")
print(" ")


from sklearn.svm import SVC

#params_grid = {
#    'C': [0.00001, 0.001,0.01,1, 10, 100],
#    'kernel': ['poly', 'rbf', 'sigmoid']
#}

#svc = SVC()

#grid_search = GridSearchCV(estimator = svc, 
#                           param_grid = params_grid, 
#                           scoring='f1',
#                          cv = 5, verbose = 1)

#grid_search.fit(X_train, y_train)
#best_params = grid_search.best_params_

#print(best_params)

svc2= SVC(kernel="poly", C=0.1)

svc2.fit(X_train, y_train)
y_pred_svc = svc2.predict(X_test)

precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_svc, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_svc,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_svc)
print("SVC")
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")
print(" ")

print("Con Modelos aditivos de error")

from sklearn.ensemble import BaggingClassifier

#param_grid = {'n_estimators': [2*n+1 for n in range(20)],
#     'base_estimator__max_depth' : [2*n+1 for n in range(10) ] }

#Bag = BaggingClassifier(base_estimator = DecisionTreeClassifier(), random_state=0, bootstrap=True)

#search = GridSearchCV(estimator=Bag, param_grid=param_grid, scoring='accuracy', cv=3)

#search.fit(X_train, y_train)

#print(search.best_score_)

#print(search.best_params_)

Bag = BaggingClassifier(base_estimator = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=123), n_estimators=31, random_state=0, bootstrap=True)

Bag.fit(X_train, y_train)

y_pred_bag_dt = Bag.predict(X_test)

#Modelo perfecto
precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_bag_dt, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_bag_dt,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_bag_dt)
print("Bagging con Arbol de Decision")
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")
print(" ")


#Random Forest tambien es un aditivo solo para Clasificador de Arbol de decision
from sklearn.ensemble import RandomForestClassifier

#param_grid = {'n_estimators': [2*n+1 for n in range(20)],
#             'max_depth' : [2*n+1 for n in range(10) ],
#             'max_features':["auto", "sqrt", "log2"]}

#RFC = RandomForestClassifier()

#search = GridSearchCV(estimator=RFC, param_grid=param_grid,scoring='accuracy', cv=5)
#search.fit(X_train, y_train)

#print(search.best_score_)

#print(search.best_params_)

RFC2 = RandomForestClassifier(max_depth=7, max_features="log2" , n_estimators=15, random_state=123)

RFC2.fit(X_train,y_train)

y_pred_RFC2 = RFC2.predict(X_test)

precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_RFC2, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_RFC2,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_RFC2)
print("Random Forest")
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")
print(" ")


from sklearn.ensemble import ExtraTreesClassifier

#EF = ExtraTreesClassifier(oob_score=True, 
#                          random_state=42, 
#                          warm_start=True,
#                          bootstrap=True,
#                          n_jobs=-1)

#oob_list = list()

#Para ver cual es la mejor cantidad de arboles con menos error
#for n_trees in [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]:
    
#    EF.set_params(n_estimators=n_trees)
#    EF.fit(X_train, y_train)

    # oob error
#    oob_error = 1 - EF.oob_score_
#    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))

#et_oob_df = pd.concat(oob_list, axis=1).T.set_index('n_trees')

#Muestra los errores para extra tree
#print(et_oob_df)

ETC2 = ExtraTreesClassifier(oob_score=True, 
                          random_state=42, 
                          warm_start=True,
                          bootstrap=True,
                          n_jobs=-1,
                          n_estimators=100)

ETC2.fit(X_train, y_train)

y_pred_ETC2 = ETC2.predict(X_test)

precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_ETC2, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_ETC2,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_ETC2)
print("Random Forest")
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")
print(" ")

from sklearn.ensemble import StackingClassifier

#Los estimadores se deben poner manual y tambien elegir el estimador final, los parametros tambien se pueden poner
#pondria los anteriores aunque en este caso no fue necesario
estimators = [('SVM',SVC(random_state=42)),('KNN',KNeighborsClassifier()),('dt',DecisionTreeClassifier())]

clf = StackingClassifier( estimators=estimators, final_estimator= LogisticRegression())

clf.fit(X_train, y_train)

y_pred_clf = clf.predict(X_test)

precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_clf, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_clf,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_clf)
print("Stacking")
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")
print(" ")

from sklearn.ensemble import GradientBoostingClassifier
#El menor error son 100 y menos arboles
#error_list = list()
#tree_list = [1,5,10,50,100,150,200,400]
#for n_trees in tree_list:
    
    # Initialize the gradient boost classifier
#   GBC = GradientBoostingClassifier(n_estimators=n_trees, random_state=42)

    #Fit the model
#   print(f'Fitting model with {n_trees} trees')
#   GBC.fit(X_train, y_train)
#   y_pred = GBC.predict(X_test)

    # Get the error
#   error = 1.0 - accuracy_score(y_test, y_pred)
    
    # Store it
#   error_list.append(pd.Series({'n_trees': n_trees, 'error': error}))

#error_df = pd.concat(error_list, axis=1).T.set_index('n_trees')

#print(error_df)

#param_grid = {'n_estimators': tree_list,
#              'learning_rate': [0.1, 0.01, 0.001, 0.0001],
#              'subsample': [0.5],
#              'max_features': [4]}

#GV_GBC = GridSearchCV(GradientBoostingClassifier(random_state=42), 
#                      param_grid=param_grid, 
#                      scoring='accuracy',
#                      n_jobs=-1)

#GV_GBC = GV_GBC.fit(X_train, y_train)

#print(GV_GBC.best_estimator_)

#learning_rate=0.001, max_features=4, n_estimators=400, subsample=0.5

GBC1 = GradientBoostingClassifier(n_estimators=400, max_features=4, learning_rate=0.001, random_state=123, subsample=0.5)

GBC1.fit(X_train, y_train)

y_pred_GBC1 = GBC1.predict(X_test)

precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_GBC1, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_GBC1,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_GBC1)
print("GradientBoosting")
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")
print(" ")

from sklearn.ensemble import AdaBoostClassifier

#ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))

#param_grid = {'n_estimators': [1,2,5,10,50,100,200],
#              'learning_rate': [0.0001,0.001,0.01,0.1, 1]}

#GV_ABC = GridSearchCV(ABC,
#                      param_grid=param_grid, 
#                      scoring='accuracy',
#                      n_jobs=-1)

#GV_ABC = GV_ABC.fit(X_train, y_train)

#learning_rate=0.01, n_estimators=200
#print(GV_ABC.best_estimator_)

#Se pueden poner otros estimadores base como SVC
ABC1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=200, learning_rate=0.01)

ABC1.fit(X_train, y_train)

y_pred_ABC1 = ABC1.predict(X_test)

precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_ABC1, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_ABC1,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_ABC1)
print("AdaBoost")
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")
print(" ")

from xgboost import XGBClassifier

#model =XGBClassifier(objective='multi:logistic',eval_metric='mlogloss')

#param_grid = {'learning_rate': [0.1*(n+1) for n in range(10)],
#             'n_estimators' : [2*n+1 for n in range(10)]}
             
from sklearn import preprocessing             

#search = GridSearchCV(estimator=model, param_grid=param_grid, scoring="neg_log_loss")

le2 = preprocessing.LabelEncoder()
y_train2 = le2.fit_transform(y_train)

#search.fit(X_train, y_train2)

#print(search.best_score_)

#print(search.best_params_)

XGB1 =XGBClassifier(objective='multi:logistic',eval_metric='mlogloss', learning_rate=0.8, n_estimators=15)

XGB1.fit(X_train, y_train2)

y_pred_XGB = XGB1.predict(X_test)

precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_XGB, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_XGB,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_XGB)
print("XGBoost (No funciono)")
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")
print(" ")

print("Los que tienen mejores rendimiento son los que tienen modelos aditivos, pero estos pierden explicabilidad")

from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
















