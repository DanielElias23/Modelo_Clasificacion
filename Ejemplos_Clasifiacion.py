                                         ___________________________________
                                         |                                  |
                                         |             INDICE               |
                                         |__________________________________|

        1. REGRESION LOGISTICA                             "Ejemplo de regresion logistica"
        2. REGRESION LOGISTICA EJEMPLO 2                   "Otros ejemplos de regresion logistica con otras setup"
        3. KNN                                             "Ejemplo de KNN" 
        4. KNN ELEGIR MEJOR K                              "Elegir mejor la cantidad de k en KNN"
        5. MAQUINA DE VECTORES DE SOPORTE                  "Ejemplo de SVC"
        6. SVC EJEMPLO 2                                   "SVC ejemplo 2 y Gridsearch" 
        7. ARBOLES DE DECISION                             "Arboles de decision"
        8. ARBOLES DE DECISION EJEMPLO 2                   "Arboles de decision con Gridsearch"
        9. BAGGING                                         "Modelos de adicion, bagging"  
       10. RANDOM FOREST Y EXTRA TREE                      "Modelos de adicion para arbol de decision"
       11. RANDOM FOREST EJEMPLO 2                         "Ejemplo 2 de random forest"
       12. STACKING                                        "Stacking ocupar varios modelos juntos"
       13. BOOSTING Y STACKING                             "Otro ejemplo de stacking"
       14. GRADIENTE Y XBOOST                              "Modelos de adicion gradiente y xboost"
       15. ADABOOST                                        "Modelo Adaboost"
       16. MODELO EXPLICATIVOS                             "Modelos que pueden explicar modelos de caja negra"
       17. DATOS DESEQUILIBRADOS                           "Como tratar datos desequilibrados, para estos modelos"
 

#Cada seccion tiene ejemplo completos sobre el tema

##########################################################################################################################

                                             #REGRESION LOGISTICA
                                             
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,ConfusionMatrixDisplay, precision_recall_fscore_support, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns  

rs = 123

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items.csv"
food_df = pd.read_csv(dataset_url)

print(food_df.dtypes)

print(food_df.head(10))

feature_cols = food_df.columns.tolist()
print(feature_cols)

print(food_df.iloc[:, :-1].describe())

print(food_df.info())

#Este es importante porque meustra el porcentaje de datos de cada clase
print(food_df.iloc[:, -1:].value_counts(normalize=True))

food_df.iloc[:, -1:].value_counts().plot.bar(color=['yellow', 'red', 'green'])
plt.show()

#Esta deifiniendo "X" y "y", siendo -1 el label, la ultima columna que de clasificacion
X_raw = food_df.iloc[:, :-1]
y_raw = food_df.iloc[:, -1:]

scaler = MinMaxScaler()

#Escala todo X ya que todas las columnas son numericas
X = scaler.fit_transform(X_raw)

label_encoder = LabelEncoder()

#La matrix de valores la junta y la aplana si diviciones, luego las transforma
y = label_encoder.fit_transform(y_raw.values.ravel())

print(np.unique(y, return_counts=True))

#Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)

print(f"Training dataset shape, X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing dataset shape, X_test: {X_test.shape}, y_test: {y_test.shape}")

#forma de penalizar el modelo
penalty= 'l2'

#Es el tipo de regresion logistica, binomial 2 clases, multinomial 3 clases o mas
multi_class = 'multinomial'

solver = 'lbfgs'

#Numero de iteraciones, es importante para que muestra diferentes formas en los graficos
max_iter = 1000

l2_model = LogisticRegression(random_state=rs, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)

l2_model.fit(X_train, y_train)

l2_preds = l2_model.predict(X_test)


#Una funcion que mide las metricas siendo accuary la mas importantes, midiendo un accuracy para todas las clases juntas, en
#cambio recall, precision, f_beta son para las clases individuales, por lo que existiran 3 calculos diferentes al haber 3
#clases, accuracy es general para saber como se comporta el modelo
def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp)
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos

print(pd.DataFrame(evaluate_metrics(y_test, l2_preds)))

#Notamos que el modelo tiene un mal rendimiento en la clase 3 esto se debe porque es la minoria de datos son solo un 7%


#Creamos otro modelo para ver como se comporta con penalizacion

penalty= 'l1'
multi_class = 'multinomial'
solver = 'saga'
max_iter = 1000

l1_model = LogisticRegression(random_state=rs, penalty=penalty, multi_class=multi_class, solver=solver, max_iter = 1000)

l1_model.fit(X_train, y_train)

l1_preds = l1_model.predict(X_test)

#Para ver la distribucion de probabilidad, ya que se una regresion multinomial, nos da las probabilidades, en
#este caso la probabilidad de que sea clase 2 es de 96.4%, clase 1 3.5%, calse 3 0.00006%
odd_ratios = l1_model.predict_proba(X_test[:1, :])[0]
print(odd_ratios)

l1_model.predict(X_test[:1, :])[0]
print(X_test[:1, :])
print(l1_model.predict(X_test[:1, :])[0])


#Lo que muestra es que mejoro bastante el rendimiento con el anterior gracias a la penalizacion l1, una respuesta puede
#ser porque muchas caracteristicas de hacen 0, evitando el sobreajuste
print(pd.DataFrame(evaluate_metrics(y_test, l1_preds)))


#Hace una matriz de confusion con porcentajes, no con la cantidad de datos, por defecto es clase 0, clase 1 y clase 2
cf = confusion_matrix(y_test, l1_preds, normalize='true')

sns.set_context('talk')
disp = ConfusionMatrixDisplay(confusion_matrix=cf,display_labels=l1_model.classes_)
disp.plot()
plt.show()

print(l1_model.coef_)       

# Extrae y ordena los coeficientes
def get_feature_coefs(regression_model, label_index, columns):
    coef_dict = {}
    for coef, feat in zip(regression_model.coef_[label_index, :], columns):
        if abs(coef) >= 0.01:
            coef_dict[feat] = coef
    # Sort coefficients
    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    return coef_dict

# Genera un grafico de barras
def get_bar_colors(values):
    color_vals = []
    for val in values:
        if val <= 0:
            color_vals.append('r')
        else:
            color_vals.append('g')
    return color_vals

# Visualiza los coeficientes
def visualize_coefs(coef_dict):
    features = list(coef_dict.keys())
    values = list(coef_dict.values())
    y_pos = np.arange(len(features))
    color_vals = get_bar_colors(values)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y_pos, values, align='center', color=color_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    # labels read top-to-bottom
    ax.invert_yaxis()  
    ax.set_xlabel('Feature Coefficients')
    ax.set_title('')
    plt.show()

#Ordena primeros los coeficientes le decimos: modelo, que clase es (0 a 2), y feature_cols
coef_dict = get_feature_coefs(l1_model, 3, feature_cols) 
#Visualiza los coeficientes 
visualize_coefs(coef_dict)                                 

######################################################################################################################

                                       #MODELO 2 DE REGRESION LOGISTICA
                                       
import seaborn as sns
import pandas as pd
import numpy as np                                       
                                       
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/Human_Activity_Recognition_Using_Smartphones_Data.csv", sep=',')                                       

data = pd.DataFrame(data)
                 
print(data.shape)

#Cuenta los tipos diferentes de cada columna, nos dice que todos excepto una columna es "float64" y la ultima es "object"                  
#EL conteo es de columnas, value_count cuenta un resumen de las repeticiones
print(data.dtypes.value_counts())                                      

#Es el tipo de dato de los 5 ultimas columnas, siendo la ultima "object"
print(data.dtypes.tail())

#Muestra el minimo valor que mas se repite, todas considen en que -1 es el minimo valor de todas las columnas, no contando
#la ultima columna
print(data.iloc[:, :-1].min().value_counts())                                       

#Muestra que el maximo valor que mas repite es 1 en todas las columnas expeto la ultima                   
print(data.iloc[:, :-1].max().value_counts())                                       

#Son 6 categorias con alrededor de 1500 valores cada una, una categorizacion muy pareja
print(data.Activity.value_counts())                                       


#No se acepta la matriz con string para los modelos de maching learning deben ser numeros                                       
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Activity'] = le.fit_transform(data.Activity)
#Es solo una muestra de la columna posicion y la categoria en numeros
print(data['Activity'].sample(5))                                       
                                       

#Calcular la correlacion de las columnas en una matriz
feature_cols = data.columns[:-1]
corr_values = data[feature_cols].corr()

print(pd.DataFrame(corr_values))
print(corr_values.shape)
# Me parece que ordena en dos filas, las conjugaciones de indices que hizo anteriormente en la matriz
tril_index = np.tril_indices_from(corr_values)
print(pd.DataFrame(tril_index))

# Make the unused values NaNs
for coord in zip(*tril_index):
    corr_values.iloc[coord[0], coord[1]] = np.NaN
    
#A las correlaciones calculadas anteriormente, como es una matriz las apila con stack(), pasa a dataframe, resetea el indice
#luego renombra las columnas
corr_values = (corr_values
               .stack()
               .to_frame()
               .reset_index()
               .rename(columns={'level_0':'feature1',
                                'level_1':'feature2',
                                0:'correlation'}))

#Pasa la columna que contiene las correlaciones en todas a valores absolutos
corr_values['abs_correlation'] = corr_values.correlation.abs()                                       
                                       
import matplotlib.pyplot as plt
import seaborn as sns                                       
                                       
sns.set_context('talk')
sns.set_style('white')

#Plotea las correlaciones en un histograma en version stack, son 157079 correlaciones
print(corr_values.abs_correlation)
ax = corr_values.abs_correlation.hist(bins=50, figsize=(12, 8))
ax.set(xlabel='Absolute Correlation', ylabel='Frequency')                                       
plt.show()        
                                       
#Solo muestra las que son mayores a 0.8 que son cerca de 20000             
print(corr_values.sort_values('correlation', ascending=False).query('abs_correlation>0.8'))                                       
                                       
                                       
from sklearn.model_selection import StratifiedShuffleSplit



#Lo que hace es apilar las muestras y revolverlas, separa tambien conjunto de entrenamiento y pruebas 
strat_shuf_split = StratifiedShuffleSplit(n_splits=1, 
                                          test_size=0.3, 
                                          random_state=42)

#Reconoce que es para todas las columnas "X", recordando que X se le llamo feature_cols y el label sera la segunda 
#que pongamos "y"
train_idx, test_idx = next(strat_shuf_split.split(data[feature_cols], data.Activity))

#Es una columna que contiene 70% del conjunto de entrenamiento
print(train_idx.shape)

#Es una columna que contiene 30% del conjunto de prueba
print(test_idx.shape)

# Crea dataframes
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'Activity']

X_test  = data.loc[test_idx, feature_cols]
y_test  = data.loc[test_idx, 'Activity']                                       

#Lo que hizo StratiefiedShuffleSplit es hacer una mezcla equitativa en porcentaje lo mas posible entre datos 
#de entrenamiento y prueba para cada categoria para que fuera equilibrada               
print(y_train.value_counts(normalize=True))                                       
print(y_test.value_counts(normalize=True))                                       
                                       
#De la clase 0 por ejemplo entrenamiento y prueba tienen 18%, la clase 4 los dos tienen 13% tanto entrenamiento
#y prueba en sus respectivos conjunto de datos



from sklearn.linear_model import LogisticRegression

# Standard logistic regression
lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
print("1")
from sklearn.linear_model import LogisticRegressionCV

#Con regulacion l1 IMPORTANTE ESTE MODELO DEMORO ALREDEDOR DE 15 MINUTOS
lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear').fit(X_train, y_train)
print("2")
#Con regulacion l2 IMPORTANTE ESTE MODELO DEMORO ALREDEDOR DE 15 MINUTOS
lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2', solver='liblinear').fit(X_train, y_train)
print("3")

#Combinando todos los coeficientes en un dataframe
coefficients = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

#Grafica la dispercion de coeficientes tanto negativos como positivos
for lab,mod in zip(coeff_labels, coeff_models):
                   coeffs = mod.coef_
                   coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]], codes=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
                   coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))

coefficients = pd.concat(coefficients, axis=1)

print(coefficients.sample(10))

fig, axList = plt.subplots(nrows=3, ncols=2)
axList = axList.flatten()
fig.set_size_inches(10,10)

for ax in enumerate(axList):
   loc = ax[0]
   ax = ax[1]
    
   data = coefficients.xs(loc, level=1, axis=1)
   data.plot(marker='o', ls='', ms=2.0, ax=ax, legend=False)
    
   if ax is axList[0]:
       ax.legend(loc=4)
        
   ax.set(title='Coefficient Set '+str(loc))

plt.tight_layout()
plt.show()

print("a")

y_pred = list()
y_prob = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

print("b")

for lab,mod in zip(coeff_labels, coeff_models):
    print("c")
    y_pred.append(pd.Series(mod.predict(X_test), name=lab))
    print("d")
    y_prob.append(pd.Series(mod.predict_proba(X_test).max(axis=1), name=lab))
    print("e")

print("f")    
y_pred = pd.concat(y_pred, axis=1)
y_prob = pd.concat(y_prob, axis=1)

print(y_pred.head())



from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

metrics = list()
cm = dict()


#Para calcular metricas de los coeficientes anteriores, en general el muestra excelentes puntucaciones para los 3 modelos
#de clasificacion 0.98 de puntucacion en general, no es necesario hacer los que demoran mas si el primero funciona bien
for lab in coeff_labels:

    # Preciision, recall, f-score from the multi-class support function
    precision, recall, fscore, _ = score(y_test, y_pred[lab], average='weighted')
    
    # The usual way to calculate accuracy
    accuracy = accuracy_score(y_test, y_pred[lab])
    
    # ROC-AUC scores can be calculated by binarizing the data
    auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),
              label_binarize(y_pred[lab], classes=[0,1,2,3,4,5]), 
              average='weighted')
    
    # Last, the confusion matrix
    cm[lab] = confusion_matrix(y_test, y_pred[lab])
    
    metrics.append(pd.Series({'precision':precision, 'recall':recall, 
                              'fscore':fscore, 'accuracy':accuracy,
                              'auc':auc}, 
                             name=lab))

metrics = pd.concat(metrics, axis=1)

print(metrics)


#Para graficar la matriz de confusion para cada modelo con diferentes regulaciones por separado,
#muy buen codigo
fig, axList = plt.subplots(nrows=2, ncols=2)
axList = axList.flatten()
fig.set_size_inches(12, 10)

axList[-1].axis('off')

for ax,lab in zip(axList[:-1], coeff_labels):
    sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d');
    ax.set(title=lab);
    
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np

saludo = pd.DataFrame(np.array([[1,3,5,7,"hola"], [2,4,6,8,"hola"], [11,2,3,5,"chao"], [15,21,33,40,"chao"], [12,10,1,55,"hola"]]))


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


#IMPORTANTE recordar que la transoformacion es por abecedario por la primera letra que empieze, en este caso chao = 0,
#hola=1
saludo.iloc[:,-1] = le.fit_transform(saludo.iloc[:,-1])

print(saludo.iloc[:,-1])

X=saludo.iloc[:,:-1]
y=saludo.iloc[:,-1]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
saludo.iloc[:,-1] = le.fit_transform(saludo.iloc[:,-1])
from sklearn.linear_model import LogisticRegression
model_l1_2 = LogisticRegression(penalty="l1", solver="liblinear", multi_class="ovr").fit(X,y)
X_test = pd.DataFrame(np.array([[5,30,10,50], [30,10,1,5]]))
y_pred=model_l1_2.predict(X_test)


y_test = pd.DataFrame(np.array([["chao"], ["chao"]]))
y_test = le.fit_transform(y_test)

print(saludo.iloc[:,-1].value_counts()) 
#Muestra la probabilidad de que ocurra cada categoria
y_prob = model_l1_2.predict_proba(X_test)

#Muestra el resultado de la prediccion
print(y_pred)

#Muestra la probabilidad de que ocurra cada categoria, es por columna, primera columna categoria chao=0, segunda columna
#hola=1, es una matriz en el primera intento es mas probable de que ocurra "hola", en el segundo de que ocurra "chao"
print(y_prob)

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

accuracy = accuracy_score(y_test, y_pred)
f1=score(y_test, y_pred)
print(accuracy)
print(f1)

#Solo mostrara 2 datos si los dos tienen 2 valores, pero funciona perfecto
CM = confusion_matrix(y_pred, y_test)

sns.heatmap(CM, annot=True, fmt='d')
plt.show()




#########################################################################################################################

                                                  #KNN

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Evaluation metrics related methods
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

rs = 123

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/tumor.csv"
tumor_df = pd.read_csv(dataset_url)

print(tumor_df)

print(tumor_df.head())

print(tumor_df.columns)

#son todos numericos, la ultima es la clasificacion
print(tumor_df.info())

#Se quiere predecir si el tumor es benigno o maligno

X = tumor_df.iloc[:, :-1]
y = tumor_df.iloc[:, -1:]

print(X.describe())

print(y.value_counts(normalize=True))

#tumor_benigno es 0, tumor maligo es 1
y.value_counts().plot.bar(color=['green', 'red'])
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)

#Modelo de vecinos cercanos en 2 grupos por tener dos categorias
knn_model = KNeighborsClassifier(n_neighbors=2)

knn_model.fit(X_train, y_train.values.ravel())

preds = knn_model.predict(X_test)

#Para evvaluar las metricas
def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average='binary')
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos

#Consiguio buenas puntuaciones en todas las metricas
print(evaluate_metrics(y_test, preds))

#Ahora codificando para 5 vecinos cercanos
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train.values.ravel())
preds = model.predict(X_test)

#Mejoro bastante la puntuacion increiblemente
print(evaluate_metrics(y_test, preds))


#Ahora intentando muchos diferentes valores para k y saber su puntuacion

max_k = 50
f1_scores = []

for k in range(1, max_k + 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn = knn.fit(X_train, y_train.values.ravel())
    preds = knn.predict(X_test)
    f1 = f1_score(preds, y_test)
    f1_scores.append((k, round(f1_score(y_test, preds), 4)))

#Conviertiendo la lista en un dataframe
f1_results = pd.DataFrame(f1_scores, columns=['K', 'F1 Score'])
f1_results.set_index('K')

#Graficar los resultados, muy bueno para probar diferentes k, se ve que k=5 es el que tiene la mejor puntuacion
ax = f1_results.plot(figsize=(12, 12))
ax.set(xlabel='Num of Neighbors', ylabel='F1 Score')
ax.set_xticks(range(1, max_k, 2));
plt.ylim((0.85, 1))
plt.title('KNN F1 Score')
plt.show()




#######################################################################################################################

                                          #KNN CON VARIOS K
                                        
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, os, sys 
import seaborn as sns                                          
                                          
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/churndata_processed.csv")

print(round(df.describe(),2))

df_uniques = pd.DataFrame([[i, len(df[i].unique())] for i in df.columns], columns=['Variable', 'Unique Values']).set_index('Variable')
print(df_uniques)

binary_variables = list(df_uniques[df_uniques['Unique Values'] == 2].index)
print(binary_variables)

categorical_variables = list(df_uniques[(6 >= df_uniques['Unique Values']) & (df_uniques['Unique Values'] > 2)].index)
print(categorical_variables)

#Muestra los valores unicos de las variables de la
print([[i, list(df[i].unique())] for i in categorical_variables])

#Solo esta tomando los string
ordinal_variables = ['contract', 'satisfaction']

#Los valores unicos de la columna meses
print(df['months'].unique())

print(ordinal_variables.append('months'))

#set elimina todos los datos duplicados de las columnas y los ordena de menor a mayor
#Solo se resta los nombres no la columna entera
numeric_variables = list(set(df.columns) - set(ordinal_variables) - set(categorical_variables) - set(binary_variables))
print(numeric_variables)

#grafica dos columnas que resultaron de la operaciones anteiores, las columnas que se restan son: todas las columnas que
#tengan solo 2 categorias, que tenga de 2 a 6 categorias, otra que resta nuevamente nombres que sobrevivieron a la anteior
#En resumen solo grafica las dos columnas que tienen mas datos unicos de 50 y mayor
df[numeric_variables].hist(figsize=(12, 6))
plt.show()

df['months'] = pd.cut(df['months'], bins=5)
#Lo que hace es cortar los datos de esa columnas en 5 intervalos, no tienen igual cantidad de datos como muestra el histograma
print(df['months'])

###Codificar y scalado de KNN es necesario

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OrdinalEncoder

lb = LabelBinarizer()
le = LabelEncoder()

#Codifica las columnas que tienen entre 3 a 6 datos diferentes con LabelEncoder,  son solo 3 columnas
for column in ordinal_variables:
    df[column] = le.fit_transform(df[column])

print(df[ordinal_variables].astype('category').describe())

#Codifica con Labelbinarizer las columnas que tiene solo 2 datos diferentes
for column in binary_variables:
    df[column] = lb.fit_transform(df[column])

#La resta queda en 0 estas tienen los mismos nombres de columnas
categorical_variables = list(set(categorical_variables) - set(ordinal_variables))


#Categorical variabeles no tiene nada dentro, por lo que solo toma df completo, con los datos codificados
df = pd.get_dummies(df, columns= categorical_variables, drop_first=True)

#Muestra que todas las columnas estan codicadas excepto las que tienen mayor cantidad de datos unicos, dos mayores a 50
print(df.describe().T)

### Escalando los datos

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

#Escala las dos variables con datos unicos altos (numeric_variables), tambien los que tienen 3 a 6 datos unicos
for column in [ordinal_variables + numeric_variables]:
    df[column] = mm.fit_transform(df[column])

print(round(df.describe().T, 3))

#Guarda esto en un archivo, correr solo una vez
#outputfile = 'churndata_processed.csv'
#df.to_csv(outputfile, index=False)

###Creacion modelo de KNN, ahora que estan todos los datos codificados y escalados con las puntuaciones

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

#Definimos las variables
y = df['churn_value']
X = df.drop(columns='churn_value')

#Separamos los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#Definimos un modelos KNN de 3 vecinos
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#Puntuacion del modelo, puede ser mejorable
print(classification_report(y_test, y_pred))
print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 2))
print('F1 Score: ', round(f1_score(y_test, y_pred), 2))

#Graficar la matriz de confusion
sns.set_palette(sns.color_palette())
_, ax = plt.subplots(figsize=(12,12))
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', annot_kws={"size": 40, "weight": "bold"})  
labels = ['False', 'True']
ax.set_xticklabels(labels, fontsize=25);
ax.set_yticklabels(labels[::-1], fontsize=25);
ax.set_ylabel('Prediction', fontsize=30);
ax.set_xlabel('Ground Truth', fontsize=30)
plt.show()

#Creando otro modelo con 5 vecinos con distancia
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Preciision, recall, f-score from the multi-class support function
print(classification_report(y_test, y_pred))
print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 2))
print('F1 Score: ', round(f1_score(y_test, y_pred), 2))
#El modelo mejora, pero muy poco con mas vecinos y con medicion de distancia

_, ax = plt.subplots(figsize=(12,12))
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', annot_kws={"size": 40, "weight": "bold"})  
labels = ['False', 'True']
ax.set_xticklabels(labels, fontsize=25);
ax.set_yticklabels(labels[::-1], fontsize=25);
ax.set_ylabel('Prediction', fontsize=30);
ax.set_xlabel('Ground Truth', fontsize=30)
plt.show()


###¿Cual es el mejor K?


#Itineramos valores de k hasta 40, para ver cual tiene mejor puntuacion
max_k = 40
f1_scores = list()
error_rates = list() # 1-accuracy

for k in range(1, max_k):
    
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn = knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    f1 = f1_score(y_pred, y_test)
    f1_scores.append((k, round(f1_score(y_test, y_pred), 4)))
    error = 1-round(accuracy_score(y_test, y_pred), 4)
    error_rates.append((k, error))
    
f1_results = pd.DataFrame(f1_scores, columns=['K', 'F1 Score'])
error_results = pd.DataFrame(error_rates, columns=['K', 'Error Rate'])


#Grafica las puntuaciones para cada K, la mejor puntuacion esta entre 21 y 22
sns.set_context('talk')
sns.set_style('ticks')

plt.figure(dpi=300)
ax = f1_results.set_index('K').plot(figsize=(12, 12), linewidth=6)
ax.set(xlabel='K', ylabel='F1 Score')
ax.set_xticks(range(1, max_k, 2));
plt.title('KNN F1 Score')
plt.savefig('knn_f1.png')
plt.show()

#Hay que graficar tanto el error como las puntuaciones, puesto no siempre son coicientes, el grafico de arriba tenia
#dos graficos tienes dos picos de puntuacion iguales, pero solo uno tiene un error mas bajo es 21-22
sns.set_context('talk')
sns.set_style('ticks')

plt.figure(dpi=300)
ax = error_results.set_index('K').plot(figsize=(12, 12), linewidth=6)
ax.set(xlabel='K', ylabel='Error Rate')
ax.set_xticks(range(1, max_k, 2))
plt.title('KNN Elbow Curve')
plt.savefig('knn_elbow.png')
plt.show()



#########################################################################################################################

                                         #MAQUINA DE VECTORES DE SOPORTE
                                         
                                       #rendimiento, C, kernel y graficas

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler

rs = 123

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items_binary.csv"
food_df = pd.read_csv(dataset_url)

print(food_df.head(10))

feature_cols = list(food_df.iloc[:, :-1].columns)
print(feature_cols)

X = food_df.iloc[:, :-1]
y = food_df.iloc[:, -1:]

print(X.describe())

y.value_counts(normalize=True)

#Tiene dos clases con menos frecuencia y mas frecuencia, pero hay pocos datos para una clase y la otra tiene muchas, por
#lo que se necesitan emparejar para que el modelo funcione mejor
y.value_counts().plot.bar(color=['red', 'green'])
plt.show()

###Construyendo el modelo

#Separamos en datos de entrenamiento y prueba con el modo estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)

model = SVC()

model.fit(X_train, y_train.values.ravel())

preds = model.predict(X_test)

def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average='binary')
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos

#En general tuvo puntuacion bastante buena
print(evaluate_metrics(y_test, preds))


###Ahora con regulacion C

#El parametro C reduce el error y mantiene el nivel de decision lo mas equilibrado posible, ya que tenemos muy pocos
#datos de una categoria seria lo ideal, pero tambien esto puede causar sobreajuste si C es muy grande, reduce demaciado
#el margen, en caso de C ser muy pequeño, el margen puede ser mayor, esto probocaria mayor datos erroneos, pero esto
#podria generalizar mejor el modelo

#Algunos kernel mas usados, "rbf" con caracterisiticas gaussianas, "poly" con caracterisiticas polinomicas, "sigmoid"
#para datos que tienen dos categorias

#1. con C=10, kernel="rbf"

print("C=10, rbf")
model = SVC(C=10, kernel='rbf')
model.fit(X_train, y_train.values.ravel())
preds = model.predict(X_test)
print(evaluate_metrics(y_test, preds))
#Mejoro bastante la puntuacion


### Con Gridsearch

params_grid = {
    'C': [1, 10, 100],
    'kernel': ['poly', 'rbf', 'sigmoid']
}

model = SVC()

#Automaticamente elije los mejores hiperparametros
grid_search = GridSearchCV(estimator = model, 
                           param_grid = params_grid, 
                           scoring='f1',
                           cv = 5, verbose = 1)
# Search the best parameters with training data
grid_search.fit(X_train, y_train.values.ravel())
best_params = grid_search.best_params_

#Los mejores hiperparametros son C=100 kernel="rbf"
print(best_params)

model = SVC(C=100, kernel='rbf')
model.fit(X_train, y_train.values.ravel())
preds = model.predict(X_test)

#Es efectivamente el mejor en la accuracy=0.966
print(evaluate_metrics(y_test, preds))


#####Graficar para ver como se separan los puntos


#Ahora solo 3 columnas, dos en X y una "y" categorica, la categorica distingue los colores en la grafica
#Para graficar solo se puede graficar maximo con 3 columnas en X, mas variables no se pueden graficar
simplified_food_df = food_df[['Calories', 'Dietary Fiber', 'class']]

#Solo con mil filas
X = simplified_food_df.iloc[:1000, :-1].values
y = simplified_food_df.iloc[:1000, -1:].values

#Remuestramos X e y para que varie un poco
under_sampler = RandomUnderSampler(random_state=rs)
X_under, y_under = under_sampler.fit_resample(X, y)

#Muestra la cantidad de filas y columnas de X, y
print(f"Dataset resampled shape, X: {X_under.shape}, y: {y_under.shape}")

scaler = MinMaxScaler()
X_under = scaler.fit_transform(X_under)

#linear para poder graficarlo
linear_svm = SVC(C=1000, kernel='linear')
linear_svm.fit(X_under, y_under)

#Metodo para graficar la linea del vector de decicion y sus margenes
def plot_decision_boundry(X, y, model):
    plt.figure(figsize=(16, 12))
    #Grafica los puntos con las columnas de X, definiendo las clases y, la grafica de los puntos no tiene que ver
    #con el espacio del kernel que se usara
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
    )

    # plot support vectors
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.show()

plot_decision_boundry(X_under, y_under, linear_svm)

#Si queremos un especio no lineal quizas queramos ocupar "rbf"

svm_rbf_kernel = SVC(C=100, kernel='rbf')
svm_rbf_kernel.fit(X_under, y_under)

plot_decision_boundry(X_under, y_under, svm_rbf_kernel)

svm_rbf_kernel = SVC(C=100, kernel='poly')
svm_rbf_kernel.fit(X_under, y_under)
plot_decision_boundry(X_under, y_under, svm_rbf_kernel)


######################################################################################################################

                                   #Vectores ejemplos y medicion de tiempo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/Wine_Quality_Data.csv", sep=',')

print(data.shape)

print(data.info())

#Muestra las correlaciones solo con un tipo de color, la ultima columna
y = (data['color'] == 'red').astype(int)
fields = list(data.columns[:-1])  # everything except "color"
correlations = data[fields].corrwith(y)
correlations.sort_values(inplace=True)
print(correlations)

sns.set_context('talk')
#sns.set_palette(palette)
sns.set_style('white')

#Recordando que pairplot muestra todas las posibles correlaciones entre columna, en este caso
#no existen correlaciones claras, muy poca correlacion entre columnas
#sns.pairplot(data, hue='color')
#plt.show()

from sklearn.preprocessing import MinMaxScaler

#En la primra parte topa los indices de las de las dos ultimas columnas que tienen los valores mas altos
fields = correlations.map(abs).sort_values().iloc[-2:].index
print(fields)
X = data[fields]

#Escala esas columnas
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#Le cambia de nombre a una
X = pd.DataFrame(X, columns=['%s_scaled' % fld for fld in fields])
print(X.columns)


###LinearSVC

#Graficando las dos columnas tomadas anteiormente

from sklearn.svm import LinearSVC

LSVC = LinearSVC()
LSVC.fit(X, y)

X_color = X.sample(300, random_state=45)
y_color = y.loc[X_color.index]
y_color = y_color.map(lambda r: 'red' if r == 1 else 'yellow')
ax = plt.axes()
ax.scatter(
    X_color.iloc[:, 0], X_color.iloc[:, 1],
    color=y_color, alpha=1)
# -----------
x_axis, y_axis = np.arange(0, 1.005, .005), np.arange(0, 1.005, .005)
xx, yy = np.meshgrid(x_axis, y_axis)
xx_ravel = xx.ravel()
yy_ravel = yy.ravel()
X_grid = pd.DataFrame([xx_ravel, yy_ravel]).T
y_grid_predictions = LSVC.predict(X_grid)
y_grid_predictions = y_grid_predictions.reshape(xx.shape)
ax.contourf(xx, yy, y_grid_predictions, cmap=plt.cm.autumn_r, alpha=.3)
# -----------
ax.set(
    xlabel=fields[0],
    ylabel=fields[1],
    xlim=[0, 1],
    ylim=[0, 1],
    title='decision boundary for LinearSVC')
plt.show()


###kernel Gaussian 

#Codigo creado para graficar dos columnas en X
def plot_decision_boundary(estimator, X, y):
    estimator.fit(X, y)
    X_color = X.sample(300)
    y_color = y.loc[X_color.index]
    y_color = y_color.map(lambda r: 'red' if r == 1 else 'yellow')
    x_axis, y_axis = np.arange(0, 1, .005), np.arange(0, 1, .005)
    xx, yy = np.meshgrid(x_axis, y_axis)
    xx_ravel = xx.ravel()
    yy_ravel = yy.ravel()
    X_grid = pd.DataFrame([xx_ravel, yy_ravel]).T
    y_grid_predictions = estimator.predict(X_grid)
    y_grid_predictions = y_grid_predictions.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contourf(xx, yy, y_grid_predictions, cmap=plt.cm.autumn_r, alpha=.3)
    ax.scatter(X_color.iloc[:, 0], X_color.iloc[:, 1], color=y_color, alpha=1)
    ax.set(
        xlabel=fields[0],
        ylabel=fields[1],
        title=str(estimator))
    plt.show()
   
   
from sklearn.svm import SVC

#Los gammas hacen que el modelos SVC con kernel gaussiano se curve, entre mas valor de gamma mas curvo seran los limites
#de decision, esto tambien es afectado por el valor de C, un gamma bajo es practicamente lineal
gammas = [.5, 1, 2, 10]

#Probando diferentes gamma
#for gamma in gammas:
#    SVC_Gaussian = SVC(kernel='rbf', gamma=gamma)
#    plot_decision_boundary(SVC_Gaussian, X, y)

#Probando diferentes valores de C
#Cs = [.1, 1, 10]
#for C in Cs:
#    SVC_Gaussian = SVC(kernel='rbf', gamma=2, C=C)
#    plot_decision_boundary(SVC_Gaussian, X, y)
    
    
###Comparacion de tiempo de demora de los modelos

from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

y = data.color == 'red'
X = data[data.columns[:-1]] #Toda la data excepto la ultima columna

#Tenemos 3 modelos
kwargs = {'kernel': 'rbf'}
svc = SVC(**kwargs)

#Aproxima un mapa de los kernel
nystroem = Nystroem(**kwargs)

#Es un modelo de gradiente estocastico, se le pueden poner diferentes funciones de error y penalizaciones, entre otros
#parametros y modelos
sgd = SGDClassifier() 

import timeit

#Las medidas de tiempo en los diferentes sistemas operativos se miden de forma diferente, ademas la medida por defecto
#tiene tiempos arbitrarios por lo que no tiene sentido imprimir start
start = timeit.default_timer()
svc.fit(X,y)  

#Lo que importa es esta diferencia de tiempo, nos dara la diferencia con "start" y por ende cuando demoro en ese lapso
#de tiempo, esta medido en segundos y decimales que son fracciones de segundos
print("time end fit svc: ", timeit.default_timer() - start)
  
start = timeit.default_timer()
X_transformed = nystroem.fit_transform(X)
sgd.fit(X_transformed, y)    
print("time end transform fit sgd: ", timeit.default_timer() - start)

#La cantidad de filas se multiplica por 5, lo que hace tener una gran cantidad de datos al dataframe
#Con el fin de medir tiempo    
X2 = pd.concat([X]*5)
y2 = pd.concat([y]*5)

print(X2.shape)
print(y2.shape)    

#Este caso tuvo una super mal efecto con el aumento de datos, el tiempo se multiplico mas de 5 veces, alrededor de 22 veces
start = timeit.default_timer()
svc.fit(X2, y2)    
print("time end fit svc whit X, y * 5: ", timeit.default_timer() - start)  
   
#Tuvo buen rendimiento al aumento de datos por 5, el tiempo solo se multiplico por 3.5 veces
start = timeit.default_timer()
X2_transformed = nystroem.fit_transform(X2)
sgd.fit(X2_transformed, y2)    
print("time end transform fit sgd with X,y * 5: ", timeit.default_timer() - start) 


#########################################################################################################################

                                      #Arboles de decision y GridSearch
                                            
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns                                            
                                            
rs = 123                                            
                                            
dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/tumor.csv"
tumor_df = pd.read_csv(dataset_url)       

print(tumor_df.head())
print(tumor_df.info())


#Definimos X, y
X = tumor_df.iloc[:, :-1]
y = tumor_df.iloc[:, -1:]

#Separamos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)

model = DecisionTreeClassifier(random_state=rs)

model.fit(X_train, y_train.values.ravel())

preds = model.predict(X_test)

#Para evaluar metricas
def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average='binary')
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos

#El modelo tiene buen rendimiento para estos datos con accuracy 0.956
print(evaluate_metrics(y_test, preds))

#Graficar el arbol de decision
def plot_decision_tree(model, feature_names):
    plt.subplots(figsize=(25, 20)) 
    tree.plot_tree(model, 
                       feature_names=feature_names,  
                       filled=True)
    plt.show()

#Para obtener los nombres de cada columnas
feature_names = X.columns.values

#Grafica el arbol muy correctamente con todos sus valores
#EL color naranjo significa que la mayoria de datos pertenecen a la clase 0
#El color azul significa que la mayoria de datos perteecen a la clase 1
#El arbol tiene tantas ramificaciones como caracterisicas tenga la data, cada ramificacion es una caracteristica
plot_decision_tree(model, feature_names)

#Los arboles con tantas ramificaciones pueden tener sobreajuste, un modelo simplificado de este puede tener incluso
#un mejor rendimiento

#Se pueden ajustar los hiperparametros del modelo, siendo un modelo simplificado
custom_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=3, random_state=rs)

custom_model.fit(X_train, y_train.values.ravel())
preds = custom_model.predict(X_test)

#Evaluamos el modelo mejor en general un poco mas en casi todas las puntuaciones excepto f1
print(evaluate_metrics(y_test, preds))

plot_decision_tree(custom_model, feature_names)


#Probando un nuevo modelo ahora con el criterio de gini
custom_model = DecisionTreeClassifier(criterion='gini', max_depth=15, min_samples_leaf=5, random_state=rs)
custom_model.fit(X_train, y_train.values.ravel())
preds = custom_model.predict(X_test)

#En general mejoro aun mas las puntuaciones excepto presicion, pero el arbol se inclino a una sola clase
print(evaluate_metrics(y_test, preds))
plot_decision_tree(custom_model, feature_names)


### GridSearch

#Encontrando el mas alto f1

params_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5]
}

model = DecisionTreeClassifier(random_state=rs)

grid_search = GridSearchCV(estimator = model, 
                        param_grid = params_grid, 
                        scoring='f1',
                        cv = 5, verbose = 1)
                        
grid_search.fit(X_train, y_train.values.ravel())
best_params = grid_search.best_params_

#Nos dice que los mejores hiperparametros
#criterio=gini, max_depth=10, min_samples_depth=5
print(best_params)

#creando el modelo
custom_model = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=5, random_state=rs)
custom_model.fit(X_train, y_train.values.ravel())
preds = custom_model.predict(X_test)

#Aunque la puntuacion no mejoro al modelo anteior es identica
print(evaluate_metrics(y_test, preds))
plot_decision_tree(custom_model, feature_names)


####################################################################################################################

                                 #Arbol de decision ejemplos 2 y graficas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                                    
  
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/Wine_Quality_Data.csv", sep=',')  
  
print(data.head())  
  
print(data.dtypes)
  
#Codifico los valores, pero usando metodo de remplazar
data['color'] = data.color.replace('white',0).replace('red',1).astype(int)  

#Todas los nombres excepto el de las columnas  
feature_cols = [x for x in data.columns if x not in 'color']  
  

from sklearn.model_selection import StratifiedShuffleSplit

#Podemos definir explicitamente el tamaño de la prueba
strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=42)

#Separa en entrenamiento y pruebas con la misma cantidad
train_idx, test_idx = next(strat_shuff_split.split(data[feature_cols], data['color']))

#definimos conjunto de entrenamiento
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'color']

#definimos conjunto de prueba
X_test = data.loc[test_idx, feature_cols]
y_test = data.loc[test_idx, 'color']  
  
#Lo que se hace es mostrar cuanto es el porcentaje de cada categoria
print(y_train.value_counts(normalize=True).sort_index())
print(y_test.value_counts(normalize=True).sort_index())  

#Creamos el modelo
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt = dt.fit(X_train, y_train)

#Nos dice la cantidad de nodos del arbol
print(dt.tree_.node_count)

#Nos dice la profundidad de el arbol
print(dt.tree_.max_depth)
  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def measure_error(y_true, y_pred, label):
    return pd.Series({'accuracy':accuracy_score(y_true, y_pred),
                      'precision': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'f1': f1_score(y_true, y_pred)},
                      name=label)

y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),
                              measure_error(y_test, y_test_pred, 'test')],
                              axis=1)
#Calcula las puntuaciones tanto para el error de los datos de entrenamiento como los datos de prueba
print(train_test_full_error)


#Encontrando los mejores hiperparametros
from sklearn.model_selection import GridSearchCV

#que itener los valores tanto del maximo de profundiad y maximo de caracteristicas, range(inicio(opcional), final, pasos(opcional))
param_grid = {'max_depth':range(1, dt.tree_.max_depth+1, 2),
              'max_features': range(1, len(dt.feature_importances_)+1)}

GR = GridSearchCV(DecisionTreeClassifier(random_state=42),
                  param_grid=param_grid,
                  scoring='accuracy',
                  n_jobs=-1)

GR = GR.fit(X_train, y_train)

print("mejor cantidad de nodos: ",GR.best_estimator_.tree_.node_count)
print("mejor produndidad: ",GR.best_estimator_.tree_.max_depth)

y_train_pred_gr = GR.predict(X_train)
y_test_pred_gr = GR.predict(X_test)

train_test_gr_error = pd.concat([measure_error(y_train, y_train_pred_gr, 'train'),
                                 measure_error(y_test, y_test_pred_gr, 'test')],
                                axis=1)

print(train_test_gr_error)

#todos excepto residual_sugar
feature_cols = [x for x in data.columns if x != 'residual_sugar']


X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'residual_sugar']

X_test = data.loc[test_idx, feature_cols]
y_test = data.loc[test_idx, 'residual_sugar']

#Creamos un nuevo modelo con la version de arbol de decision regresor
from sklearn.tree import DecisionTreeRegressor

dr = DecisionTreeRegressor().fit(X_train, y_train)

param_grid = {'max_depth':range(1, dr.tree_.max_depth+1, 2),
              'max_features': range(1, len(dr.feature_importances_)+1)}

GR_sugar = GridSearchCV(DecisionTreeRegressor(random_state=42),
                     param_grid=param_grid,
                     scoring='neg_mean_squared_error',
                      n_jobs=-1)

GR_sugar = GR_sugar.fit(X_train, y_train)

#Soprendentemente dice que hay que aumentar los nodos a muchos mas 7953 y la profundiad a 25
print("mejor cantidad de nodos: ",GR_sugar.best_estimator_.tree_.node_count)
print("mejor profundidad. ", GR_sugar.best_estimator_.tree_.max_depth)

#Midiendo el error
from sklearn.metrics import mean_squared_error

y_train_pred_gr_sugar = GR_sugar.predict(X_train)
y_test_pred_gr_sugar  = GR_sugar.predict(X_test)

train_test_gr_sugar_error = pd.Series({'train': mean_squared_error(y_train, y_train_pred_gr_sugar),
                                         'test':  mean_squared_error(y_test, y_test_pred_gr_sugar)},
                                          name='MSE').to_frame().T

print(train_test_gr_sugar_error) #El error es bajo aumentan un poco en los datos de prueba


#Muestran una correlacion muy alta entre las predicciones y los datos de prueba
sns.set_context('notebook')
sns.set_style('white')
fig = plt.figure(figsize=(6,6))
ax = plt.axes()

ph_test_predict = pd.DataFrame({'test':y_test.values,
                                'predict': y_test_pred_gr_sugar}).set_index('test').sort_index()

ph_test_predict.plot(marker='o', ls='', ax=ax)
ax.set(xlabel='Test', ylabel='Predict', xlim=(0,35), ylim=(0,35))
plt.show()


###Esto era para graficar una imagen

from io import StringIO
from IPython.display import Image, display, display_png
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.image as mpimg


#Creando un output de destino
dot_data = StringIO()

export_graphviz(dt, out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# View the tree image
filename = 'wine_tree.png'
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
plt.show()
#imagen1 = Image(filename=filename)
#display(imagen1)


dot_data = StringIO()

export_graphviz(GR.best_estimator_, out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# View the tree image
filename = 'wine_tree_prune.png'
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
plt.show()
#Image(filename=filename) 



########################################################################################################################

                                   #Bagging (Boostraping Agreggation) (Embolsado)
                                         #Poca mejora, puntuaciones
                                       
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics 
# Give loops a progress bar
from tqdm import tqdm                                       
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning) 

#sirve para omitir las advertencia cuando esta itinerando, son demaciadas
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
                                       
def get_accuracy(X_train, X_test, y_train, y_test, model):
    return  {"test Accuracy":metrics.accuracy_score(y_test, model.predict(X_test)),"train Accuracy": metrics.accuracy_score(y_train, model.predict(X_train))}

# Plot tree helper libraries
from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

#Crea una funcion que visualiza arbol de decision
def plot_tree(model,filename = "tree.png"):
    #global churn_df 

    dot_data = StringIO()
  

    featureNames = [colunm  for colunm in churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']].columns]
    out=tree.export_graphviz(model,feature_names=featureNames, out_file=dot_data, class_names= ['left','stay'], filled=True,  special_characters=True,rotate=False)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(filename)
    img = mpimg.imread(filename)
    plt.figure(figsize=(100, 200))
    plt.imshow(img,interpolation='nearest')
    plt.show()


#Funcion que grafica los arboles de decision y sus puntuaciones de entrenamiento y prueba usando BaggingClassifier
def get_accuracy_bag(X,y,title,times=20,xlabel='Number Estimators'):
    #Iterate through different number of estimators and average out the results  


    N_estimators=[n for n in range(1,70)]
    times=20
    train_acc=np.zeros((times,len(N_estimators)))
    test_acc=np.zeros((times,len(N_estimators)))
    
    train_time=np.zeros((times,len(N_estimators)))
    test_time=np.zeros((times,len(N_estimators)))
     #average out the results
    for n in tqdm(range(times)):
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)
        for n_estimators in N_estimators:
            #Iterate through different number of estimators and average out the results   
        
            Bag= BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth = 10),n_estimators=n_estimators,bootstrap=True,random_state=0)
            Bag.fit(X_train,y_train)
          
            
             
            Accuracy=get_accuracy(X_train, X_test, y_train, y_test,  Bag)
           
            
            
  
            train_acc[n,n_estimators-1]=Accuracy['train Accuracy']
            test_acc[n,n_estimators-1]=Accuracy['test Accuracy']
        
        
        
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(train_acc.mean(axis=0))
    ax2.plot(test_acc.mean(axis=0),c='r')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Training accuracy',color='b')
    ax2.set_ylabel('Testing accuracy', color='r')
    plt.title(title)
    plt.show()
    

churn_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv")

print(churn_df.head())
print(churn_df.shape)

#Solo toma algunas columnas
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]

#Cambia de tipo a churn
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())
print(churn_df.shape)   


###Bagging de forma artesanal

#Bagging es extraer una cantidad de datos repetidos y remplazarle valores para saber el mejor parametro

from sklearn.utils import resample

#Tomamos solo 5 filas del archivo
print(churn_df[0:5])

#Lo que hace es tomar las 5 filas revolverlas, eliminar una fila al azar o mas de una en esta sub muestra, y la remplaza
#Por una fila repetida pertenecientes a la submuestra, 2 por ejemplo, habra 3 filas iguales en la submuestra
print(resample(churn_df[0:5]))

#Cada iteracion lo hace nuevamente
print(resample(churn_df[0:5]))

#Definiendo solo algunas columnas para X, y
X = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']]

print(X.head())

y = churn_df['churn']
print(y.head())

#separando en datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)
print ('Train set', X_train.shape,  y_train.shape)
print ('Test set', X_test.shape,  y_test.shape)


###Arbol de deciosion con Bagging

#El arbol de decision tiene el problema que se sobreajustan con facilidad y tiene a generalizar mal los datos
#Bagging intenta solucionar este problema

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#Cramos el modelo de arbol de decision
max_depth=5
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=10)
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = max_depth,random_state=10)
Tree
Tree.fit(X_train,y_train)

yhat = Tree.predict(X_test)
print(yhat)

#El modelo tiene buena puntuacion en los datos de entrenamiento, pero mala en los datos de prueba
print(get_accuracy(X_train, X_test, y_train, y_test,  Tree))

plot_tree(filename = "tree.png",model=Tree)

#El modelo muestra el mismo sobreajuste
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=5)
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4 ,random_state=2)
Tree.fit(X_train,y_train)
print(get_accuracy(X_train, X_test, y_train, y_test,  Tree))
plot_tree(filename = "tree1.png",model=Tree)


#AHORA CON BAGGING

from sklearn.ensemble import BaggingClassifier

#Ocupamos el mismo anteior
Bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth = 4,random_state=2),n_estimators=30,random_state=0,bootstrap=True)

Bag.fit(X_train,y_train)

Bag.predict(X_test)

#La puntuacion mejoro, pero muy poco, con los datos de entrenamiento tiene buena puntuacion, pero con con los
#datos de prueba no funciona bien
print(get_accuracy(X_train, X_test, y_train, y_test,  Bag))

#Se puede ver el efecto que tiene la cantidad de estimadores, estabilizan la puntuacion y tienden a cierto valor,
#la puntuacion es mejorada, pero no en gran medida, a partir de 20 a 30 estimadores, las puntuaciones se tienden
#a cierto, valor, pero la de entrenamiento es mas estable, la de prueba varia mucho mas
#print(get_accuracy_bag(X, y, "Customer Churn"))


#La varianza baja ---------> no mejora el resultado bagging
#La varianza alto ---------> si mejora el resultado bagging

from sklearn.svm import SVC

clf=SVC(kernel='linear',gamma='scale')
clf.fit(X_train, y_train) 
print(get_accuracy(X_train, X_test, y_train, y_test,  clf))

Bag = BaggingClassifier(base_estimator=SVC(kernel='linear',gamma='scale'),n_estimators=10,random_state=0,bootstrap=True)
Bag.fit(X_train,y_train)
print(get_accuracy(X_train, X_test, y_train, y_test,  Bag))

#SVM con bagging no hace casi nada

###Otro ejemplo

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv")

print(df.head())

#Elimina una columna
df = df[df["BareNuc"] != "?"]

#Se queda colo con estas columnas
X =  df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]

print(X.head())

#El y es solo una columna
y = df['Class']

print(y.head())

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [2*n+1 for n in range(20)],
     'base_estimator__max_depth' : [2*n+1 for n in range(10) ] }

Bag = BaggingClassifier(base_estimator = DecisionTreeClassifier(), random_state=0, bootstrap=True)

search = GridSearchCV(estimator=Bag, param_grid=param_grid, scoring='accuracy', cv=3)

search.fit(X_train, y_train)

print(search.best_score_)

#Muestra que los mejores para metros son profundidad=5 y numero de estimadores=11
print(search.best_params_)

#La puntuacion mejora, pero no tanto
print(get_accuracy(X_train, X_test, y_train, y_test, search.best_estimator_))

#En este caso muestra que entre el numero de estimadores 60 y 70 maxima bastante la puntuacion
#print(get_accuracy_bag(X, y, "Cancer Data"))

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv", delimiter=",")
print(df.head())

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

y = df["Drug"]
print(y[0:5])

le_sex = preprocessing.LabelEncoder()

#Le dice los valores que debe tomar como numeros, F=0 y M=1
le_sex.fit(['F','M'])

#Luego se lo aplica a la columna Sex
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()

#Le dice los valores que debe tomar High=0, Low=1, normal=2
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])

#Transforma la columna BP
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()

#Dice los valores que debe tomar High=1, normal=0
le_Chol.fit([ 'NORMAL', 'HIGH'])

#Se lo aplica a la columna cholesterol
X[:,3] = le_Chol.transform(X[:,3]) 
     
print(X[0:5])     

#Se paramos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)     
     
param_grid = {'n_estimators': [2*n+1 for n in range(20)],
     'base_estimator__max_depth' : [2*n+1 for n in range(10) ]}     
     
Bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,bootstrap=True)     
     
search = GridSearchCV(estimator=Bag, param_grid=param_grid,scoring='accuracy', cv=3)     
     
search.fit(X_train, y_train)     

#La puntuacion es bastante buena
print(search.best_score_)  
     
print(search.best_params_)    
     
#Los mejores parametros que encontro fueron profundidad=4, numero de estimadores=9
print(get_accuracy(X_train, X_test, y_train, y_test, search.best_estimator_))  

#Muestra como varia, mejora un poco nuevamente, en este caso dependiendo el estimador se mantiene estable un poco
#print(get_accuracy_bag(X, y, "Drug Data"))




########################################################################################################################

                                       #Bagging con random forest y extra tree
                       #Mejor numero de estimadores, puntuaciones, caracteristicas mas importantes       
                                              
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                                              

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/churndata_processed.csv")

print(data.head())

round(data.describe().T, 2)

print(data.dtypes)

fig, ax = plt.subplots(figsize=(15,10)) 
sns.heatmap(data.corr())
plt.show()

target = 'churn_value'
data[target].value_counts()

data[target].value_counts(normalize=True)

print(data.shape)

#Garantizamos que los datos sean parejos para las clases
from sklearn.model_selection import StratifiedShuffleSplit

feature_cols = [x for x in data.columns if x != target]


# De los 7000 filas solamente tomamos 1500
# This creates a generator
strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=1500, random_state=42)

# Get the index values from the generator
train_idx, test_idx = next(strat_shuff_split.split(data[feature_cols], data[target]))

# Create the data sets
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, target]

X_test = data.loc[test_idx, feature_cols]
y_test = data.loc[test_idx, target]

#La mayoria es de una sola clase 0.73 porciento
print(y_train.value_counts(normalize=True))

#Lo mismo en los datos de prueba
print(y_test.value_counts(normalize=True))


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)



###Con Random Forest

from sklearn.ensemble import RandomForestClassifier

#Creamos nuestro modelo de random forest
RF = RandomForestClassifier(oob_score=True, 
                            random_state=42, 
                            warm_start=True,
                            n_jobs=-1)

oob_list = list()

#Para itinerar todos los posibles arboles
for n_trees in [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]:
    
    RF.set_params(n_estimators=n_trees)

    RF.fit(X_train, y_train)

    #Para obtener los errores
    oob_error = 1 - RF.oob_score_
    
    #La puntuacion    
    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))

rf_oob_df = pd.concat(oob_list, axis=1).T.set_index('n_trees')

#El error mas bajo se logra con 150 arboles
print(rf_oob_df)

import matplotlib.pyplot as plt
import seaborn as sns

#Para confirmar graficamente que el menor error fuera de bolsa es 150
sns.set_context('talk')
sns.set_style('white')

ax = rf_oob_df.plot(legend=False, marker='o', figsize=(14, 7), linewidth=5)
ax.set(ylabel='out-of-bag error')
plt.show()


###Con Extra Tree


from sklearn.ensemble import ExtraTreesClassifier

# Initialize the random forest estimator
# Note that the number of trees is not setup here
EF = ExtraTreesClassifier(oob_score=True, 
                          random_state=42, 
                          warm_start=True,
                          bootstrap=True,
                          n_jobs=-1)

oob_list = list()

#Para probar la diferente cantidad de arboles
for n_trees in [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]:
    
    EF.set_params(n_estimators=n_trees)
    EF.fit(X_train, y_train)

    # oob error
    oob_error = 1 - EF.oob_score_
    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))

et_oob_df = pd.concat(oob_list, axis=1).T.set_index('n_trees')

#Muestra los errores para extra tree
print(et_oob_df)

#Para mostrar la comparacion con datos de random forest y extra tree
oob_df = pd.concat([rf_oob_df.rename(columns={'oob':'RandomForest'}),
                    et_oob_df.rename(columns={'oob':'ExtraTrees'})], axis=1)

print(oob_df)


#La comparacion para mostrar que random forest logra mucho menor error que extra tree
sns.set_context('talk')
sns.set_style('white')

ax = oob_df.plot(marker='o', figsize=(14, 7), linewidth=5)
ax.set(ylabel='out-of-bag error')
plt.show()



# Random forest with 100 estimators
model = RF.set_params(n_estimators=100)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

#Muestra un resumen de todas las puntuaciones por clase
cr = classification_report(y_test, y_pred)
print(cr)

#Muetra un resumen tambien de las puntuaciones generales, las mas importantes
score_df = pd.DataFrame({'accuracy': accuracy_score(y_test, y_pred),
                         'precision': precision_score(y_test, y_pred),
                         'recall': recall_score(y_test, y_pred),
                         'f1': f1_score(y_test, y_pred),
                         'auc': roc_auc_score(y_test, y_pred)},
                         index=pd.Index([0]))

print(score_df)
       
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix,ConfusionMatrixDisplay

#Plot de la matriz de confusion
sns.set_context('talk')
cm = confusion_matrix(y_test, y_pred,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
disp.plot()
plt.show()                              
       


#Graficar la curva ROC y AUC
sns.set_context('talk')

fig, axList = plt.subplots(ncols=2)
fig.set_size_inches(16, 8)

# Get the probabilities for each of the two categories
y_prob = model.predict_proba(X_test)

# Plot the ROC-AUC curve
ax = axList[0]

fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
ax.plot(fpr, tpr, linewidth=5)
# It is customary to draw a diagonal dotted line in ROC plots.
# This is to indicate completely random prediction. Deviation from this
# dotted line towards the upper left corner signifies the power of the model.
ax.plot([0, 1], [0, 1], ls='--', color='black', lw=.3)
ax.set(xlabel='False Positive Rate',
       ylabel='True Positive Rate',
       xlim=[-.01, 1.01], ylim=[-.01, 1.01],
       title='ROC curve')
ax.grid(True)

# Plot the precision-recall curve
ax = axList[1]

precision, recall, _ = precision_recall_curve(y_test, y_prob[:,1])
ax.plot(recall, precision, linewidth=5)
ax.set(xlabel='Recall', ylabel='Precision',
       xlim=[-.01, 1.01], ylim=[-.01, 1.01],
       title='Precision-Recall curve')
ax.grid(True)

plt.tight_layout()
plt.show()        



#Grafico CIRCULO de las caracterisitcas mas importantes
feature_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(16, 6))
ax.pie(feature_imp, labels=None, autopct=lambda pct: '{:1.1f}%'.format(pct) if pct > 5.5 else '')
ax.set(ylabel='Relative Importance')
ax.set(xlabel='Feature')

# Adjust the layout to prevent label overlapping
plt.tight_layout()

# Move the legend outside the chart
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),labels=feature_imp.index)

plt.show()



########################################################################################################################

                                              #RANDOM FOREST

#Los bosques aleatorios tienen menos correlacion entre sus predicciones lo que mejora su accuracy.
#Una de las diferencias de Random Forest con bagging es que Random Forest al incluir mas modelos no sufre sobreajuste.

import pandas as pd
import pylab as plt
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

#Funcion de puntuaciones
def get_accuracy(X_train, X_test, y_train, y_test, model):
    return  {"test Accuracy":metrics.accuracy_score(y_test, model.predict(X_test)),"trian Accuracy": metrics.accuracy_score(y_train, model.predict(X_train))}


#Funcion de correlaciones
def get_correlation(X_test, y_test,models):
    #This function calculates the average correlation between predictors  
    n_estimators=len(models.estimators_)
    prediction=np.zeros((y_test.shape[0],n_estimators))
    predictions=pd.DataFrame({'estimator '+str(n+1):[] for n in range(n_estimators)})
    
    for key,model in zip(predictions.keys(),models.estimators_):
        predictions[key]=model.predict(X_test.to_numpy())
    
    corr=predictions.corr()
    print("Average correlation between predictors: ", corr.mean().mean()-1/n_estimators)
    return corr


###Ver diferencias de Random Forest con Bagging

churn_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv")

print(churn_df.head())

#Solo tomamos algunas columnas
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

print(churn_df.head())

X=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']]

M=X.shape[1]
#Muestra que M tiene 7 columnas
print(M)

m=3

#Muestra la que existe un rango de valores de 0 a 7
feature_index= range(M)
print(feature_index)

from sklearn.utils import resample
import random

#Remuestrea de 0 a 7 las columnas, puede salir cualquiera, con cantidad de columnas =3
random.sample(feature_index,m)

#Itinera 5 veces, solo pueden salir las primeras 5 filas
for n in range(5):

    print("sample {}".format(n))
    print(resample(X[0:5]).iloc[:,random.sample(feature_index,m)])

#Generas pequeñas submuestras de el datasetanterior

#Ahora con Random Forest modelo

y = churn_df['churn']
print(y.head())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)
print ('Train set', X_train.shape,  y_train.shape)
print ('Test set', X_test.shape,  y_test.shape)


#Comparacion de Random Forest con Bagging

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#Vemos la prediccion de un modelo de clasificacion bagging con arbol de decision
n_estimators=20
Bag= BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth = 4,random_state=2),n_estimators=n_estimators,random_state=0,bootstrap=True)


Bag.fit(X_train,y_train)

Bag.predict(X_test).shape

#En los datos de entrenamiento es bastante buena, pero en la prueba es mala no generaliza bien
print(get_accuracy(X_train, X_test, y_train, y_test,  Bag))

#La correlacion entre la prediccion es muy baja
get_correlation(X_test, y_test,Bag).style.background_gradient(cmap='coolwarm')

#Ahora con randomforest
from sklearn.ensemble import RandomForestClassifier

n_estimators=20

M_features=X.shape[1]

#Metodo para determinar el m que ocupamos
max_features=round(np.sqrt(M_features))-1
print(max_features)

print(y_test)

model = RandomForestClassifier( max_features=max_features,n_estimators=n_estimators, random_state=0)

model.fit(X_train,y_train)

#Tuvo mejor rendimiento, redujo el sobreajuste y tuvo mejor puntuacion
#Los datos de entrenamiento tienen puntuacion bastante buena, pero se reduce en los datos de prueba
print(get_accuracy(X_train, X_test, y_train, y_test, model))

#La correlacion sigue siendo bastante baja
get_correlation(X_test, y_test,model).style.background_gradient(cmap='coolwarm')


###Gridsearch mejores hiperparametros

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv")

print(df.head())

df= df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]

X =  df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]

print(X.head())

y=df['Class']
print(y.head())

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier()
model.get_params().keys()

param_grid = {'n_estimators': [2*n+1 for n in range(20)],
             'max_depth' : [2*n+1 for n in range(10) ],
             'max_features':["auto", "sqrt", "log2"]}

search = GridSearchCV(estimator=model, param_grid=param_grid,scoring='accuracy')
search.fit(X_train, y_train)

print(search.best_score_)

#Como mejor parametro encuentra profunidad=15, max_feature=log2, cantidad de estimadores=31
print(search.best_params_)

print(get_accuracy(X_train, X_test, y_train, y_test, search.best_estimator_))


from sklearn import preprocessing
import pandas as pd
import pylab as plt
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
def get_accuracy(X_train, X_test, y_train, y_test, model):
    return  {"test Accuracy":metrics.accuracy_score(y_test, model.predict(X_test)),"trian Accuracy": metrics.accuracy_score(y_train, model.predict(X_train))}
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv", delimiter=",")
print(df.head())

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

y = df["Drug"]
print(y[0:5])

#Codificando las columnas



le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

print(X[0:5])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

param_grid = {'n_estimators': [2*n+1 for n in range(20)],
             'max_depth' : [2*n+1 for n in range(10) ],
             'max_features':["auto", "sqrt", "log2"]}

model = RandomForestClassifier()

search = GridSearchCV(estimator=model, param_grid=param_grid,scoring='accuracy', cv=3)
search.fit(X_train, y_train)

print(search.best_score_)

#Los mejores hiperparametros que encontro: profundiad=7, max_features=auto, numero de estimadores=19
print(search.best_params_)

print(get_accuracy(X_train, X_test, y_train, y_test, search.best_estimator_))      


#######################################################################################################################

                                         #Stacking (apilamiento)
                             #Encontrar mejores hiperparametros con GridSearch

#Stacking toma varios modelos de clasificacion como entrada, los llama learners (aprendices) y con estos
#Crea un nuevo clasificador unico meta-clasificador

#La idea de tomar todos los datos para todos los clasificadores no funciona esto crea sobreajuste, por lo que se toman
#pliegues

#Cada columna puede tomar un clasificador diferente

import pandas as pd
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

def get_accuracy(X_train, X_test, y_train, y_test, model):
    return  {"test Accuracy":metrics.accuracy_score(y_test, model.predict(X_test)),"trian Accuracy": metrics.accuracy_score(y_train, model.predict(X_train))}

#Tomamos el dataset del vino

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/wine.data",names= ['Class', 'Alcohol', 'Malic acid', 'Ash',
         'Alcalinity of ash' ,'Magnesium', 'Total phenols',
         'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',     'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
         'Proline'])

print(df.head())

print(df.dtypes)

print(df['Class'].unique())

#Para sacar las correlaciones de la data
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

#Existen muy pocas que podrian ser regressiones
#sns.pairplot(df, hue="Class", diag_kws={'bw': 0.2})
#plt.show()

#La lista dos todas las columnas
features=list(df)
print(features[1:])

#Definiendo X e y
y=df[features[0]]
X=df[features[1:]]

#Escalamos X
scaler = preprocessing.StandardScaler().fit(X)
X= scaler.transform(X)

print(X.mean(axis=0))

print(X.std(axis=0))

#Separamos en datos de entrenamiento y de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)
print ('Train set', X_train.shape,  y_train.shape)
print ('Test set', X_test.shape,  y_test.shape)


#Creamos un diccionario con los estimadores
estimators = [('SVM',SVC(random_state=42)),('KNN',KNeighborsClassifier()),('dt',DecisionTreeClassifier())]
print(estimators)

#Apilamos los estimadores, para el meta-clasificador necesita un parametro, lo ponemos con regression logistica
clf = StackingClassifier( estimators=estimators, final_estimator= LogisticRegression())
clf.fit(X_train, y_train)
print(clf)

yhat=clf.predict(X_test)
print(yhat)

#El accuracy es bastante bueno para los datos de entrenamiento como los de prueba
print(get_accuracy(X_train, X_test, y_train, y_test, clf))

#La puntuacion es bastante alta
print(metrics.accuracy_score(y_test, yhat))

#Proponemos otro ejemplo, en este caso la puntuacion es variable, en cada intento da diferentes puntuaciones
#pero la puntuacione es muy buena, puede llegar a ser perfecta
estimators2 = [('SVM',SVC(random_state=42)),('KNN',KNeighborsClassifier()),('lr',LogisticRegression())]
clf2 = StackingClassifier( estimators=estimators2, final_estimator= DecisionTreeClassifier())
clf2.fit(X_train, y_train)
yhat2 = clf2.predict(X_test)

print(get_accuracy(X_train, X_test, y_train, y_test, clf2))
print(metrics.accuracy_score(y_test, yhat2))
#Esto tambien es propenso a sobreajuste


###Ejemplo 2 con GridSearch

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv", delimiter=",")
df.head()

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

y = df["Drug"]
print(y[0:5])


from sklearn import preprocessing

#Codigicando las columnas
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

print(X[0:5])

scaler = preprocessing.StandardScaler().fit(X)
X= scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

estimators = [('SVM',SVC(random_state=42)),('knn',KNeighborsClassifier()),('dt',DecisionTreeClassifier())]
print(estimators)

clf = StackingClassifier( estimators=estimators, final_estimator= LogisticRegression())
print(clf)

#Le proponemos los hipermarametros de los modelos que elijamos para el stacking
param_grid = {'dt__max_depth': [n for n in range(10)],'dt__random_state':[0],'SVM__C':[0.01,0.1,1],'SVM__kernel':['linear', 'poly', 'rbf'],'knn__n_neighbors':[1,4,8,9] }

search = GridSearchCV(estimator=clf, param_grid=param_grid,scoring='accuracy')
search.fit(X_train, y_train)

#Tiene puntcuacion perfecta
print(search.best_score_)

#Los mejores hiperparametros son: SVM__C=0.01, SVM_kernel="linear", profundidad=4, random_state=0, knn_n=1
print(search.best_params_)

#La puntuacion es bastante buena casi perfecta
print(get_accuracy(X_train, X_test, y_train, y_test, search))


#######################################################################################################################

                                        #BOOSTING Y STACKING

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/Human_Activity_Recognition_Using_Smartphones_Data.csv", sep=',')

print(data.shape)

print(data.dtypes.value_counts())

float_columns = (data.dtypes == float)

#Verificar si los datos estan escalados
print( (data.loc[:,float_columns].max()==1.0).all() )
print( (data.loc[:,float_columns].min()==-1.0).all() )

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['Activity'] = le.fit_transform(data['Activity'])

print(le.classes_)

print(data.Activity.unique())

#Separamos datos de entrenamiento y datos de prueba
from sklearn.model_selection import train_test_split


feature_columns = [x for x in data.columns if x != 'Activity']

X_train, X_test, y_train, y_test = train_test_split(data[feature_columns], data['Activity'],
                 test_size=0.3, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

error_list = list()

#Este proceso se demoro alrededor de 1 hora
#Vemos cual es le mejor cantidad de arboles
#Habia mas opciones lo reduci para que no se demorara tanto
tree_list = [400]
#El menor error lo tiene el de 400 arboles
for n_trees in tree_list:
    
    # Initialize the gradient boost classifier
   GBC = GradientBoostingClassifier(n_estimators=n_trees, random_state=42)

    #Fit the model
   print(f'Fitting model with {n_trees} trees')
   GBC.fit(X_train.values, y_train.values)
   y_pred = GBC.predict(X_test)

    # Get the error
   error = 1.0 - accuracy_score(y_test, y_pred)
    
    # Store it
   error_list.append(pd.Series({'n_trees': n_trees, 'error': error}))

error_df = pd.concat(error_list, axis=1).T.set_index('n_trees')

print(error_df)

#sns.set_context('talk')
#sns.set_style('white')

# Create the plot
#ax = error_df.plot(marker='o', figsize=(12, 8), linewidth=5)

# Set parameters
#ax.set(xlabel='Number of Trees', ylabel='Error')
#ax.set_xlim(0, max(error_df.index)*1.1)
#plt.show()

#Encontrar los mejores parametro y tasas de aprendizaje
from sklearn.model_selection import GridSearchCV

# The parameters to be fit
param_grid = {'n_estimators': tree_list,
              'learning_rate': [0.1, 0.01, 0.001, 0.0001],
              'subsample': [0.5],
              'max_features': [4]}

# The grid search object
GV_GBC = GridSearchCV(GradientBoostingClassifier(random_state=42), 
                      param_grid=param_grid, 
                      scoring='accuracy',
                      n_jobs=-1)

# Do the grid search
GV_GBC = GV_GBC.fit(X_train, y_train)

#Los mejores hiperparametros son, maxima caracterisitcas=4, numero de estimadores=400, random state=42, subsample=0.5
print(GV_GBC.best_estimator_)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
y_pred = GV_GBC.predict(X_test)

#Las puntuaciones para todas las clases es bastante buena, la mejor con la ultima
print(classification_report(y_pred, y_test))

#Muy grafico muy represetentativos, la cantidad de aciertos es muy alta
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.show()

###Con AdaBoost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))

param_grid = {'n_estimators': [100],
              'learning_rate': [0.01]}

GV_ABC = GridSearchCV(ABC,
                      param_grid=param_grid, 
                      scoring='accuracy',
                      n_jobs=-1)

GV_ABC = GV_ABC.fit(X_train, y_train)

#Los mejores hiperparametros son profunidad=1, learning_rate=0.01, n_estimadores=100
print(GV_ABC.best_estimator_)

y_pred = GV_ABC.predict(X_test)

#La puntuacion es buena, pero es la peor de los 4 modelos
print(classification_report(y_pred, y_test))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.show()


###Con VoltingClassifier

from sklearn.linear_model import LogisticRegression

# L2 regularized logistic regression
LR_L2 = LogisticRegression(penalty='l2', max_iter=500, solver='saga').fit(X_train, y_train)

y_pred = LR_L2.predict(X_test)

#La regression logistica por si solo tiene buenos resultados
print(classification_report(y_pred, y_test))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.show()

from sklearn.ensemble import VotingClassifier

# The combined model--logistic regression and gradient boosted trees
estimators = [('LR_L2', LR_L2), ('GBC', GV_GBC)]

# Though it wasn't done here, it is often desirable to train 
# this model using an additional hold-out data set and/or with cross validation
VC = VotingClassifier(estimators, voting='soft')
VC = VC.fit(X_train, y_train)

y_pred = VC.predict(X_test)

#Mejora los resulados mas que regression logistica por si sola
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.show()

sns.set_context('talk')
cm = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cm, annot=True, fmt='d')
plt.show()



#######################################################################################################################

                                      #Gradiente boosting para clasificadores
                                              #XBOOST GRADIENTE
                                      
#Gradiente boosting es una MODELO ADITIVO como funcion de perdida                               
#Adaboost es un caso particular de gradiente boosting clasificacion                                         

#Gradiente boosting ocupa diferente funciones de perdida, se ocupa para reducir el efecto
#de sobre ajuste de arboles de decision por ejemplo

#En general comparandola con otros modelos, es de los que obtiene mejores resultados en general con random forest tuneados

import pandas as pd
import pylab as plt
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def get_accuracy(X_train, X_test, y_train, y_test, model):
    return  {"test Accuracy":metrics.accuracy_score(y_test, model.predict(X_test)),"train Accuracy": metrics.accuracy_score(y_train, model.predict(X_train))}

def get_accuracy_boost(X,y,title,times=20,xlabel='Number Estimators',Learning_rate_=[0.2,0.4,0.6,1], n_est = 100):

    lines_array=['solid','--', '-.', ':']

    N_estimators=[n*2 for n in range(1,n_est//2)]
    
    train_acc=np.zeros((times,len(Learning_rate_),len(N_estimators)))
    test_acc=np.zeros((times,len(Learning_rate_),len(N_estimators)))


    #Iterate through different number of Learning rate  and average out the results  
    
    for n in tqdm(range(times)):
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)
        for n_estimators in N_estimators:
            for j,lr in enumerate(Learning_rate_):


                model = XGBClassifier(objective=objective,learning_rate=lr,n_estimators=n_estimators,eval_metric='mlogloss')


                model.fit(X_train,y_train)



                Accuracy=get_accuracy(X_train, X_test, y_train, y_test,  model)



                train_acc[n,j,(n_estimators//2)-1]=Accuracy['train Accuracy']
                test_acc[n,j,(n_estimators//2)-1]=Accuracy['test Accuracy']
    



    fig, ax1 = plt.subplots()
    mean_test=test_acc.mean(axis=0)
    mean_train=train_acc.mean(axis=0)
    ax2 = ax1.twinx()

    for j,(lr,line) in enumerate(zip(Learning_rate_,lines_array)): 

        ax1.plot(mean_train[j,:],linestyle = line,color='b',label="Learning rate "+str(lr))
        ax2.plot(mean_test[j,:],linestyle = line, color='r',label=str(lr))

    ax1.set_ylabel('Training accuracy',color='b')
    ax1.legend()
    ax2.set_ylabel('Testing accuracy', color='r')
    ax2.legend()
    ax1.set_xlabel(xlabel)
    plt.show()

###XGBOOST

churn_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv")

churn_df.head()

#Solo tomamos algunas columnas
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())

X=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']]

y = churn_df['churn']
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)
print ('Train set', X_train.shape,  y_train.shape)
print ('Test set', X_test.shape,  y_test.shape)

from xgboost import XGBClassifier

n_estimators=5
random_state=0
objective='binary:logistic'
learning_rate=0.1

model =XGBClassifier(objective=objective,learning_rate=learning_rate,n_estimators=n_estimators,eval_metric='mlogloss')

print("learning rate:", model.learning_rate)
print("lobjective:", model.objective)
print("n_estimators:", model.n_estimators)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred 

#Con learning_rate=0.1, n_estimators=5, objetive="logistic"
#Puntuacion: train=0.77, test=0.76
print(get_accuracy(X_train, X_test, y_train, y_test,  model))

#Camiando el learning_rate=0.3
learning_rate=0.3
model =XGBClassifier(objective=objective,learning_rate=learning_rate,n_estimators=n_estimators)
model.fit(X_train, y_train)
#En este caso aumento la puntuacion es un caso particular, una tasa 0.3 es alta aumenta el sobreajuste en general
#puntuacion: train= 0.92, test=0.81
print(get_accuracy(X_train, X_test, y_train, y_test,  model))

#Cambiando el n_estimators la puntuacion es: train=1.0 , test=0.73
n_estimators=100
model =XGBClassifier(objective=objective,learning_rate=learning_rate,n_estimators=n_estimators)
model.fit(X_train, y_train)
print(get_accuracy(X_train, X_test, y_train, y_test,  model))

#Plot que muestra lineas tanto para los estimadores en los dos casos de entrenamiento y prueba
#Probando diferentes cantidad de estimadores
get_accuracy_boost(X,y,title="Training and Test Accuracy vs Weak Classifiers",times=10,xlabel='Number Estimators', n_est = 10)



eval_metric="error"
eval_set = [(X_test, y_test)]

#verbose=True es para mostrar el resultado de cada iteracion
model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

#Para tomar todos los resultados
results = model.evals_result()
print(results)

#Plotea todas las iteraciones
plt.plot(range(0, len(results['validation_0']['error'])), results['validation_0']['error'])
plt.xlabel('iterations')
plt.ylabel('Misclassified Samples')

#Con funcion de perdida logaritmica para entrenamiento funciona, pero de prueba no parece logaritmica
eval_metric='logloss'
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set,verbose=False)
results=model.evals_result()

results.keys()

fig, ax = plt.subplots()
ax.plot( results['validation_0']['logloss'], label='Train')
ax.plot( results['validation_1']['logloss'], label='Test/Validation')
ax.legend()
plt.show()
#Se muestran totalmente diferente las curvas de error, esto es muestra de sobreajuste

###PARADA TEMPRANA

#Como el sobreajuste en un principio deciende en la prueba, luego sube infinitamente, al momento que deje de bajar
#debemos parar, es un truco para evitar el sobreajuste

#Vemos que en los datos de prueba es bastante baja la puntuacion
print(get_accuracy(X_train, X_test, y_train, y_test,  model))

#Ponemos un valor para que nos muestre el error
early_stopping_rounds=10

eval_set = [(X_test, y_test)]
eval_metric='logloss'

#verbose muestra los errores que va teniendo gasta 10 iteraciones
#automaticamente reconoce cual es la menor iteracion y para en el modelo
model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True,early_stopping_rounds=early_stopping_rounds)

#La puntucacion es de la menor iteracion, mejora en comparacion en la anterior sube su puntuacion
print(get_accuracy(X_train, X_test, y_train, y_test,  model))

###Parametros para arboles

#Es una prueba de parametors, el mejor es el inicial en puntuacion

objective='binary:logistic'
learning_rate=0.1
n_estimators=10
model =XGBClassifier(objective=objective,learning_rate=learning_rate,n_estimators=n_estimators,eval_metric='mlogloss')
model.fit(X_train, y_train)
print(get_accuracy(X_train, X_test, y_train, y_test,  model))

max_depth=3

model =XGBClassifier(objective=objective,learning_rate=learning_rate,n_estimators=n_estimators,eval_metric='mlogloss',max_depth=max_depth)
model.fit(X_train, y_train)
print(get_accuracy(X_train, X_test, y_train, y_test,  model))

#min_chilg_weight es el numero minimo de instancias que necesita para cada arbol pequeño
min_child_weight=4

model =XGBClassifier(objective=objective,learning_rate=learning_rate,n_estimators=n_estimators,eval_metric='mlogloss',min_child_weight=4)
model.fit(X_train, y_train)
print(get_accuracy(X_train, X_test, y_train, y_test,  model))

#Son parametros de regulacion para reducir el sobreajuste

gamma=1
reg_lambda=2
alpha=1
model =XGBClassifier(objective=objective,learning_rate=learning_rate,n_estimators=n_estimators,eval_metric='mlogloss',gamma=gamma,reg_lambda=reg_lambda,alpha=alpha)
model.fit(X_train, y_train)
print(get_accuracy(X_train, X_test, y_train, y_test,  model))


###Ejemplo 2

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv")

print(df.head())

df= df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]

#Tomamos solo estas columnas
X =  df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]

print(X.head())

X=X.astype(int)
from sklearn import preprocessing
y=df['Class']
le_sex1 = preprocessing.LabelEncoder()
y = le_sex1.fit_transform(y)
print(pd.DataFrame(y))

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.model_selection import GridSearchCV

model =XGBClassifier(objective='binary:logistic',eval_metric='mlogloss')

param_grid = {'learning_rate': [0.1*(n+1) for n in range(5)],
             'n_estimators' : [2*n+1 for n in range(5)]}
             

search = GridSearchCV(estimator=model, param_grid=param_grid,scoring="neg_log_loss")
search.fit(X_train, y_train)

print(search.best_score_)

#Los mejores parametros son learning_rate=0.5, n_estimators=9
print(search.best_params_)

#La puntuacion es buena
print(get_accuracy(X_train, X_test, y_train, y_test, search.best_estimator_))


###Ejemplo 3

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv", delimiter=",")
print(df.head())

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

y = df["Drug"]
le_sex2 = preprocessing.LabelEncoder()
y = le_sex2.fit_transform(y)
print(y[0:5])

#Codificamos los datos
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

print(X[0:5])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

param_grid = {'learning_rate': [0.1*(n+1) for n in range(2)],
             'n_estimators' : [2*n+1 for n in range(2)] }

#Como solo hay dos clases lo hacemos con regresion logistica binaria
model =XGBClassifier(objective='binary:logistic',eval_metric='mlogloss')

#Buscamos los mejores hiperparametros
search = GridSearchCV(estimator=model, param_grid=param_grid,scoring="neg_log_loss")
search.fit(X_train, y_train)

#Me dio negativa por alguna razon
print(search.best_score_)

#Mejores parametros learning_rate=0.2, n_estimator=3
print(search.best_params_)

#Puntuacion bastante buena
print(get_accuracy(X_train, X_test, y_train, y_test, search.best_estimator_))


###Como trabaja el gradiente

x = np.linspace(-1, 2)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, x**2, linewidth=2)
ax1.set_title('convex function')
ax2.plot(x, x**2 + np.exp(-5*(x - .5)**2), linewidth=2)
ax2.set_title('non-convex function')
plt.show()

plt.plot(x, x**2, linewidth=2)
plt.text(-.7, 3, '$ \ell(H_{T}+h_{t},y_i)$', size=20)
plt.plot(x, 2*x - 1)
plt.plot(1, 1, 'k+')
plt.text(.3, -.75, 'approximation', size=15)
plt.show()

X=np.linspace(0, 3,num=100)
 
y=np.zeros(X.shape)
y[X>1]=1
y[X>2]=2
plt.plot(X,y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

reg = DecisionTreeRegressor(max_depth=1)
reg.fit(X.reshape(-1,1),y)
h_1=reg.predict(X.reshape(-1,1))

plt.plot(X,y,label="y")
plt.plot(X,h_1,label="$h_{1}(x)$")
plt.legend()
plt.show()

gamma=1
r=y-gamma*reg.predict(X.reshape(-1,1))

def predict(y,weak_learners,gamma):
    yhat=np.zeros(y.shape)
    for h in weak_learners:
        yhat+=h.predict(X.reshape(-1,1))
    return yhat

weak_learners =[]
gamma=1
r=y
weak_learners.append(reg)
for t_ in range(0,10):
    #train weak learner 
    reg=DecisionTreeRegressor(max_depth=1)
    reg.fit(X.reshape(-1,1),r)
    weak_learners.append(reg)
    
    #Calculate r_i,t for each iteration  
    r=r-gamma*reg.predict(X.reshape(-1,1))
    #plot function   
    plt.plot(X,y,label="y")
    plt.plot(X,predict (y,weak_learners,gamma),label="$H_{}(x)$".format(t_+1))
    plt.legend()
    plt.show()
    


####################################################################################################################### 
    
                                           #Adaboost (Adaptative Boost)
                                               
#Adaboost es parte de la familia de algoritmos de boosteo como bagging y random forest
#Adaboost combina muchos clasificadores donde cada uno de ellos tiene un rendimiento menor,
#lo que se llama aprendiz mas ligero, todos combinados logran un gran clasificador mas potente
#Para ajustar adaboost se requiere ajustar hiperparametros, pero es mucho mas rapido que otros boost
#La combinacion de clasificadores, es una regresion lineal de clasificadores debiles, en este caso
#al agregar mas estimadores hay mas riesgo de sobreajuste.
#Pero entre mas clasificadores debiles, la precision es mejor
                               
import pandas as pd
import pylab as plt
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm                                               
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def get_accuracy(X_train, X_test, y_train, y_test, model):
    return  {"test Accuracy":metrics.accuracy_score(y_test, model.predict(X_test)),"train Accuracy": metrics.accuracy_score(y_train, model.predict(X_train))}

def get_accuracy_bag(X,y,title,times=20,xlabel='Number Estimators',Learning_rate_=[0.2,0.4,0.6,1]):

    lines_array=['solid','--', '-.', ':']

    N_estimators=[n for n in range(1,100)]
    
    times=20
    train_acc=np.zeros((times,len(Learning_rate_),len(N_estimators)))
    test_acc=np.zeros((times,len(Learning_rate_),len(N_estimators)))


    #Iterate through different number of Learning rate  and average out the results  
    for n in tqdm(range(times)):
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)
        for n_estimators in N_estimators:
            for j,lr in enumerate(Learning_rate_):


                model = AdaBoostClassifier(n_estimators=n_estimators+1,random_state=0,learning_rate=lr)


                model.fit(X_train,y_train)



                Accuracy=get_accuracy(X_train, X_test, y_train, y_test,  model)



                train_acc[n,j,n_estimators-1]=Accuracy['train Accuracy']
                test_acc[n,j,n_estimators-1]=Accuracy['test Accuracy']




    fig, ax1 = plt.subplots()
    mean_test=test_acc.mean(axis=0)
    mean_train=train_acc.mean(axis=0)
    ax2 = ax1.twinx()

    for j,(lr,line) in enumerate(zip(Learning_rate_,lines_array)): 

        ax1.plot(mean_train[j,:],linestyle = line,color='b',label="Learning rate "+str(lr))
        ax2.plot(mean_test[j,:],linestyle = line, color='r',label=str(lr))

    ax1.set_ylabel('Training accuracy',color='b')
    ax1.set_xlabel('No of estimators')
    ax1.legend()
    ax2.set_ylabel('Testing accuracy', color='r')
    ax2.legend()
    plt.show()

churn_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv")

print(churn_df.head())

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())

X=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']]

y = churn_df['churn']
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)
print ('Train set', X_train.shape,  y_train.shape)
print ('Test set', X_test.shape,  y_test.shape)

from sklearn.ensemble import AdaBoostClassifier

n_estimators=5
random_state=0

model = AdaBoostClassifier(n_estimators=n_estimators,random_state=random_state)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred) 

#Una puntuacion regular
print(get_accuracy(X_train, X_test, y_train, y_test,  model))

model.base_estimator_

model.estimators_

[ ("for weak classifiers {} the we get ".format(i+1),get_accuracy(X_train, X_test, y_train, y_test,  weak_classifiers)) for i,weak_classifiers in enumerate(model.estimators_)]

n_estimators=100
random_state=0

model = AdaBoostClassifier(n_estimators=n_estimators,random_state=random_state)
model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

#Mejora en datos de entrenamiento con mas estimadores, pero no mejora, incluso baja mas
print(get_accuracy(X_train, X_test, y_train, y_test, model))

#Muestra que existe sobreajuste, las curvas de entrenamiento y prueba no tienen relacion
get_accuracy_bag(X,y,title="Training and Test Accuracy vs Weak Classifiers",Learning_rate_=[1],times=20,xlabel='Number Estimators')

n_estimators=100
random_state=0
learning_rate=0.7

model = AdaBoostClassifier(n_estimators=n_estimators,random_state=random_state,learning_rate=learning_rate)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(get_accuracy(X_train, X_test, y_train, y_test, model))

#Muestra que entre mas pequeño la tasa de aprenzaje mejonos sobreajuste
get_accuracy_bag(X,y,title="Training and Test Accuracy vs Weak Classifiers",Learning_rate_=[0.2,0.4,0.6,1],times=20,xlabel='Number Estimators')


###Cambiando algoritmo base

from sklearn.svm import SVC

base_estimator=SVC(kernel='rbf',gamma=1)

base_estimator.fit(X_train, y_train)

print(get_accuracy(X_train, X_test, y_train, y_test, base_estimator))

#print(base_estimator.predict_proba(X_train))

algorithm='SAMME'

model =AdaBoostClassifier(n_estimators=5, base_estimator=base_estimator,learning_rate=1,algorithm='SAMME' )

model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)
print(get_accuracy(X_train, X_test, y_train, y_test, model))

###Ejemplo 2

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv")

df.head()

df= df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]

X =  df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]

X.head()

y=df['Class']
y.head()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.model_selection import GridSearchCV

model = AdaBoostClassifier()
model.get_params().keys()

param_grid = {'learning_rate': [0.1*(n+1) for n in range(10)],
             'n_estimators' : [2*n+1 for n in range(10)],
              'algorithm':['SAMME', 'SAMME.R']}  

search = GridSearchCV(estimator=model, param_grid=param_grid,scoring='accuracy')
search.fit(X_train, y_train)

print(search.best_score_)

print(search.best_params_)

print(get_accuracy(X_train, X_test, y_train, y_test, search.best_estimator_))

###Ejemplo 3

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv", delimiter=",")
print(df.head())

#Ocupamos solo algunas columnas para X
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

y = df["Drug"]
print(y[0:5])

#Codificamos
from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

print(X[0:5])

#Separamos en datos de entrenamiento
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

model = RandomForestClassifier()   

param_grid = {'learning_rate': [0.1*(n+1) for n in range(10)],
             'n_estimators' : [2*n+1 for n in range(10)],
              'algorithm':['SAMME', 'SAMME.R']} 
  
           

#Encontramos los mejores hiperparametros          
search = GridSearchCV(estimator=model, param_grid=param_grid,scoring='accuracy', cv=3)
search.fit(X_train, y_train) 

print(search.best_score_)

print(search.best_params_)

print(get_accuracy(X_train, X_test, y_train, y_test, search.best_estimator_))


#######################################################################################################################

                                           #Modelo explicativo

#Un modelo explicativo intenta explicar modelos que ya no es posible interpretar o se entienden muy poco

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import lime.lime_tabular

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML201EN-SkillsNetwork/labs/module_4/datasets/hr_new_job_processed.csv"
job_df=pd.read_csv(url)

print(job_df.describe())

#X son 11 columas
X = job_df.loc[:, job_df.columns != 'target']
y = job_df[['target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = 12)

black_box_model = RandomForestClassifier(random_state = 123, max_depth=25, 
                             max_features=10, n_estimators=100, 
                             bootstrap=True)

black_box_model.fit(X_train, y_train.values.ravel())

y_blackbox = black_box_model.predict(X_test)

#Tuvo una puntuacion de 0.8 el cual es regular
print(metrics.roc_auc_score(y_test, y_blackbox))


###Permutacion de caracterisitcas mas importantes

#Una de las formas de explicar un modelo es atraves de sus caracterisitcas mas importantes
#Mezcalmos los valores de estas caracteristicas y vemos el resultado
#La puntuacion del modelo se hara antes y despues del modelo

feature_importances = permutation_importance(estimator=black_box_model, X = X_train, y = y_train, n_repeats=5,
                                random_state=123, n_jobs=2)


#Muestra las dimensiones matriz con las puntuaciones de importancia
print(feature_importances.importances.shape)

#Muestra la matriz explicitamente
print(feature_importances.importances)

#En este caso la matriz tiene 11 filas, de 11 caracteristicas, por 5 columnas, cada columna es una permutacion

def visualize_feature_importance(importance_array):
    # Sort the array based on mean value
    sorted_idx = importance_array.importances_mean.argsort()
    # Visualize the feature importances using boxplot
    fig, ax = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(10)
    fig.tight_layout()
    ax.boxplot(importance_array.importances[sorted_idx].T,
               vert=False, labels=X_train.columns[sorted_idx])
    ax.set_title("Permutation Importances (train set)")
    plt.show()

#Saca un promedio de las puntuaciones en general varian super poco, asi que se mantienen estables a pesar de la
#cantidad de iteraciones
visualize_feature_importance(feature_importances)

#Con 10 repeticiones, es la cantidad de permutaciones puestas explicitamente
feature_importances = permutation_importance(estimator=black_box_model, X = X_train, y = y_train, n_repeats=10,
                                random_state=123, n_jobs=2)
                                
print(feature_importances.importances.shape)

#Graficando las permutaciones
visualize_feature_importance(feature_importances)


###Grafico de dependencia parcial

#Al existir para un modelo muchas columnas, es posible que en un limite no se puedan graficar tantes, por lo que
#solo tomaremos las mas importantes a libre discrecion en cuanto a la cantidad

#Tomamos estas columnas y las graficamos esta, consigo misma despues de pasar por el modelo

#Tomamos las dos caracterisitcas mas importantes, tienen relaciones lineales aproximadamente negativas
important_features = ['city_development_index', 'experience']
"arguments: "
" - estimator: the black box model"
" - X is the training data X"
" - features are the important features we are interested"
#Lo de arriba nos muestra las entradas que necesita

#Grafica la dependencia de las dos caracteristicas mas importantes
PartialDependenceDisplay.from_estimator(estimator=black_box_model, 
                        X=X_train, 
                        features=important_features,
                        random_state=123)
plt.show()

#solo horas de entrenamiento se comporta como una relacion lineal, las demas casi nada
important_features = ['company_size', 'education_level', 'training_hours']

PartialDependenceDisplay.from_estimator(estimator=black_box_model, 
                        X=X_train, 
                        features=important_features,
                        random_state=123)
plt.show()


### EXPLICANDO CON REGRESSION LOGISITCA

#Nomralizamos los datos
min_max_scaler = StandardScaler()
X_test_minmax = min_max_scaler.fit_transform(X_test)

#Le agregamos una penalizacion para explicar mejor
lm_surrogate = LogisticRegression(max_iter=1000, 
                                  random_state=123, penalty='l1', solver='liblinear')
lm_surrogate.fit(X_test_minmax, y_blackbox)

y_surrogate = lm_surrogate.predict(X_test_minmax)

#Obtiene una puntuacio de 0.75, significa que pudo interpretar el 75% de los datos de caja negra
print(metrics.accuracy_score(y_blackbox, y_surrogate))


#Ahora tomamos solo los coeficientes de la regression, para poder explicarlo mejor
def get_feature_coefs(regression_model):
    coef_dict = {}
    # Filter coefficients less than 0.01
    for coef, feat in zip(regression_model.coef_[0, :], X_test.columns):
        if abs(coef) >= 0.01:
            coef_dict[feat] = coef
    # Sort coefficients
    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    return coef_dict

#surrogate son las predicciones, muestra todos los coeficientes
coef_dict = get_feature_coefs(lm_surrogate)
print(coef_dict)

#Para la funcion de abajo y poder graficarla
def get_bar_colors(values):
    color_vals = []
    for val in values:
        if val <= 0:
            color_vals.append('r')
        else:
            color_vals.append('g')
    return color_vals

#Una funcion para visualizar los coeficientes
def visualize_coefs(coef_dict):
    features = list(coef_dict.keys())
    values = list(coef_dict.values())
    y_pos = np.arange(len(features))
    color_vals = get_bar_colors(values)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y_pos, values, align='center', color=color_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    # labels read top-to-bottom
    ax.invert_yaxis()  
    ax.set_xlabel('Feature Coefficients')
    ax.set_title('')
    plt.show()

#Grafica los coeficientes mostrando que una feature tiene el coeficiente por lejos mas grande
visualize_coefs(coef_dict)

### EXPLICACION CON ARBOL DE DECISION

tree_surrogate = DecisionTreeClassifier(random_state=123, 
                                         max_depth=5, 
                                         max_features=10)

tree_surrogate.fit(X_test, y_blackbox)
y_surrogate = tree_surrogate.predict(X_test)

#Mejoro la puntucacion con respecto al anterior
print(metrics.accuracy_score(y_blackbox, y_surrogate))

#Es solo el texto de un arbol
tree_exp = export_text(tree_surrogate, feature_names=list(X_train.columns))

print(tree_exp)


###LIME (para explicar modelos de caja negra)

#Se genera un dataset modificado a traves de las permutaciones
#Se crean dos datasets nuevos
#Uno se crea con pesos para las instancias
#El otro se crea con un modelo de caja negra que crea predicciones (estas predicciones son conclusiones sobre varias features, es como un resumen, del datasets con una conclusion) (intenta explicar lo sucedido)
#Luego se toman las diferencias

explainer = lime.lime_tabular.LimeTabularExplainer(
    # Set the training dataset to be X_test.values (2-D Numpy array)
    training_data=X_test.values,
    # Set the mode to be classification
    mode='classification',
    # Set class names to be `Not Changing` and `Changing`
    class_names = ['Not Changing', 'Changing'],
    # Set feature names
    feature_names=list(X_train.columns),
    random_state=123,
    verbose=True)

instance_index = 19
selected_instance = X_test.iloc[[instance_index]]
lime_test_instance = selected_instance.values.reshape(-1)
print(selected_instance)

exp = explainer.explain_instance(
                                 # Instance to explain
                                 lime_test_instance, 
                                 # The prediction from black-box model
                                 black_box_model.predict_proba,
                                 # Use max 10 features
                                 num_features=10)

#Muestra un grafico de barras que es un VS, los rojos es una creencia y los verdes otro creencia opuesta, en este caso
#como el modelo de caja negra respondio que en conclusion es poco probable que los empleados dejen la empresa, con lime
#se intenta explicar lo mismo ratificando en este VS que la opcion roja es la mayoria de los motivos indican que los
#empleados no dejaran la empresa
exp.as_pyplot_figure()
plt.show()



#######################################################################################################################

                                            #Datos desequilibrados

import pandas as pd
import numpy as np 
import imblearn
from matplotlib.pyplot import figure
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from collections import Counter


rs = 123
# Grid search hyperparameters for a logistic regression model
def grid_search_lr(X_train, y_train):
    params_grid = {
    'class_weight': [{0:0.05, 1:0.95}, {0:0.1, 1:0.9}, {0:0.2, 1:0.8}]
    }
    lr_model = LogisticRegression(random_state=rs, max_iter=1000)
    grid_search = GridSearchCV(estimator = lr_model, 
                           param_grid = params_grid, 
                           scoring='f1',
                           cv = 5, verbose = 1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

# Grid search hyperparameters for a random forest model
def grid_search_rf(X_train, y_train):
    params_grid = {
    'max_depth': [5, 10, 15, 20],
    'n_estimators': [25, 50, 100],
    'min_samples_split': [2, 5],
    'class_weight': [{0:0.1, 1:0.9}, {0:0.2, 1:0.8}, {0:0.3, 1:0.7}]
    }
    rf_model = RandomForestClassifier(random_state=rs)
    grid_search = GridSearchCV(estimator = rf_model, 
                           param_grid = params_grid, 
                           scoring='f1',
                           cv = 5, verbose = 1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

def split_data(df):
    X = df.loc[ : , df.columns != 'Class']
    y = df['Class'].astype('int')
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)

credit_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML201EN-SkillsNetwork/labs/module_4/datasets/im_credit.csv", index_col=False)

print(credit_df.head())

print(credit_df['Class'].value_counts())

#Muestra que de una clase hay 200000 valores y en la otra sola 200
credit_df['Class'].value_counts().plot.bar(color=['green', 'red'])
plt.show()

#Un modelo rapido para ver como funciona con datos desequilibrados
X_train, X_test, y_train, y_test = split_data(credit_df)

model = LogisticRegression(random_state=rs, 
                              max_iter = 1000)

# Train the model
model.fit(X_train, y_train)
preds = model.predict(X_test)

#Dice que el modelo es excelente, pero es mentira, es excelente solo para una clase
print(accuracy_score(y_test, preds))

#Algunas puntuaciones muestra que el modelo es malo
accuracy = accuracy_score(y_test, preds)
precision, recall, fbeta, support = precision_recall_fscore_support(y_test, preds, beta=5, pos_label=1, average='binary')
auc = roc_auc_score(y_test, preds)
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {fbeta:.2f}")
print(f"AUC is: {auc:.2f}")

###SMOTE (sobremuestreo sin repeticiones)

#Con SMOTE creamos mas datos similares a la clase en desventaja, lo hace con distancia euclidiana
from imblearn.over_sampling import RandomOverSampler, SMOTE

smote_sampler = SMOTE(random_state = rs)

print(X_test.shape)
print(X_train.shape)

X_smo, y_smo = smote_sampler.fit_resample(X_train, y_train)

print(X_train.shape)

#SMOTE igualo a 160000 las dos clases solo para el entrenamiento
y_smo.value_counts().plot.bar(color=['green', 'red'])
plt.show()

model.fit(X_smo, y_smo)
preds = model.predict(X_test)

#La puntuacion mejoro en muchas metricas, pero la precision es muy baja =0.03
#La explicacion es que los datos iniciales estaban muy demaciado sesgado, este sesgo empeoro con la creacion de datos
#Por lo tanto la puntuacion bajo bastante, esto debido a muchos falsos positivos
precision, recall, f_beta, support = precision_recall_fscore_support(y_test, preds, beta=5, pos_label=1, average='binary')
auc = roc_auc_score(y_test, preds)
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")

#Es por esto que SMOTE no funciona con datos sesgados, hay que aplicar otro metodo

### Reponderacion

#Tambien podemos hacerlo por filas (instancias) poniendole cierto peso a cada dato
#Esto funciona en una clasificacion binaria, donde cada clase es opuesta, la clase con menos filas
#le aplicamos un peso de 0.9 sin son el 10% de los datos y a las que son mayoria un peso de 0.1

class_weight = {}

# Assign weight of class 0 to be 0.1
class_weight[0] = 0.1

# Assign weight of class 1 to be 0.9
class_weight[1] = 0.9

#La regression logisitca tiene para poner cierto peso a los datos
model = LogisticRegression(random_state=rs, 
                              max_iter = 1000,
                              class_weight=class_weight)

model.fit(X_train, y_train)
preds = model.predict(X_test)


#Mejoro bastante la precision y las puntuaciones en general
precision, recall, f_beta, support = precision_recall_fscore_support(y_test, preds, beta=5, pos_label=1, average='binary')
auc = roc_auc_score(y_test, preds)
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy is: {accuracy:.2f}")
print(f"Precision is: {precision:.2f}")
print(f"Recall is: {recall:.2f}")
print(f"Fscore is: {f_beta:.2f}")
print(f"AUC is: {auc:.2f}")



###Funciones rapidas para hacer modelos

rs = 123
# Build a logistic regression model
def build_lr(X_train, y_train, X_test, threshold=0.5, best_params=None):
    
    model = LogisticRegression(random_state=rs, 
                              max_iter = 1000)
    # If best parameters are provided
    if best_params:
        model = LogisticRegression(penalty = 'l2',
                              random_state=rs, 
                              max_iter = 1000,
                              class_weight=best_params['class_weight'])
    # Train the model
    model.fit(X_train, y_train)
    # If predicted probability is largr than threshold (default value is 0.5), generate a positive label
    predicted_proba = model.predict_proba(X_test)
    yp = (predicted_proba [:,1] >= threshold).astype('int')
    return yp, model

def build_rf(X_train, y_train, X_test, threshold=0.5, best_params=None):
    
    model = RandomForestClassifier(random_state = rs)
    # If best parameters are provided
    if best_params:
        model = RandomForestClassifier(random_state = rs,
                                   # If bootstrap sampling is used
                                   bootstrap = best_params['bootstrap'],
                                   # Max depth of each tree
                                   max_depth = best_params['max_depth'],
                                   # Class weight parameters
                                   class_weight=best_params['class_weight'],
                                   # Number of trees
                                   n_estimators=best_params['n_estimators'],
                                   # Minimal samples to split
                                   min_samples_split=best_params['min_samples_split'])
    # Train the model   
    model.fit(X_train, y_train)
    # If predicted probability is largr than threshold (default value is 0.5), generate a positive label
    predicted_proba = model.predict_proba(X_test)
    yp = (predicted_proba [:,1] >= threshold).astype('int')
    return yp, model

rs = 123
def evaluate(yt, yp, eval_type="Original"):
    results_pos = {}
    results_pos['type'] = eval_type
    # Accuracy
    results_pos['accuracy'] = accuracy_score(yt, yp)
    # Precision, recall, Fscore
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, beta=5, pos_label=1, average='binary')
    results_pos['recall'] = recall
    # AUC
    results_pos['auc'] = roc_auc_score(yt, yp)
    # Precision
    results_pos['precision'] = precision
    # Fscore
    results_pos['fscore'] = f_beta
    return results_pos

### Para restablecer los datos a los originales

def resample(X_train, y_train):
    # SMOTE sampler (Oversampling)
    smote_sampler = SMOTE(random_state = 123)
    # Undersampling
    under_sampler = RandomUnderSampler(random_state=123)
    # Resampled datasets
    X_smo, y_smo = smote_sampler.fit_resample(X_train, y_train)
    X_under, y_under = under_sampler.fit_resample(X_train, y_train)
    return X_smo, y_smo, X_under, y_under


#Para visualizar metricas

def visualize_eval_metrics(results):
    df = pd.DataFrame(data=results)
    #table = pd.pivot_table(df, values='type', index=['accuracy', 'precision', 'recall', 'f1', 'auc'],
    #                columns=['type'])
    #df = df.set_index('type').transpose()
    print(df)
    x = np.arange(5)
    original = df.iloc[0, 1:].values
    class_weight = df.iloc[1, 1:].values
    smote = df.iloc[2, 1:].values
    under = df.iloc[3, 1:].values
    width = 0.2
    figure(figsize=(12, 10), dpi=80)
    plt.bar(x-0.2, original, width, color='#95a5a6')
    plt.bar(x, class_weight, width, color='#d35400')
    plt.bar(x+0.2, smote, width, color='#2980b9')
    plt.bar(x+0.4, under, width, color='#3498db')
    plt.xticks(x, ['Accuracy', 'Recall', 'AUC', 'Precision', 'Fscore'])
    plt.xlabel("Evaluation Metrics")
    plt.ylabel("Score")
    plt.legend(["Original", "Class Weight", "SMOTE", "Undersampling"])
    plt.show()

###Ejemplo 2

churn_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML201EN-SkillsNetwork/labs/module_4/datasets/im_churn.csv", index_col=False)

print(churn_df.head())

X_train, X_test, y_train, y_test = split_data(churn_df)

#Se puede ver que los datos estan desequilibrados, pero no tan extremo como el anterior
y_train.value_counts().plot.bar(color=['green', 'red'])
plt.show()

best_params_no_weight = {'bootstrap': True,
                         'class_weight': None, 
                         'max_depth': 10, 
                         'min_samples_split': 5, 
                         'n_estimators': 50}

results = []

#Creamos un modelo random forest con los mejores parametros encontrados por gridsearch
preds, model = build_rf(X_train, y_train, X_test, best_params=best_params_no_weight)
result = evaluate(y_test, preds, "Original")

#Muestra una buena puntcuacion, pero en algunas metricas es bajo
print(result)
results.append(result)

###Probando la agregacion de pesos

class_weight = {}
# 0.2 to Non-churn class
class_weight[0] = 0.2
# 0.8 to Churn class
class_weight[1] = 0.8

best_params_weight = {'bootstrap': True,
                         'class_weight': class_weight, 
                         'max_depth': 10, 
                         'min_samples_split': 5, 
                         'n_estimators': 50}

preds_cw, weight_model = build_rf(X_train, y_train, X_test, best_params=best_params_weight)

result = evaluate(y_test, preds_cw, "Class Weight")

#Mejora bastante las metricas, pero la precision baja
print(result)
results.append(result)


### Probando SMOTE

# X_smo is resampled from X_train using SMOTE
# y_smo is resampled from y_train using SMOTE
# X_under is resampled from X_train using Undersampling
# y_under is resampled from y_train using Undersampling
X_smo, y_smo, X_under, y_under = resample(X_train, y_train)

#Tiene una puntucacion bastante regular
preds_smo, smo_model = build_rf(X_smo, y_smo, X_test, best_params=best_params_no_weight)
result = evaluate(y_test, preds_smo, "SMOTE")
print(result)
results.append(result)

#No mejora suben y bajan otras puntuaciones de metricas
preds_under, under_model = build_rf(X_under, y_under, X_test, best_params=best_params_no_weight)
result = evaluate(y_test, preds_under, "Undersampling")
print(result)
results.append(result)

#En general cualquiera puede ser usado tienen puntuaciones bastante parecidas
visualize_eval_metrics(results)

###Ejemplo 3

tumor_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML201EN-SkillsNetwork/labs/module_4/datasets/im_cancer.csv", index_col=False)
X_train, X_test, y_train, y_test = split_data(tumor_df)

#Este datasets no necesita ser remuestrado, pero aun asi vemos los efectos que puede producir
y_train.value_counts().plot.bar(color=['green', 'red'])
plt.show()


X_smo, y_smo, X_under, y_under = resample(X_train, y_train)

best_params_weight = {'bootstrap': True,
                         'class_weight': {0: 0.2, 1: 0.8}, 
                         'max_depth': 10, 
                         'min_samples_split': 5, 
                         'n_estimators': 50}

# no class-weights
results=[]
preds, model = build_rf(X_train, y_train, X_test)
results.append(evaluate(y_test, preds))
# class weight
preds, model = build_rf(X_train, y_train, X_test, best_params=best_params_weight)
results.append(evaluate(y_test, preds))
# Resampling
preds, model = build_rf(X_smo, y_smo, X_test)
results.append(evaluate(y_test, preds))
preds, model = build_rf(X_under, y_under, X_test)
results.append(evaluate(y_test, preds))

#En este caso todas son bastante altas, pero con peso y sobremuestreo mejor un poco minimamente, a pesar de no necesitarlo
#en principio
visualize_eval_metrics(results)

###Ejemplo 4

hr_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML201EN-SkillsNetwork/labs/module_4/datasets/im_hr.csv", index_col=False)

y_train.value_counts().plot.bar(color=['green', 'red'])
best_params = {'class_weight': {0: 0.1, 1: 0.9}}
results = []
# no class-weights
preds, model = build_lr(X_train, y_train, X_test)
result = evaluate(y_test, preds)
results.append(result)
# class weight
preds, weight_model = build_lr(X_train, y_train, X_test, best_params=best_params)
result = evaluate(y_test, preds, eval_type="Class Weight")
results.append(result)
# Resampling
preds, smote_model = build_lr(X_smo, y_smo, X_test)
result = evaluate(y_test, preds, eval_type="SMOTE")
results.append(result)
preds_under, under_model = build_lr(X_under, y_under, X_test)
result = evaluate(y_test, preds_under, eval_type="Undersampling")
#metrics.plot_roc_curve(smote_model, X_test, y_test) 
results.append(result)

#En este caso las puntucaciones son bastante iguales excepto en una
visualize_eval_metrics(results)






       
