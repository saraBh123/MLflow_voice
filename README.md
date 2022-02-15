# MLflow_voice
## Importation de bibliothèque  ##
```
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn as mlflow_sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm
from sklearn import tree
```
### Importation des données et choisir les Features ###
```
data = pd.read_csv("D:/DossierSara/DEDS/semester 3/python/Projet/voice.csv")
```
```
# Obtenir des informations sur le data 
data.info()
```
```
data.isnull().values.any()
```
### Ajustement des valeurs de label (mâle = 1, femelle = 0 ) ###
Après avoir obtenu des informations sur les données, nous appellerons homme 1 et femme 0
```
data.label = [1 if each == "male" else 0 for each in data.label]
data['label'] # maintenant nous avons une Label en tant qu'entier
```
### Normalisation des données ###
```
X=data.drop(['label'],axis=1)  # composants de prédiction
y=data['label'] # principaux résultats homme ou femme
```
```
scaler = StandardScaler()
X = scaler.fit_transform(X)
```
### Opération de fractionnement pour les données d'apprentissage et de test ###
Les données sont séparées pour les opérations de formation et de test. Nous aurons 20 % de données pour le test et 80 % de données pour le train après l'opération fractionnée.
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
```
### MLflow ###
création d'une expérience
```
mlflow.create_experiment('Voice_Models_Classification')
```
accéder à l'expérience
```
mlflow.set_experiment('Voice_Models_Classification')
```
fonction de modele 
```
def log_run(run_name, model, val_x, val_y):
    
    ## Démarrer avec le nom donné
    mlflow.start_run(run_name=run_name, nested=True)
    
    ## Obtenir une prédiction sur l'ensemble de données de validation
    val_pred = model.predict(val_x)
    
    ## log tous les hyperparamètres
    mlflow.log_params(model.get_params())
        
    ## Calculer les métriques requises
    precision, recall, fscore, support = precision_recall_fscore_support(val_y, val_pred, average='micro')
    
    ## log tous les paramètres requis
    mlflow.log_metrics(
        {'precision': precision, 'recall': recall, 'fscore': fscore}
    )
    
    ## Cela enregistre les modèles basés sur sklearn en les convertissant en pickle
    mlflow_sklearn.log_model(model, run_name)
    
    mlflow.end_run()
```
### Les Modeles ###
#### random forest Classifier ####
```
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
log_run('Random_Forest_default_param', rf_model, x_test, y_test)
```
#### Decision Tree Classifier ####
```
dtc=tree.DecisionTreeClassifier()
dtc.fit(x_train,y_train)
log_run('Decision_Tree_model', dtc, x_test, y_test)
```
#### SVM ####
```
svm = svm.SVC()
svm.fit(x_train, y_train)
log_run('SVM_model', dtc, x_test, y_test)
```
