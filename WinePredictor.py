import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# Load data
load_data = pd.read_csv("WinePredictor.csv")

# Clean, Prepare and manipulate data
feature_nm=['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

print("Feture name",feature_nm)


# Creating labelEncoder
label_encoder = preprocessing.LabelEncoder()

load_data['Alcohol'] = label_encoder.fit_transform(load_data['Alcohol'])
load_data['Malic acid'] = label_encoder.fit_transform(load_data['Malic acid'])
load_data['Ash'] = label_encoder.fit_transform(load_data['Ash'])
load_data['Alcalinity of ash'] =label_encoder.fit_transform(load_data['Alcalinity of ash'])
load_data['Magnesium'] =label_encoder.fit_transform(load_data['Magnesium'])
load_data['Total phenols'] =label_encoder.fit_transform(load_data['Total phenols'])
load_data['Flavanoids'] =label_encoder.fit_transform(load_data['Flavanoids'])
load_data['Nonflavanoid phenols'] =label_encoder.fit_transform(load_data['Nonflavanoid phenols'])
load_data['Proanthocyanins'] =label_encoder.fit_transform(load_data['Proanthocyanins'])
load_data['Color intensity'] =label_encoder.fit_transform(load_data['Color intensity'])
load_data['Hue'] =label_encoder.fit_transform(load_data['Hue'])
load_data['OD280/OD315 of diluted wines'] =label_encoder.fit_transform(load_data['OD280/OD315 of diluted wines'])
load_data['Proline'] =label_encoder.fit_transform(load_data['Proline'])


# Combining weather and temp into single listof tuples
features=list(zip(load_data['Alcohol'],load_data['Malic acid'],load_data['Ash'],load_data['Alcalinity of ash'],load_data['Magnesium'],load_data['Total phenols'],load_data['Flavanoids'],load_data['Nonflavanoid phenols'],load_data['Proanthocyanins'],load_data['Color intensity'],load_data['Hue'],load_data['OD280/OD315 of diluted wines'],load_data['Proline']))
target=load_data['Class']

data_train, data_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

# train data
classifier = KNeighborsClassifier(n_neighbors=5)

#Train
classifier.fit(data_train, target_train)

# Test data
predictions = classifier.predict(data_test)
#print(predictions)

#Accuracy
print("Accuracy:",metrics.accuracy_score(target_test,predictions) * 100,"%")