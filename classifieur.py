
from data import Data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

def make_plot(element):
	""" 
	Plot l'élément (tuple) pris en argument
	"""
	nom, val = element
	longueur = range(1,11)
	# longueur = list(range(1,3))
	for key, value in val.items():
		plt.plot(longueur,value, label = key, marker="o")

	plt.legend(loc='upper left')
	plt.xlabel('Nombre de paramètres')
	plt.title(nom, fontsize=12, fontweight='bold')
	plt.subplots_adjust(hspace = 1)




donnee = Data()
donnee.resample()
# print(donnee.data.iloc[:,:10])

# metrics = ['accuracy', 'precision'] 

# Les metrics qu'on va utilisé pour valider notre modèle
metrics = {'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'accuracy': 'accuracy'}# make_scorer(recall_score, average='macro')} 

model = [
	('KNN',KNeighborsClassifier()),
	('Arbre décisionnel',DecisionTreeClassifier()),
	('Random Forest', RandomForestClassifier())
	#('Kmeans', KMeans())
]


k = 5

fit_time = {}
pred_time = {}

train_accuracy = {}
test_accuracy = {}

train_precision = {}
test_precision = {}

train_recall = {}
test_recall = {}

for nom, classifieur in model:
	print("\n \t\t\t ",nom, "\n")
	fit_time[nom] = []
	pred_time[nom] = []

	train_accuracy[nom] = []
	test_accuracy[nom] = []

	train_precision[nom] = []
	test_precision[nom] = []
	
	train_recall[nom] = []
	test_recall[nom] = []

	for i in range(1,11):
		print("\n \t \t \t Pour {} paramètre(s)\n".format(i))

		res = cross_validate(estimator=classifieur, 
			X=donnee.data.iloc[:,:i], 
			y=donnee.labels, 
			scoring=metrics, 
			cv=k,
			return_train_score=True)
		
		print(res)

		fit_time[nom].append(np.average(res['fit_time']))
		pred_time[nom].append(np.average(res['score_time']))

		train_accuracy[nom].append(np.average(res['train_accuracy']))
		test_accuracy[nom].append(np.average(res['test_accuracy']))

		train_precision[nom].append(np.average(res['train_prec_macro']))
		train_recall[nom].append(np.average(res['train_rec_macro']))

		test_precision[nom].append(np.average(res['test_prec_macro']))
		test_recall[nom].append(np.average(res['test_rec_macro']))


list_plot = [
	('Fitting time', fit_time),
	('Prediction time', pred_time),

	('Train accuracy score', train_accuracy), 
	('Test accuracy score', test_accuracy),

	('Train precision score', train_precision), 
	('Test precision score', test_precision),

	('Train recall score', train_recall),
	('Test recall score',test_recall)
]

for i in range(0,len(list_plot),2):
	plt.figure()
	make_plot(list_plot[i])
	plt.figure()
	make_plot(list_plot[i+1])
	plt.show()

	input("Appuyer sur un bouton pour afficher la suite...")


# Comparer accuracy avec truc normal pour voir si c'est égal ou non