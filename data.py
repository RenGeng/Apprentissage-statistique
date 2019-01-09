"""
Classe data permettant de charger, split et éauilibrer les données

"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

def nom_colonne():
	""" Retourne le nom des colonnes """
	# res = {0:"Elevation", 1:"Aspect", 2:"Slope",
	# 	   3:"H_Dist_Hydrology", 4:"V_Dist_Hydrology", 5:"H_Dist_Roadways",
	# 	   6:"Hillshade_9am",7:"Hillshade_Noon",8:"Hillshade_3pm",9:"H_Dist_Fire"}

	res = ["Elevation", "Aspect", "Slope",
		   "H_Dist_Hydrology", "V_Dist_Hydrology", "H_Dist_Roadways",
		   "Hillshade_9am","Hillshade_Noon","Hillshade_3pm","H_Dist_Fire"]

	for i in range(4):
		res.append("Wilderness_area" + str(i))
	for i in range(40):
		res.append("Soil_Type" + str(i))
	res.append("Cover_type")
	return res


class Data:

	def __init__(self, file="covtype.data"):
		"""
		Lecture du fichier passé en argument, par défaut on lit covtype.data

		"""
		self.nom_colonne = nom_colonne()

		self.data = pandas.read_csv("covtype.data",header=None,names=self.nom_colonne)
		self.labels = self.data[self.data.columns[-1]]
		self.data = self.data.drop(columns=["Cover_type"])

	def proportion(self):
		plt.bar(range(1,8),self.labels.value_counts().sort_index().values)
		plt.title("Nombre d'individu selon la classe")
		plt.show()

	def matrice_corr(self):
		corr = self.data.loc[:,self.nom_colonne[:10]].corr()
		corr = abs(corr) # car des nombres négatifs

		# f, ax = plt.subplots(figsize=(5, 5))

		sns.axes_style("white")
		sns.heatmap(corr, mask=np.zeros_like(corr), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, xticklabels=corr.columns, yticklabels=corr.columns,annot=True)

		plt.show()

	def resample(self, type = "over"):
		"""
		Equilibrage de données entre les classes, par défaut on fait du oversampling

		"""

		if type == 'over':
			""" Equilibre tous les classes sur un nomrbe d'individu identique """
			x, y = SMOTE().fit_resample(self.data, self.labels)
			self.data = pandas.DataFrame(x,columns=self.nom_colonne[:-1])
			self.labels = pandas.Series(y)

			import collections

			print(collections.Counter(y))
