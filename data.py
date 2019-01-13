"""
Classe data permettant de charger, split et éauilibrer les données

"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

	############## Traitement des données ##############
	def __init__(self, file="covtype.data"):
		"""
		Lecture du fichier passé en argument, par défaut on lit covtype.data

		"""
		self.nom_colonne = nom_colonne()

		self.complete_data = pandas.read_csv("covtype.data",header=None,names=self.nom_colonne)
		self.labels = self.complete_data[self.complete_data.columns[-1]]
		self.data = self.complete_data.drop(columns=["Cover_type"])

	def resample(self, sampling_type = "both"):
		"""
		Equilibrage de données entre les classes, par défaut on fait du oversampling

		"""
		if sampling_type == 'over':
			""" Equilibre tous les classes sur un nomrbe d'individu identique """
			x, y = SMOTE(random_state=0).fit_resample(self.data, self.labels)
			self.data = pandas.DataFrame(x,columns=self.nom_colonne[:-1])
			self.labels = pandas.Series(y)

		elif sampling_type == 'under':
			""" Sous échantillonne les grosses classes jusqu'à avoir le même nombre que la classe minoritaire"""
			x,y = RandomUnderSampler(random_state=0).fit_resample(self.data, self.labels)
			self.data = pandas.DataFrame(x,columns=self.nom_colonne[:-1])
			self.labels = pandas.Series(y)

		elif sampling_type == 'both':
			""" On fait d'abord un sous échantillonnage puis un sur échantillonnage"""

			# On enlève 60% pour la classe 2 et on met la classe 1 au même nombre que la classe 2 pour éviter le sur échantillonage
			# x,y = RandomUnderSampler(sampling_strategy={1:113320,2:113320}).fit_resample(self.data, self.labels)
			x,y = RandomUnderSampler(random_state=0,sampling_strategy={1:50000,2:50000}).fit_resample(self.data, self.labels)
			self.data = pandas.DataFrame(x,columns=self.nom_colonne[:-1])
			self.labels = pandas.Series(y)
			
			x, y = SMOTE(random_state=0).fit_resample(self.data, self.labels)
			self.data = pandas.DataFrame(x,columns=self.nom_colonne[:-1])
			self.labels = pandas.Series(y)
		else:
			raise ValueError("Le type d'équilibrage '{}' que vous avez demandé n'est pas encore implémenté. Veillez utiliser soit oversampling ('over'), soit undersampling ('under'), soit les deux ('both')".format(sampling_type))


	############## Analyse ##############
	def proportion(self, plot_all=False):
		plt.figure()
		plt.bar(range(1,8),self.labels.value_counts().sort_index().values)
		plt.title("Nombre d'individu selon la classe")
		if not plot_all:
			plt.show()


	def matrice_corr(self, plot_all=False):
		plt.figure()
		corr = self.data.loc[:,self.nom_colonne[:10]].corr()
		corr = abs(corr) # car des nombres négatifs

		# f, ax = plt.subplots(figsize=(5, 5))

		sns.axes_style("white")
		sns.heatmap(corr, mask=np.zeros_like(corr), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, xticklabels=corr.columns, yticklabels=corr.columns,annot=True)
		plt.tight_layout()
		if not plot_all:
			plt.show()

	def boxplot(self, plot_all=False):
		plt.figure()
		temp_data = self.complete_data.loc[:,self.nom_colonne[:10]]

		wilderness = self.complete_data.loc[:,self.nom_colonne[10:14]]
		wilderness = [np.argmax(i,axis=None,out=None) for i in np.array(wilderness)]

		soil = self.complete_data.loc[:,self.nom_colonne[14:-1]]
		soil = [np.argmax(i,axis=None,out=None) for i in np.array(soil)] # va de 0 à 39

		temp_data["wilderness_area"] = wilderness
		temp_data["soil_type"] = soil
		temp_data["Cover_type"] = self.labels

		for index, nom_colonne in enumerate(list(temp_data)[:-1]):
			graphe_pos = plt.subplot(3,4,index+1)
			# graphe_pos.get_xaxis().set_visible(False)
			graphe_pos.get_xaxis().set_ticklabels([])
			graphe = temp_data.boxplot(column = nom_colonne, by = "Cover_type", ax = graphe_pos)

			plt.subplots_adjust(hspace = 0.5)
		# ax = sns.boxplot(data=pandas.melt(self.data),dodge=False)
		if not plot_all:
			plt.show()

	def plot_all(self):
		self.proportion(True)
		self.matrice_corr(True)
		self.boxplot(True)
		plt.show()

	def ACP(self):
		data_norme = StandardScaler().fit_transform(self.data.iloc[:,:10])

		acp = PCA(svd_solver='full')
		coordonnee = acp.fit_transform(data_norme)
		# print(acp.explained_variance_ratio_)
		plt.plot(range(1,len(acp.explained_variance_ratio_)+1), np.cumsum(acp.explained_variance_ratio_))
		plt.xlabel('Nombre de composants')
		plt.grid()
		plt.title("Pourcentage de variance expliqué")

		n = len(self.labels)
		corvar = np.zeros((data_norme.shape[1],data_norme.shape[1]))
		sqrt_eigval = np.sqrt((n-1)/n*acp.explained_variance_)

		for k in range(data_norme.shape[1]):
			corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]
		
		fig,axes = plt.subplots(figsize=(10,10))
		axes.set_xlim(-1,1)
		axes.set_ylim(-1,1)

		for j in range(data_norme.shape[1]):
			plt.annotate(self.data.columns[j],(corvar[j,0],corvar[j,1]))

		plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
		plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

		cercle = plt.Circle((0,0),1,color='blue',fill=False)
		axes.add_artist(cercle)
		plt.show()

		return coordonnee


if __name__ == '__main__':
	data = Data()
	# data.plot_all()
	data.resample()
	data.ACP()


	from sklearn.pipeline import Pipeline
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import cross_val_score

	pipe1 = Pipeline([('pca', PCA(n_components=10)), # Nombre de composant à garder
                 ('tree', DecisionTreeClassifier())])
	print("\n Résultat de l'arbre décisionnel avec 6 composants de ACP\n",cross_val_score(pipe1, data.data, data.labels, cv=5))

	pipe2 = Pipeline([('pca', PCA(n_components=10)), # Nombre de composant à garder
                 ('knn', KNeighborsClassifier())])
	print("\nRésultat de KNN avec 6 composants de ACP\n",cross_val_score(pipe2, data.data, data.labels, cv=5))

	pipe3 = Pipeline([('pca', PCA(n_components=10)), # Nombre de composant à garder
                 ('Random forest', RandomForestClassifier())])
	print("\nRésultat de random forest avec 6 composants de ACP\n",cross_val_score(pipe3, data.data, data.labels, cv=5))


