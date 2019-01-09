# -*- coding: utf-8 -*

import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# lecture
X = pandas.read_csv("covtype.data",header=None)

data = X.iloc[:,:10].rename(index=str,columns={0: "Elevation", 1: "Aspect", 2: "Slope",
								    3: "H_Dist_Hydrology", 4: "V_Dist_Hydrology", 5: "H_Dist_Roadways",
								    6: "Hillshade_9am", 7: "Hillshade_Noon", 8: "Hillshade_3pm", 9 : "H_Dist_Fire"})

wilderness = X.iloc[:,10:14]
wilderness = [np.argmax(i,axis=None,out=None) for i in np.array(wilderness)]

soil = X.iloc[:,14:-1]
soil = [np.argmax(i,axis=None,out=None) for i in np.array(soil)] # va de 0 à 39

classe = X.iloc[:,54] # dernière colonne

data["wilderness_area"] = wilderness
data["soil_type"] = soil
data[["wilderness_area","soil_type"]] = data[["wilderness_area","soil_type"]].astype('category')

print(classe)

# Plot
# plt.bar(range(1,8),classe.value_counts().sort_index().values)
# plt.title("Nombre de cover_type selon la classe")
# plt.show()

# OneVsRestClassifier(LinearSVC(random_state=0)).fit(data.values, classe.values).predict(classe.values) 