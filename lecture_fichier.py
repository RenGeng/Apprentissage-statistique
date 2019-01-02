# -*- coding: utf-8 -*

import pandas
import matplotlib

X = pandas.read_csv("covtype.data",header=None)

classe = X.iloc[:,54] # dernière colonne
