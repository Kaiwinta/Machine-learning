import numpy as np
import pandas as pd
import matplotlib as plt

#L'on ajoute le nom des colonnes à chaque colonne
cols = ("fLength", "fWidth", "fSize", "fConc" , "fConc1" , "fAsym", "fM3Long", "fM3Trans" , "fAlpha" , "fDist" , "class")
df =pd.read_csv("magic04.data", names=cols)

#class est soit g ou f mais du coup on transformme en binaire 
df["class"] = (df["class"] == "g").astype(int)
