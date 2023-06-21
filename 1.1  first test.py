import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Le dataset importer parle de rayon lumineux d'un télescope
#Les class sont g pour gamma et f pour l'autre
#Toute la doc sur:  https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
#Le reste des colonnes sont les features que l'on mettras dans le model

"""
Different type of machine learning :

    01) Supervised Learning:
        For all input we have the output so the model can compare
        what he calculate and what is the real answer in order to learn

        Learning task:
            Classification:
                Classification                  Pizza/ Keba/ Orange
                Binary classificatiion          Spam/not Spam

            Regression:
                Predict continues values ex: crypto 

    02) Unsupervised learning 
        The goal for the computer is to regonize pattern 
        He dont provide an anwser but put all the data in some categories
        Bassed on what he calculated

    03) Renforcement Learning
        Based on reward an punition 
        We reward the computer for doing something that may be good
"""


"""
    Bunch of input that go into a model
    And the ouput is a prediction
    noted Y

    All the input are called features
    And are noted X
    
    Different types of features:
        Qualitative :   Categorical data ex:(gender) 
            It can have an order or not ex: rating can have number but not nationality

        Quantitative : Numerical valued date

"""

"""
    We divide the datasets into 3 parts

    The training datasets
    The validation dataset
    The testing dataset

"""
"""
    L1 Loss:    loss = sum(abs(y_real - y_predicted))
    L2 Loss:    loss = sum((y_real - y_predicted)**2)
    Binary Cross entropy Loss  ==> un bordel
"""

#L'on ajoute le nom des colonnes à chaque colonne
cols = ("fLength", "fWidth", "fSize", "fConc" , "fConc1" , "fAsym", "fM3Long", "fM3Trans" , "fAlpha" , "fDist" , "class")
df =pd.read_csv("magic04.data", names=cols)

#class est soit g ou f mais du coup on transformme en binaire 
df["class"] = (df["class"] == "g").astype(int)
#print(df.head())

for label in cols[:-1]:
    plt.hist(df[df['class']==1][label], color = 'blue', label = 'gamma', alpha = 0.7 , density = True)
    plt.hist(df[df['class']==0][label], color = 'red', label = 'hadron', alpha = 0.7 , density = True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    #plt.show()

train , valid , test = np.split(df.sample(frac=1), [int(0.6* len(df)), int(0.8* len(df))])