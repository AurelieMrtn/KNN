## Import des bibliothèques et données

import pandas as pd
import math
import numpy as np


chemin = "C:\\Users\\aurel\\OneDrive - De Vinci\\Bureau\\Cours\\3) Datascience et IA\\2) TD\\Projets\\Projet 2_KNN\\"

# On importe les différents jeux de données
liste = [str(i) for i in range(11)]
df = pd.read_csv(chemin + "data.txt",sep=";",names=liste)
df_test = pd.read_csv(chemin + "preTest.txt",sep=";",names=liste)
df_final = pd.read_csv(chemin + "finalTest.txt",sep=";",names=liste)


## Fonctions

# On calcule la distance euclidienne entre un individu et les données
def distance_euclidienne(individu, donnees):
    return math.sqrt(sum(list(map( lambda x,y : (float(x) - float(y))**2 ,donnees,individu))))


# Centrer réduire les données
def normaliser(dataframe):
    # On prend une variable temporaire pour la colonne de la classe
    df_classe = dataframe["10"]

    # On soustrait chaque valeur de la colonne par la moyenne de la colonne
    dataframe_temp = dataframe.sub(dataframe.mean())

    # On divise chaque valeur de la colonne par l'ecart-type de la colonne
    dataframe = dataframe_temp.div(dataframe.std())

    # On remet la classe car elle a ete normalisee dans l'operation
    dataframe["10"] = df_classe

    # On retourne le dataframe normalise
    return dataframe


# Fonction du KNN
def knn(data,ech,k):
    prediction = []

    # Pour chaque ligne dans l'echantillon a predire :
    for e in ech.iloc:

        # On cree un tableau de k lignes et 2 colonnes trié par distance,
        # la premiere colonne correspondant à la distance,
        # la seconde à la prediction pour cette distance
        dist_classe = np.array(sorted(list( map(lambda x : (distance_euclidienne(e[:10],x[:10]),x[10]),data.iloc) ))[:k])

        # On compte le nombre d'occurences de 0 et de 1 dans les k plus proches voisins
        proportion_classe = np.unique(dist_classe.transpose()[1],return_counts=True)

        # La prediction retenue est celle dont la proportion est la plus grande
        predit = proportion_classe[0][list(proportion_classe[1]).index(max(list(proportion_classe[1])))]

        # On ajoute la prediction a la liste de predictions
        prediction.append(predit)

    # On retourne la liste des predictions
    return prediction


# On calcule la proportion d'erreur de prédiction (utilisé sur les tests)
def verif(knn_predit,data_verif):
    return 1- sum(list(map(lambda x,y : int(x[10]!=y),data_verif.iloc,knn_predit)))/len(data_verif)


# On exporte le fichier au bon format
def exportation(prediction):
    fileD = open(chemin + "Deveze_groupeJ.txt","w")
    fileD.write("\n".join(list(map(lambda x : str(int(x)),prediction))))
    fileD.close()

    fileM = open(chemin + "Martin_groupeJ.txt","w")
    fileM.write("\n".join(list(map(lambda x : str(int(x)),prediction))))
    fileM.close()


## Main


# On normalise les donnees
df = normaliser(df)
df_test = normaliser(df_test)
df_final = normaliser(df_final)


# On definit k
k_choisi = 37


# On test le knn sur le jeu de donnees test et on affiche la precision du programme
def knn_test():
    knn_t = knn(df,df_test,k_choisi)
    correct = verif(knn_t,df_test)
    print("Le programme est precis a {}%".format(correct*100))


# On regarde si le test est concluant
#knn_test()


# On applique le knn sur le jeu de donnees final et on l'exporte
knn_final = knn(df,df_final,k_choisi)
exportation(knn_final)