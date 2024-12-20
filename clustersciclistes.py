"""
@ IOC - CE IABD
"""
import os
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

def load_dataset(path):
    """
    Carrega el dataset de registres dels ciclistes

    arguments:
        path -- dataset

    Returns: dataframe
    """
    # 
    df = pd.read_csv(path, sep=",")
    return df

def EDA(df):
    """
    Exploratory Data Analysis del dataframe

    arguments:
        df -- dataframe

    Returns: None
    """
    logging.debug(f"\n {df.shape}")
    logging.debug(f"\n {df[:5]}")
    logging.debug(f"\n {df.columns}")
    logging.debug(f"\n {df.info()}")
    return None

def clean(df):
    """
    Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

    arguments:
        df -- dataframe

    Returns: dataframe
    """
    df = df.drop(['id','tt'], axis=1)
    logging.debug(df.columns)
    return df

def extract_true_labels(df):
    """
    Guardem les etiquetes dels ciclistes (BEBB, ...)

    arguments:
        df -- dataframe

    Returns: numpy ndarray (true labels)
    """
    true_labels =  df['label'].to_numpy()
    logging.debug(true_labels)
    return true_labels

def visualitzar_pairplot(df):
    """
    Genera una imatge combinant entre sí tots els parells d'atributs.
    Serveix per apreciar si es podran trobar clústers.

    arguments:
        df -- dataframe

    Returns: None
    """
    sns.pairplot(df)
    try:
        os.makedirs(os.path.dirname('img/'))
    except FileExistsError:
        pass
    plt.savefig("img/pairplot.png")
    plt.show()

def clustering_kmeans(data, n_clusters=4):
    """
    Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
    Entrena el model

    arguments:
        data -- les dades: tp i tb

    Returns: model (objecte KMeans)
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data)
    return model

def visualitzar_clusters(data, labels):
    """
    Visualitza els clusters en diferents colors. Provem diferents combinacions de parells d'atributs

    arguments:
        data -- el dataset sobre el qual hem entrenat
        labels -- l'array d'etiquetes a què pertanyen les dades (hem assignat les dades a un dels 4 clústers)

    Returns: None
    """
    try:
        os.makedirs(os.path.dirname('img/'))
    except FileExistsError:
        pass
    # Com només tenim 2 columnes es crea un sol gràfic
    sns.scatterplot(x="tp", y="tb", data=data, hue=labels, palette="pastel")
    plt.xlabel("Temps de pujada")
    plt.ylabel("Temps de baixada")
    plt.savefig("img/grafica1.png")
    plt.show()

def associar_clusters_patrons(tipus, model):
    """
    Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
    S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

    arguments:
    tipus -- un array de tipus de patrons que volem actualitzar associant els labels
    model -- model KMeans entrenat

    Returns: array de diccionaris amb l'assignació dels tipus als labels
    """
    # proposta de solució

    dicc = {'tp':0, 'tb': 1}

    logging.info('Centres:')
    for j in range(len(tipus)):
        logging.info('{:d}:\t(tp: {:.1f}\ttb: {:.1f})'.format(j, model.cluster_centers_[j][dicc['tp']], model.cluster_centers_[j][dicc['tb']]))

    # Procés d'assignació
    ind_label_0 = -1
    ind_label_1 = -1
    ind_label_2 = -1
    ind_label_3 = -1

    suma_max = 0
    suma_min = 50000

    for j, center in enumerate(model.cluster_centers_):
        suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
        if suma_max < suma:
            suma_max = suma
            ind_label_3 = j
        if suma_min > suma:
            suma_min = suma
            ind_label_0 = j

    tipus[0].update({'label': ind_label_0})
    tipus[3].update({'label': ind_label_3})

    lst = [0, 1, 2, 3]
    lst.remove(ind_label_0)
    lst.remove(ind_label_3)

    if model.cluster_centers_[lst[0]][0] < model.cluster_centers_[lst[1]][0]:
        ind_label_1 = lst[0]
        ind_label_2 = lst[1]
    else:
        ind_label_1 = lst[1]
        ind_label_2 = lst[0]

    tipus[1].update({'label': ind_label_1})
    tipus[2].update({'label': ind_label_2})

    logging.info('\nHem fet l\'associació')
    logging.info('\nTipus i labels:\n%s', tipus)
    return tipus

def generar_informes(df, tipus):
    """
    Generació dels informes a la carpeta informes/. Tenim un dataset de ciclistes i 4 clústers, i generem
    4 fitxers de ciclistes per cadascun dels clústers

    arguments:
        df -- dataframe
        tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

    Returns: None
    """
    # Separem els ciclistes per label
    ciclistes_label = [
        df[df['label'] == 0],
        df[df['label'] == 1],
        df[df['label'] == 2],
        df[df['label'] == 3],
    ]
    try:
        os.makedirs(os.path.dirname('informes/'))
    except FileExistsError:
        pass
    
    for tip in tipus:
        nom_fitxer = tip['name'] + ".txt"
        logging.debug(nom_fitxer)
        f = open("informes/" + nom_fitxer, "w")
        t = [t for t in tipus if t['name'] == tip['name']]
        logging.debug(t)
        ciclistes = ciclistes_label[t[0]['label']].index
        for ciclista in ciclistes:
            f.write(str(ciclista) + "\n")
        f.close()
        
    logging.info('S\'han generat els informes en la carpeta informes/\n')

def nova_prediccio(dades, model):
    """
    Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers

    arguments:
        dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
        model -- clustering model
    Returns: (dades agrupades, prediccions del model)
    """
    df_nous_ciclistes = pd.DataFrame(columns=['id', 'tp', 'tb', 'tt'], data=dades)
    # Eliminem les columnes que no hem tingut en compte
    df_nous_ciclistes = df_nous_ciclistes.drop(['id','tt'], axis=1)
    pred = model.predict(df_nous_ciclistes)
    
    return df_nous_ciclistes, pred

# ----------------------------------------------

if __name__ == "__main__":
    
    path_dataset = './data/ciclistes.csv'
    
    # s'afegeix la configuració per veure els logs
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    
    # Carreguem dades, fem l'anàlisi i netegem dades
    ciclistes_data = load_dataset(path_dataset)
    EDA(ciclistes_data)
    ciclistes_data = clean(ciclistes_data)
    
    # Es crea una nova variable per les etiquetes i s'eliminen del dataframe
    true_labels = extract_true_labels(ciclistes_data)
    ciclistes_data = ciclistes_data.drop('label', axis=1)
    
    # Fem un primer gràfic per començar a visualitzar els clústers
    visualitzar_pairplot(ciclistes_data)
    
    # Generem un model KMeans i l'entrenem, després guardem el model a un arxiu
    model = clustering_kmeans(ciclistes_data)
    with open('model/clustering_model.pkl', 'wb') as f:
        pickle.dump(clustering_kmeans, f)
    data_labels = model.labels_

    # Calculem mètriques i les guardem
    logging.info('\nHomogeneïtat: %.3f', homogeneity_score(true_labels, data_labels))
    logging.info('Completesa: %.3f', completeness_score(true_labels, data_labels))
    logging.info('V-measure: %.3f', v_measure_score(true_labels, data_labels))
    with open('model/scores.pkl', 'wb') as f:
        pickle.dump({
        "h": homogeneity_score(true_labels, data_labels),
        "c": completeness_score(true_labels, data_labels),
        "v": v_measure_score(true_labels, data_labels)
        }, f)
    visualitzar_clusters(ciclistes_data, data_labels)
    # array de diccionaris que assignarà els tipus als labels
    tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]

    # Afegim les etiquetes donades pel model al dataframe i les associem al tipus de ciclista
    ciclistes_data['label'] = model.labels_.tolist()
    logging.debug('\nLabels:\n%s', ciclistes_data[:5])
    tipus = associar_clusters_patrons(tipus, model)
    with open('model/tipus_dict.pkl', 'wb') as f:
        pickle.dump(tipus, f)
    logging.info('\nTipus i labels:\n%s', tipus)
    
    # Generem els informes
    generar_informes(ciclistes_data, tipus)
    
    # Classificació de nous valors
    nous_ciclistes = [
        [500, 3230, 1430, 4670], # BEBB
        [501, 3300, 2120, 5420], # BEMB
        [502, 4010, 1510, 5520], # MEBB
        [503, 4350, 2200, 6550] # MEMB
    ]
    df_nous_ciclistes, pred = nova_prediccio(nous_ciclistes, model)
    logging.info('\nPredicció dels valors:\n%s', pred)
    
    #Assignació dels nous valors als tipus
    for i, p in enumerate(pred):
        t = [t for t in tipus if t['label'] == p]
        logging.info('tipus %s (%s) - classe %s', df_nous_ciclistes.index[i], t[0]['name'], p)

