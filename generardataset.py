import os
import logging
import numpy as np
import logging

# s'afegeix la configuració per veure els logs
logging.basicConfig(format='%(message)s', level=logging.DEBUG)

def generar_dataset(num, ind, dicc):
    """
    Genera els temps dels ciclistes, de forma aleatòria, però en base a la informació del diccionari
    arguments:
        num (int) -- número de ciclistes a generar
        ind (list) -- llistat amb els identificadors del ciclistes
        dicc (list) -- diccionari per crear els diferents ciclistes

    Returns:
        None
    """
    # Es verifica que el directori existeix i si no el tenim el crea
    os.makedirs("data", exist_ok = True)

    f = open("data/ciclistes.csv", "w")
    f.write("id,label,tp,tb,tt\n")
    for i in range(num):
        tipus = np.random.choice(dicc) # s'agafa una fila aleatòria del diccionari
        label = tipus["name"]
        dorsal = ind[i]
        # es generen temps aleatoris agafant la mitjana que va amb el tipus de ciclista
        pujada = int(np.random.normal(tipus["mu_p"], tipus["sigma"])) 
        baixada = int(np.random.normal(tipus["mu_b"], tipus["sigma"]))
        total = pujada + baixada
        str = f"{dorsal},{label},{pujada},{baixada},{total}"
        f.write(str + "\n")
        logging.debug(str)
    f.close()
    return None

if __name__ == "__main__":

    str_ciclistes = 'data/ciclistes.csv'


    # BEBB: bons escaladors, bons baixadors
    # BEMB: bons escaladors, mal baixadors
    # MEBB: mal escaladors, bons baixadors
    # MEMB: mal escaladors, mal baixadors

    # Port del Cantó (18 Km de pujada, 18 Km de baixada)
    # pujar a 20 Km/h són 54 min = 3240 seg
    # pujar a 14 Km/h són 77 min = 4268 seg
    # baixar a 45 Km/h són 24 min = 1440 seg
    # baixar a 30 Km/h són 36 min = 2160 seg
    mu_p_be = 3240 # mitjana temps pujada bons escaladors
    mu_p_me = 4268 # mitjana temps pujada mals escaladors
    mu_b_bb = 1440 # mitjana temps baixada bons baixadors
    mu_b_mb = 2160 # mitjana temps baixada mals baixadors
    sigma = 240 # 240 s = 4 min

    dicc = [
        {"name":"BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
        {"name":"BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
        {"name":"MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
        {"name":"MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
    ]
    
    # definim el total de ciclistes i generem un array de dorsals
    total_ciclistes = 1000
    dorsals = np.arange(1, total_ciclistes+1, 1)
    generar_dataset(total_ciclistes,dorsals,dicc)
    f = open(str_ciclistes, 'r')
    ciclistes = f.readlines()
    f.close()
    
    logging.info("s'ha generat data/ciclistes.csv")
