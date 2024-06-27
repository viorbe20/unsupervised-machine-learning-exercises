import numpy as np
import pandas as pd
from sklearn import metrics

def get_bic_values(K, grupos, X):
    """
    K: Número de grupos (clusters)
    grupos: Vector que contiene los grupos de los datos
    X: Matriz de datos
    """
    N = X.shape[0] # Número de datos
    P = X.shape[1] # Número de variables
    xi = np.zeros((1,K)) # Vector xi

    # Calculamos el sumario de xi en la fórmula
    for k in range(0, K):
        suma = 0
        for j in range(0, P):
            sigma = np.square(np.std(X[:, j]))
            sigma_j = np.square(np.std(X[grupos==k, j]))
            suma += 0.5*np.log(sigma + sigma_j)

        n_k = sum(grupos==k) # Número de elementos en el grupo k
        xi[0, k] = -n_k*suma

    bic = -2*np.sum(xi) + 2*K*P*np.log(N)
    return bic

def plot_bic(K, grupos, X):
    """
    K: Número de grupos (clusters)
    grupos: Vector que contiene los grupos de los datos
    X: Matriz de datos
    """
    N = X.shape[0] # Número de datos
    P = X.shape[1] # Número de variables
    xi = np.zeros((1,K)) # Vector xi

    # Calculamos el sumario de xi en la fórmula
    for k in range(0, K):
        suma = 0
        for j in range(0, P):
            sigma = np.square(np.std(X[:, j]))
            sigma_j = np.square(np.std(X[grupos==k, j]))
            suma += 0.5*np.log(sigma + sigma_j)

        n_k = sum(grupos==k) # Número de elementos en el grupo k
        xi[0, k] = -n_k*suma

    bic = -2*np.sum(xi) + 2*K*P*np.log(N)
    return bic

def get_internal_validation(data, labels_pred):
    """
    Calcula medidas de validación interna para evaluar la calidad de los clusters generados.
    Retorna un DataFrame con las medidas de validación interna.
    """
    silhouette_score = metrics.silhouette_score(data, labels_pred)
    calinski_harabasz_score = metrics.calinski_harabasz_score(data, labels_pred)
    
    validation_data = {
        "Medida": ["Coeficiente de Silhouette", "Índice Calinski-Harabasz"],
        "Valor": [silhouette_score, calinski_harabasz_score]
    }
    
    df_validation = pd.DataFrame(validation_data)
    return df_validation

    
def get_external_validation(labels_pred, labels_true):
    """
    Calcula medidas de validación externa para comparar los clusters generados con una clasificación verdadera conocida.
    Retorna un DataFrame con las medidas de validación externa.
    """
    homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
    completeness = metrics.completeness_score(labels_true, labels_pred)
    v_measure = metrics.v_measure_score(labels_true, labels_pred)
    adjusted_rand_index = metrics.adjusted_rand_score(labels_true, labels_pred)
    mutual_information = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    
    validation_data = {
        "Medida": ["Homogeneidad", "Exhaustividad", "Media armónica", "Adjusted Rand Index", "Mutual Information"],
        "Valor": [homogeneity, completeness, v_measure, adjusted_rand_index, mutual_information]
    }
    
    df_validation = pd.DataFrame(validation_data)
    return df_validation
