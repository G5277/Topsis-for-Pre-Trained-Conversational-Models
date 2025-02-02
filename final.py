import numpy as np
import pandas as pd

models = ['GPT-3', 'BERT', 'T5', 'DialoGPT', 'BlenderBot', 'LaMDA']
criteria = ['Accuracy', 'Speed', 'Context Retention', 'Flexibility', 'Resource Efficiency', 'User Satisfaction']

decision_matrix = np.array([
    [9, 7, 8, 9, 6, 8],  # GPT-3
    [8, 6, 7, 8, 7, 7],  # BERT
    [8, 8, 8, 8, 8, 8],  # T5
    [7, 9, 7, 7, 7, 7],  # DialoGPT
    [8, 7, 8, 9, 6, 8],  # BlenderBot
    [9, 8, 9, 9, 5, 9]   # LaMDA
])

weights = np.array([0.3, 0.2, 0.2, 0.15, 0.1, 0.05])

def normalize_matrix(matrix):
    norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))  # Normalize column-wise
    return norm_matrix

def weighted_matrix(norm_matrix, weights):
    return norm_matrix * weights

def ideal_negative_ideal(matrix):
    ideal = np.max(matrix, axis=0)
    negative_ideal = np.min(matrix, axis=0)
    return ideal, negative_ideal

def euclidean_distance(matrix, ideal, negative_ideal):
    D_plus = np.sqrt(np.sum((matrix - ideal) ** 2, axis=1)) 
    D_minus = np.sqrt(np.sum((matrix - negative_ideal) ** 2, axis=1))  
    return D_plus, D_minus

def relative_closeness(D_plus, D_minus):
    return D_minus / (D_plus + D_minus)

norm_matrix = normalize_matrix(decision_matrix)
weighted_matrix = weighted_matrix(norm_matrix, weights)
ideal, negative_ideal = ideal_negative_ideal(weighted_matrix)
D_plus, D_minus = euclidean_distance(weighted_matrix, ideal, negative_ideal)
closeness = relative_closeness(D_plus, D_minus)

ranking = pd.DataFrame({
    'Model': models,
    'Closeness': closeness
})

ranking = ranking.sort_values(by='Closeness', ascending=False)
print(ranking)

