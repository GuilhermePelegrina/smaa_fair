# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:32:45 2025

@author: guipe
"""

import pandas as pd
import numpy as np
from math import log2
from scipy.optimize import linear_sum_assignment

def generator(nCrit, N):
    """
    This function generates N weights vector.
    Output: Matrix N x nCrit.
    """
    weights_matrix = np.zeros((N, nCrit))

    for i in range(N):
        rand_num = np.random.rand(nCrit - 1)
        q = np.concatenate(([0], np.sort(rand_num), [1]))
        weights_rand = np.array([q[j] - q[j - 1] for j in range(1, nCrit + 1)])
        weights_matrix[i, :] = weights_rand

    return weights_matrix

def generator_ordinal(order, N):
    """
    This function generates N weights vector, given a predefined preference order among weights.
    Output: Matrix N x nCrit.
    """
    criteria = len(order)
    weights_matrix = np.zeros((N, criteria))

    for i in range(N):
        rand_num = np.random.rand(criteria - 1)
        q = np.concatenate(([0], np.sort(rand_num), [1]))
        weights_rand = np.array([q[j] - q[j - 1] for j in range(1, criteria + 1)])

        # Ordering weights according to preferential order among criteria
        weights_sorted = np.sort(weights_rand)
        weights = np.zeros(criteria)
        for j in range(criteria):
            k = criteria - order[j] + 1
            weights[j] = weights_sorted[k - 1]

        weights_matrix[i, :] = weights

    return weights_matrix


# Fairness metrics
def statistical_parity_at_k(ranking, group_labels, group_value, k):
    """
    Statistical Parity
    ranking: array of ordered alternatives
    group_labels: vetor of sensitive group labels
    group_value: value of the protected group
    k: first k positions to evaluate (top-k)
    """
    top_k = ranking[:k]
    prop_top_k = np.mean(group_labels[top_k] == group_value)
    prop_total = np.mean(group_labels == group_value)
    return abs(prop_top_k - prop_total)

def rKL(ranking, group_labels):
    """
    Normalized Discounted KL Divergence (rKL)
    ranking: array of ordered alternatives
    group_labels: vetor of sensitive group labels
    """
    # Visibility function
    def visibility(pos):
        return 1 / log2(pos + 2)

    # Expected distribution
    group_dist_total = pd.Series(group_labels).value_counts(normalize=True).sort_index()

    # Observed distribution
    vis = np.array([visibility(i) for i in range(len(ranking))])
    group_vis_sum = {}
    for g in group_dist_total.index:
        group_vis_sum[g] = vis[group_labels[ranking] == g].sum()
    total_vis = sum(group_vis_sum.values())
    group_dist_obs = pd.Series({g: v / total_vis for g, v in group_vis_sum.items()}).sort_index()

    # KL Divergence
    eps = 1e-12
    P = group_dist_obs.values + eps
    Q = group_dist_total.values + eps
    kl = np.sum(P * np.log(P / Q))
   
    return kl

def rKL2(ranking, group_labels, k):
    """
    Prefix-based rKL@k

    Computes the cumulative KL divergence between the group
    distribution observed in each top-i prefix (i=1,...,k)
    and the global group distribution.

    Parameters
    ----------
    ranking : array
        Ordered alternatives (best → worst)
    group_labels : array-like
        Sensitive group membership of alternatives
    k : int
        Top-k cutoff

    Returns
    -------
    float
        Discounted cumulative KL divergence up to top-k
    """

    eps = 1e-12
    k = min(k, len(ranking))

    # Target (population) distribution
    group_dist_total = (
        pd.Series(group_labels)
        .value_counts(normalize=True)
        .sort_index()
    )

    groups = group_dist_total.index.tolist()

    rkl_sum = 0.0

    for i in range(1, k + 1):

        prefix = ranking[:i]
        prefix_groups = group_labels[prefix]

        # Empirical distribution in top-i
        prefix_dist = (
            pd.Series(prefix_groups)
            .value_counts(normalize=True)
            .reindex(groups, fill_value=0.0)
        )

        # KL divergence at prefix i
        kl_i = 0.0
        for g in groups:
            p = max(prefix_dist[g], eps)
            q = max(group_dist_total[g], eps)
            kl_i += p * np.log(p / q)

        # Logarithmic discount (top positions matter more)
        discount = 1.0 / log2(i + 1)

        rkl_sum += discount * kl_i

    return rkl_sum


def ndkl(ranking, group_labels):
    """
    Normalized Discounted Cumulative KL-Divergence (ndkl)
    ranking: array of ordered alternatives
    group_labels: vetor of sensitive group labels
    """
    m = len(ranking)

    # Groups and expected distributions
    group_dist_target = (
        pd.Series(group_labels)
        .value_counts(normalize=True)
        .sort_index()
    )

    groups = group_dist_target.index.tolist()
    eps = 1e-12

    ndkl_sum = 0.0
    Z = 0.0  # Normalization

    for i in range(1, m + 1):
        prefix = ranking[:i]
        prefix_groups = group_labels[prefix]

        # Empirical distribution in top-i
        prefix_dist = (
            pd.Series(prefix_groups)
            .value_counts(normalize=True)
            .reindex(groups, fill_value=0.0)
        )

        # KL divergence in prefix i
        kl_i = 0.0
        for g in groups:
            p = max(prefix_dist[g], eps)
            q = max(group_dist_target[g], eps)
            kl_i += p * np.log(p / q)

        # Log discount
        discount = 1.0 / log2(i + 1)
        ndkl_sum += discount * kl_i
        Z += discount

    return ndkl_sum / Z


def central_ranking_expected_rank(A, labels=None):
    """
    Method: Expected ranking
    This function geenrates a single ranking from the acceptability matrix
    """
    nAlt = A.shape[0]
    pos = np.arange(1, nAlt+1)
    exp_pos = A @ pos  # E[posição] para cada alternativa
    order = np.argsort(exp_pos)  # menor posição esperada é melhor
    names = labels if labels is not None else [f"Alt_{i+1}" for i in range(nAlt)]
    df = pd.DataFrame({"Alt": names, "E_pos": exp_pos})
    df["rank"] = 1 + df["E_pos"].rank(method="dense").astype(int) - df["E_pos"].rank(method="dense").min()
    df = df.iloc[order].reset_index(drop=True)
    return order, df

def central_ranking_mar(A, labels=None):
    """
    Maximum Acceptability Ranking (MAR)
    solved as a linear assignment problem
    using the Hungarian algorithm.

    MUCH faster than MILP formulation.
    """

    A = np.array(A)
    m = A.shape[0]

    # Hungarian solves minimization → convert
    cost = -A

    row_ind, col_ind = linear_sum_assignment(cost)

    assigned_pos = col_ind
    order = np.argsort(assigned_pos)

    mar_score = np.array([A[i, assigned_pos[i]] for i in range(m)])

    names = labels if labels is not None else [f"Alt_{i+1}" for i in range(m)]

    df = pd.DataFrame({
        "Alt": names,
        "mar_score": mar_score
    })

    df["rank"] = 1 + (-df["mar_score"]).rank(method="dense").astype(int) - (-df["mar_score"]).rank(method="dense").min()

    df = df.iloc[order].reset_index(drop=True)

    return order, df  

def get_groups_from_alts(alt_names, groups):
    indices = [int(a.split("_")[1]) - 1 for a in alt_names]  # pega o número depois de Alt_
    return groups.values[indices]

def generate_synthetic_bias1(n=100, seed=42):
    """
    Bias conditioned to a group (ex: different schools)
    """
    np.random.seed(seed)

    groups = np.array(['A'] * (n//2) + ['B'] * (n//2))

    # Critérios não enviesados
    C1 = np.random.normal(0.6, 0.1, n)
    C2 = np.random.normal(0.6, 0.1, n)

    # Critério enviesado (independente do mérito)
    bias = np.where(groups == 'A', 0.15, -0.15)
    C3 = np.random.normal(0.5, 0.1, n) + bias

    data = pd.DataFrame({
        'C1_merit': C1,
        'C2_merit': C2,
        'C3_structural_bias': C3
    })

    return data, pd.Series(groups)


def generate_synthetic_bias2(n=100, seed=20):
    """
    Bias cinditionedViés condicional ao mérito (ex: cartas de recomendação)
    """
    np.random.seed(seed)

    groups = np.array(['A'] * (n//2) + ['B'] * (n//2))

    # Mérito real
    C1 = np.random.normal(0.6, 0.1, n)
    C2 = np.random.normal(0.6, 0.1, n)

    # Viés condicional
    bias = np.where(groups == 'A', 0.15, -0.15)
    C3 = 0.6 * C1 + bias

    data = pd.DataFrame({
        'C1_merit': C1,
        'C2_merit': C2,
        'C3_conditional_bias': C3
    })

    return data, pd.Series(groups)

def fairness_weighted_central_weight_vector(weights_matrix,rankings,fairness_values,fairness_max):
    """
    Fairness-weighted SMAA central weight vectors.

    This function computes a fairness-aware version of the SMAA central weight vector.
    """

    N, nCrit = weights_matrix.shape
    nAlt = rankings.shape[1]

    central_weights = np.zeros((nAlt, nCrit))

    for alt in range(nAlt):

        # Simulations where alternative is ranked first
        idx = np.where(rankings[:, 0] == alt)[0]

        if len(idx) == 0:
            continue

        # Fairness-adjusted weights
        fairness_adjustment = fairness_max - fairness_values[idx]

        # Avoid division by zero
        if fairness_adjustment.sum() == 0:
            fairness_adjustment = np.ones(len(idx))

        # Weighted average of weight vectors
        central_weights[alt] = np.average(weights_matrix[idx],axis=0,weights=fairness_adjustment)

    return central_weights

# Selecting the dataset
# Options: "real", "synthetic_bias1", "synthetic_bias2"

DATA_MODE = "real"  

if DATA_MODE == "real":
    data = pd.read_excel('data_countries_2020_norm.xlsx')
    groups = data.Grupo
    data = data.drop(columns=['Country','Grupo'])

elif DATA_MODE == "synthetic_bias1":
    data, groups = generate_synthetic_bias1(n=100)

elif DATA_MODE == "synthetic_bias2":
    data, groups = generate_synthetic_bias2(n=100)

else:
    raise ValueError("Unknown DATA_MODE")

# Normalizing the dataset (assuming that all criteria are to be maximized)
data = (data-data.min())/(data.max()-data.min())
nAlt,nCrit = data.shape

# SMAA parameters
N = 10000 # Number of realizations
weights_matrix = generator(nCrit,N) # Generating the set of weights

# Calculating global scores
global_scores = data.values @ weights_matrix.T

# Defining top-k
k = 20

# Fairness analysis
StatPar_all = []
rKL_all = []
ndkl_all = []
all_rankings = []

# Calculating SMAA acceptability matrix
acceptability_matrix = np.zeros((nAlt, nAlt)) # Classical SMAA
acceptability_matrix_StatPar = np.zeros((nAlt, nAlt))
acceptability_matrix_rKL = np.zeros((nAlt, nAlt))
acceptability_matrix_ndkl = np.zeros((nAlt, nAlt))

for j in range(N):
    scores = global_scores[:, j]  # Scores for weight j
    ranking = np.argsort(scores)[::-1]  # Decreasing order
    all_rankings.append(ranking)
    
    # Fairness metrics
    StatPar_all.append(statistical_parity_at_k(ranking, groups.values, groups.unique()[0], k))
    rKL_all.append(rKL2(ranking, groups.values, k))
    #rKL_all.append(rKL(ranking, groups.values))
    ndkl_all.append(ndkl(ranking, groups.values))
    
    for pos, alt in enumerate(ranking):
        
        acceptability_matrix[alt, pos] += 1
    
        # Acceptability matrices for SMAA-fair - adjusting to the higher the better
        acceptability_matrix_StatPar[alt, pos] += StatPar_all[j]
        acceptability_matrix_rKL[alt, pos] += rKL_all[j]
        acceptability_matrix_ndkl[alt, pos] += ndkl_all[j]

StatPar_all = np.array(StatPar_all)
rKL_all = np.array(rKL_all)
ndkl_all = np.array(ndkl_all)
all_rankings = np.array(all_rankings)

# Extracting the maximum fairness metrics (lowest fairness degrees)
StatPar_all_max = np.max(StatPar_all)
rKL_all_max = np.max(rKL_all)
ndkl_all_max = np.max(ndkl_all)

# Adjusting to the higher the better
acceptability_matrix_StatPar = StatPar_all_max*acceptability_matrix - acceptability_matrix_StatPar
acceptability_matrix_rKL = rKL_all_max*acceptability_matrix - acceptability_matrix_rKL
acceptability_matrix_ndkl = ndkl_all_max*acceptability_matrix - acceptability_matrix_ndkl

print(f"Statistical Parity - média: {StatPar_all.mean():.4f}")
print(f"rKL - média: {rKL_all.mean():.4f}")
print(f"ndkl - média: {ndkl_all.mean():.4f}")
        

# Normalization
acceptability_matrix /= N
acceptability_matrix_StatPar /= np.sum(acceptability_matrix_StatPar,axis=1)
acceptability_matrix_rKL /= np.sum(acceptability_matrix_rKL,axis=1)
acceptability_matrix_ndkl /= np.sum(acceptability_matrix_ndkl,axis=1)

# Dataframes for visualization
acceptability_df = pd.DataFrame(acceptability_matrix,
    index=[f"Alt_{i+1}" for i in range(nAlt)],
    columns=[f"Pos_{p+1}" for p in range(nAlt)]
)

acceptability_df_StatPar = pd.DataFrame(
    acceptability_matrix_StatPar,
    index=[f"Alt_{i+1}" for i in range(nAlt)],
    columns=[f"Pos_{p+1}" for p in range(nAlt)]
)

acceptability_df_rKL = pd.DataFrame(
    acceptability_matrix_rKL,
    index=[f"Alt_{i+1}" for i in range(nAlt)],
    columns=[f"Pos_{p+1}" for p in range(nAlt)]
)

acceptability_df_ndkl = pd.DataFrame(
    acceptability_matrix_ndkl,
    index=[f"Alt_{i+1}" for i in range(nAlt)],
    columns=[f"Pos_{p+1}" for p in range(nAlt)]
)

# Defining a single ranking from the acceptability matrix
# 1) Expected ranking
ord_exp, df_exp = central_ranking_expected_rank(acceptability_df, acceptability_df.index)
ord_exp_StatPar, df_exp_StatPar = central_ranking_expected_rank(acceptability_df_StatPar, acceptability_df_StatPar.index)
ord_exp_rKL, df_exp_rKL = central_ranking_expected_rank(acceptability_df_rKL, acceptability_df_rKL.index)
ord_exp_ndkl, df_exp_ndkl = central_ranking_expected_rank(acceptability_df_ndkl, acceptability_df_ndkl.index)

# 2) Maximum acceptability ranking
order_mar, df_mar = central_ranking_mar(acceptability_df, acceptability_df.index)
order_mar_StatPar, df_mar_StatPar = central_ranking_mar(acceptability_df_StatPar, acceptability_df_StatPar.index)
order_mar_rKL, df_mar_rKL = central_ranking_mar(acceptability_df_rKL, acceptability_df_rKL.index)
order_mar_ndkl, df_mar_ndkl = central_ranking_mar(acceptability_df_ndkl, acceptability_df_ndkl.index)

# Extracting the fairness-aware weights central vector
central_weights_classic = fairness_weighted_central_weight_vector(weights_matrix,all_rankings,np.zeros(N),1)
central_weights_StatPar = fairness_weighted_central_weight_vector(weights_matrix,all_rankings,StatPar_all,StatPar_all_max)
central_weights_rKL = fairness_weighted_central_weight_vector(weights_matrix,all_rankings,rKL_all,rKL_all_max)
central_weights_ndkl = fairness_weighted_central_weight_vector(weights_matrix,all_rankings,ndkl_all,ndkl_all_max)

summary = pd.DataFrame({
    'Rank_ER': df_exp.Alt,
    'Group_ER': get_groups_from_alts(df_exp.Alt, groups),
    'Rank_MAR': df_mar.Alt,
    'Group_MAR': get_groups_from_alts(df_mar.Alt, groups),
    'Rank_ER_StatPar': df_exp_StatPar.Alt,
    'Group_ER_StatPar': get_groups_from_alts(df_exp_StatPar.Alt, groups),
    'Rank_MAR_StatPar': df_mar_StatPar.Alt,
    'Group_MAR_StatPar': get_groups_from_alts(df_mar_StatPar.Alt, groups),
    'Rank_ER_rKL': df_exp_rKL.Alt,
    'Group_ER_rKL': get_groups_from_alts(df_exp_rKL.Alt, groups),
    'Rank_MAR_rKL': df_mar_rKL.Alt,
    'Group_MAR_rKL': get_groups_from_alts(df_mar_rKL.Alt, groups),
    'Rank_ER_ndkl': df_exp_ndkl.Alt,
    'Group_ER_ndkl': get_groups_from_alts(df_exp_ndkl.Alt, groups),
    'Rank_MAR_ndkl': df_mar_ndkl.Alt,
    'Group_MAR_ndkl': get_groups_from_alts(df_mar_ndkl.Alt, groups)
})