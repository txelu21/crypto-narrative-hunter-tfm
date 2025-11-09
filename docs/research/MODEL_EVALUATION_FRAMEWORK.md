# Model Evaluation Framework
**Crypto Narrative Hunter - Master Thesis**

**Version:** 1.0
**Date:** October 8, 2025
**Status:** Active - Validation Reference

---

## Purpose

This document defines the comprehensive evaluation framework for assessing clustering quality, statistical significance, and hypothesis validation. It directly addresses **Tutor Question 6: "How will you measure if your model works?"** by providing detailed metrics, formulas, interpretation guidelines, and acceptance criteria.

---

## Overview

**Evaluation Dimensions:**
1. **Clustering Quality** - Are clusters well-formed and meaningful?
2. **Statistical Significance** - Are findings statistically valid?
3. **Cluster Stability** - Are results robust across different samples?
4. **Cross-Validation** - Do Tier 1 findings generalize to Tier 2?
5. **Interpretability** - Are clusters actionable and domain-relevant?

**Problem Type:** Unsupervised clustering (no ground truth labels)
**Primary Algorithm:** HDBSCAN (density-based, auto-determines k)
**Comparison Algorithm:** K-Means (classical partitioning, manual k selection)

---

## 1. Clustering Quality Metrics

### 1.1 Silhouette Score

**What It Measures:** Cluster cohesion (how similar points are within clusters) vs separation (how different clusters are from each other)

**Formula:**
```python
for each point i:
    a(i) = average distance to other points in same cluster
    b(i) = average distance to points in nearest neighbor cluster

    silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))

# Overall score
silhouette_avg = mean(silhouette(i) for all points)
```

**Implementation:**
```python
from sklearn.metrics import silhouette_score, silhouette_samples

# Overall score
silhouette_avg = silhouette_score(X, cluster_labels)

# Per-cluster scores
cluster_scores = {}
for k in unique_clusters:
    cluster_mask = (cluster_labels == k)
    cluster_scores[k] = silhouette_samples(X, cluster_labels)[cluster_mask].mean()
```

**Interpretation:**
| Range | Interpretation | Action |
|-------|----------------|--------|
| 0.7 - 1.0 | Strong structure | Excellent separation ✅ |
| 0.5 - 0.7 | Reasonable structure | Good separation ✅ |
| 0.25 - 0.5 | Weak structure | Marginal separation ⚠️ |
| < 0.25 | No substantial structure | Poor separation ❌ |

**Acceptance Criteria:**
- ✅ **Overall silhouette ≥ 0.5** (H1 hypothesis acceptance threshold)
- ✅ **Per-cluster silhouette ≥ 0.4** (no weak clusters)
- ✅ **No cluster with negative silhouette** (misclassified points)

**Visualization:**
- Silhouette plot: One bar per cluster showing score distribution
- Identify clusters with low scores → candidates for refinement

**Limitations:**
- Sensitive to cluster shape (assumes convex clusters)
- Computationally expensive for large datasets (O(n²))
- May penalize elongated or irregular clusters

---

### 1.2 Davies-Bouldin Index (DBI)

**What It Measures:** Average similarity ratio of each cluster with its most similar cluster. Lower is better.

**Formula:**
```python
for each cluster i:
    # Within-cluster scatter
    S_i = (1/n_i) * sum(distance(x, centroid_i) for x in cluster_i)

    # Between-cluster distance
    for each other cluster j:
        M_ij = distance(centroid_i, centroid_j)

        # Similarity ratio
        R_ij = (S_i + S_j) / M_ij

    # Maximum similarity (worst case)
    D_i = max(R_ij for all j != i)

# Overall DBI
DBI = (1/k) * sum(D_i for all clusters)
```

**Implementation:**
```python
from sklearn.metrics import davies_bouldin_score

dbi = davies_bouldin_score(X, cluster_labels)
```

**Interpretation:**
| Range | Interpretation | Action |
|-------|----------------|--------|
| 0.0 - 0.5 | Excellent separation | Very distinct clusters ✅ |
| 0.5 - 1.0 | Good separation | Well-defined clusters ✅ |
| 1.0 - 2.0 | Moderate separation | Acceptable ⚠️ |
| > 2.0 | Poor separation | Clusters overlap ❌ |

**Acceptance Criteria:**
- ✅ **DBI < 1.0** (H1 hypothesis acceptance threshold)
- ✅ **Lower is better** (minimize overlapping clusters)

**Advantages:**
- Computationally efficient (O(n·k))
- Intuitive interpretation (lower = better separation)
- Works well with K-Means (centroid-based)

**Limitations:**
- Assumes spherical clusters (less suitable for HDBSCAN)
- Sensitive to outliers in centroid calculation

---

### 1.3 Calinski-Harabasz Score (Variance Ratio)

**What It Measures:** Ratio of between-cluster variance to within-cluster variance. Higher is better.

**Formula:**
```python
# Between-cluster variance
B = sum(n_k * distance(centroid_k, global_centroid)^2 for all clusters k)

# Within-cluster variance
W = sum(sum(distance(x, centroid_k)^2 for x in cluster_k) for all clusters k)

# Variance ratio
CH = (B / W) * ((n - k) / (k - 1))
```
where:
- n = total number of points
- k = number of clusters

**Implementation:**
```python
from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(X, cluster_labels)
```

**Interpretation:**
- **No absolute threshold** (use for relative comparison)
- **Higher is better** (more compact, well-separated clusters)
- Compare across different k values (K-Means) or parameter settings (HDBSCAN)

**Use Cases:**
- Selecting optimal k for K-Means (elbow method companion)
- Comparing HDBSCAN parameter settings (min_cluster_size, min_samples)
- Validating that chosen solution is better than alternatives

**Acceptance Criteria:**
- ✅ **Use for relative comparison** (no absolute threshold)
- ✅ **Chosen model has higher CH than alternatives**

**Advantages:**
- Fast computation (O(n·k))
- Works well for convex clusters
- Helpful for parameter tuning

**Limitations:**
- Not reliable for density-based clusters (HDBSCAN)
- Sensitive to outliers

---

### 1.4 Elbow Method (K-Means Only)

**What It Measures:** Within-cluster sum of squares (inertia) vs number of clusters k

**Formula:**
```python
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia[k] = kmeans.inertia_  # Sum of squared distances to nearest centroid
```

**Implementation:**
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
silhouettes = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# Plot elbow curve
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Plot silhouette curve
plt.plot(k_range, silhouettes, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs k')
plt.show()
```

**Interpretation:**
- Look for "elbow" in inertia curve (diminishing returns)
- Cross-reference with silhouette plot (choose k where silhouette is high AND elbow appears)

**Acceptance Criteria:**
- ✅ **Elbow visible between k=3 and k=7** (expected range)
- ✅ **Silhouette maximized near elbow point**

---

## 2. Statistical Validation Metrics

### 2.1 Chi-Square Test of Independence (H2: Cluster-Narrative Affinity)

**What It Measures:** Whether cluster membership and narrative preference are statistically independent

**Null Hypothesis (H0):** Cluster and narrative are independent (no affinity)
**Alternative Hypothesis (Ha):** Cluster and narrative are dependent (affinity exists)

**Formula:**
```python
# Contingency table
observed = pd.crosstab(cluster_labels, narrative_categories)

# Expected frequencies (under independence)
expected = (observed.sum(axis=1)[:, None] * observed.sum(axis=0)) / observed.sum().sum()

# Chi-square statistic
chi2 = sum((observed - expected)^2 / expected)

# Degrees of freedom
df = (n_clusters - 1) * (n_narratives - 1)

# p-value
p_value = 1 - chi2.cdf(chi2, df)
```

**Implementation:**
```python
from scipy.stats import chi2_contingency

# Create contingency table (clusters × narratives)
contingency_table = pd.crosstab(
    df['cluster'],
    df['dominant_narrative']  # Most allocated narrative per wallet
)

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Effect size (Cramér's V)
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n * min_dim))

print(f"Chi-square: {chi2:.2f}, p-value: {p_value:.4f}, Cramér's V: {cramers_v:.3f}")
```

**Interpretation:**
| p-value | Interpretation | Decision |
|---------|----------------|----------|
| < 0.001 | Very strong evidence | Reject H0 ✅ |
| 0.001 - 0.01 | Strong evidence | Reject H0 ✅ |
| 0.01 - 0.05 | Moderate evidence | Reject H0 ✅ |
| > 0.05 | Insufficient evidence | Fail to reject H0 ❌ |

**Effect Size (Cramér's V):**
| Value | Interpretation |
|-------|----------------|
| 0.1 | Small effect |
| 0.3 | Medium effect |
| 0.5 | Large effect |

**Acceptance Criteria:**
- ✅ **p-value < 0.05** (reject null, accept H2 hypothesis)
- ✅ **Cramér's V ≥ 0.3** (medium-to-large effect size)
- ✅ **At least 2 clusters show >20% overweight in a narrative**

**Post-hoc Analysis:**
- Adjusted residuals: Identify which cluster-narrative pairs drive the association
- Bonferroni correction for multiple comparisons

---

### 2.2 ANOVA / Kruskal-Wallis (Performance Differences Across Clusters)

**What It Measures:** Whether performance metrics (ROI, Sharpe) differ significantly across clusters

**Parametric (ANOVA):**
```python
from scipy.stats import f_oneway

# Test if ROI differs across clusters
cluster_rois = [df[df.cluster == k]['roi_pct'] for k in unique_clusters]
f_stat, p_value = f_oneway(*cluster_rois)
```

**Non-Parametric (Kruskal-Wallis):**
```python
from scipy.stats import kruskal

# Use if data not normally distributed
h_stat, p_value = kruskal(*cluster_rois)
```

**Interpretation:**
- p < 0.05: At least one cluster differs significantly from others

**Post-hoc (Pairwise Comparisons):**
```python
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Pairwise comparisons
p_values = []
for i in range(len(unique_clusters)):
    for j in range(i+1, len(unique_clusters)):
        _, p = mannwhitneyu(cluster_rois[i], cluster_rois[j])
        p_values.append(p)

# Bonferroni correction
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
```

**Acceptance Criteria:**
- ✅ **p-value < 0.05** (clusters differ in performance)
- ✅ **Effect size (Cohen's d) ≥ 0.5** for at least one pairwise comparison

---

### 2.3 Independent t-test / Mann-Whitney U (H3, H5)

**Use Cases:**
- H3: Early adopters vs late adopters performance
- H5: Add-on-dips wallets vs others performance

**Parametric (t-test):**
```python
from scipy.stats import ttest_ind

# Compare two groups
t_stat, p_value = ttest_ind(group1_roi, group2_roi)

# Effect size (Cohen's d)
mean_diff = group1_roi.mean() - group2_roi.mean()
pooled_std = np.sqrt((group1_roi.std()**2 + group2_roi.std()**2) / 2)
cohens_d = mean_diff / pooled_std
```

**Non-Parametric (Mann-Whitney U):**
```python
from scipy.stats import mannwhitneyu

# Use if data not normally distributed
u_stat, p_value = mannwhitneyu(group1_roi, group2_roi, alternative='two-sided')
```

**Interpretation:**
| p-value | Cohen's d | Interpretation |
|---------|-----------|----------------|
| < 0.05 | ≥ 0.8 | Large, significant difference ✅ |
| < 0.05 | 0.5 - 0.8 | Medium, significant difference ✅ |
| < 0.05 | 0.2 - 0.5 | Small, significant difference ⚠️ |
| ≥ 0.05 | Any | No significant difference ❌ |

**Acceptance Criteria:**
- ✅ **p-value < 0.05**
- ✅ **Cohen's d ≥ 0.5** (medium effect size)

---

### 2.4 Correlation Analysis (H4: Concentration vs Performance)

**Linear Correlation (Pearson):**
```python
from scipy.stats import pearsonr

# Test linear relationship
r, p_value = pearsonr(df['portfolio_hhi'], df['sharpe_ratio'])

print(f"Pearson r: {r:.3f}, p-value: {p_value:.4f}")
```

**Interpretation:**
| |r| | Interpretation |
|------|----------------|
| 0.7 - 1.0 | Strong correlation |
| 0.4 - 0.7 | Moderate correlation |
| 0.1 - 0.4 | Weak correlation |
| < 0.1 | No correlation |

**Non-Linear Relationship (Polynomial Regression):**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Test for inverted U-curve (quadratic)
X = df[['portfolio_hhi']]
y = df['sharpe_ratio']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

r_squared = model.score(X_poly, y)
beta_quadratic = model.coef_[2]  # Coefficient for HHI^2 term

print(f"R²: {r_squared:.3f}, β₂ (quadratic): {beta_quadratic:.4f}")
```

**Acceptance Criteria (H4):**
- ✅ **Linear: |r| > 0.3 AND p < 0.05** OR
- ✅ **Quadratic: R² > 0.2 AND β₂ significant (p < 0.05)**
- ✅ **Inverted U-curve:** β₂ < 0 (negative quadratic term)

---

## 3. Cluster Stability Metrics

### 3.1 Bootstrap Validation

**What It Measures:** Stability of clustering across different random samples

**Method:**
```python
from sklearn.metrics import adjusted_rand_score

def bootstrap_clustering(X, n_iterations=1000):
    """Bootstrap validation for cluster stability"""
    ari_scores = []

    # Original clustering
    original_labels = hdbscan_cluster(X)

    for i in range(n_iterations):
        # Resample with replacement
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_bootstrap = X[indices]

        # Cluster bootstrap sample
        bootstrap_labels = hdbscan_cluster(X_bootstrap)

        # Map back to original indices
        bootstrap_labels_aligned = map_labels_to_original(bootstrap_labels, indices)

        # Calculate Adjusted Rand Index
        ari = adjusted_rand_score(original_labels, bootstrap_labels_aligned)
        ari_scores.append(ari)

    return {
        'mean_ari': np.mean(ari_scores),
        'std_ari': np.std(ari_scores),
        'ci_95': np.percentile(ari_scores, [2.5, 97.5])
    }
```

**Interpretation:**
| Mean ARI | Interpretation |
|----------|----------------|
| 0.9 - 1.0 | Excellent stability ✅ |
| 0.7 - 0.9 | Good stability ✅ |
| 0.5 - 0.7 | Moderate stability ⚠️ |
| < 0.5 | Poor stability ❌ |

**Acceptance Criteria:**
- ✅ **Mean ARI > 0.7** (H1 hypothesis validation)
- ✅ **95% CI does not include 0** (consistently better than random)

---

### 3.2 Adjusted Rand Index (ARI)

**What It Measures:** Similarity between two clusterings, adjusted for chance

**Formula:**
```python
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```
where:
- RI = Rand Index (percentage of point pairs correctly clustered)
- E[RI] = Expected RI under random labeling

**Range:** -1 to 1
- 1.0 = Perfect agreement
- 0.0 = Random labeling
- Negative = Worse than random

**Use Cases:**
- Comparing HDBSCAN vs K-Means results
- Bootstrap stability validation
- Tier 1 vs Tier 2 validation (if labels can be inferred)

---

## 4. Cross-Validation Strategy

### 4.1 Tier 1 vs Tier 2 Validation

**Tier 1 (Deep Analysis):** 2,343 wallets with full transaction data
**Tier 2 (Aggregate Validation):** 22,818 wallets with pool participation data

**Validation Approach:**
```python
# Step 1: Cluster Tier 1 wallets
tier1_labels = hdbscan_cluster(tier1_features)

# Step 2: Assign cluster labels based on dominant characteristics
cluster_profiles = {}
for k in unique_clusters:
    cluster_profiles[k] = {
        'avg_trade_frequency': tier1[tier1.cluster == k]['trade_frequency'].mean(),
        'dominant_narrative': tier1[tier1.cluster == k]['dominant_narrative'].mode(),
        'avg_roi': tier1[tier1.cluster == k]['roi_pct'].mean()
    }

# Step 3: Classify Tier 2 wallets using pool participation
# (Cannot do full clustering, but can validate aggregate trends)
for narrative in ['AI', 'DeFi', 'Gaming', 'Meme']:
    tier1_exposure = tier1.groupby('cluster')[f'{narrative}_exposure_pct'].mean()
    tier2_exposure = tier2[f'{narrative}_pool_volume_pct'].mean()

    # Compare trends (qualitative validation)
    print(f"{narrative}: Tier 1 top cluster = {tier1_exposure.max():.1f}%, Tier 2 avg = {tier2_exposure:.1f}%")
```

**Validation Checks:**
- ✅ Narrative preferences in Tier 1 clusters align with Tier 2 pool participation
- ✅ High-performing clusters in Tier 1 correspond to high-volume wallets in Tier 2
- ✅ Trading frequency distributions similar across tiers

**Limitations:**
- Tier 2 lacks individual transactions → cannot replicate full clustering
- Validation is directional/qualitative, not quantitative

---

## 5. Interpretability Checks

### 5.1 Cluster Profiling

**For Each Cluster:**
```python
def profile_cluster(df, cluster_id):
    cluster_data = df[df.cluster == cluster_id]

    profile = {
        'size': len(cluster_data),
        'avg_roi_pct': cluster_data['roi_pct'].mean(),
        'median_sharpe': cluster_data['sharpe_ratio'].median(),
        'dominant_narrative': cluster_data['dominant_narrative'].mode()[0],
        'avg_trade_frequency': cluster_data['trade_frequency'].mean(),
        'avg_holding_period': cluster_data['avg_holding_period_days'].mean(),
        'avg_portfolio_hhi': cluster_data['portfolio_hhi'].mean()
    }

    return profile
```

**Interpretability Criteria:**
- ✅ **Clusters have distinct profiles** (not overlapping characteristics)
- ✅ **Labels are domain-meaningful** (e.g., "Early AI Adopter", not "Cluster 3")
- ✅ **Actionable insights** (investors can learn from cluster strategies)

---

### 5.2 Feature Importance (Post-Clustering)

**Identify which features best separate clusters:**
```python
from sklearn.ensemble import RandomForestClassifier

# Train classifier on cluster labels (post-hoc)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, cluster_labels)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

**Use Cases:**
- Identify key differentiators between clusters
- Simplify cluster descriptions (focus on top 3-5 features)
- Validate that clustering uses meaningful features (not noise)

---

## 6. Acceptance Thresholds Summary

### Hypothesis-Level Thresholds

| Hypothesis | Metric | Threshold | Status |
|------------|--------|-----------|--------|
| **H1: Clustering** | Silhouette Score | ≥ 0.5 | Required |
| **H1: Clustering** | Davies-Bouldin Index | < 1.0 | Required |
| **H1: Clustering** | Bootstrap ARI | > 0.7 | Required |
| **H2: Affinity** | Chi-square p-value | < 0.05 | Required |
| **H2: Affinity** | Cramér's V | ≥ 0.3 | Required |
| **H3: Early Adoption** | t-test p-value | < 0.05 | Required |
| **H3: Early Adoption** | Cohen's d | ≥ 0.5 | Required |
| **H4: Concentration** | Pearson r OR R² | \|r\| > 0.3 OR R² > 0.2 | Required |
| **H5: Accumulation** | t-test p-value | < 0.05 | Required |
| **H5: Accumulation** | Cohen's d | ≥ 0.5 | Required |

---

## 7. Evaluation Pipeline (Epic 5)

### Step 1: Clustering Quality (Epic 4.3 output)
```python
# HDBSCAN results
silhouette_hdbscan = silhouette_score(X, hdbscan_labels)
dbi_hdbscan = davies_bouldin_score(X, hdbscan_labels)
ch_hdbscan = calinski_harabasz_score(X, hdbscan_labels)

# K-Means results (k=3, 4, 5)
for k in [3, 4, 5]:
    silhouette_kmeans = silhouette_score(X, kmeans_labels[k])
    # ... record all metrics

# Select best model
if silhouette_hdbscan >= 0.5 and dbi_hdbscan < 1.0:
    final_labels = hdbscan_labels
    print("✅ H1 accepted: HDBSCAN produces valid clusters")
else:
    # Try K-Means alternatives
    pass
```

### Step 2: Statistical Validation (Epic 4.4 + Epic 5)
```python
# H2: Cluster-Narrative Affinity
chi2, p, cramers_v = test_cluster_narrative_affinity(df)
if p < 0.05 and cramers_v >= 0.3:
    print("✅ H2 accepted: Significant cluster-narrative affinity")

# H3-H5: Performance hypotheses
test_early_adoption_hypothesis(df)
test_concentration_hypothesis(df)
test_accumulation_hypothesis(df)
```

### Step 3: Bootstrap Stability (Epic 5)
```python
bootstrap_results = bootstrap_clustering(X, n_iterations=1000)
if bootstrap_results['mean_ari'] > 0.7:
    print("✅ Clusters are stable across bootstrap samples")
```

### Step 4: Generate Report
```python
# outputs/documentation/EVALUATION_REPORT.md
generate_evaluation_report(
    clustering_metrics={'silhouette': 0.63, 'dbi': 0.82, ...},
    hypothesis_results={'H1': 'accepted', 'H2': 'accepted', ...},
    bootstrap_stability={'mean_ari': 0.78, 'ci_95': (0.72, 0.84)},
    tier2_validation={'narrative_alignment': 'confirmed', ...}
)
```

---

## 8. Failure Modes and Contingency Plans

### Failure Mode 1: Silhouette < 0.5
**Cause:** Features don't separate wallets well
**Action:**
- Re-engineer features (add interaction terms, transformations)
- Try different normalization (Robust scaler, Min-Max)
- Reduce feature set (PCA, feature selection)
- Revisit smart money filtering criteria

### Failure Mode 2: No Cluster-Narrative Affinity (H2 fails)
**Cause:** Wallets don't specialize in narratives
**Action:**
- Test affinity at finer granularity (sub-narratives like "LLM AI" vs "Gaming AI")
- Relax threshold (0.2 < Cramér's V < 0.3 = weak affinity, still report)
- Pivot research question: "Do archetypes exist independent of narratives?"

### Failure Mode 3: Unstable Clusters (ARI < 0.7)
**Cause:** Sample size too small, or data too noisy
**Action:**
- Increase min_cluster_size in HDBSCAN (stricter clustering)
- Use ensemble clustering (consensus across multiple runs)
- Document limitation: "Clusters exist but lack bootstrap stability in 1-month window"

---

## 9. Reporting Standards

### Metrics to Report in Thesis
For each clustering result:
- Silhouette Score (overall + per-cluster)
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Number of clusters
- Cluster sizes (n wallets per cluster)

For each hypothesis test:
- Test statistic (χ², t, F, etc.)
- p-value
- Effect size (Cramér's V, Cohen's d, R²)
- Confidence intervals (95%)
- Interpretation (accept/reject H0)

For stability:
- Bootstrap ARI (mean, std, 95% CI)
- Number of iterations

---

## 10. Alignment with Tutor Questions

| Tutor Question | Document Section |
|----------------|------------------|
| **6. How will you measure success?** | Sections 1-3 (all metrics with formulas) |
| **6a. Classification** | N/A (unsupervised problem) |
| **6b. Clustering** | Section 1 (Silhouette Score as primary metric) |
| **6c. Time Series** | N/A (colleague's work) |

**Additional Coverage:**
- Statistical rigor (Section 2: Hypothesis testing)
- Stability validation (Section 3: Bootstrap)
- Interpretability (Section 5: Cluster profiling)

---

## 11. References

### Internal Documents
- `RESEARCH_HYPOTHESES.md` - Hypotheses validated by these metrics
- `FEATURE_ENGINEERING_SPEC.md` - Features used in clustering
- `DATA_DICTIONARY.md` - Data schema

### Statistical Methods
- **Silhouette:** Rousseeuw, P.J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis"
- **Davies-Bouldin:** Davies, D.L. & Bouldin, D.W. (1979). "A Cluster Separation Measure"
- **Calinski-Harabasz:** Caliński, T. & Harabasz, J. (1974). "A dendrite method for cluster analysis"
- **Chi-square:** Pearson, K. (1900). "On the criterion that a given system of deviations..."
- **ARI:** Hubert, L. & Arabie, P. (1985). "Comparing partitions"
- **Effect sizes:** Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences"

---

## Document Version Control

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | Oct 8, 2025 | Initial evaluation framework | Dev Agent |

---

**Status:** ACTIVE - Reference for Epic 5 validation and thesis results chapter
**Next Review:** After Epic 4.3 (clustering complete, apply metrics)
