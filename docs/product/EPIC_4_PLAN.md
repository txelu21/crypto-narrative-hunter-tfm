# Epic 4: Feature Engineering & Clustering
**Crypto Narrative Hunter - Master Thesis Project**

**Date:** October 19, 2025 (Updated: October 25, 2025)
**Status:** ✅ **STORY 4.1 COMPLETE - READY FOR CLUSTERING**
**Prerequisites:** ✅ ALL COMPLETE
- ✅ Epic 1-3: Data collection complete
- ✅ Story 4.2: Narrative classification complete (0% "Other")
- ✅ Codebase cleanup complete
- ✅ Data quality: A+ (98%)
- ✅ **Story 4.1: Feature engineering complete (33 features)**
- ✅ **Dataset validated and cleaned (ML-ready)**

---

## Epic Overview

Transform raw blockchain data into meaningful features for wallet clustering and archetype identification.

### Goals
1. Generate 42 engineered features per wallet
2. Identify 5-7 distinct smart money archetypes via clustering
3. Analyze cluster-narrative affinities
4. Achieve Silhouette Score ≥ 0.5
5. Answer all 5 research questions with statistical significance

### Timeline
- **Story 4.1:** Wallet Feature Engineering (3-4 days)
- **Story 4.2:** ✅ COMPLETE (Narrative Classification)
- **Story 4.3:** Clustering Analysis (2-3 days)
- **Story 4.4:** Cluster-Narrative Affinity (1-2 days)
- **Total:** 6-9 working days

---

## Story 4.1: Wallet Feature Engineering

### Objective
Calculate 42 features per wallet across 7 categories to enable clustering.

### Input Data
- ✅ `wallets.csv` - 2,343 Tier 1 wallets
- ✅ `transactions.csv` - 34,034 transactions
- ✅ `wallet_token_balances.csv` - 1,767,738 snapshots
- ✅ `tokens.csv` - 1,495 tokens (100% classified)
- ✅ `eth_prices.csv` - 729 hourly prices

### Feature Categories (Prioritized)

#### Category 1: Performance Metrics (7 features) - **CRITICAL**
*Priority: P0 - Required for clustering*

| # | Feature | Formula | Data Source |
|---|---------|---------|-------------|
| 1 | `roi_percent` | (final_value - initial_value) / initial_value × 100 | Balance snapshots + ETH prices |
| 2 | `win_rate` | profitable_trades / total_trades | Transactions |
| 3 | `sharpe_ratio` | mean(daily_returns) / std(daily_returns) × √365 | Balance snapshots |
| 4 | `max_drawdown_pct` | max((peak - trough) / peak) | Balance snapshots |
| 5 | `total_pnl_usd` | Σ(realized_gains) + unrealized_gains | Transactions + balances |
| 6 | `avg_trade_size_usd` | total_volume / total_trades | Transactions |
| 7 | `volume_consistency` | std(daily_volume) / mean(daily_volume) | Transactions |

**Implementation Priority:** FIRST
**Estimated Time:** 1 day

---

#### Category 2: Behavioral Features (8 features) - **CRITICAL**
*Priority: P0 - Core archetype differentiator*

| # | Feature | Formula | Archetype Relevance |
|---|---------|---------|---------------------|
| 8 | `trade_frequency` | total_trades / active_days | Active vs. Passive |
| 9 | `avg_holding_period_days` | mean(sell_time - buy_time) | Long-term vs. Short-term |
| 10 | `diamond_hands_score` | tokens_held_30d+ / total_traded | HODL behavior |
| 11 | `rotation_frequency` | token_switches / week | Portfolio churn |
| 12 | `weekend_activity_ratio` | weekend_trades / weekday_trades | Professional indicator |
| 13 | `night_trading_ratio` | (8pm-8am trades) / total | Geographic/lifestyle |
| 14 | `gas_optimization_score` | percentile_rank(gas_efficiency) | Sophistication |
| 15 | `dex_diversity_score` | shannon_entropy(dex_usage) | Platform specialization |

**Implementation Priority:** SECOND
**Estimated Time:** 1 day

---

#### Category 3: Portfolio Concentration (6 features) - **CRITICAL**
*Priority: P0 - Required for RQ4*

| # | Feature | Formula | Clustering Value |
|---|---------|---------|------------------|
| 16 | `portfolio_hhi` | Σ(token_value_i / total_value)² | Concentration index |
| 17 | `portfolio_gini` | gini_coefficient(holdings) | Inequality measure |
| 18 | `top3_concentration_pct` | sum(top_3_values) / total × 100 | Top-heavy indicator |
| 19 | `num_tokens_avg` | mean(daily_token_count) | Diversification level |
| 20 | `num_tokens_std` | std(daily_token_count) | Portfolio volatility |
| 21 | `portfolio_turnover` | tokens_changed / avg_size | Churn rate |

**Implementation Priority:** THIRD
**Estimated Time:** 0.5 day

---

#### Category 4: Narrative Exposure (6 features) - **CRITICAL**
*Priority: P0 - Required for RQ2 (NOW UNBLOCKED)*

| # | Feature | Formula | RQ2 Relevance |
|---|---------|-------------|---------------|
| 22 | `narrative_diversity_score` | shannon_entropy(narrative_dist) | Specialization |
| 23 | `primary_narrative_pct` | max(narrative_exposure) | Dominant interest |
| 24 | `defi_exposure_pct` | defi_volume / total_volume | DeFi focus |
| 25 | `ai_exposure_pct` | ai_volume / total_volume | AI narrative focus |
| 26 | `meme_exposure_pct` | meme_volume / total_volume | Meme coin interest |
| 27 | `stablecoin_usage_ratio` | stablecoin_volume / total | Risk aversion |

**✅ UNBLOCKED:** 100% token classification complete
**Implementation Priority:** FOURTH
**Estimated Time:** 0.5 day

---

#### Category 5: Accumulation/Distribution (6 features) - **CRITICAL**
*Priority: P0 - Required for RQ5*

| # | Feature | Formula | Pattern Detected |
|---|---------|---------|------------------|
| 28 | `accumulation_phase_days` | days_with_net_positive_change | Buying pressure |
| 29 | `distribution_phase_days` | days_with_net_negative_change | Selling pressure |
| 30 | `accumulation_intensity` | avg_daily_increase / portfolio_value | Conviction strength |
| 31 | `distribution_intensity` | avg_daily_decrease / portfolio_value | Exit aggression |
| 32 | `balance_volatility` | std(daily_value_changes) | Portfolio stability |
| 33 | `trend_direction` | linear_regression_slope(portfolio_value) | Growth trajectory |

**Implementation Priority:** FIFTH
**Estimated Time:** 0.5 day

---

#### Category 6: Timing & Sophistication (5 features) - **OPTIONAL**
*Priority: P1 - Nice to have, not critical*

| # | Feature | Formula | Sophistication Signal |
|---|---------|---------|----------------------|
| 34 | `early_entry_score` | mean(days_from_launch_to_first_buy) | Early adopter tendency |
| 35 | `avg_entry_price_percentile` | percentile(buy_price, 30d_range) | Entry timing quality |
| 36 | `avg_exit_price_percentile` | percentile(sell_price, 30d_range) | Exit timing quality |
| 37 | `market_timing_score` | correlation(trade_size, optimal_timing) | Timing skill |
| 38 | `contrarian_score` | inverse_corr(buy_volume, sentiment) | Contrarian behavior |

**Implementation Priority:** OPTIONAL (if time permits)
**Estimated Time:** 1 day

---

#### Category 7: Risk Metrics (4 features) - **OPTIONAL**
*Priority: P1 - Nice to have, not critical*

| # | Feature | Formula | Risk Profile |
|---|---------|---------|--------------|
| 39 | `portfolio_beta` | cov(portfolio_returns, eth_returns) / var(eth_returns) | Market correlation |
| 40 | `value_at_risk_95` | 5th_percentile(daily_returns) | Tail risk |
| 41 | `risk_adjusted_return` | total_return / portfolio_std | Sharpe-like metric |
| 42 | `leverage_indicator` | max_position_size / portfolio_value | Position sizing |

**Implementation Priority:** OPTIONAL (if time permits)
**Estimated Time:** 0.5 day

---

## Implementation Architecture

### Service Structure
```
services/
└── feature_engineering/
    ├── __init__.py
    ├── performance_calculator.py     # Category 1
    ├── behavioral_analyzer.py        # Category 2
    ├── concentration_calculator.py   # Category 3
    ├── narrative_exposure.py         # Category 4
    ├── accumulation_detector.py      # Category 5
    ├── timing_analyzer.py            # Category 6 (optional)
    ├── risk_calculator.py            # Category 7 (optional)
    └── feature_orchestrator.py       # Main coordinator
```

### CLI Interface
```bash
# Generate all features
python cli_feature_engineering.py --wallets all --output outputs/csv/wallet_features.csv

# Generate specific categories
python cli_feature_engineering.py --categories performance,behavioral

# Resume from checkpoint
python cli_feature_engineering.py --resume
```

### Output Schema
```csv
wallet_address,roi_percent,win_rate,sharpe_ratio,...[42 features total]
0x123...,45.2,0.65,1.23,...
```

---

## Story 4.3: Clustering Analysis

### Objective
Identify 5-7 distinct wallet archetypes using unsupervised learning.

### Algorithms

#### 1. HDBSCAN (Primary)
**Why:** Density-based, no k assumption, handles outliers
```python
# Parameters to tune
min_cluster_size = [25, 50, 100]
min_samples = [5, 10, 20]
metric = 'euclidean'
```

**Expected Output:**
- 5-8 clusters + noise points
- Cluster labels per wallet
- Cluster quality metrics

#### 2. K-Means (Validation)
**Why:** Centroid-based, easier interpretation
```python
# Parameters
k = [3, 5, 7, 10]
n_init = 50
max_iter = 300
```

**Expected Output:**
- Compare with HDBSCAN results
- Validate cluster count
- Ensure consistency

### Evaluation Metrics

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **Silhouette Score** | ≥ 0.5 | Cluster cohesion & separation |
| **Davies-Bouldin Index** | ≤ 1.0 | Lower = better separation |
| **Calinski-Harabasz Score** | Maximize | Between/within cluster variance ratio |

### Preprocessing Pipeline
```python
1. Handle missing values (median imputation)
2. Outlier treatment (IQR capping)
3. Log transform skewed features
4. StandardScaler (mean=0, std=1)
5. Optional: PCA to 15-20 components if 42 features
```

### Output
```csv
# wallet_clusters.csv
wallet_address,cluster_id,cluster_name,silhouette_score
0x123...,0,Diamond Hands HODLers,0.72
0x456...,1,Active Day Traders,0.68
```

---

## Story 4.4: Cluster-Narrative Affinity Analysis

### Objective
Analyze narrative preferences by wallet archetype (RQ2).

### Methodology

#### 1. Narrative Exposure by Cluster
```python
# For each cluster, calculate:
- Mean narrative exposure percentages
- Dominant narrative (max exposure)
- Narrative diversity (entropy)
```

#### 2. Statistical Significance Testing
```python
# Chi-square test
H0: No association between cluster and narrative preference
H1: Cluster membership affects narrative preference
alpha = 0.05
```

#### 3. Temporal Adoption Analysis
```python
# For each cluster-narrative pair:
- First adoption date (min transaction timestamp)
- Adoption velocity (time to 10% exposure)
- Persistence (% still holding after 30 days)
```

#### 4. Performance by Cluster-Narrative
```python
# Cross-tabulate:
- Mean ROI by cluster × narrative
- Sharpe ratio by cluster × narrative
- Win rate by cluster × narrative
```

### Visualizations

1. **Cluster-Narrative Heatmap**
   - Rows: Clusters (5-7)
   - Columns: Narratives (10)
   - Values: Mean exposure %
   - Color: Diverging scale (blue-white-red)

2. **Radar Charts per Cluster**
   - Axes: 10 narrative categories
   - Values: Normalized exposure
   - One chart per archetype

3. **Performance Comparison**
   - Box plots: ROI distribution per cluster
   - Scatter: Narrative diversity vs. Sharpe ratio
   - Bar chart: Win rate by dominant narrative

### Output Tables
```csv
# cluster_narrative_affinity.csv
cluster_id,narrative,mean_exposure_pct,mean_roi,mean_sharpe,p_value
0,DeFi,45.2,32.1,1.45,0.001
0,Meme,12.3,15.2,0.82,0.234
```

---

## Success Criteria

### Minimum Viable Results
- ✅ Generate ≥ 27 features (Categories 1-5) for 2,343 wallets
- ✅ Identify 3-5 distinct archetypes with Silhouette ≥ 0.4
- ✅ Demonstrate statistically significant narrative preferences (p < 0.05)
- ✅ Label archetypes with interpretable names
- ✅ Export clean datasets for thesis

### Stretch Goals
- 🎯 Generate all 42 features (Categories 1-7)
- 🎯 Achieve Silhouette Score ≥ 0.5
- 🎯 Identify 5-7 archetypes
- 🎯 Create interactive visualizations
- 🎯 Temporal narrative adoption patterns

---

## Deliverables

### Datasets
1. `wallet_features.csv` - 2,343 wallets × 27-42 features
2. `wallet_clusters.csv` - Cluster assignments + metadata
3. `cluster_profiles.csv` - Mean feature values per cluster
4. `cluster_narrative_affinity.csv` - Affinity matrix
5. `cluster_performance.csv` - Performance metrics per cluster

### Visualizations
1. Cluster silhouette plots
2. t-SNE 2D projection
3. Cluster-narrative heatmaps
4. Radar charts (archetype profiles)
5. Performance comparison charts

### Documentation
1. Feature engineering methodology
2. Clustering evaluation report
3. Archetype descriptions
4. Statistical test results

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Low Silhouette Score (<0.4) | Try different feature sets, PCA, scaling methods |
| Too many/few clusters | Grid search HDBSCAN parameters, compare K-Means |
| Non-interpretable clusters | Add domain knowledge labels, validate with narrative exposure |
| Missing data in features | Median imputation, or exclude wallets with >30% missing |
| Computational time | Process in batches, save checkpoints, use vectorized operations |

---

## Implementation Checklist

### Story 4.1: Feature Engineering ✅ **COMPLETE (Oct 25, 2025)**
- [x] Create `services/feature_engineering/` structure
- [x] Implement Category 1: Performance (7 features)
- [x] Implement Category 2: Behavioral (8 features)
- [x] Implement Category 3: Concentration (6 features)
- [x] Implement Category 4: Narrative Exposure (6 features)
- [x] Implement Category 5: A/D Indicators (6 features)
- [x] Create CLI interface (`cli_feature_engineering.py`)
- [x] Generate `wallet_features_master.csv` (2,159 wallets × 34 features)
- [x] Comprehensive EDA validation (12,000+ word report)
- [x] Data cleanup pipeline (fixed 7 critical issues)
- [x] Generate cleaned dataset (2,159 wallets × 41 features - ML-ready)
- [x] Validate feature quality (0 missing values, 0 duplicates)
- [x] Document feature formulas and cleanup methodology

**Deliverables:**
- ✅ `wallet_features_master_20251022_195455.csv` (original, 34 features)
- ✅ `wallet_features_cleaned_20251025_121221.csv` (cleaned, 41 features, ML-ready)
- ✅ `EDA_VALIDATION_REPORT.md` (comprehensive analysis)
- ✅ `cleanup_report_20251025_121221.md` (before/after comparison)
- ✅ `cleanup_wallet_features.py` (reproducible cleanup script)
- ✅ 5 EDA visualization charts

### Story 4.3: Clustering
- [ ] Implement preprocessing pipeline
- [ ] Implement HDBSCAN clustering
- [ ] Implement K-Means clustering
- [ ] Calculate evaluation metrics
- [ ] Generate cluster labels
- [ ] Create cluster profile summaries
- [ ] Export `wallet_clusters.csv`
- [ ] Visualize clusters (t-SNE, silhouette plots)

### Story 4.4: Affinity Analysis
- [ ] Calculate narrative exposure by cluster
- [ ] Chi-square significance testing
- [ ] Temporal adoption analysis
- [ ] Performance by cluster-narrative
- [ ] Generate heatmaps
- [ ] Generate radar charts
- [ ] Export `cluster_narrative_affinity.csv`
- [ ] Document findings

---

## Timeline

```
Week 1:
- Days 1-2: Story 4.1 Categories 1-3 (Performance, Behavioral, Concentration)
- Days 3-4: Story 4.1 Categories 4-5 (Narrative, A/D) + testing

Week 2:
- Days 1-2: Story 4.3 Clustering implementation + evaluation
- Day 3: Story 4.4 Affinity analysis
- Day 4: Visualizations + documentation
```

**Total: 8 working days**

---

## Next Immediate Action

✅ **COMPLETED:** Story 4.1 - Feature Engineering & Data Validation

**Achievements:**
- Generated 33 features across 5 categories for 2,159 wallets
- Identified and fixed 7 critical data quality issues
- Created ML-ready cleaned dataset (41 features after transformations)
- Produced comprehensive validation documentation
- Achieved 100/100 data quality score

📋 **NEXT:** Story 4.3 - Clustering Analysis

**Immediate Tasks:**
1. Load cleaned dataset: `wallet_features_cleaned_20251025_121221.csv`
2. Implement preprocessing pipeline (scaling, PCA optional)
3. Execute HDBSCAN clustering with parameter tuning
4. Validate with K-Means for comparison
5. Calculate evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
6. Generate cluster labels and profiles
7. Export `wallet_clusters.csv`

**Required Input:** `outputs/features/wallet_features_cleaned_20251025_121221.csv`
**Expected Output:** 5-7 distinct wallet archetypes with Silhouette ≥ 0.5
**Estimated Time:** 2-3 days

---

**Epic Status:** ✅ **Story 4.1 COMPLETE - 50% Epic Progress**
**Last Updated:** October 25, 2025
**Next Review:** After Story 4.3 completion (Clustering)
