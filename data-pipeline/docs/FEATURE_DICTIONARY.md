# Feature Dictionary - Wallet Clustering Features
**Epic 4: Feature Engineering & Clustering**
**Date:** October 22, 2025
**Total Features:** 33
**Total Wallets:** 2,159

---

## Overview

This document describes all 33 engineered features used for wallet clustering analysis. Features are organized into 5 categories that capture different aspects of wallet behavior.

---

## Category 1: Performance Metrics (7 features)

**Purpose:** Measure wallet profitability and trading success

### 1. `roi_percent`
- **Description:** Return on Investment - percentage change in portfolio value
- **Formula:** `((Final Portfolio Value - Initial Portfolio Value) / Initial Portfolio Value) × 100`
- **Units:** Percentage (%)
- **Range:** -100% to +∞
- **Interpretation:**
  - Positive = Profitable wallet
  - Negative = Losing wallet
  - 0% = Break-even
- **Example:** `79.44%` = Portfolio grew 79% over tracking period

### 2. `win_rate`
- **Description:** Percentage of profitable trades (using FIFO accounting)
- **Formula:** `(Profitable Trades / Total Evaluated Trades) × 100`
- **Units:** Percentage (%)
- **Range:** 0% to 100%
- **Interpretation:**
  - 100% = All trades profitable
  - 50% = Break-even trader
  - <50% = More losing than winning trades
- **Example:** `68.42%` = 68 out of 100 trades were profitable
- **Note:** Uses FIFO matching of buy/sell pairs

### 3. `sharpe_ratio`
- **Description:** Risk-adjusted returns (higher = better risk/reward)
- **Formula:** `(Mean Daily Return - Risk Free Rate) / Std Dev of Returns`
- **Units:** Ratio (unitless)
- **Range:** -∞ to +∞ (typically -3 to +5)
- **Interpretation:**
  - >3 = Excellent risk-adjusted returns
  - 1-3 = Good returns for risk taken
  - <1 = Poor risk-adjusted returns
  - Negative = Losing money with volatility
- **Example:** `3.50` = Strong risk-adjusted performance

### 4. `max_drawdown_pct`
- **Description:** Maximum peak-to-trough decline in portfolio value
- **Formula:** `Max((Peak Value - Trough Value) / Peak Value) × 100`
- **Units:** Percentage (%)
- **Range:** 0% to 100%
- **Interpretation:**
  - 0% = No drawdown (always growing)
  - 10-20% = Moderate risk tolerance
  - >50% = High risk/volatility
- **Example:** `16.63%` = Portfolio dropped 16.63% at worst point

### 5. `total_pnl_usd`
- **Description:** Total realized profit/loss from closed positions
- **Formula:** `Sum of (Sell Price - Buy Price) × Amount for all closed positions`
- **Units:** US Dollars ($)
- **Range:** -∞ to +∞
- **Interpretation:**
  - Positive = Net profitable
  - Negative = Net losses
  - 0 = Break-even or no closed trades
- **Example:** `$57,826,923` = $57.8M in realized profits
- **Note:** Only tracks realized PnL (completed buy→sell cycles)

### 6. `avg_trade_size_usd`
- **Description:** Average dollar value of tokens traded per transaction
- **Formula:** `Average(Token Amount × Estimated Price USD)`
- **Units:** US Dollars ($)
- **Range:** $0 to +∞
- **Interpretation:**
  - <$10k = Small retail trader
  - $10k-$100k = Medium trader
  - >$100k = Large trader/whale
- **Example:** `$186,085` = Average $186k per trade

### 7. `volume_consistency`
- **Description:** Coefficient of variation of daily trading volume
- **Formula:** `Std Dev(Daily Volume) / Mean(Daily Volume)`
- **Units:** Ratio (unitless)
- **Range:** 0 to +∞
- **Interpretation:**
  - 0 = Perfectly consistent trading
  - <0.5 = Low variability
  - 0.5-1.0 = Moderate variability
  - >1.0 = Erratic/irregular trading
- **Example:** `1.17` = Moderately erratic trading volume

---

## Category 2: Behavioral Metrics (8 features)

**Purpose:** Capture trading habits, patterns, and style

### 8. `trade_frequency`
- **Description:** Number of trades per active day
- **Formula:** `Total Transactions / Number of Days with Transactions`
- **Units:** Trades per day
- **Range:** 1 to +∞
- **Interpretation:**
  - 1-5 = Casual trader
  - 5-50 = Active trader
  - >100 = Bot/professional trader
- **Example:** `394.4` = Hyperactive trader (likely bot)

### 9. `avg_holding_period_days`
- **Description:** Average time tokens are held before selling
- **Formula:** `Average(Sell Timestamp - Buy Timestamp)` using FIFO
- **Units:** Days
- **Range:** 0 to +∞
- **Interpretation:**
  - <1 day = Day trader
  - 1-7 days = Swing trader
  - >30 days = Long-term holder
- **Example:** `6.16 days` = Holds for about a week

### 10. `diamond_hands_score`
- **Description:** Tendency to hold vs sell quickly (0-100 scale)
- **Formula:** `(Holding Period Score / 2) + (Sell Frequency Score / 2)`
- **Units:** Score (0-100)
- **Range:** 0 to 100
- **Interpretation:**
  - 0-25 = "Paper hands" (sells quickly)
  - 25-50 = Moderate holding
  - 50-75 = Strong holder
  - 75-100 = "Diamond hands" (rarely sells)
- **Example:** `10.26` = Paper hands (sells quickly)

### 11. `rotation_frequency`
- **Description:** How often portfolio composition changes
- **Formula:** `Average(Tokens Added + Tokens Removed per Day)`
- **Units:** Tokens changed per day
- **Range:** 0 to +∞
- **Interpretation:**
  - 0 = Static portfolio
  - <1 = Stable
  - 1-5 = Active rebalancing
  - >5 = Constantly churning
- **Example:** `0.0` = Very stable portfolio

### 12. `weekend_activity_ratio`
- **Description:** Proportion of trading done on weekends (Sat-Sun)
- **Formula:** `Weekend Transactions / Total Transactions`
- **Units:** Ratio (0-1)
- **Range:** 0 to 1
- **Interpretation:**
  - 0 = No weekend trading (bot/professional?)
  - 0.29 = ~29% weekend (typical retail)
  - 1 = Only weekends (hobby trader?)
- **Example:** `0.11` = 11% weekend trading

### 13. `night_trading_ratio`
- **Description:** Proportion of trading during night hours (22:00-06:00 UTC)
- **Formula:** `Night Transactions / Total Transactions`
- **Units:** Ratio (0-1)
- **Range:** 0 to 1
- **Interpretation:**
  - 0 = No night trading
  - 0.5 = Half at night (different timezone?)
  - 1 = Only night trading
- **Example:** `0.22` = 22% night trading

### 14. `gas_optimization_score`
- **Description:** How well wallet optimizes gas fees (0-100 scale)
- **Formula:** Based on average gas paid vs median gas price
- **Units:** Score (0-100)
- **Range:** 0 to 100
- **Interpretation:**
  - 100 = Always pays low gas
  - 50 = Average gas optimization
  - 0 = Always pays high gas
- **Example:** `50.0` = Average optimization
- **Note:** Low variance in dataset due to stable gas prices

### 15. `dex_diversity_score`
- **Description:** Diversity of DEXs used (Shannon entropy)
- **Formula:** `-Σ(p_i × log2(p_i))` where p_i = proportion of trades on DEX i
- **Units:** Entropy bits
- **Range:** 0 to ~3.5
- **Interpretation:**
  - 0 = Only one DEX
  - 1-2 = Uses 2-4 DEXs
  - >2 = Highly diversified
- **Example:** `0.0` = Single DEX usage
- **Note:** Low in dataset (most use single aggregator)

---

## Category 3: Portfolio Concentration (6 features)

**Purpose:** Measure diversification vs concentration of holdings

### 16. `portfolio_hhi`
- **Description:** Herfindahl-Hirschman Index - portfolio concentration
- **Formula:** `Σ(market_share_i²) × 10000`
- **Units:** HHI score
- **Range:** 0 to 10,000
- **Interpretation:**
  - <1,500 = Diversified
  - 1,500-2,500 = Moderately concentrated
  - >2,500 = Highly concentrated
  - 10,000 = All in one token
- **Example:** `9,997.78` = Extremely concentrated

### 17. `portfolio_gini`
- **Description:** Gini coefficient - inequality in token holdings
- **Formula:** `(2 × Σ(i × x_i)) / (n × Σ(x_i)) - (n+1)/n`
- **Units:** Coefficient (0-1)
- **Range:** 0 to 1
- **Interpretation:**
  - 0 = Perfect equality (all tokens same size)
  - 0.5 = Moderate inequality
  - 1 = Perfect inequality (one token dominates)
- **Example:** `0.50` = Moderate inequality

### 18. `top3_concentration_pct`
- **Description:** Percentage of portfolio value in top 3 tokens
- **Formula:** `(Sum of Top 3 Token Values / Total Portfolio Value) × 100`
- **Units:** Percentage (%)
- **Range:** 0% to 100%
- **Interpretation:**
  - <50% = Well diversified
  - 50-80% = Moderately concentrated
  - >80% = Highly concentrated in few tokens
  - 100% = Only 1-3 tokens
- **Example:** `100%` = Top 3 = entire portfolio

### 19. `num_tokens_avg`
- **Description:** Average number of tokens held across tracking period
- **Formula:** `Average(Daily Token Count)`
- **Units:** Count
- **Range:** 1 to +∞
- **Interpretation:**
  - 1-10 = Focused portfolio
  - 10-50 = Moderate diversification
  - >50 = Highly diversified
- **Example:** `50.0` = Holds ~50 tokens on average

### 20. `num_tokens_std`
- **Description:** Standard deviation of token count (portfolio size volatility)
- **Formula:** `Std Dev(Daily Token Count)`
- **Units:** Count
- **Range:** 0 to +∞
- **Interpretation:**
  - 0 = Stable portfolio size
  - <5 = Slight variation
  - >10 = Frequently changing size
- **Example:** `0.0` = Very stable portfolio size

### 21. `portfolio_turnover`
- **Description:** Rate of portfolio composition change
- **Formula:** `Average((Tokens Added + Tokens Removed) / Avg Portfolio Size)`
- **Units:** Ratio (0-1+)
- **Range:** 0 to +∞
- **Interpretation:**
  - 0 = No turnover (buy and hold)
  - <0.1 = Low turnover
  - 0.1-0.5 = Moderate rebalancing
  - >0.5 = High turnover
- **Example:** `0.0` = No portfolio turnover

---

## Category 4: Narrative Exposure (6 features)

**Purpose:** Identify token category preferences and beliefs

### 22. `narrative_diversity_score`
- **Description:** Diversity of narrative exposure (Shannon entropy)
- **Formula:** `-Σ(p_i × log2(p_i))` where p_i = proportion in narrative i
- **Units:** Entropy bits
- **Range:** 0 to ~3.5
- **Interpretation:**
  - 0 = Only one narrative
  - 1-2 = Focused on 2-3 narratives
  - >2 = Highly diversified across narratives
- **Example:** `0.21` = Focused on 1-2 narratives

### 23. `primary_narrative_pct`
- **Description:** Percentage of portfolio in dominant narrative
- **Formula:** `(Value in Primary Narrative / Total Portfolio Value) × 100`
- **Units:** Percentage (%)
- **Range:** 0% to 100%
- **Interpretation:**
  - <50% = Balanced across narratives
  - 50-80% = Preference for one narrative
  - >80% = Strongly focused on one narrative
  - 100% = Only one narrative
- **Example:** `93.29%` = Strong focus on primary narrative

### 24. `defi_exposure_pct`
- **Description:** Percentage of portfolio in DeFi tokens
- **Formula:** `(DeFi Token Value / Total Portfolio Value) × 100`
- **Units:** Percentage (%)
- **Range:** 0% to 100%
- **Interpretation:**
  - 0% = No DeFi exposure
  - 1-25% = Low DeFi exposure
  - 25-75% = Moderate DeFi focus
  - >75% = DeFi specialist
- **Example:** `0.68%` = Minimal DeFi exposure

### 25. `ai_exposure_pct`
- **Description:** Percentage of portfolio in AI/LLM tokens
- **Formula:** `(AI Token Value / Total Portfolio Value) × 100`
- **Units:** Percentage (%)
- **Range:** 0% to 100%
- **Interpretation:**
  - 0% = No AI exposure
  - 1-25% = Low AI interest
  - 25-75% = Moderate AI focus
  - >75% = AI narrative believer
- **Example:** `0.0%` = No AI exposure

### 26. `meme_exposure_pct`
- **Description:** Percentage of portfolio in meme coins
- **Formula:** `(Meme Token Value / Total Portfolio Value) × 100`
- **Units:** Percentage (%)
- **Range:** 0% to 100%
- **Interpretation:**
  - 0% = Avoids memes
  - 1-25% = Occasional meme gambles
  - 25-75% = Significant meme exposure
  - >75% = Meme coin degen
- **Example:** `14.08%` = Moderate meme exposure

### 27. `stablecoin_usage_ratio`
- **Description:** Proportion of days holding stablecoins
- **Formula:** `Days with Stablecoins / Total Days`
- **Units:** Ratio (0-1)
- **Range:** 0 to 1
- **Interpretation:**
  - 0 = Never holds stablecoins
  - 0.5 = Holds stables half the time
  - 1 = Always holds stablecoins
- **Example:** `0.30` = Holds stables 30% of time

---

## Category 5: Accumulation/Distribution (6 features)

**Purpose:** Identify buying vs selling phases and trends

### 28. `accumulation_phase_days`
- **Description:** Number of days with net positive portfolio growth
- **Formula:** `Count(Days where Portfolio Value Increased)`
- **Units:** Days
- **Range:** 0 to Total Days
- **Interpretation:**
  - High value = Accumulation phase
  - Low value = Not actively building
- **Example:** `0.40` days = Minimal accumulation

### 29. `distribution_phase_days`
- **Description:** Number of days with net negative portfolio growth
- **Formula:** `Count(Days where Portfolio Value Decreased)`
- **Units:** Days
- **Range:** 0 to Total Days
- **Interpretation:**
  - High value = Distribution/selling phase
  - Low value = Not actively selling
- **Example:** `1.0` days = Some distribution

### 30. `accumulation_intensity`
- **Description:** Average rate of growth on accumulation days
- **Formula:** `Average(% Change on Positive Days)`
- **Units:** Percentage per day
- **Range:** 0% to +∞
- **Interpretation:**
  - <5% = Slow accumulation
  - 5-20% = Moderate accumulation
  - >20% = Aggressive accumulation
- **Example:** `0.02%` = Very slow accumulation

### 31. `distribution_intensity`
- **Description:** Average rate of decline on distribution days
- **Formula:** `Average(Abs(% Change) on Negative Days)`
- **Units:** Percentage per day
- **Range:** 0% to 100%
- **Interpretation:**
  - <5% = Slow distribution
  - 5-20% = Moderate selling
  - >20% = Aggressive dumping
- **Example:** `0.17%` = Very slow distribution

### 32. `balance_volatility`
- **Description:** Standard deviation of daily portfolio value changes
- **Formula:** `Std Dev(Daily % Changes)`
- **Units:** Percentage
- **Range:** 0% to +∞
- **Interpretation:**
  - 0% = No volatility (stable)
  - <10% = Low volatility
  - 10-30% = Moderate volatility
  - >30% = High volatility
- **Example:** `0.06%` = Very low volatility

### 33. `trend_direction`
- **Description:** Overall trend: accumulating (+1) or distributing (-1)
- **Formula:** `tanh((Final Value - Initial Value) / Initial Value)`
- **Units:** Score (-1 to +1)
- **Range:** -1 to +1
- **Interpretation:**
  - +1 = Strong accumulation trend
  - 0 = Neutral/flat
  - -1 = Strong distribution trend
- **Example:** `-0.006` = Slightly distributing

---

## Data Quality Notes

### Missing Values
- **Total Missing:** 0 (after cleaning)
- **Originally:** 1 missing value in `balance_volatility` (filled with 0)

### Wallet Coverage
- **Total Wallets:** 2,159
- **Original Dataset:** 2,343 wallets
- **Excluded:** 184 wallets (8%) due to insufficient balance data for Categories 3-5

### Known Limitations

1. **Price Estimation:**
   - Using rough ETH equivalence (0.1% of ETH) for tokens without historical prices
   - May cause outliers in PnL and trade size metrics

2. **Low Variance Features:**
   - `gas_optimization_score` = 50 for all (stable gas period)
   - `dex_diversity_score` = 0 for all (single aggregator usage)
   - `num_tokens_std` = 0 for most (stable portfolio sizes)
   - `portfolio_turnover` = 0 for most (buy-and-hold behavior)

3. **Data Period:**
   - September 2024 (30 days)
   - Reflects bull market conditions
   - May not capture bear market behavior

---

## File Location

**Master Dataset (Dual Format):**
- **CSV:** `outputs/features/wallet_features_master_YYYYMMDD_HHMMSS.csv` (719 KB)
- **Parquet:** `outputs/features/wallet_features_master_YYYYMMDD_HHMMSS.parquet` (314 KB, 56% smaller)

**Parquet Benefits:**
- 56% smaller file size (better for storage/transfer)
- Faster to read/write (columnar format)
- Better compression (Snappy algorithm)
- Native support in pandas, scikit-learn, etc.

**Source Files:**
- `outputs/features/performance_features_*.csv` (Category 1)
- `outputs/features/behavioral_features_*.csv` (Category 2)
- `outputs/features/concentration_features_*.csv` (Category 3)
- `outputs/features/narrative_features_*.csv` (Category 4)
- `outputs/features/accumulation_features_*.csv` (Category 5)

---

## Usage for Clustering

### Recommended Preprocessing

1. **Feature Scaling:** Use RobustScaler or StandardScaler
2. **Outlier Handling:** Consider winsorization at 95th/99th percentile
3. **Feature Selection:** May drop low-variance features
4. **Dimensionality Reduction:** Consider PCA if needed

### Clustering Algorithms

- **HDBSCAN:** For finding natural clusters with noise
- **K-Means:** For fixed number of clusters
- **Hierarchical:** For understanding cluster relationships

---

**Last Updated:** October 22, 2025
**Version:** 1.0
**Status:** Ready for Clustering Analysis
