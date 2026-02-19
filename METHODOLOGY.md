# Vol Trading Methodology — Claude Project Context

> Upload this to your Claude Project as knowledge so Claude understands
> how you think about vol. Edit/expand as needed.

## Overview

I run relative volatility options strategies across agricultural commodity futures:
**SOY, MEAL, OIL, CORN, WHEAT**. My trading day ends at 2:30pm.

## Core Data

- **master_vol_skew.csv**: Daily IV and skew data for all commodities/contracts going back to 2012+. Columns include date, commodity, contract_code, expiry, dirty_vol (ATM IV), skew points at various deltas (-1.5, -.5, .5, 1.5, 3), and trading DTE.
- **all_commodity_prices.csv**: Daily OHLC + settle + volume for all contracts, ~2007-present.
- **mapping.csv**: Maps futures contract months to options months (e.g., SOY March futures → H options month).
- **USDA report calendar**: Report dates back to 2007 with report types (WASDE, Stocks, Crop Progress, etc.).

## How I Think About Opportunities

### Implied Vol
- I look at current ATM IV as a percentile of its own history (same contract month, 5-year lookback default).
- IV percentile below 20% or above 80% are interesting.
- Week-over-week IV changes flag momentum.

### Skew
- I compare current skew points to their historical medians for the same contract month.
- Skew dislocations (current vs median > 1 vol point) are potential trades.

### Term Structure
- Front-to-back IV ratio flags calendar spread opportunities.
- I look at term structure shape (contango/backwardation in vol).

### Variance Ratios
- Ratio of realized variance across different time periods.
- Calculated for each commodity across multiple lookback windows (1-12 years + all).
- Used to assess whether current IV is cheap/rich relative to what's been realized.

### Realized Vol
- Calculated as annualized standard deviation of log returns.
- Standard windows: 10d, 20d, 30d, 60d.
- IV vs RV spread is a key indicator.

### Cross-Commodity
- SOY vs MEAL, SOY vs OIL (crush relationships).
- CORN vs WHEAT (feed substitution).
- Vol spreads between related commodities flag relative value.

### USDA Reports
- Reports add variance. I calculate variance multipliers by comparing 60-day pre-report variance (index = 1) to actual report-day variance.
- Like reports compared to like reports (Jan WASDE to Jan WASDE, etc.).
- Upcoming reports influence forward vol expectations.

## What I Want Claude to Help With

When I upload a snapshot, I want Claude to:
1. Identify anything that looks anomalous or interesting across the book.
2. Flag opportunities based on percentile extremes, skew dislocations, term structure kinks, or cross-commodity divergences.
3. Consider upcoming USDA events and their historical variance impact.
4. Suggest specific trades (spreads, calendars, skew trades) with rationale.
5. Challenge my assumptions — if something looks interesting but has a structural reason, note it.

## Conventions
- Vol is always annualized, expressed in percentage points (e.g., 22.5 = 22.5%).
- Skew points are vol differentials at specific delta offsets from ATM.
- Contract months use standard codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec.
