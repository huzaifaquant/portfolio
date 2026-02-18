-- ============================================================================
-- CONSOLIDATED QUESTDB QUERIES FOR API ENDPOINTS
-- ============================================================================
-- These queries return all values needed for each API endpoint in a single call
-- Replace {portfolioId}, {userId}, {date}, {endDate} with actual values
-- Only includes values that can be calculated from QuestDB schema
-- ============================================================================

-- ============================================================================
-- API 1: GET /api/portfolios/stats/{portfolioId}
-- Returns: accountBalance, availableFunds, equity, holdingCount, holdingMarketValue,
--          realizedPnl, totalGainPercentage, unrealizedPnl, value
-- ============================================================================
WITH
  portfolio_cash AS (
    SELECT accountBalance, availableFund
    FROM ofd_portfolio
    WHERE id = '{portfolioId}'
  ),
  positions_summary AS (
    SELECT
      COUNT(DISTINCT tvId) AS holdingCount,
      COALESCE(SUM(marketValue), 0) AS holdingMarketValue,
      COALESCE(SUM(unrealizedPnl), 0) AS unrealizedPnl
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
  ),
  pending_orders AS (
    SELECT COALESCE(SUM(price * ABS(quantity)), 0) AS pendingValue
    FROM ofd_oms_orders
    WHERE portfolioId = '{portfolioId}'
      AND status IN ('PENDING', 'PARTIAL')
      AND pendingQty > 0
  ),
  initial_avg_prices AS (
    SELECT DISTINCT
      tvId,
      averagePrice AS initialAvgPrice
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}'
      AND averagePrice IS NOT NULL
      AND averagePrice > 0
  ),
  numbered_transactions AS (
    SELECT
      t.id,
      t.tvId,
      t.entryDate,
      t.side,
      t.direction,
      t.quantity,
      t.price,
      t.entryPrice,
      COALESCE(iap.initialAvgPrice, t.entryPrice) AS initialAvgPrice,
      ROW_NUMBER() OVER (PARTITION BY t.tvId ORDER BY entryDate, id) AS seq
    FROM ofd_oms_transactions t
    LEFT JOIN initial_avg_prices iap ON t.tvId = iap.tvId
    WHERE t.portfolioId = '{portfolioId}'
      AND t.status = 'EXECUTED'
      AND t.side IN ('buy', 'sell')
  ),
  transactions_pass1 AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      curr.initialAvgPrice,
      curr.seq,
      SUM(
        CASE 
          WHEN curr.direction = 'LONG' AND curr.side = 'buy' THEN ABS(curr.quantity)
          WHEN curr.direction = 'LONG' AND curr.side = 'sell' THEN -ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'sell' THEN ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'buy' THEN -ABS(curr.quantity)
          ELSE 0
        END
      ) OVER (
        PARTITION BY curr.tvId 
        ORDER BY entryDate, id 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS cumulative_quantity,
      SUM(
        CASE 
          WHEN curr.direction = 'LONG' AND curr.side = 'buy' 
          THEN curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'LONG' AND curr.side = 'sell' 
          THEN -curr.entryPrice * ABS(curr.quantity)  -- Will use prev_avg_price in PASS 2
          WHEN curr.direction = 'SHORT' AND curr.side = 'sell' 
          THEN curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'buy' 
          THEN -curr.entryPrice * ABS(curr.quantity)  -- Will use prev_avg_price in PASS 2
          ELSE 0
        END
      ) OVER (
        PARTITION BY curr.tvId 
        ORDER BY entryDate, id 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS cumulative_cost_basis_pass1
    FROM numbered_transactions curr
  ),
  transactions_with_prev_avg AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      curr.initialAvgPrice,
      curr.seq,
      curr.cumulative_quantity,
      curr.cumulative_cost_basis_pass1,
      CASE 
        WHEN curr.seq = 1 THEN curr.initialAvgPrice
        WHEN prev.cumulative_quantity != 0 AND prev.cumulative_quantity IS NOT NULL
        THEN prev.cumulative_cost_basis_pass1 / prev.cumulative_quantity
        ELSE curr.entryPrice
      END AS prev_avg_price
    FROM transactions_pass1 curr
    LEFT JOIN transactions_pass1 prev ON curr.tvId = prev.tvId AND prev.seq = curr.seq - 1
  ),
  transactions_with_calc AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      curr.prev_avg_price,
      curr.seq,
      curr.cumulative_quantity,
      SUM(
        CASE 
          -- Opening positions: use entryPrice
          WHEN curr.direction = 'LONG' AND curr.side = 'buy' 
          THEN curr.entryPrice * ABS(curr.quantity)
          -- Closing positions: use prev_avg_price (CRITICAL FIX)
          WHEN curr.direction = 'LONG' AND curr.side = 'sell' 
          THEN -COALESCE(curr.prev_avg_price, curr.entryPrice) * ABS(curr.quantity)
          -- Opening positions: use entryPrice
          WHEN curr.direction = 'SHORT' AND curr.side = 'sell' 
          THEN curr.entryPrice * ABS(curr.quantity)
          -- Closing positions: use prev_avg_price (CRITICAL FIX)
          WHEN curr.direction = 'SHORT' AND curr.side = 'buy' 
          THEN -COALESCE(curr.prev_avg_price, curr.entryPrice) * ABS(curr.quantity)
          ELSE 0
        END
      ) OVER (
        PARTITION BY curr.tvId 
        ORDER BY entryDate, id 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS cumulative_cost_basis_approx
    FROM transactions_with_prev_avg curr
  ),
  transactions_with_prev AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      CASE 
        WHEN prev.cumulative_quantity != 0 AND prev.cumulative_quantity IS NOT NULL
        THEN prev.cumulative_cost_basis_approx / prev.cumulative_quantity
        ELSE NULL
      END AS prev_avg_price
    FROM transactions_with_calc curr
    LEFT JOIN transactions_with_calc prev ON curr.tvId = prev.tvId AND prev.seq = curr.seq - 1
  ),
  realized_pnl_per_transaction AS (
    SELECT
      tvId,
      CASE
        -- Long position sell: (current price - prev_avg_price) * quantity
        WHEN side = 'sell' AND direction = 'LONG' AND prev_avg_price IS NOT NULL
        THEN (price - prev_avg_price) * ABS(quantity)
        -- Short position buy (cover): (prev_avg_price - current price) * quantity
        WHEN side = 'buy' AND direction = 'SHORT' AND prev_avg_price IS NOT NULL
        THEN (prev_avg_price - price) * ABS(quantity)
        ELSE 0
      END AS realizedPnL
    FROM transactions_with_prev
  ),
  realized_total AS (
    SELECT
      COALESCE(SUM(realizedPnL), 0) AS totalRealizedPnL
    FROM realized_pnl_per_transaction
  ),
  current_equity AS (
    SELECT equity
    FROM ofd_portfolio_equity_snapshot
    WHERE portfolioId = '{portfolioId}'
    ORDER BY cts DESC
    LIMIT 1
  ),
  initial_capital AS (
    SELECT initialCapital
    FROM ofd_portfolio_initial_capital
    WHERE portfolioId = '{portfolioId}'
    ORDER BY cts ASC
    LIMIT 1
  )
SELECT
  pc.accountBalance,
  pc.availableFund AS availableFunds,
  COALESCE(ce.equity, pc.accountBalance + ps.holdingMarketValue ) AS equity,
  ps.holdingCount,
  ps.holdingMarketValue,
  rt.totalRealizedPnL AS realizedPnl,
  CASE 
    WHEN ic.initialCapital IS NULL THEN NULL  -- Error case: initial capital not set
    WHEN ic.initialCapital > 0
    THEN ((rt.totalRealizedPnL + ps.unrealizedPnl) / ic.initialCapital) * 100
    ELSE 0
  END AS totalGainPercentage,
  ps.unrealizedPnl,
  COALESCE(ce.equity, pc.accountBalance + ps.holdingMarketValue + po.pendingValue) AS value
FROM portfolio_cash pc
LEFT JOIN positions_summary ps ON 1=1
LEFT JOIN pending_orders po ON 1=1
LEFT JOIN realized_total rt ON 1=1
LEFT JOIN current_equity ce ON 1=1
LEFT JOIN initial_capital ic ON 1=1;


-- ============================================================================
-- API 2: GET /api/portfolios (Portfolio List with Metrics)
-- Returns: accountBalance, availableFund, lineItems (holdings), portfolioMetrics,
--          assetsTypeCounts, assetTypeInvestments, realizePnl, unrealizePnl,
--          value, totalGainPercent, YTDPnl, winnings
-- ============================================================================
-- Note: This query returns data for a single portfolio. For list endpoint,
--       call this query for each portfolioId in the list.
-- ============================================================================
WITH
  portfolio_cash AS (
    SELECT accountBalance, availableFund
    FROM ofd_portfolio
    WHERE id = '{portfolioId}'
  ),
  -- Latest prices (pre-computed to avoid correlated subqueries)
  -- Using unified latest_prices table with tvId
  latest_prices_cte AS (
    SELECT 
      tvId,
      price AS latestPrice
    FROM latest_prices
    LATEST ON cts PARTITION BY tvId
  ),
  -- Holdings (lineItems)
  holdings_data AS (
    SELECT
      p.tvId,
      p.assetClass,
      p.direction,
      ABS(p.quantity) AS shares,
      p.averagePrice AS avgCostPerShare,
      p.marketValue,
      p.unrealizedPnl AS totalGain,
      CASE 
        WHEN p.tradeValue > 0 
        THEN (p.unrealizedPnl / p.tradeValue) * 100 
        ELSE 0 
      END AS totalGainPercent,
      COALESCE(
        lp.latestPrice,
        p.marketValue / NULLIF(ABS(p.quantity), 0)
      ) AS currentPrice
    FROM ofd_oms_positions p
    LEFT JOIN latest_prices_cte lp ON p.tvId = lp.tvId
    WHERE p.portfolioId = '{portfolioId}' AND p.quantity != 0
  ),
  -- Asset type counts and investments
  asset_types AS (
    SELECT
      assetClass,
      COUNT(DISTINCT tvId) AS counts,
      COALESCE(SUM(tradeValue), 0) AS investment
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
    GROUP BY assetClass
  ),
  -- Portfolio metrics - bySymbol
  by_symbol AS (
    SELECT
      tvId,
      assetClass,
      SUM(ABS(quantity)) AS quantity,
      SUM(ABS(quantity) * price) AS value,
      SUM(CASE WHEN side = 'buy' THEN ABS(quantity) ELSE 0 END) AS boughtQuantity,
      SUM(CASE WHEN side = 'sell' THEN ABS(quantity) ELSE 0 END) AS soldQuantity,
      SUM(CASE WHEN side = 'buy' THEN ABS(quantity) * price ELSE 0 END) AS boughtValue,
      SUM(CASE WHEN side = 'sell' THEN ABS(quantity) * price ELSE 0 END) AS soldValue,
      COUNT(1) AS count,
      -- Quantity-weighted average prices
      SUM(CASE WHEN side = 'buy' THEN ABS(quantity) * price ELSE 0 END) / 
        NULLIF(SUM(CASE WHEN side = 'buy' THEN ABS(quantity) ELSE 0 END), 0) AS averageBoughtPrice,
      SUM(CASE WHEN side = 'sell' THEN ABS(quantity) * price ELSE 0 END) / 
        NULLIF(SUM(CASE WHEN side = 'sell' THEN ABS(quantity) ELSE 0 END), 0) AS averageSoldPrice
    FROM ofd_oms_transactions
    WHERE portfolioId = '{portfolioId}' AND status = 'EXECUTED'
    GROUP BY tvId, assetClass
  ),
  -- Current positions for remainingQuantity
  current_positions AS (
    SELECT tvId, assetClass, quantity AS remainingQuantity
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
  ),
  -- Most/Least traded symbols (both buy and sell sides)
  symbol_trade_counts AS (
    SELECT
      tvId,
      assetClass,
      SUM(ABS(quantity)) AS totalVolume,
      COUNT(1) AS tradeCount
    FROM ofd_oms_transactions
    WHERE portfolioId = '{portfolioId}' AND status = 'EXECUTED'
    GROUP BY tvId, assetClass
  ),
  -- Realized P&L using prev_avg_price logic (Correct Formulae) - API 2
  -- Formula: 
  --   Long sell: (price - prev_avg_price) * quantity
  --   Short buy: (prev_avg_price - price) * quantity
  -- Where prev_avg_price is calculated from previous transactions sequentially
  -- 
  -- CRITICAL: Closing positions (sell long, buy short) must use prev_avg_price in cost basis,
  -- not entryPrice. This requires a two-pass calculation to resolve the circular dependency.
  
  -- Get initial average price from ofd_oms_positions for each tvId (if available)
  initial_avg_prices_api2 AS (
    SELECT DISTINCT
      tvId,
      averagePrice AS initialAvgPrice
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}'
      AND averagePrice IS NOT NULL
      AND averagePrice > 0
  ),
  numbered_transactions_api2 AS (
    SELECT
      t.id,
      t.tvId,
      t.entryDate,
      t.side,
      t.direction,
      t.quantity,
      t.price,
      t.entryPrice,
      COALESCE(iap.initialAvgPrice, t.entryPrice) AS initialAvgPrice,
      ROW_NUMBER() OVER (PARTITION BY t.tvId ORDER BY entryDate, id) AS seq
    FROM ofd_oms_transactions t
    LEFT JOIN initial_avg_prices_api2 iap ON t.tvId = iap.tvId
    WHERE t.portfolioId = '{portfolioId}'
      AND t.status = 'EXECUTED'
      AND t.side IN ('buy', 'sell')
  ),
  -- PASS 1: Calculate cumulative quantity and cost basis using entryPrice for all transactions
  transactions_pass1_api2 AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      curr.initialAvgPrice,
      curr.seq,
      SUM(
        CASE 
          WHEN curr.direction = 'LONG' AND curr.side = 'buy' THEN ABS(curr.quantity)
          WHEN curr.direction = 'LONG' AND curr.side = 'sell' THEN -ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'sell' THEN ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'buy' THEN -ABS(curr.quantity)
          ELSE 0
        END
      ) OVER (
        PARTITION BY curr.tvId 
        ORDER BY entryDate, id 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS cumulative_quantity,
      SUM(
        CASE 
          WHEN curr.direction = 'LONG' AND curr.side = 'buy' 
          THEN curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'LONG' AND curr.side = 'sell' 
          THEN -curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'sell' 
          THEN curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'buy' 
          THEN -curr.entryPrice * ABS(curr.quantity)
          ELSE 0
        END
      ) OVER (
        PARTITION BY curr.tvId 
        ORDER BY entryDate, id 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS cumulative_cost_basis_pass1
    FROM numbered_transactions_api2 curr
  ),
  -- Calculate prev_avg_price from PASS 1 results (FIXED: prev.seq = curr.seq - 1)
  transactions_with_prev_avg_api2 AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      curr.initialAvgPrice,
      curr.seq,
      curr.cumulative_quantity,
      curr.cumulative_cost_basis_pass1,
      CASE 
        WHEN curr.seq = 1 THEN curr.initialAvgPrice
        WHEN prev.cumulative_quantity != 0 AND prev.cumulative_quantity IS NOT NULL
        THEN prev.cumulative_cost_basis_pass1 / prev.cumulative_quantity
        ELSE curr.entryPrice
      END AS prev_avg_price
    FROM transactions_pass1_api2 curr
    LEFT JOIN transactions_pass1_api2 prev ON curr.tvId = prev.tvId AND prev.seq = curr.seq - 1
  ),
  -- PASS 2: Recalculate cumulative cost basis using prev_avg_price for closing positions
  transactions_with_calc_api2 AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      curr.prev_avg_price,
      curr.seq,
      curr.cumulative_quantity,
      SUM(
        CASE 
          WHEN curr.direction = 'LONG' AND curr.side = 'buy' 
          THEN curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'LONG' AND curr.side = 'sell' 
          THEN -COALESCE(curr.prev_avg_price, curr.entryPrice) * ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'sell' 
          THEN curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'buy' 
          THEN -COALESCE(curr.prev_avg_price, curr.entryPrice) * ABS(curr.quantity)
          ELSE 0
        END
      ) OVER (
        PARTITION BY curr.tvId 
        ORDER BY entryDate, id 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS cumulative_cost_basis_approx
    FROM transactions_with_prev_avg_api2 curr
  ),
  transactions_with_prev_api2 AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      CASE 
        WHEN prev.cumulative_quantity != 0 AND prev.cumulative_quantity IS NOT NULL
        THEN prev.cumulative_cost_basis_approx / prev.cumulative_quantity
        ELSE NULL
      END AS prev_avg_price
    FROM transactions_with_calc_api2 curr
    LEFT JOIN transactions_with_calc_api2 prev ON curr.tvId = prev.tvId AND prev.seq = curr.seq - 1
  ),
  -- Calculate realized P&L using prev_avg_price
  realized_pnl_per_transaction_api2 AS (
    SELECT
      tvId,
      CASE
        -- Long position sell: (current price - prev_avg_price) * quantity
        WHEN side = 'sell' AND direction = 'LONG' AND prev_avg_price IS NOT NULL
        THEN (price - prev_avg_price) * ABS(quantity)
        -- Short position buy (cover): (prev_avg_price - current price) * quantity
        WHEN side = 'buy' AND direction = 'SHORT' AND prev_avg_price IS NOT NULL
        THEN (prev_avg_price - price) * ABS(quantity)
        ELSE 0
      END AS realizedPnL
    FROM transactions_with_prev_api2
  ),
  realized_total AS (
    SELECT
      COALESCE(SUM(realizedPnL), 0) AS totalRealizedPnL
    FROM realized_pnl_per_transaction_api2
  ),
  -- Unrealized PnL
  unrealized_total AS (
    SELECT COALESCE(SUM(unrealizedPnl), 0) AS totalUnrealizedPnL
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
  ),
  -- Total valuee
  total_value AS (
    SELECT
      COALESCE(SUM(marketValue), 0) AS totalMarketValue
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
  ),
  pending_orders_value AS (
    SELECT COALESCE(SUM(price * ABS(quantity)), 0) AS pendingValue
    FROM ofd_oms_orders
    WHERE portfolioId = '{portfolioId}'
      AND status IN ('PENDING', 'PARTIAL')
      AND pendingQty > 0
  ),
  -- Cumulative return
  current_equity AS (
    SELECT equity
    FROM ofd_portfolio_equity_snapshot
    WHERE portfolioId = '{portfolioId}'
    ORDER BY cts DESC
    LIMIT 1
  ),
  initial_capital AS (
    SELECT initialCapital
    FROM ofd_portfolio_initial_capital
    WHERE portfolioId = '{portfolioId}'
    ORDER BY cts ASC
    LIMIT 1
  ),
  -- YTD PnL (using YEAR/MONTH/DAY functions - these are supported in QuestDB)
  ytd_start AS (
    SELECT equity AS ytdStartEquity
    FROM ofd_portfolio_equity_snapshot
    WHERE portfolioId = '{portfolioId}'
      AND YEAR(cts) = YEAR(now())
      AND MONTH(cts) = 1 
      AND DAY(cts) = 1
    ORDER BY cts
    LIMIT 1
  ),
  -- Win rate and win/loss ratio
  long_trades AS (
    SELECT 
      (s.price - b.entryPrice) * ABS(s.quantity) AS tradePnL
    FROM ofd_oms_transactions s
    JOIN ofd_oms_transactions b ON s.tvId = b.tvId AND s.portfolioId = b.portfolioId
    WHERE s.portfolioId = '{portfolioId}'
      AND s.side = 'sell' AND b.side = 'buy'
      AND s.status = 'EXECUTED' AND b.status = 'EXECUTED'
      AND s.entryDate >= b.entryDate
  ),
  short_trades AS (
    SELECT 
      (s.entryPrice - b.price) * ABS(b.quantity) AS tradePnL
    FROM ofd_oms_transactions b
    JOIN ofd_oms_transactions s ON b.tvId = s.tvId AND b.portfolioId = s.portfolioId
    WHERE b.portfolioId = '{portfolioId}'
      AND b.side = 'buy' AND s.side = 'sell'
      AND b.status = 'EXECUTED' AND s.status = 'EXECUTED'
      AND b.entryDate >= s.entryDate
  ),
  all_trades AS (
    SELECT tradePnL FROM long_trades
    UNION ALL
    SELECT tradePnL FROM short_trades
  ),
  trade_stats AS (
    SELECT
      COUNT(1) AS totalTrades,
      SUM(CASE WHEN tradePnL > 0 THEN 1 ELSE 0 END) AS winningTrades,
      SUM(CASE WHEN tradePnL <= 0 THEN 1 ELSE 0 END) AS loosingTrades
    FROM all_trades
  ),
  -- Stock type ratio
  stock_type_totals AS (
    SELECT
      assetClass,
      SUM(marketValue) AS totalValue
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
    GROUP BY assetClass
  ),
  total_portfolio_value AS (
    SELECT SUM(marketValue) AS total
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
  ),
  -- Total counts (moved from SELECT subqueries to CTEs)
  total_stock AS (
    SELECT COUNT(DISTINCT tvId) AS totalStock
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0 AND assetClass = 'stocks'
  ),
  total_crypto AS (
    SELECT COUNT(DISTINCT tvId) AS totalCrypto
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0 AND assetClass = 'crypto'
  ),
  total_traded_asset_count AS (
    SELECT COUNT(DISTINCT tvId) AS totalTradedAssetCount
    FROM ofd_oms_transactions
    WHERE portfolioId = '{portfolioId}' AND status = 'EXECUTED'
  ),
  total_trades_count AS (
    SELECT COUNT(1) AS totalTradesCount
    FROM ofd_oms_transactions
    WHERE portfolioId = '{portfolioId}' AND status = 'EXECUTED'
  )
SELECT
  -- Basic portfolio info
  pc.accountBalance,
  pc.availableFund AS availableFund,
  COALESCE(ce.equity, pc.accountBalance + tv.totalMarketValue + pov.pendingValue) AS value,
  rt.totalRealizedPnL AS realizePnl,
  ut.totalUnrealizedPnL AS unrealizePnl,
  CASE 
    WHEN ic.initialCapital IS NULL THEN NULL  -- Error case: initial capital not set
    WHEN ic.initialCapital > 0
    THEN ((rt.totalRealizedPnL + ut.totalUnrealizedPnL) / ic.initialCapital) * 100
    ELSE 0
  END AS totalGainPercent,
  CASE 
    WHEN ytd.ytdStartEquity IS NULL THEN NULL  -- Error case: YTD start equity not available
    WHEN ytd.ytdStartEquity > 0
    THEN ((COALESCE(ce.equity, pc.accountBalance + tv.totalMarketValue) - 
           ytd.ytdStartEquity) * 100.0 / ytd.ytdStartEquity)
    ELSE 0
  END AS YTDPnl,
  -- Winnings
  CASE 
    WHEN ts.totalTrades > 0 
    THEN (ts.winningTrades * 100.0 / ts.totalTrades)
    ELSE 0
  END AS winRate,
  CASE 
    WHEN ts.loosingTrades > 0
    THEN CAST(ts.winningTrades AS DOUBLE) / NULLIF(ts.loosingTrades, 0)
    ELSE NULL
  END AS winLossRatio,
  ts.winningTrades,
  ts.loosingTrades,
  CONCAT(ts.winningTrades, '/', ts.loosingTrades) AS winnLossRatioString,
  -- Total counts (from CTEs)
  tst.totalStock,
  tc.totalCrypto,
  ttac.totalTradedAssetCount,
  ttc.totalTradesCount
FROM portfolio_cash pc
CROSS JOIN total_value tv
CROSS JOIN pending_orders_value pov
CROSS JOIN realized_total rt
CROSS JOIN unrealized_total ut
CROSS JOIN current_equity ce
CROSS JOIN initial_capital ic
CROSS JOIN ytd_start ytd
CROSS JOIN trade_stats ts
CROSS JOIN total_stock tst
CROSS JOIN total_crypto tc
CROSS JOIN total_traded_asset_count ttac
CROSS JOIN total_trades_count ttc;

-- ============================================================================
-- API 3: GET /api/portfolios/cumulativeReturn/{portfolioId}
-- Returns: chart (daily equity), dailyReturns, maxDD, sharpeRatio, sortinoRatio,
--          var, vol, weeklyReturns, winRT
-- ============================================================================
-- Provided this in the breakDown

-- Note: For chart and dailyReturns arrays, you'll need separate queries:
-- SELECT cts AS date, equity, dailyPnL, dailyPnLPct, cumulativeReturnPct
-- FROM ofd_portfolio_equity_snapshot WHERE portfolioId = '{portfolioId}' ORDER BY cts;

-- ============================================================================
-- API 4: GET /api/portfolios/tradeVolume/{portfolioId}
-- Returns: tradeVolumeByPeriod (buy/sell by year)
-- ============================================================================
-- Note: Using cts (designated timestamp) for SAMPLE BY, but grouping by year of entryDate
-- Alternative: Use SAMPLE BY 1y on cts if year grouping by trade date is acceptable
SELECT
  YEAR(entryDate) AS year,
  SUM(CASE WHEN side = 'buy' THEN ABS(quantity) * price ELSE 0 END) AS buy,
  SUM(CASE WHEN side = 'sell' THEN ABS(quantity) * price ELSE 0 END) AS sell
FROM ofd_oms_transactions
WHERE portfolioId = '{portfolioId}'
  AND status = 'EXECUTED'
  AND entryDate IS NOT NULL
GROUP BY YEAR(entryDate)
ORDER BY year;


-- ============================================================================
-- API 5: GET /api/portfolios/compositionChart/{portfolioId}
-- Returns: allocationPct by tvId, assetType, date
-- ============================================================================
-- Using SAMPLE BY for month aggregation (requires designated timestamp)
-- Note: Both tables use cts as designated timestamp

WITH
  position_values_by_date AS (
    SELECT
      cts AS month,
      tvId,
      assetClass AS assetType,
      SUM(marketValue) AS symbolValue
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
    SAMPLE BY 1M
    -- Note: tvId and assetClass are implicit grouping keys with SAMPLE BY
    -- QuestDB groups by time bucket (1M) + any non-aggregated columns (tvId, assetClass)
  ),
  total_value_by_date AS (
    SELECT
      cts AS month,
      SUM(marketValue) AS total
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
    SAMPLE BY 1M
  )
SELECT
  pvbd.month AS date,
  pvbd.tvId,
  pvbd.assetType,
  (pvbd.symbolValue / NULLIF(tvbd.total, 0)) * 100 AS allocationPct
FROM position_values_by_date pvbd
JOIN total_value_by_date tvbd ON pvbd.month = tvbd.month
ORDER BY pvbd.month, pvbd.tvId;

-- ============================================================================
-- API 6: GET /api/portfolios/weightedReturn/{portfolioId}
-- Returns: contribution by tvId, assetType, date
-- ============================================================================
-- Formula: contribution = weight × period_return
-- weight = position market value / total portfolio value  
-- period_return = (current_price - previous_price) / previous_price for the specific period
-- NOTE: This calculates TRUE period-specific returns using price changes, not cumulative P&L
-- ============================================================================
WITH
  -- Get position snapshots with asset class and price info
  position_snapshots AS (
    SELECT
      CAST(cts AS DATE) AS date,
      tvId,
      assetClass,
      marketValue,
      quantity,
      averagePrice,
      marketValue / NULLIF(ABS(quantity), 0) AS currentPrice,
      ROW_NUMBER() OVER (PARTITION BY tvId ORDER BY cts) AS rn
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
  ),
  -- Calculate period return based on price change (not cumulative P&L)
  position_returns AS (
    SELECT
      curr.date,
      curr.tvId,
      curr.assetClass,
      curr.marketValue AS currentValue,
      curr.currentPrice,
      COALESCE(prev.currentPrice, curr.currentPrice) AS previousPrice,
      CASE 
        WHEN COALESCE(prev.currentPrice, curr.currentPrice) > 0
        THEN ((curr.currentPrice - COALESCE(prev.currentPrice, curr.currentPrice)) / 
              NULLIF(COALESCE(prev.currentPrice, curr.currentPrice), 0))
        ELSE 0
      END AS periodReturn
    FROM position_snapshots curr
    LEFT JOIN position_snapshots prev ON curr.tvId = prev.tvId AND prev.rn = curr.rn - 1
  ),
  -- Calculate total portfolio value per date
  daily_totals AS (
    SELECT
      date,
      SUM(currentValue) AS totalMarketValue
    FROM position_returns
    GROUP BY date
  )
SELECT
  pr.date,
  pr.tvId,
  pr.assetClass AS assetType,
  CASE 
    WHEN dt.totalMarketValue > 0
    THEN (pr.currentValue / NULLIF(dt.totalMarketValue, 0)) * pr.periodReturn
    ELSE 0
  END AS contribution
FROM position_returns pr
JOIN daily_totals dt ON pr.date = dt.date
ORDER BY pr.date, pr.tvId;

-- ============================================================================
-- API 7: GET /api/portfolios/highlights/{portfolioId}
-- BROKEN DOWN INTO SEPARATE QUERIES
-- ============================================================================
-- NOTE: This API has been broken down into 6 separate queries (see api7_queries_breakdown.sql)
-- Each query returns data that matches the expected response format:
--   - Query 7.1: highestLowestNumberofTradesbySymbol (highestTraded, lowestTraded)
--   - Query 7.2: highestLowestVolumebySymbol (highestVolume, lowestVolume)
--   - Query 7.3: lastDaysPnL
--   - Query 7.4: mostProfitable, leastProfitable
--   - Query 7.5: perTradeAveragePnL, avgWinningPnL, avgLosingPnL
--   - Query 7.6: totalGainPercentage, totalPnL, totalValue, winLossRatio
-- ============================================================================
-- 
-- For the complete breakdown, see: api7_queries_breakdown.sql
-- ============================================================================


-- ============================================================================
-- API 8: GET /api/portfolios/pastPerformance/{portfolioId}
-- Returns: avgGain, avgHoldingPeriod, avgTradesPerMonth, holding, loosingTrades,
--          openPosition, spBeat, stockTypeRatio, totalGain, totalTrades,
--          winLossRatio, winningTrades, winRate
-- ============================================================================
WITH
  -- Trade stats using prev_avg_price logic (Correct Formulae) - API 8
  -- Get initial average price from oms_positions for each tvId (if available)
  initial_avg_prices_api8 AS (
    SELECT DISTINCT
      tvId,
      averagePrice AS initialAvgPrice
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}'
      AND averagePrice IS NOT NULL
      AND averagePrice > 0
  ),
  numbered_transactions_api8 AS (
    SELECT
      t.id,
      t.tvId,
      t.entryDate,
      t.side,
      t.direction,
      t.quantity,
      t.price,
      t.entryPrice,
      COALESCE(iap.initialAvgPrice, t.entryPrice) AS initialAvgPrice,
      ROW_NUMBER() OVER (PARTITION BY t.tvId ORDER BY entryDate, id) AS seq
    FROM ofd_oms_transactions t
    LEFT JOIN initial_avg_prices_api8 iap ON t.tvId = iap.tvId
    WHERE t.portfolioId = '{portfolioId}'
      AND t.status = 'EXECUTED'
      AND t.side IN ('buy', 'sell')
  ),
  -- PASS 1: Calculate cumulative quantity and cost basis using entryPrice for all transactions
  transactions_pass1_api8 AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      curr.initialAvgPrice,
      curr.seq,
      SUM(
        CASE 
          WHEN curr.direction = 'LONG' AND curr.side = 'buy' THEN ABS(curr.quantity)
          WHEN curr.direction = 'LONG' AND curr.side = 'sell' THEN -ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'sell' THEN ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'buy' THEN -ABS(curr.quantity)
          ELSE 0
        END
      ) OVER (
        PARTITION BY curr.tvId 
        ORDER BY entryDate, id 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS cumulative_quantity,
      SUM(
        CASE 
          WHEN curr.direction = 'LONG' AND curr.side = 'buy' 
          THEN curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'LONG' AND curr.side = 'sell' 
          THEN -curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'sell' 
          THEN curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'buy' 
          THEN -curr.entryPrice * ABS(curr.quantity)
          ELSE 0
        END
      ) OVER (
        PARTITION BY curr.tvId 
        ORDER BY entryDate, id 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS cumulative_cost_basis_pass1
    FROM numbered_transactions_api8 curr
  ),
  -- Calculate prev_avg_price from PASS 1 results (FIXED: prev.seq = curr.seq - 1)
  transactions_with_prev_avg_api8 AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      curr.initialAvgPrice,
      curr.seq,
      curr.cumulative_quantity,
      curr.cumulative_cost_basis_pass1,
      CASE 
        WHEN curr.seq = 1 THEN curr.initialAvgPrice
        WHEN prev.cumulative_quantity != 0 AND prev.cumulative_quantity IS NOT NULL
        THEN prev.cumulative_cost_basis_pass1 / prev.cumulative_quantity
        ELSE curr.entryPrice
      END AS prev_avg_price
    FROM transactions_pass1_api8 curr
    LEFT JOIN transactions_pass1_api8 prev ON curr.tvId = prev.tvId AND prev.seq = curr.seq - 1
  ),
  -- PASS 2: Recalculate cumulative cost basis using prev_avg_price for closing positions
  transactions_with_calc_api8 AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      curr.prev_avg_price,
      curr.seq,
      curr.cumulative_quantity,
      SUM(
        CASE 
          WHEN curr.direction = 'LONG' AND curr.side = 'buy' 
          THEN curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'LONG' AND curr.side = 'sell' 
          THEN -COALESCE(curr.prev_avg_price, curr.entryPrice) * ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'sell' 
          THEN curr.entryPrice * ABS(curr.quantity)
          WHEN curr.direction = 'SHORT' AND curr.side = 'buy' 
          THEN -COALESCE(curr.prev_avg_price, curr.entryPrice) * ABS(curr.quantity)
          ELSE 0
        END
      ) OVER (
        PARTITION BY curr.tvId 
        ORDER BY entryDate, id 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS cumulative_cost_basis_approx
    FROM transactions_with_prev_avg_api8 curr
  ),
  transactions_with_prev_api8 AS (
    SELECT
      curr.id,
      curr.tvId,
      curr.entryDate,
      curr.side,
      curr.direction,
      curr.quantity,
      curr.price,
      curr.entryPrice,
      CASE 
        WHEN prev.cumulative_quantity != 0 AND prev.cumulative_quantity IS NOT NULL
        THEN prev.cumulative_cost_basis_approx / prev.cumulative_quantity
        ELSE NULL
      END AS prev_avg_price
    FROM transactions_with_calc_api8 curr
    LEFT JOIN transactions_with_calc_api8 prev ON curr.tvId = prev.tvId AND prev.seq = curr.seq - 1
  ),
  realized_trades AS (
    SELECT 
      CASE
        WHEN side = 'sell' AND direction = 'LONG' AND prev_avg_price IS NOT NULL
        THEN (price - prev_avg_price) * ABS(quantity)
        WHEN side = 'buy' AND direction = 'SHORT' AND prev_avg_price IS NOT NULL
        THEN (prev_avg_price - price) * ABS(quantity)
        ELSE 0
      END AS tradeReturn
    FROM transactions_with_prev_api8
    WHERE (side = 'sell' AND direction = 'LONG') OR (side = 'buy' AND direction = 'SHORT')
  ),
  trade_stats AS (
    SELECT
      AVG(tradeReturn) AS avgGain,
      COUNT(1) AS totalTrades,
      SUM(CASE WHEN tradeReturn > 0 THEN 1 ELSE 0 END) AS winningTrades,
      SUM(CASE WHEN tradeReturn <= 0 THEN 1 ELSE 0 END) AS loosingTrades,
      SUM(tradeReturn) AS totalGain
    FROM realized_trades
  ),
  -- Average holding period - simplified approach
  -- Uses average time between first buy and last sell per tvId
  holding_periods_per_symbol AS (
    SELECT
      tvId,
      MIN(CASE WHEN side = 'buy' THEN entryDate END) AS firstBuy,
      MAX(CASE WHEN side = 'sell' THEN entryDate END) AS lastSell
    FROM ofd_oms_transactions
    WHERE portfolioId = '{portfolioId}'
      AND status = 'EXECUTED'
      AND side IN ('buy', 'sell')
    GROUP BY tvId
  ),
  valid_holding_periods AS (
    SELECT 
      AVG(DATEDIFF('d', firstBuy, lastSell)) AS avgHoldingPeriod
    FROM holding_periods_per_symbol
    WHERE firstBuy IS NOT NULL 
      AND lastSell IS NOT NULL
      AND DATEDIFF('d', firstBuy, lastSell) >= 0
  ),
  -- Average trades per month (using CAST to DATE and grouping)
  months_active AS (
    SELECT
      COUNT(DISTINCT CAST(entryDate AS DATE)) AS monthCount
    FROM ofd_oms_transactions
    WHERE portfolioId = '{portfolioId}' AND status = 'EXECUTED'
  ),
  avg_trades_per_month AS (
    SELECT
      CASE 
        WHEN ma.monthCount > 0
        THEN CAST(ts.totalTrades AS DOUBLE) / ma.monthCount
        ELSE 0
      END AS avgTradesPerMonth
    FROM trade_stats ts
    CROSS JOIN months_active ma
  ),
  -- Holdings count
  holdings_count AS (
    SELECT COUNT(DISTINCT tvId) AS holding
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
  ),
  -- Open positions
  open_positions AS (
    SELECT COUNT(1) AS openPosition
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
  ),
  -- Stock type ratio
  stock_type_totals AS (
    SELECT
      assetClass,
      SUM(marketValue) AS totalValue
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
    GROUP BY assetClass
  ),
  total_portfolio_value AS (
    SELECT SUM(marketValue) AS total
    FROM ofd_oms_positions
    WHERE portfolioId = '{portfolioId}' AND quantity != 0
  ),
  stock_type_ratio AS (
    SELECT
      COALESCE(SUM(CASE WHEN assetClass = 'stocks' THEN totalValue ELSE 0 END) / NULLIF(total, 0) * 100, 0) AS stocks,
      COALESCE(SUM(CASE WHEN assetClass = 'crypto' THEN totalValue ELSE 0 END) / NULLIF(total, 0) * 100, 0) AS crypto,
      COALESCE(SUM(CASE WHEN assetClass = 'ETFs' THEN totalValue ELSE 0 END) / NULLIF(total, 0) * 100, 0) AS ETFs,
      COALESCE(SUM(CASE WHEN assetClass = 'futures' THEN totalValue ELSE 0 END) / NULLIF(total, 0) * 100, 0) AS futures,
      COALESCE(SUM(CASE WHEN assetClass = 'indices' THEN totalValue ELSE 0 END) / NULLIF(total, 0) * 100, 0) AS indices,
      COALESCE(SUM(CASE WHEN assetClass = 'options' THEN totalValue ELSE 0 END) / NULLIF(total, 0) * 100, 0) AS options
    FROM stock_type_totals
    CROSS JOIN total_portfolio_value
  ),
  -- SP Beat (would need SPY data - placeholder)
  sp_beat AS (
    SELECT 0.01 AS spBeat  -- Placeholder: would need SPY comparison
  ),
  -- Win/Loss ratio
  win_loss_ratio AS (
    SELECT
      CASE 
        WHEN ts.loosingTrades > 0
        THEN CAST(ts.winningTrades AS DOUBLE) / NULLIF(ts.loosingTrades, 0)
        ELSE 0
      END AS winLossRatio,
      CASE 
        WHEN ts.totalTrades > 0 
        THEN (ts.winningTrades * 100.0 / ts.totalTrades)
        ELSE 0
      END AS winRate
    FROM trade_stats ts
  )
SELECT
  COALESCE(ts.avgGain, 0) AS avgGain,
  COALESCE(vhp.avgHoldingPeriod, 0) AS avgHoldingPeriod,
  COALESCE(atpm.avgTradesPerMonth, 0) AS avgTradesPerMonth,
  hc.holding,
  ts.loosingTrades,
  op.openPosition,
  sb.spBeat,
  str.stocks,
  str.crypto,
  str.ETFs,
  str.futures,
  str.indices,
  str.options,
  COALESCE(ts.totalGain, 0) AS totalGain,
  ts.totalTrades,
  wlr.winLossRatio,
  ts.winningTrades,
  wlr.winRate
FROM trade_stats ts
CROSS JOIN valid_holding_periods vhp
CROSS JOIN avg_trades_per_month atpm
CROSS JOIN holdings_count hc
CROSS JOIN open_positions op
CROSS JOIN stock_type_ratio str
CROSS JOIN sp_beat sb
CROSS JOIN win_loss_ratio wlr;

-- ============================================================================
-- API 9: GET /api/portfolios/rankings/{portfolioId}
-- BROKEN DOWN INTO SEPARATE QUERIES FOR ARRAYS
-- ============================================================================
-- NOTE: This API has been broken down into 4 separate queries (see api9_queries_breakdown.sql)
-- Each query returns an array that matches the expected response format:
--   - Query 9.1: mostTraded array
--   - Query 9.2: biggestInv array
--   - Query 9.3: mostBought array
--   - Query 9.4: topHoldings array
-- ============================================================================
-- 
-- For the complete breakdown, see: api9_queries_breakdown.sql
-- ============================================================================

-- Note: For holdings array, use separate query:
-- SELECT tvId, direction, averagePrice AS average_price, quantity, assetClass AS asset_type
-- FROM ofd_oms_positions WHERE portfolioId = '{portfolioId}' AND quantity != 0;

-- ============================================================================
-- API 10: GET /api/portfolios/tradeSummery/{portfolioId}
-- Returns: trade summary with orders (already exists as Query #38)
-- Use the existing tradeSummary query from questdb_portfolio_queries_cte_format.sql
-- ============================================================================

WITH
  buy_totals AS (
    SELECT
      tvId,
      assetClass,
      AVG(price) AS avgOpenPrice,
      SUM(quantity) AS openQty,
      MIN(entryDate) AS entryDate
    FROM ofd_oms_transactions
    WHERE portfolioId = '{portfolioId}'
      AND side = 'buy'
      AND status = 'EXECUTED'
    GROUP BY tvId, assetClass
  ),
  sell_totals AS (
    SELECT
      tvId,
      assetClass,
      AVG(price) AS avgClosePrice,
      SUM(quantity) AS closeQty,
      MAX(entryDate) AS exitDate
    FROM ofd_oms_transactions
    WHERE portfolioId = '{portfolioId}'
      AND side = 'sell'
      AND status = 'EXECUTED'
    GROUP BY tvId, assetClass
  )
SELECT
  COALESCE(bt.tvId, st.tvId) AS tvId,
  COALESCE(bt.assetClass, st.assetClass) AS assetClass,
  CASE 
    WHEN st.tvId IS NOT NULL THEN 'sell'
    ELSE 'buy'
  END AS side,
  bt.avgOpenPrice,
  st.avgClosePrice,
  bt.entryDate,
  st.exitDate,
  bt.openQty,
  st.closeQty,
  COALESCE(st.avgClosePrice * st.closeQty, 0) - COALESCE(bt.avgOpenPrice * bt.openQty, 0) AS totalReturn,
  CASE 
    WHEN bt.avgOpenPrice * bt.openQty > 0
    THEN ((COALESCE(st.avgClosePrice * st.closeQty, 0) - COALESCE(bt.avgOpenPrice * bt.openQty, 0)) * 100.0 / 
          (bt.avgOpenPrice * bt.openQty))
    ELSE 0
  END AS returnPct,
  CASE 
    WHEN st.tvId IS NOT NULL THEN 'COMPLETED'
    ELSE 'OPEN'
  END AS status,
  DATEDIFF('d', bt.entryDate, COALESCE(st.exitDate, NOW())) AS holdingDays
FROM buy_totals bt
LEFT JOIN sell_totals st ON bt.tvId = st.tvId AND bt.assetClass = st.assetClass
UNION ALL
SELECT
  st.tvId AS tvId,
  st.assetClass AS assetClass,
  'sell' AS side,
  NULL AS avgOpenPrice,
  st.avgClosePrice,
  NULL AS entryDate,
  st.exitDate,
  NULL AS openQty,
  st.closeQty,
  COALESCE(st.avgClosePrice * st.closeQty, 0) AS totalReturn,
  0 AS returnPct,
  'COMPLETED' AS status,
  NULL AS holdingDays
FROM sell_totals st
LEFT JOIN buy_totals bt ON st.tvId = bt.tvId AND st.assetClass = bt.assetClass
WHERE bt.tvId IS NULL
ORDER BY COALESCE(entryDate, exitDate) DESC;

-- ============================================================================
-- ADDITIONAL: Line Items (Holdings) for Portfolio List
-- Returns: holdings with current prices, PnL, etc.
-- ============================================================================
WITH
  -- Use LATEST ON instead of subquery (QuestDB-compatible)
  -- Using unified latest_prices table with tvId
  latest_prices_cte3 AS (
    SELECT 
      tvId,
      price AS latestPrice
    FROM latest_prices
    LATEST ON cts PARTITION BY tvId
  ),
  positions AS (
    SELECT 
      p.tvId,
      p.assetClass,
      p.direction,
      ABS(p.quantity) AS shares,
      p.averagePrice AS avgCostPerShare,
      p.marketValue,
      p.unrealizedPnl AS totalGain,
      CASE 
        WHEN p.tradeValue > 0 
        THEN (p.unrealizedPnl / p.tradeValue) * 100 
        ELSE 0 
      END AS totalGainPercent,
      COALESCE(
        lp.latestPrice,
        p.marketValue / NULLIF(ABS(p.quantity), 0)
      ) AS currentPrice
    FROM ofd_oms_positions p
    LEFT JOIN latest_prices_cte3 lp ON p.tvId = lp.tvId
    WHERE p.portfolioId = '{portfolioId}' AND p.quantity != 0
  )
SELECT
  tvId,
  assetClass AS stockType,
  shares,
  currentPrice,
  avgCostPerShare AS originalPrice,
  marketValue,
  totalGain,
  totalGainPercent,
  direction,
  -- Day gain (would need previous day snapshot)
  0 AS dayGain,
  0 AS dayGainPercent
FROM positions
ORDER BY marketValue DESC;


