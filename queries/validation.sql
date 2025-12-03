-- ============================================================================
-- COMPLETE PORTFOLIO ANALYSIS - PRODUCTION-READY (QUESTDB)
-- ============================================================================
WITH step1_raw_data AS (
    SELECT
        t.id,
        t.tradeId,
        t.tvId,
        t.side,
        t.quantity,
        t.price,
        t.pending,
        ABS(t.quantity) AS executed_qty,
        t.mts,
        t.cts,
        t.portfolioId,
    FROM qat_oms_trades t
    WHERE t.portfolioId = '5deb9038-5bd8-459d-b060-22ac987050ff'
      AND t.tvId = 'NVDA_NASDAQ'
),
step2_cumulative_qty AS (
    SELECT *,
        SUM(quantity) OVER (PARTITION BY portfolioId, tvId ORDER BY mts ASC, id ASC) AS current_quantity
    FROM step1_raw_data
),
step2b_prev_values AS (
    SELECT *,
        LAG(current_quantity) OVER (PARTITION BY portfolioId, tvId ORDER BY mts ASC, id ASC) AS prev_quantity,
        LAG(price) OVER (PARTITION BY portfolioId, tvId ORDER BY mts ASC, id ASC) AS prev_price
    FROM step2_cumulative_qty
),
step3_segment_id AS (
    SELECT *,
        SUM(CASE
            WHEN prev_quantity IS NULL AND current_quantity != 0 THEN 1
            WHEN prev_quantity = 0 AND current_quantity != 0 THEN 1
            WHEN prev_quantity > 0 AND current_quantity < 0 THEN 1
            WHEN prev_quantity > 0 AND current_quantity = 0 THEN 1
            WHEN prev_quantity < 0 AND current_quantity > 0 THEN 1
            WHEN prev_quantity < 0 AND current_quantity = 0 THEN 1
            ELSE 0
        END) OVER (PARTITION BY portfolioId, tvId ORDER BY mts ASC, id ASC) AS segment_id
    FROM step2b_prev_values
),
step4_position_type AS (
    SELECT *,
        CASE WHEN current_quantity > 0 THEN 'long'
             WHEN current_quantity < 0 THEN 'short'
             ELSE 'flat' END AS position_direction
    FROM step3_segment_id
),
step5_split_trades AS (
    SELECT *,
        CASE
            WHEN prev_quantity IS NULL THEN 0
            WHEN side = 'sell' AND prev_quantity > 0
                 THEN LEAST(executed_qty, ABS(prev_quantity))
            WHEN side = 'buy' AND prev_quantity < 0
                 THEN LEAST(executed_qty, ABS(prev_quantity))
            ELSE 0
        END AS closed_qty,
        CASE
            WHEN prev_quantity IS NULL THEN executed_qty
            WHEN side = 'sell' AND prev_quantity > 0
                 THEN GREATEST(0, executed_qty - ABS(prev_quantity))
            WHEN side = 'buy' AND prev_quantity < 0
                 THEN GREATEST(0, executed_qty - ABS(prev_quantity))
            WHEN side = 'buy' THEN executed_qty
            WHEN side = 'sell' THEN executed_qty
            ELSE 0
        END AS open_qty
    FROM step4_position_type
),
step6_segment_buys AS (
    SELECT *,
        SUM(CASE
            WHEN side = 'buy' AND position_direction = 'long' THEN open_qty * price
            WHEN side = 'sell' AND position_direction = 'short' THEN open_qty * price
            ELSE 0
        END) OVER (PARTITION BY portfolioId, tvId, segment_id ORDER BY mts ASC, id ASC
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_buy_cost,
        SUM(CASE
            WHEN side = 'buy' AND position_direction = 'long' THEN open_qty
            WHEN side = 'sell' AND position_direction = 'short' THEN open_qty
            ELSE 0
        END) OVER (PARTITION BY portfolioId, tvId, segment_id ORDER BY mts ASC, id ASC
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_buy_qty
    FROM step5_split_trades
),
step7_cost_basis AS (
    SELECT *,
        CASE
            WHEN cumulative_buy_qty = 0 THEN 0
            ELSE (cumulative_buy_cost / cumulative_buy_qty) * ABS(current_quantity)
        END AS cost_basis
    FROM step6_segment_buys
),
step8_avg_price AS (
    SELECT *,
        CASE WHEN current_quantity != 0
             THEN cost_basis / ABS(current_quantity)
             ELSE 0
        END AS avg_price
    FROM step7_cost_basis
),
step9_prev_avg_price AS (
    SELECT *,
        LAG(avg_price) OVER (PARTITION BY portfolioId, tvId ORDER BY mts ASC, id ASC) AS prev_avg_price
    FROM step8_avg_price
),
step10_unrealized_pnl AS (
    SELECT *,
        ABS(current_quantity) * price AS position_value,
        CASE
            WHEN current_quantity < 0 THEN cost_basis + (current_quantity * price)
            ELSE (current_quantity * price) - cost_basis
        END AS unrealized_pnl
    FROM step9_prev_avg_price
),
step11_realized_pnl AS (
    SELECT *,
        CASE
            WHEN side = 'sell'
                 AND prev_quantity > 0
                 AND prev_avg_price IS NOT NULL
                 THEN (price - prev_avg_price) * closed_qty
            WHEN side = 'buy'
                 AND prev_quantity < 0
                 AND prev_avg_price IS NOT NULL
                 THEN (prev_avg_price - price) * closed_qty
            ELSE NULL
        END AS realized_pnl_point_in_time
    FROM step10_unrealized_pnl
),
step12_cumulative_realized AS (
    SELECT *,
        SUM(CASE WHEN realized_pnl_point_in_time IS NOT NULL
                 THEN realized_pnl_point_in_time
                 ELSE 0
            END) OVER (PARTITION BY portfolioId, tvId ORDER BY mts ASC, id ASC) AS realized_pnl_lifetime,
        SUM(CASE WHEN realized_pnl_point_in_time IS NOT NULL
                 THEN realized_pnl_point_in_time
                 ELSE 0
            END) OVER (PARTITION BY portfolioId, tvId, segment_id ORDER BY mts ASC, id ASC) AS realized_pnl_segment
    FROM step11_realized_pnl
),
step13_return_pct AS (
    SELECT *,
        CASE
            WHEN cost_basis = 0 THEN 0.0
            ELSE (unrealized_pnl / cost_basis) * 100
        END AS unrealized_return_pct,
        CASE
            WHEN cost_basis = 0 THEN 0.0
            ELSE ((realized_pnl_segment + unrealized_pnl) / cost_basis) * 100
        END AS total_return_pct_segment,
        CASE
            WHEN cost_basis = 0 THEN 0.0
            ELSE ((realized_pnl_lifetime + unrealized_pnl) / cost_basis) * 100
        END AS total_return_pct_lifetime
    FROM step12_cumulative_realized
),
step14_ticker_pv AS (
    SELECT *,
        ABS(current_quantity) * price AS ticker_pv
    FROM step13_return_pct
),
step15_all_timestamps AS (
    SELECT DISTINCT portfolioId, mts FROM step14_ticker_pv
),
step16_all_symbols AS (
    SELECT DISTINCT tvId FROM step14_ticker_pv
),
step17_all_mts_symbol_combinations AS (
    SELECT
        ts.portfolioId,
        ts.mts,
        s.tvId
    FROM step15_all_timestamps ts
    CROSS JOIN (SELECT DISTINCT tvId, tvId FROM step14_ticker_pv) s
),
step18_get_latest_state_per_combination AS (
    SELECT
        c.portfolioId,
        c.mts,
        c.tvId,
        t.current_quantity,
        t.price,
        ROW_NUMBER() OVER (PARTITION BY c.portfolioId, c.mts, c.tvId ORDER BY t.mts DESC, t.id DESC) as rn
    FROM step17_all_mts_symbol_combinations c
    LEFT JOIN step14_ticker_pv t
        ON c.portfolioId = t.portfolioId
        AND c.tvId = t.tvId
        AND t.mts <= c.mts
),
step19_latest_state_deduped AS (
    SELECT
        portfolioId,
        mts,
        tvId,
        current_quantity,
        price
    FROM step18_get_latest_state_per_combination
    WHERE rn = 1
),
step20_total_pv_per_timestamp AS (
    SELECT
        portfolioId,
        mts,
        SUM(CASE WHEN current_quantity != 0 THEN ABS(current_quantity) * price ELSE 0 END) as total_pv,
        COUNT(DISTINCT CASE WHEN current_quantity != 0 THEN tvId END) as holding_count
    FROM step19_latest_state_deduped
    GROUP BY portfolioId, mts
),
step21_merge_total_pv AS (
    SELECT
        t.*,
        COALESCE(tp.total_pv, 0) AS total_pv,
        COALESCE(tp.holding_count, 0) AS holding_count
    FROM step14_ticker_pv t
    LEFT JOIN step20_total_pv_per_timestamp tp
        ON t.portfolioId = tp.portfolioId
        AND t.mts = tp.mts
),
step22_cash_change AS (
    SELECT *,
        CASE
            WHEN side = 'buy'
                THEN
                    (CASE WHEN prev_quantity < 0 AND closed_qty > 0
                          THEN (2 * prev_avg_price - price) * closed_qty
                          ELSE 0
                     END)
                    +
                    (CASE WHEN open_qty > 0
                          THEN -(price * open_qty)
                          ELSE 0
                     END)
            WHEN side = 'sell'
                THEN
                    (CASE WHEN prev_quantity > 0 AND closed_qty > 0
                          THEN (price * closed_qty)
                          ELSE 0
                     END)
                    +
                    (CASE WHEN open_qty > 0
                          THEN -(price * open_qty)
                          ELSE 0
                     END)
            ELSE 0
        END AS cash_change
    FROM step21_merge_total_pv
),
step23_cumulative_cash_change AS (
    SELECT *,
        SUM(cash_change) OVER (PARTITION BY portfolioId ORDER BY mts ASC, id ASC) AS total_cash_change
    FROM step22_cash_change
),
step24_pending_orders AS (
    SELECT
        portfolioId,
        COALESCE(SUM(price * ABS(quantity)), 0) AS pending_value
    FROM qat_oms_orders
    WHERE portfolioId = '5deb9038-5bd8-459d-b060-22ac987050ff'
      AND status IN ('PENDING', 'PARTIAL')
      AND pendingQty > 0
    GROUP BY portfolioId
),
step25_remaining_cash AS (
    SELECT *,
        50000.0 AS initialCapital,
        50000.0 + total_cash_change AS remaining_cash
    FROM step23_cumulative_cash_change
),
step26_account_balance AS (
    SELECT
        t.*,
        po.pending_value,
        ROUND(total_pv + remaining_cash + COALESCE(po.pending_value, 0), 2) AS account_balance
    FROM step25_remaining_cash t
    LEFT JOIN step24_pending_orders po ON t.portfolioId = po.portfolioId
)
SELECT
    mts,
    tvId,
    side,
    executed_qty AS quantity,
    price,
    current_quantity,
    position_direction,
    segment_id,
    closed_qty,
    open_qty,
    avg_price,
    cost_basis,
    holding_count,
    ROUND(position_value, 2) AS position_value,
    ROUND(unrealized_pnl, 2) AS unrealized_pnl,
    ROUND(unrealized_return_pct, 2) AS unrealized_return_pct,
    ROUND(realized_pnl_point_in_time, 2) AS realized_pnl_point_in_time,
    ROUND(realized_pnl_segment, 2) AS realized_pnl_segment,
    ROUND(realized_pnl_lifetime, 2) AS realized_pnl_lifetime,
    ROUND(total_return_pct_segment, 2) AS total_return_pct_segment,
    ROUND(total_return_pct_lifetime, 2) AS total_return_pct_lifetime,
    ROUND(initialCapital, 2) AS initialCapital,
    ROUND(cash_change, 2) AS cash_change,
    ROUND(total_cash_change, 2) AS total_cash_change,
    ROUND(remaining_cash, 2) AS remaining_cash,
    ROUND(ticker_pv, 2) AS equity_per_symbol,
    ROUND(total_pv, 2) AS total_pv,
    ROUND(COALESCE(pending_value, 0), 2) AS pending_value,
    ROUND(account_balance, 2) AS account_balance,
    ROUND(((realized_pnl_lifetime + unrealized_pnl) / initialCapital) * 100, 2) AS totalGainPercentage
FROM step26_account_balance
ORDER BY mts ASC;