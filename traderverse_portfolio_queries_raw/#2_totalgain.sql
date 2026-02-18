query for overview stats
WITH 
        positions AS (
            SELECT 
                p.tvId, 
                p.symbol, 
                p.assetClass, 
                lp.price AS price, 
                p.direction, 
                (p.averagePrice * p.quantity) AS totalCost, 
                (p.quantity * lp.price) AS marketValue,
                pr.accountBalance,
                pr.availableFund,
                pic.initialCapital,
                p.portfolioId
            FROM ${omsPositions} p 
            JOIN ${portfolio} pr ON pr.id = p.portfolioId
            JOIN ${portfolioInitialCapital} pic ON pic.portfolioId = p.portfolioId
            LEFT JOIN latest_prices lp ON lp.tvId = p.tvId 

            WHERE p.portfolioId = ${portfolioId}
                AND p.quantity != 0
        ), 
        withUnrealizedPnl AS (
            SELECT 
                *,
                CASE 
                    WHEN direction = 'long' THEN marketValue - totalCost 
                    ELSE totalCost - marketValue 
                END AS unrealizedPnl 
            FROM positions
        ), 
        withEquity AS (
            SELECT 
                *,
                (totalCost + unrealizedPnl) AS equity 
            FROM withUnrealizedPnl
        ), 
        withGroupSum AS (
            SELECT 
                SUM(availableFund) AS availableFund,
                SUM(initialCapital) AS initialCapital,
                SUM(accountBalance) AS accountBalance,
                COUNT(tvId) AS holdingCount,
                SUM(unrealizedPnl) as unrealizedPnl,
                SUM(equity) AS equity
            FROM withEquity 
            group by portfolioId
        ),
        withAccountValue AS (
            SELECT 
            availableFund,
            accountBalance,
            equity,
            holdingCount,
            unrealizedPnl,
            -- TODO: Add realized PnL
            '0' as realizedPnl,
            (accountBalance + equity) AS accountValue,
            COALESCE((accountBalance + equity) / NULLIF(initialCapital, 0), 0) * 100 AS totalGainPercentage
            from withGroupSum
        )
        SELECT * FROM withAccountValue;