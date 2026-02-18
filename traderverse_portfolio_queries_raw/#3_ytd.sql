WITH 
        positions AS (
            SELECT 
                p.tvId, 
                p.symbol, 
                p.assetClass, 
                lp.price AS price, 
                p.direction, 
               (p.averagePrice * ABS(p.quantity)) AS totalCost, 
                (ABS(p.quantity) * lp.price) AS marketValue,
                pr.accountBalance,
                pr.availableFund,
                pic.initialCapital,
                pic.initialCapital as startValueOfYear,
                p.portfolioId
            FROM ofd_oms_positions p 
            JOIN ofd_portfolio pr ON pr.id = p.portfolioId
            JOIN ofd_portfolio_initial_capital pic ON pic.portfolioId = p.portfolioId
            LEFT JOIN latest_prices lp ON lp.tvId = p.tvId 

            WHERE p.portfolioId = '5deb9038-5bd8-459d-b060-22ac987050ff'
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
                LAST(startValueOfYear) as startValueOfYear,
                LAST(accountBalance) as accountBalance,
                SUM(equity) AS equity
            FROM withEquity 
            group by portfolioId
        ),
        withAccountValue AS (
            SELECT
             startValueOfYear,
            (accountBalance + equity) AS accountValue,
            from withGroupSum
        ),
        withYTDPnl as (
            SELECT 
            (accountValue - startValueOfYear) as ytdPnl,
            Coalesce((accountValue/startValueOfYear)-1,0) *100  as ytdPnlPercentage
            FROM
            withAccountValue
        )
        SELECT * FROM withYTDPnl;