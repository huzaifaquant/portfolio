WITH positions AS (
            SELECT 
                p.tvId, 
                p.symbol, 
                p.assetClass, 
                lp.close AS price, 
                p.direction, 
                (p.averagePrice * p.quantity) AS totalCost, 
                (p.quantity * lp.close) AS marketValue, 
                sv1.sector, 
                sv1.industry, 
                CASE 
                    WHEN sv1.marketCap < 50000000 THEN 'Nano-Cap' 
                    WHEN sv1.marketCap < 300000000 THEN 'Micro-Cap' 
                    WHEN sv1.marketCap < 2000000000 THEN 'Small-Cap' 
                    WHEN sv1.marketCap < 10000000000 THEN 'Mid-Cap' 
                    WHEN sv1.marketCap < 200000000000 THEN 'Large-Cap' 
                    ELSE 'Mega-Cap' 
                END AS marketCapClassification 
            FROM ofd_oms_positions p 
            LEFT JOIN (
                SELECT tvId, close FROM crypto_price_1m LATEST BY tvId
                UNION ALL
                SELECT tvId, close FROM price_1m LATEST BY tvId
            ) lp ON lp.tvId = p.tvId 
            LEFT JOIN stocks_v1 sv1 ON sv1.tvId = p.tvId
            WHERE p.portfolioId = 'f57e0b85-698f-45e1-bcd6-d6b3ce9b9847'
                AND p.quantity != 0
        ), 
        withUnrealizedPnl AS (
            SELECT 
                tvId,
                symbol,
                assetClass,
                price,
                direction,
                totalCost,
                marketValue,
                sector,
                industry,
                marketCapClassification,
                CASE 
                    WHEN direction = 'long' THEN marketValue - totalCost 
                    ELSE totalCost - marketValue 
                END AS unrealizedPnl 
            FROM positions
        ), 
        withEquity AS (
            SELECT 
                tvId,
                symbol,
                assetClass,
                price,
                direction,
                totalCost,
                marketValue,
                sector,
                industry,
                marketCapClassification,
                unrealizedPnl,
                (totalCost + unrealizedPnl) AS equity 
            FROM withUnrealizedPnl
        ), 
        withGroupSum AS (
            SELECT 
               -- replace with sector, industry, marketCapClassification, tvId, assetClass, 
                assetClass AS diversificationKey,
                SUM(equity) AS equity, 
                SUM(totalCost) AS cost 
            FROM withEquity 
            GROUP BY assetClass
        ), 
        withTotal AS (
            SELECT 
                diversificationKey,
                equity,
                cost,
                SUM(cost) OVER () AS totalCost, 
                SUM(equity) OVER () AS totalEquity 
            FROM withGroupSum
        ), 
        withDiversification AS (
            SELECT 
                CASE 
                    WHEN diversificationKey IS NULL THEN 'N/A' 
                    ELSE CAST(diversificationKey AS TEXT) 
                END AS diversificationKey, 
                equity, 
                cost, 
                COALESCE(equity / NULLIF(totalEquity, 0), 0) * 100 AS equityDiversificationPercentage, 
                COALESCE(cost / NULLIF(totalCost, 0), 0) * 100 AS costDiversificationPercentage
            FROM withTotal
        ) 
        SELECT 
            diversificationKey,
            equity,
            cost,
            equityDiversificationPercentage,
            costDiversificationPercentage
        FROM withDiversification
        ORDER BY equity DESC