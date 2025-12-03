Top Holding

WITH
      positionsWithPrice AS (
        SELECT 
          p.portfolioId,
          p.tvId,
          p.symbol,
          p.assetClass,
          p.direction,
          ABS(p.quantity) * p.averagePrice AS totalCost,
          lp.price * ABS(p.quantity) AS currentMarketValue,
          lp.price AS currentPrice
        FROM ${omsPositions} p
        LEFT JOIN latest_prices lp ON p.tvId = lp.tvId
        WHERE p.portfolioId = ${portfolioId}  AND p.quantity != 0
        ORDER BY p.cts DESC
      ),
      withAllCalculations AS (
      SELECT
          portfolioId,
          tvId,
          symbol,
          assetClass,
          totalCost,
          CASE WHEN direction = 'long' THEN currentMarketValue - totalCost ELSE totalCost - currentMarketValue END AS unrealizedPnl
        FROM positionsWithPrice
      ),
      withMarketValue AS (
        SELECT 
            wac.tvId,
            wac.symbol,
            wac.assetClass as assetType,
            (wac.totalCost + wac.unrealizedPnl) AS marketValue
        FROM withAllCalculations wac
        ORDER BY marketValue DESC
      )
      SELECT * FROM withMarketValue limit 10;

Most Bought

SELECT
        tvId,
        LAST(symbol) as symbol,
        LAST(assetClass) AS assetType,
        SUM(ABS(quantity)) AS shares
        FROM ${omsOrders}
        WHERE portfolioId = ${portfolioId}
        AND side = 'buy'
        AND status = 'EXECUTED'
        GROUP BY tvId
        ORDER BY shares DESC
        LIMIT 10;

Biggest Investment

SELECT
        tvId,
        symbol,
        assetClass AS assetType,
        averagePrice * ABS(quantity) AS investment
        FROM ${omsPositions}
        WHERE portfolioId = ${portfolioId} AND quantity != 0
        ORDER BY investment DESC
        LIMIT 10;

Most Traded (Pre requisite tradeSummary)
SELECT
        tvId,
        LAST(symbol) as symbol,
        LAST(assetClass) AS assetType,
        COUNT(1) AS count
        FROM ${portfolioTradeSummary}
        WHERE portfolioId = ${portfolioId} AND status = 'closed'
        GROUP BY tvId
        ORDER BY count DESC
        LIMIT 10;