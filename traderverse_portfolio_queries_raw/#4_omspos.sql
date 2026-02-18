WITH
      positionsWithPrice AS (
        SELECT 
          p.portfolioId,
          p.tvId,
          p.symbol,
          p.assetClass,
          p.direction,
          ABS(p.quantity) AS shares,
          p.averagePrice,
          ABS(p.quantity) * p.averagePrice AS totalCost,
          lp.price * ABS(p.quantity) AS currentMarketValue,
          lp.price AS currentPrice
        FROM ofd_oms_positions p
        LEFT JOIN latest_prices lp ON p.tvId = lp.tvId
        WHERE p.portfolioId = '5deb9038-5bd8-459d-b060-22ac987050ff'  AND p.quantity != 0
        ORDER BY p.cts DESC
      ),
      withAllCalculations AS (
      SELECT
          portfolioId,
          tvId,
          symbol,
          assetClass,
          shares,
          currentPrice,
          averagePrice,
          direction,
          totalCost,
          CASE WHEN direction = 'long' THEN currentMarketValue - totalCost ELSE totalCost - currentMarketValue END AS unrealizedPnl,
          COALESCE(
            CASE WHEN direction = 'long' THEN COALESCE((currentMarketValue - totalCost) / totalCost,0) ELSE COALESCE((totalCost - currentMarketValue) / totalCost,0) END,
            0
          ) * 100 AS totalGainPercent,
          -- Day gain (would need previous day snapshot)
          0 as changePercent24h,
          0 AS dayGain,
          0 AS dayGainPercent
        FROM positionsWithPrice
        LIMIT 10 , 0
      ),
      takeProfitAndStopLossOrders AS (
        SELECT
          tvId,
          price,
          orderType,
          status,
          ROW_NUMBER() OVER (PARTITION BY tvId, orderType ORDER BY cts DESC) AS rn
        FROM ofd_oms_orders
          WHERE portfolioId = '5deb9038-5bd8-459d-b060-22ac987050ff'
        AND (orderType = 'takeprofit' OR orderType = 'stoploss')
      ),
      takeProfitAndStopLossInfo AS (
        SELECT
          tvId,
          MAX(price) AS price,
          orderType,
          status
        FROM takeProfitAndStopLossOrders
        WHERE rn = 1
        GROUP BY tvId, orderType, status
      ),
      withStopLossAndTakeProfit AS (
      SELECT 
        wac.tvId,
        wac.symbol,
        wac.assetClass,
        wac.shares,
        wac.currentPrice,
        wac.averagePrice,
        (wac.totalCost + wac.unrealizedPnl) AS marketValue,
        wac.direction,
        wac.totalCost,
        wac.unrealizedPnl,
        wac.totalGainPercent,
        wac.dayGain,
        wac.dayGainPercent,
        wac.changePercent24h,
        CASE WHEN tpsli.orderType = 'takeprofit' AND tpsli.status = 'PENDING' THEN tpsli.price ELSE NULL END AS takeProfit,
        CASE WHEN tpsli.orderType = 'stoploss' AND tpsli.status = 'PENDING' THEN tpsli.price ELSE NULL END AS stopLoss
      FROM withAllCalculations wac
      LEFT JOIN takeProfitAndStopLossInfo tpsli ON wac.tvId = tpsli.tvId
      )
      SELECT * FROM withStopLossAndTakeProfit;


--- unrealized gain has a problem