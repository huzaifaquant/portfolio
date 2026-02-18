Trade Summary Fetch Query

SELECT
  id,
  symbol,
  tvId,
  side,
  entryDate,
  assetClass,
  investment,
  status,
  avgClosePrice,
  avgOpenPrice,
  openQty,
  closeQty,
  exitDate,
  portfolioId,
  deletedOn,
  createdAt,
  price,
  remainingQty,
  cost,
  totalCost,
  currentMarketValue,
  unrealizedPnl,
  totalReturn,
  returnPct
FROM (
  SELECT
    p.id,
    p.symbol,
    p.tvId,
    p.side,
    p.entryDate,
    p.assetClass,
    p.investment,
    p.status,
    p.avgClosePrice,
    p.avgOpenPrice,
    p.openQty,
    p.closeQty,
    p.exitDate,
    p.portfolioId,
    p.deletedOn,
    p.createdAt,
    lp.price,
    (p.openQty - p.closeQty) AS remainingQty,
    (p.openQty - p.closeQty) * p.avgOpenPrice AS cost,
    p.avgOpenPrice * p.openQty AS totalCost,
    (p.openQty - p.closeQty) * COALESCE(lp.price, 0) AS currentMarketValue,
    CASE
      WHEN p.side = 'buy' THEN (p.openQty - p.closeQty) * COALESCE(lp.price, 0) - (p.openQty - p.closeQty) * p.avgOpenPrice
      ELSE (p.openQty - p.closeQty) * p.avgOpenPrice - (p.openQty - p.closeQty) * COALESCE(lp.price, 0)
    END AS unrealizedPnl,
    p.totalReturn + 
      CASE
        WHEN p.side = 'buy' THEN (p.openQty - p.closeQty) * (COALESCE(lp.price, 0) - p.avgOpenPrice)
        ELSE (p.openQty - p.closeQty) * (p.avgOpenPrice - COALESCE(lp.price, 0))
      END AS totalReturn,
    NULLIF(
      p.totalReturn + 
        CASE
          WHEN p.side = 'buy' THEN (p.openQty - p.closeQty) * (COALESCE(lp.price, 0) - p.avgOpenPrice)
          ELSE (p.openQty - p.closeQty) * (p.avgOpenPrice - COALESCE(lp.price, 0))
        END,
      0
    ) / NULLIF(p.avgOpenPrice * p.openQty, 0) AS returnPct
  FROM
    ofd_portfolio_trade_summary p
    LEFT JOIN latest_prices lp ON p.tvId = lp.tvId
  WHERE
    p.portfolioId = ${portfolioId}
  ${sql.raw(limitQuery)}
) AS computed;