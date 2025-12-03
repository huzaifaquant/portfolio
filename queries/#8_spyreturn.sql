WITH positions AS (
  SELECT 
      p.tvId,  
      p.direction, 
      (p.averagePrice * ABS(p.quantity)) AS totalCost, 
      (ABS(p.quantity) * lp.price) AS marketValue,
      pr.accountBalance,
  FROM ofd_portfolio pr
  LEFT JOIN ofd_oms_positions p ON p.portfolioId = pr.id
  LEFT JOIN latest_prices lp ON lp.tvId = p.tvId 
  WHERE pr.id = '41a708ed-8046-4941-b9a2-7d658a4dc323'
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
      LAST(accountBalance) AS accountBalance,
      SUM(equity) AS equity
  FROM withEquity 
),
withAccountValue AS (
  SELECT 
  (accountBalance + equity) AS portfolioValue,
  DATE_TRUNC('day',NOW()) AS date
  from withGroupSum
),
withCurrentPortfolioValue AS (
 SELECT * from (SELECT 
    date,
    value 
  FROM ofd_portfolio_valuation
  WHERE deletedAt IS NULL 
      AND portfolioId = '41a708ed-8046-4941-b9a2-7d658a4dc323'
  ORDER BY date ASC) withFilter

  UNION

  SELECT 
      date,
      portfolioValue AS value
  FROM withAccountValue
),
prevValuation as (
  select
  date,
  round(value,2) as portfolioValue,
  -- portfolioId,
  LAG(value) OVER(ORDER BY date) as prevPortfolioValue,
  -- (value - LAG(value) OVER (ORDER BY date)) / LAG(value) OVER (ORDER BY date) AS dailyReturn,
  --   ((value - LAG(value) OVER (ORDER BY date)) / LAG(value) OVER (ORDER BY date)) * 100 AS dailyReturnPercentage
   from withCurrentPortfolioValue 
),
withDailyReturn AS (
  select
  date,
  portfolioValue,
  round(portfolioValue - prevPortfolioValue,2)  AS dailyReturnPrice,
  round((portfolioValue - prevPortfolioValue) / prevPortfolioValue,4) * 100 AS dailyReturnPercentage
  from prevValuation
),
withCumulativeReturnPercentage AS(
  SELECT 
  date,
  portfolioValue,
  SUM(dailyReturnPercentage) OVER (ORDER BY date asc) AS portfolioCRP
  FROM withDailyReturn
),
SPYDaily AS(
  select 
    LAST(close) as close ,
    timestamp
  FROM price_1m where tvId='SPY_AMEX' SAMPLE BY 1d ORDER BY timestamp 
),
WithJoinPortfolio AS (
 SELECT 
 sd.close,
 sd.timestamp,
 wcrp.date,
 wcrp.portfolioValue,
 wcrp.portfolioCRP
 FROM 
 SPYDaily sd 
 JOIN withCumulativeReturnPercentage wcrp ON wcrp.date=sd.timestamp 
),
WithSPYPrevValue AS (
  SELECT 
    close,
    timestamp,
    date,
    portfolioValue,
    portfolioCRP,
    LAG(close) OVER (ORDER BY timestamp ASC) as prevClose
    FROM WithJoinPortfolio
),
withDailyReturnPercentage AS (
  SELECT
  close,
  prevClose,
  coalesce(((close - prevClose) / prevClose),0) *100  as SPYdailyReturnPercentage,
  timestamp,
  date,
  portfolioValue,
  portfolioCRP
  from WithSPYPrevValue
),
withSPYCumulativeReturn As (
  SELECT 
  portfolioValue,
  portfolioCRP,
  SUM(SPYdailyReturnPercentage) OVER(ORDER BY timestamp) AS SPYCRP,
  timestamp as date
  FROM withDailyReturnPercentage
)
select 
*
from withSPYCumulativeReturn;