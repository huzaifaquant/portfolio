TradeVolByYear
SELECT
    YEAR(entryDate) AS year,
    SUM(CASE WHEN side = 'buy' THEN ABS(quantity) * price ELSE 0 END) AS buy,
    SUM(CASE WHEN side = 'sell' THEN ABS(quantity) * price ELSE 0 END) AS sell
    FROM ${omsTransactions}
    WHERE portfolioId = ${portfolioId}
    AND status = 'EXECUTED'
    AND entryDate IS NOT NULL
    GROUP BY YEAR(entryDate)
    ORDER BY year;