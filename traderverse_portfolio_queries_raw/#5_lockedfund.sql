pendingOrders as (
    select 
    SUM(o.quantity * (
        CASE WHEN o.entryPrice IS NULL THEN lp.price ELSE o.entryPrice END
        )) as totalCost, 
    SUM(o.quantity) as quantity, 
    o.tvId, 
    o.side,
    CASE WHEN side = 'buy' THEN 'long' ELSE 'short' END as direction,
    LAST(o.portfolioId) as portfolioId 
    FROM ofd_oms_orders o
    LEFT JOIN latest_prices lp ON lp.tvId=o.tvId
    where o.portfolioId='5deb9038-5bd8-459d-b060-22ac987050ff' and 
    (orderType='limit' OR orderType='market') and 
    status='PENDING' 
    group by side,tvId
),
withPositionsAndLockedFund as (
    select po.*,
    po.totalCost/po.quantity AS entryPrice,
    p.direction as holdingDirection,
    p.quantity as holdingQuantity,
    CASE 
    WHEN p.direction = po.direction THEN totalCost
    WHEN ABS(p.quantity) >= po.quantity THEN 0
    ELSE (po.quantity - ABS(p.quantity)) * (po.totalCost/po.quantity) END as lockedFund
    FROM pendingOrders po
    LEFT JOIN ofd_oms_positions p ON p.tvId=po.tvId AND p.portfolioId=po.portfolioId
),
withGroup AS (
    select SUM(lockedFund) as lockedFund
    FROM withPositionsAndLockedFund
    GROUP BY portfolioId
)
select * from withGroup;