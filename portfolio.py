import pandas as pd
from collections import defaultdict

# ---------- Display (optional) ----------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.max_colwidth', None)

# ---------- Global State ----------

portfolio_state = {
    'cash': 200,                     # constant after reset
    'remaining': 200,                # evolves with trades
    'quantities': defaultdict(int),  # {ticker: quantity}
    'cost_basis': defaultdict(float),# {ticker: absolute cost basis}
    'avg_price': defaultdict(float), # {ticker: average entry price}
    'realized_pnl': 0.0,
    'last_price': {},                # {ticker: last seen price}
    'portfolio_df': None             # accumulator
}

COLUMNS = [
    'Date','Ticker','Buy/Sell','Position','Cash','Buyable/Sellable',
    'Quantity Buy','Remaining','Current Quantity','Price',
    'Avg Price','Cost Basis','Position Value PV',
    'PnL (Long) Unrealized','PnL (Short) Unrealized','Pnl Unrealized','PnL Unrealized Value',
    'PV (Long)','PV (Short)','Open Position','Open PV',
    'Total PV','Total PV + Remaining','PnL Realized'
]

# ---------- Lifecycle ----------

def reset_portfolio(initial_cash=200):
    global portfolio_state
    portfolio_state = {
        'cash': initial_cash,
        'remaining': initial_cash,
        'quantities': defaultdict(int),
        'cost_basis': defaultdict(float),
        'avg_price': defaultdict(float),
        'realized_pnl': 0.0,
        'last_price': {},
        'portfolio_df': pd.DataFrame(columns=COLUMNS)
    }

def get_portfolio_df():
    return portfolio_state['portfolio_df'].copy()

# ---------- Helpers ----------

def normalize_quantity(q):
    """
    Accepts:
      - numeric: 10, -5
      - strings: "-(-10)" -> 10,  "(10)" -> 10, "-10" -> -10, "  -(-5)  " -> 5
    """
    if isinstance(q, (int, float)):
        return float(q)
    s = str(q).strip().replace(' ', '')
    if s.startswith('-(') and s.endswith(')'):
        inner = s[2:-1]
        return -float(inner)
    if s.startswith('(') and s.endswith(')'):
        inner = s[1:-1]
        return float(inner)
    return float(s)

# ---------- Core Single-Trade Calculations ----------

def calculate_cash_single():
    # Cash is constant after initialization
    return portfolio_state['cash']

def calculate_remaining_single(action, position, price, q_in, old_quantity, old_cost_basis):
    """
    Remaining update (prev = prior Remaining, p = current price, q = shares):
    - Buy Long:   prev - p*q
    - Sell Long:  prev + p*q
    - Sell Short: prev - p*q
    - Buy Short:  prev + [ initial + (initial - final) ]
                  where initial = avg * close_qty, final = p * close_qty
    
    Uses previous avg price (from old_cost_basis / old_quantity) for closing calculations.
    """
    rem = portfolio_state['remaining']
    a = str(action).lower()
    qty = abs(q_in) if q_in < 0 else q_in
    if qty == 0 or a == 'hold':
        return rem

    # Previous avg price (0 if no position)
    prev_avg = abs(old_cost_basis / old_quantity) if old_quantity else 0.0

    if a == 'buy':
        if old_quantity < 0:
            # Buy Short (cover up to held short) using prev_avg for initial
            cover = min(qty, abs(old_quantity))
            if cover > 0:
                initial = prev_avg * cover
                final = price * cover
                delta = initial + (initial - final)
                rem += delta
            # Any excess turns into Buy Long at market
            excess = qty - cover
            if excess > 0:
                rem -= price * excess
        else:
            # Buy Long
            rem -= price * qty
        return rem

    if a == 'sell':
        if old_quantity > 0:
            # Sell Long up to held long
            close = min(qty, old_quantity)
            if close > 0:
                rem += price * close
            # Any excess turns into Sell Short at market
            excess = qty - close
            if excess > 0:
                rem -= price * excess
        else:
            # Sell Short (open/increase short)
            rem -= price * qty
        return rem

    return rem

def calculate_current_quantity_single(ticker, action, position, q_in, old_quantity):
    """
    Quantity change depends on action:
    - Buy: increases quantity (moves longward)
    - Sell: decreases quantity (moves shortward)
    """
    a = str(action).lower()
    qty = abs(q_in) if q_in < 0 else q_in

    if a == 'hold':
        new_q = old_quantity
    elif a == 'buy':
        new_q = old_quantity + qty
    elif a == 'sell':
        new_q = old_quantity - qty
    else:
        new_q = old_quantity

    portfolio_state['quantities'][ticker] = new_q
    return new_q

def calculate_avg_price_and_cost_basis_single(ticker, action, position, price, q_in, old_quantity, new_quantity, old_cost_basis):
    """
    Maintain absolute Cost Basis and Avg Price for net position.
    Conventions:
    - Long: Cost Basis = dollars spent on current net long shares
    - Short: Cost Basis = dollars received from current net short shares (short proceeds)
    - Avg Price = abs(Cost Basis / Quantity) when quantity != 0
    - Crossing 0: prior side closes, new side starts fresh
    """
    a = str(action).lower()
    qty = abs(q_in) if q_in < 0 else q_in
    cb = old_cost_basis

    if a == 'hold':
        pass

    elif a == 'buy':
        if old_quantity >= 0:
            # adding/opening long
            cb = cb + qty * price
        else:
            # buying to cover short
            to_cover = min(qty, abs(old_quantity))
            if qty > to_cover:
                # fully cover then open long with residual
                open_long = qty - to_cover
                cb = open_long * price
            else:
                # still short; proportionally reduce short proceeds
                if abs(old_quantity) > 0:
                    remaining_short = abs(old_quantity) - to_cover
                    cb = cb * (remaining_short / abs(old_quantity)) if remaining_short > 0 else 0.0

    elif a == 'sell':
        if old_quantity > 0:
            # selling from long
            sell_qty = qty
            if sell_qty < old_quantity:
                remaining_long = old_quantity - sell_qty
                cb = cb * (remaining_long / old_quantity)
            elif sell_qty == old_quantity:
                cb = 0.0
            else:
                # flipped to short: close long then open short with extra
                open_short = sell_qty - old_quantity
                cb = open_short * price
        else:
            # short selling or increasing short
            cb = cb + qty * price

    if new_quantity != 0:
        avg_price = abs(cb / new_quantity)
    else:
        avg_price = 0.0
        cb = 0.0

    portfolio_state['cost_basis'][ticker] = cb
    portfolio_state['avg_price'][ticker] = avg_price
    return avg_price, cb

def calculate_realized_pnl_single(ticker, action, position, price, q_in, old_quantity):
    """
    Realized PnL on closing legs - cumulative across ALL tickers.
    - Closing long by selling: (sell_price - avg_entry) * shares_closed
    - Covering short by buying: (avg_entry - cover_price) * shares_closed
    
    Uses previous avg price from state (reads directly before updating).
    Long position: calculate realized PnL when action is 'sell'
    Short position: calculate realized PnL when action is 'buy'
    """
    realized = portfolio_state['realized_pnl']
    a = str(action).lower()
    pos = str(position).lower()
    
    # Read old avg price from state (before it gets updated)
    prev_avg_price = portfolio_state['avg_price'][ticker]

    # Long position: calculate realized PnL when selling
    if pos == 'long' and a == 'sell' and old_quantity > 0:
        closed = abs(q_in) if q_in < 0 else q_in
        if closed > 0 and prev_avg_price > 0:
            realized += (price - prev_avg_price) * closed

    # Short position: calculate realized PnL when buying/covering
    if pos == 'short' and a == 'buy' and old_quantity < 0:
        closed = abs(q_in) if q_in < 0 else q_in
        if closed > 0 and prev_avg_price > 0:
            realized += (prev_avg_price - price) * closed

    portfolio_state['realized_pnl'] = realized
    return realized

# ---------- Derived per-trade ----------

def position_value_from_position(position, new_quantity, price):
    pos = str(position).lower()
    if pos == 'short':
        # show positive PV for shorts
        return abs(new_quantity) * price
    # long/hold → normal signed PV
    return new_quantity * price

def pnl_unrealized_components(new_quantity, price, avg_price):
    if new_quantity > 0 and avg_price > 0:
        long_u = (price - avg_price) * new_quantity
    else:
        long_u = 0.0

    if new_quantity < 0 and avg_price > 0:
        short_u = (avg_price - price) * abs(new_quantity)
    else:
        short_u = 0.0

    return long_u, short_u, (long_u + short_u)

def open_positions_str():
    """Open Position: current quantities of all tickers being held"""
    parts = []
    for t, q in portfolio_state['quantities'].items():
        if q != 0:
            parts.append(f"{t} {q}")
    return ", ".join(parts) if parts else "None"

def open_pv_str(current_ticker, current_price, current_position):
    """
    Open PV: PV for each ticker calculated as Cost Basis + Unrealized PnL.
    - Current ticker: use current_price and current_position
    - Other tickers: use last_price and determine position from quantity sign
    PV = Cost Basis + Unrealized PnL for each ticker
    """
    parts = []
    for t, q in portfolio_state['quantities'].items():
        if q != 0:
            # Get cost basis and avg price for this ticker
            cb = portfolio_state['cost_basis'][t]
            avg = portfolio_state['avg_price'][t]
            
            # Determine current price
            if t == current_ticker:
                p = current_price
                pos = str(current_position).lower()
            else:
                p = portfolio_state['last_price'].get(t, 0.0)
                pos = 'short' if q < 0 else 'long'
            
            # Calculate PV = Cost Basis + Unrealized PnL
            if q > 0 and avg > 0:  # Long position
                long_u = (p - avg) * q
                ticker_pv = cb + long_u
            elif q < 0 and avg > 0:  # Short position
                short_u = (avg - p) * abs(q)
                ticker_pv = cb + short_u
            else:
                ticker_pv = 0.0
            
            parts.append(f"{t} {ticker_pv}")
    return ", ".join(parts) if parts else "None"

def calculate_pv_for_current_ticker(current_ticker, current_price, current_position, new_q, avg_p, cb):
    """
    Calculate PV (Long) and PV (Short) ONLY for the current ticker being traded.
    PV = Cost Basis + Unrealized PnL
    """
    pos = str(current_position).lower()
    
    if pos == 'long' and new_q > 0 and avg_p > 0:
        long_u = (current_price - avg_p) * new_q
        pv_long = cb + long_u
        return pv_long, 0.0
    elif pos == 'short' and new_q < 0 and avg_p > 0:
        short_u = (avg_p - current_price) * abs(new_q)
        pv_short = cb + short_u
        return 0.0, pv_short
    else:
        return 0.0, 0.0

def calculate_total_pv_all_tickers(current_ticker, current_price, current_position):
    """
    Calculate Total PV = sum of all long PVs + sum of all short PVs
    across all tickers in the portfolio (cumulative).
    PV = Cost Basis + Unrealized PnL for each position
    """
    total_long_pv = 0.0
    total_short_pv = 0.0
    
    for t, q in portfolio_state['quantities'].items():
        if q != 0:
            # Get cost basis and avg price for this ticker
            cb = portfolio_state['cost_basis'][t]
            avg = portfolio_state['avg_price'][t]
            
            # Determine current price
            if t == current_ticker:
                p = current_price
            else:
                p = portfolio_state['last_price'].get(t, 0.0)
            
            # Calculate unrealized PnL and PV
            if q > 0 and avg > 0:  # Long position
                long_u = (p - avg) * q
                ticker_pv_long = cb + long_u
                total_long_pv += ticker_pv_long
            elif q < 0 and avg > 0:  # Short position
                short_u = (avg - p) * abs(q)
                ticker_pv_short = cb + short_u
                total_short_pv += ticker_pv_short
    
    return total_long_pv + total_short_pv

def open_pnl_unrealized_str(current_ticker, current_price, current_position):
    """
    Pnl Unrealized: Unrealized PnL for each ticker (long + short combined per ticker).
    - Current ticker: use current_price and current_position
    - Other tickers: use last_price and determine position from quantity sign
    Shows each ticker with its total unrealized PnL.
    """
    parts = []
    for t, q in portfolio_state['quantities'].items():
        if q != 0:
            # Get avg price for this ticker
            avg = portfolio_state['avg_price'][t]
            
            # Determine current price
            if t == current_ticker:
                p = current_price
            else:
                p = portfolio_state['last_price'].get(t, 0.0)
            
            # Calculate unrealized PnL for this ticker (long + short combined)
            if q > 0 and avg > 0:  # Long position
                ticker_unrealized = (p - avg) * q
            elif q < 0 and avg > 0:  # Short position
                ticker_unrealized = (avg - p) * abs(q)
            else:
                ticker_unrealized = 0.0
            
            parts.append(f"{t} {ticker_unrealized}")
    
    return ", ".join(parts) if parts else "None"

# ---------- Main entry per trade ----------

def process_trade(ticker, action, position, price, quantity_buy, date=None):
    # normalize quantity (handles "-(-10)" etc.)
    q_in = normalize_quantity(quantity_buy)

    # read old state
    old_q = portfolio_state['quantities'][ticker]
    old_cb = portfolio_state['cost_basis'][ticker]

    # constant cash + new remaining
    cash = calculate_cash_single()
    new_remaining = calculate_remaining_single(action, position, price, q_in, old_q, old_cb)

    # quantity
    new_q = calculate_current_quantity_single(ticker, action, position, q_in, old_q)
    
    # Calculate realized PnL FIRST (uses old avg_price from state before updating)
    realized_pnl = calculate_realized_pnl_single(
        ticker, action, position, price, q_in, old_q
    )
    
    # THEN calculate avg_p and cb (this updates the state)
    avg_p, cb = calculate_avg_price_and_cost_basis_single(
        ticker, action, position, price, q_in, old_q, new_q, old_cb
    )

    # derived
    buyable_sellable = (new_remaining / price) if price > 0 else 0.0
    pv = position_value_from_position(position, new_q, price)
    long_u, short_u, u_total = pnl_unrealized_components(new_q, price, avg_p)
    open_pos = open_positions_str()
    open_pv = open_pv_str(ticker, price, position)
    open_pnl = open_pnl_unrealized_str(ticker, price, position) 

    # PV (Long) and PV (Short) - only for current ticker
    pv_long_current, pv_short_current = calculate_pv_for_current_ticker(ticker, price, position, new_q, avg_p, cb)
    
    # Total PV - cumulative across all tickers
    total_pv = calculate_total_pv_all_tickers(ticker, price, position)
    total_pv_plus_remaining = total_pv + new_remaining

    # result row
    row = {
        'Date': date,
        'Ticker': ticker,
        'Buy/Sell': action,
        'Position': position,
        'Cash': cash,
        'Buyable/Sellable': buyable_sellable,
        'Quantity Buy': q_in,
        'Remaining': new_remaining,
        'Current Quantity': new_q,
        'Price': price,
        'Avg Price': avg_p,
        'Cost Basis': cb,
        'Position Value PV': pv,
        'PnL (Long) Unrealized': long_u,
        'PnL (Short) Unrealized': short_u,
        'Pnl Unrealized': open_pnl,
        'PnL Unrealized Value': u_total,
        'PV (Long)': pv_long_current,
        'PV (Short)': pv_short_current,
        'Open Position': open_pos,
        'Open PV': open_pv,
        'Total PV': total_pv,
        'Total PV + Remaining': total_pv_plus_remaining,
        'PnL Realized': realized_pnl
    }

    # persist state for next trade
    portfolio_state['remaining'] = new_remaining
    portfolio_state['last_price'][ticker] = price

    # append to DF
    df_row = pd.DataFrame([row], columns=COLUMNS)
    portfolio_state['portfolio_df'] = pd.concat(
        [portfolio_state['portfolio_df'], df_row],
        ignore_index=True
    )

    return row

def add_trade(ticker, action, position, price, quantity_buy, date=None):
    process_trade(ticker, action, position, price, quantity_buy, date)
    return get_portfolio_df()