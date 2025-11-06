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
    'quantities': defaultdict(int),  # {ticker: quantity} - positive for long, negative for short
    'cost_basis': defaultdict(float),# {ticker: absolute cost basis}
    'avg_price': defaultdict(float), # {ticker: average entry price}
    'realized_pnl': 0.0,             # Cumulative realized PnL across all tickers
    'last_price': {},                # {ticker: last seen price} - used for PV calculations
    'asset_types': {},               # {ticker: asset_type} - tracks asset type per ticker
    'portfolio_df': None             # accumulator DataFrame
}

# Track trade numbers per ticker
trade_tracker = {}  # {ticker: trade_number} - tracks active trades
next_trade_number = 1  # Global counter for next new trade
investment_count = 0  # Cumulative count of positions opened (long buy or short sell)

COLUMNS = [
    'Date','Ticker', 'Asset Type','Buy/Sell','Position Taken','Current Position','Cash','Buyable/Sellable',
    'Quantity Buy','Remaining','Current Quantity','Price',
    'Avg Price','Cost Basis','Position Value PV',
    'PnL (Long) Unrealized','PnL (Short) Unrealized','Pnl Unrealized','PnL Unrealized Total Value for Current Ticker', 'Total Unrealized PnL',
    'PV (Long)','PV (Short)','Open Position','Open PV',
    'Total PV','Equity: Total PV + Remaining','PnL Realized at Point of Time','PnL Realized Cummulative','Total PnL Overall (Unrealized+Realized)',
    'Daily PnL (Unrealized+Realized)','Liquidation Price','Take Profit','Stop Loss', 
    'Last Day Pnl / Daily $', 'Daily %', 'Cumulative %', 'Investment Count', 'Performance', 'Asset Count',
    'Trade No. (Position - Trade no. - Current Quantity)', 'Total Trades', 'Win/Loss', 'Win Rate', 'Win:Loss Ratio', 
    'Trades/Month', 'Absolute Quantity Counts', 'Most Traded Symbol', 'Least Traded',
    'Avg Losing PnL', 'Avg Winning PnL', 'Most Profitable', 'Least Profitable', 'Max Drawdown',
    'Total Gain', 'Average Gain', 'Biggest Investment', 'Average Position', 'Holdings','YTD PnL',
    'Highest Traded Volume', 'Lowest Traded Volume', 'Average Holding Days',
    'Distribution', 'Distribution in %',
    'Equity Distribution (Market Cap)', 'Equity Distribution (Industry)', 'Equity Distribution (Sector)' 
]

# ---------- Lifecycle ----------

def reset_portfolio(initial_cash=200):
    """
    Reset the portfolio to initial state.
    
    Args:
        initial_cash (float): Initial cash amount. Defaults to 200.
    
    Resets:
        - All portfolio state dictionaries
        - Trade tracking variables
        - Investment count
        - Portfolio DataFrame
    """
    global portfolio_state, trade_tracker, next_trade_number, investment_count
    portfolio_state = {
        'cash': initial_cash,
        'remaining': initial_cash,
        'quantities': defaultdict(int),
        'cost_basis': defaultdict(float),
        'avg_price': defaultdict(float),
        'realized_pnl': 0.0,
        'last_price': {},
        'asset_types': {},
        'market_cap': {},      # For Equity Distribution
        'industry': {},        # For Equity Distribution
        'sector': {},         # For Equity Distribution
        'max_investment_history': defaultdict(float),
        'highest_traded_volume': None,
        'lowest_traded_volume': None,
        'position_open_period': {},
        'cumulative_holding_sum': 0.0,
        'closed_positions_count': 0,
        'current_period': 0,
        'previous_realized_pnl': 0.0,
        'portfolio_df': pd.DataFrame(columns=COLUMNS)
    }
    trade_tracker = {}
    next_trade_number = 1
    investment_count = 0  # Reset investment count

def get_portfolio_df():
    """
    Get a copy of the current portfolio DataFrame.
    
    Returns:
        pd.DataFrame: Copy of the portfolio DataFrame with all trade history.
    """
    return portfolio_state['portfolio_df'].copy()

# ---------- Helpers ----------

def normalize_quantity(q):
    """
    Normalize quantity input to float (always returns absolute value).
    
    The sign of the quantity doesn't matter - the position (long/short) determines direction.
    This function extracts the numeric value and returns its absolute value.
    
    Accepts:
        - numeric: 10, -5 -> returns 10.0, 5.0
        - strings: "-(-10)" -> 10.0,  "(10)" -> 10.0, "-10" -> 10.0, "  -(-5)  " -> 5.0
    
    Args:
        q: Quantity input (int, float, or string)
    
    Returns:
        float: Normalized quantity value (always positive/absolute)
    
    Edge cases:
        - Handles negative parentheses notation: "-(-10)" -> 10.0
        - Handles positive parentheses notation: "(10)" -> 10.0
        - Handles negative values: "-10" -> 10.0
        - Strips whitespace
        - Always returns absolute value regardless of input sign
    """
    if isinstance(q, (int, float)):
        return abs(float(q))
    s = str(q).strip().replace(' ', '')
    if s.startswith('-(') and s.endswith(')'):
        inner = s[2:-1]
        return abs(float(inner))
    if s.startswith('(') and s.endswith(')'):
        inner = s[1:-1]
        return abs(float(inner))
    return abs(float(s))

def get_or_create_trade_number(ticker, old_q, new_q, action):
    """
    Determine trade number for a ticker based on position state.
    
    Trade numbering logic:
        - If old_q == 0 and new_q != 0: Opens new trade (assign new trade number)
        - If old_q != 0 and new_q != 0: Continues existing trade (keep same trade number)
        - If old_q != 0 and new_q == 0: Closes existing trade (keep same trade number, mark as closed)
        - If old_q == 0 and new_q == 0: No trade
        - If action == 'hold': Returns existing trade number if position exists, else None
    
    Args:
        ticker (str): Ticker symbol
        old_q (float): Quantity before this trade
        new_q (float): Quantity after this trade
        action (str): Trade action ('buy', 'sell', 'hold')
    
    Returns:
        int or None: Trade number if trade exists, None otherwise
    
    Edge cases:
        - Handles position flips (long to short, short to long)
        - Handles hold actions (maintains existing trade number)
    """
    global next_trade_number
    ticker = str(ticker).upper()
    
    action_lower = str(action).lower()
    
    # If action is 'hold', check if position exists
    if action_lower == 'hold':
        if ticker in trade_tracker and old_q != 0:
            # Position exists but holding, return existing trade number
            return trade_tracker[ticker]
        else:
            return None
    
    # If opening a new position (old_q == 0, new_q != 0)
    if old_q == 0 and new_q != 0:
        # Assign new trade number
        trade_number = next_trade_number
        trade_tracker[ticker] = trade_number
        next_trade_number += 1
        return trade_number
    
    # If closing a position (old_q != 0, new_q == 0)
    if old_q != 0 and new_q == 0:
        # Return existing trade number, then remove from tracker
        trade_number = trade_tracker.get(ticker)
        if ticker in trade_tracker:
            del trade_tracker[ticker]
        return trade_number
    
    # If maintaining/changing position (old_q != 0, new_q != 0)
    if old_q != 0 and new_q != 0:
        # Continue existing trade
        return trade_tracker.get(ticker)
    
    # No position (old_q == 0, new_q == 0)
    return None

def format_trade_string(action, current_position, trade_number, new_q):
    """
    Format trade string for display.
    
    Format: "Position - Action - #TradeNo Trade - Quantity" or "... - 0 - Close"
    
    Args:
        action (str): Trade action ('buy', 'sell', 'hold')
        current_position (str): Current position ('long', 'short', 'hold')
        trade_number (int or None): Trade number
        new_q (float): New quantity after trade
    
    Returns:
        str: Formatted trade string or "No Buy/Sell" if no trade
    
    Examples:
        - "Long - Buy - #1 Trade - 10"
        - "Long - Sell - #1 Trade - 0 - Close"
        - "Short - Sell - #2 Trade - 5"
    """
    if trade_number is None:
        return "No Buy/Sell"
    
    action_str = str(action)
    pos = str(current_position)
    
    # Format quantity - use absolute value since position already explains it
    if new_q == 0:
        quantity_str = "0 - Close"
    else:
        # Use absolute value for quantity
        quantity_str = str(int(abs(new_q)))
    
    # Include Buy/Sell in the output
    return f" {pos.capitalize()} - {action_str.capitalize()} - #{trade_number} Trade - {quantity_str}"

# ---------- Core Single-Trade Calculations ----------

def calculate_cash_single():
    """
    Get the initial cash amount (constant after initialization).
    
    Formula: Cash = Initial Cash (does not change)
    
    Returns:
        float: Initial cash amount
    """
    return portfolio_state['cash']

def calculate_remaining_single(action, price, q_in, old_quantity, old_cost_basis):
    """
    Calculate remaining cash after a trade.
    
    Remaining cash update formulas:
        - Buy Long:   remaining = previous_remaining - price * quantity
        - Sell Long:  remaining = previous_remaining + price * quantity
        - Sell Short: remaining = previous_remaining - price * quantity
        - Buy Short (Cover): remaining = previous_remaining + [initial + (initial - final)]
                              where initial = avg_price * close_qty, final = price * close_qty
    
    Position flips automatically when quantity crosses zero:
        - Long selling more than owned: closes long, opens short with excess
        - Short buying more than owed: covers short, opens long with excess
    
    Args:
        action (str): Trade action ('buy', 'sell', 'hold')
        position (str): Position type ('long', 'short', 'hold')
        price (float): Trade price
        q_in (float): Quantity traded
        old_quantity (float): Quantity before trade
        old_cost_basis (float): Cost basis before trade
    
    Returns:
        float: Remaining cash after trade
    
    Edge cases:
        - Handles position flips (long to short, short to long)
        - Handles partial closes (closing some shares, keeping rest)
        - Uses previous avg price for closing calculations
    """
    rem = portfolio_state['remaining']
    a = str(action).lower()
    qty = abs(q_in) if q_in < 0 else q_in
    if qty == 0 or a == 'hold':
        return rem

    # Previous avg price (0 if no position)
    # Formula: avg_price = abs(cost_basis / quantity) when quantity != 0
    prev_avg = abs(old_cost_basis / old_quantity) if old_quantity else 0.0

    if a == 'buy':
        if old_quantity < 0:
            # Buy Short (cover up to held short) using prev_avg for initial
            cover = min(qty, abs(old_quantity))
            if cover > 0:
                initial = prev_avg * cover
                final = price * cover
                # Formula: delta = initial + (initial - final)
                # This accounts for the profit/loss from covering the short
                delta = initial + (initial - final)
                rem += delta
            # Any excess turns into Buy Long at market
            excess = qty - cover
            if excess > 0:
                rem -= price * excess
        else:
            # Buy Long: remaining decreases by price * quantity
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
            # Sell Short (open/increase short): remaining decreases
            rem -= price * qty
        return rem

    return rem

def calculate_current_quantity_single(ticker, action, q_in, old_quantity):
    """
    Calculate new quantity after trade.
    
    Quantity change formulas:
        - Buy: new_q = old_q + quantity (moves longward)
        - Sell: new_q = old_q - quantity (moves shortward)
        - Hold: new_q = old_q (no change)
    
    Position flips automatically when quantity crosses zero:
        - Long selling more than owned → flips to short
        - Short buying more than owed → flips to long
    
    Args:
        ticker (str): Ticker symbol
        action (str): Trade action ('buy', 'sell', 'hold')
        position (str): Position type ('long', 'short', 'hold')
        q_in (float): Quantity traded
        old_quantity (float): Quantity before trade
    
    Returns:
        float: New quantity after trade (positive for long, negative for short)
    
    Edge cases:
        - Handles position flips naturally through arithmetic
        - Updates global quantities dictionary
    """
    a = str(action).lower()
    qty = abs(q_in) if q_in < 0 else q_in

    if a == 'hold':
        new_q = old_quantity
    elif a == 'buy':
        new_q = old_quantity + qty  # Naturally flips short→long if buying excess
    elif a == 'sell':
        new_q = old_quantity - qty  # Naturally flips long→short if selling excess
    else:
        new_q = old_quantity

    portfolio_state['quantities'][ticker] = new_q
    return new_q

def calculate_avg_price_and_cost_basis_single(ticker, action, price, q_in, old_quantity, new_quantity, old_cost_basis):
    """
    Calculate average price and cost basis for net position.
    
    Cost Basis formulas:
        - Long: Cost Basis = dollars spent on current net long shares
        - Short: Cost Basis = dollars received from current net short shares (short proceeds)
    
    Average Price formula:
        - avg_price = abs(Cost Basis / Quantity) when quantity != 0
        - When quantity == 0: avg_price = 0, cost_basis = 0
    
    Special cases:
        - Crossing 0: prior side closes, new side starts fresh
        - Partial closes: proportionally reduce cost basis
        - Position flips: close old side, start new side with fresh cost basis
    
    Args:
        ticker (str): Ticker symbol
        action (str): Trade action ('buy', 'sell', 'hold')
        position (str): Position type ('long', 'short', 'hold')
        price (float): Trade price
        q_in (float): Quantity traded
        old_quantity (float): Quantity before trade
        new_quantity (float): Quantity after trade
        old_cost_basis (float): Cost basis before trade
    
    Returns:
        tuple: (avg_price, cost_basis) - Average price and cost basis after trade
    
    Edge cases:
        - Handles position flips (long to short, short to long)
        - Handles partial position closes
        - Resets cost basis when crossing zero
    """
    a = str(action).lower()
    qty = abs(q_in) if q_in < 0 else q_in
    cb = old_cost_basis

    if a == 'hold':
        pass

    elif a == 'buy':
        if old_quantity >= 0:
            # Adding/opening long: cost basis increases
            # Formula: new_cost_basis = old_cost_basis + quantity * price
            cb = cb + qty * price
        else:
            # Buying to cover short
            to_cover = min(qty, abs(old_quantity))
            if qty > to_cover:
                # Fully cover then open long with residual
                # New long position starts fresh
                open_long = qty - to_cover
                cb = open_long * price
            else:
                # Still short; proportionally reduce short proceeds
                # Formula: remaining_cost_basis = old_cost_basis * (remaining_short / old_short)
                if abs(old_quantity) > 0:
                    remaining_short = abs(old_quantity) - to_cover
                    cb = cb * (remaining_short / abs(old_quantity)) if remaining_short > 0 else 0.0

    elif a == 'sell':
        if old_quantity > 0:
            # Selling from long
            sell_qty = qty
            if sell_qty < old_quantity:
                # Partial close: proportionally reduce cost basis
                # Formula: remaining_cost_basis = old_cost_basis * (remaining_long / old_long)
                remaining_long = old_quantity - sell_qty
                cb = cb * (remaining_long / old_quantity)
            elif sell_qty == old_quantity:
                # Full close: cost basis becomes zero
                cb = 0.0
            else:
                # Flipped to short: close long then open short with extra
                # New short position starts fresh
                open_short = sell_qty - old_quantity
                cb = open_short * price
        else:
            # Short selling or increasing short: cost basis increases
            # Formula: new_cost_basis = old_cost_basis + quantity * price
            cb = cb + qty * price

    # Calculate average price
    # Formula: avg_price = abs(cost_basis / quantity) when quantity != 0
    if new_quantity != 0:
        avg_price = abs(cb / new_quantity)
    else:
        avg_price = 0.0
        cb = 0.0

    portfolio_state['cost_basis'][ticker] = cb
    portfolio_state['avg_price'][ticker] = avg_price
    return avg_price, cb

def calculate_realized_pnl_at_point_of_time(ticker, action, position, price, q_in, old_quantity):
    """
    Calculate realized PnL at point of time for a specific closing action.
    
    Formula for Long positions (when selling):
        realized_pnl = (sell_price - avg_entry_price) * shares_closed
    
    Formula for Short positions (when buying/covering):
        realized_pnl = (avg_entry_price - cover_price) * shares_closed
    
    This is independent and dynamic, not dependent on cumulative calculations.
    Returns PnL for this specific closing action only.
    
    Args:
        ticker (str): Ticker symbol
        action (str): Trade action ('buy', 'sell', 'hold')
        position (str): Position type ('long', 'short', 'hold')
        price (float): Trade price
        q_in (float): Quantity traded
        old_quantity (float): Quantity before trade
    
    Returns:
        float or None: Realized PnL for this trade, None if no closing occurs
    
    Edge cases:
        - Only calculates when closing positions (sell long or buy short)
        - Only realizes on shares actually closed (min of qty and owned/owed)
        - Returns None if no closing occurs or avg_price is 0
    """
    a = str(action).lower()
    pos = str(position).lower()
    
    # Read old avg price from state (before it gets updated)
    prev_avg_price = portfolio_state['avg_price'][ticker]
    
    realized_pnl_point = None

    # Long position: calculate realized PnL when selling
    if pos == 'long' and a == 'sell' and old_quantity > 0:
        qty = abs(q_in) if q_in < 0 else q_in
        # Only realize on shares actually closed (min of qty and owned)
        closed = min(qty, old_quantity)
        if closed > 0 and prev_avg_price > 0:
            # Formula: (price - avg_entry) * closed
            realized_pnl_point = (price - prev_avg_price) * closed

    # Short position: calculate realized PnL when buying/covering
    elif pos == 'short' and a == 'buy' and old_quantity < 0:
        qty = abs(q_in) if q_in < 0 else q_in
        # Only realize on shares actually covered (min of qty and owed)
        closed = min(qty, abs(old_quantity))
        if closed > 0 and prev_avg_price > 0:
            # Formula: (avg_entry - price) * closed
            realized_pnl_point = (prev_avg_price - price) * closed

    return realized_pnl_point

def calculate_realized_pnl_cumulative(ticker, action, position, price, q_in, old_quantity):
    """
    Calculate cumulative realized PnL across ALL tickers.
    
    Formulas:
        - Closing long by selling: (sell_price - avg_entry) * shares_closed
        - Covering short by buying: (avg_entry - cover_price) * shares_closed
    
    Uses previous avg price from state (reads directly before updating).
    Updates global portfolio_state['realized_pnl'].
    
    Args:
        ticker (str): Ticker symbol
        action (str): Trade action ('buy', 'sell', 'hold')
        position (str): Position type ('long', 'short', 'hold')
        price (float): Trade price
        q_in (float): Quantity traded
        old_quantity (float): Quantity before trade
    
    Returns:
        float: Cumulative realized PnL after this trade
    
    Edge cases:
        - When position flips (selling more than owned or buying more than owed),
          only the closed portion generates realized PnL, not the excess
        - Only realizes when closing positions (sell long or buy short)
    """
    realized = portfolio_state['realized_pnl']
    a = str(action).lower()
    pos = str(position).lower()
    
    # Read old avg price from state (before it gets updated)
    prev_avg_price = portfolio_state['avg_price'][ticker]

    # Long position: calculate realized PnL when selling
    if pos == 'long' and a == 'sell' and old_quantity > 0:
        qty = abs(q_in) if q_in < 0 else q_in
        # Only realize on shares actually closed (min of qty and owned)
        # If selling more than owned, only the owned shares generate realized PnL
        closed = min(qty, old_quantity)
        if closed > 0 and prev_avg_price > 0:
            # Formula: (price - avg_entry) * closed
            realized += (price - prev_avg_price) * closed

    # Short position: calculate realized PnL when buying/covering
    if pos == 'short' and a == 'buy' and old_quantity < 0:
        qty = abs(q_in) if q_in < 0 else q_in
        # Only realize on shares actually covered (min of qty and owed)
        # If buying more than owed, only the owed shares generate realized PnL
        closed = min(qty, abs(old_quantity))
        if closed > 0 and prev_avg_price > 0:
            # Formula: (avg_entry - price) * closed
            realized += (prev_avg_price - price) * closed

    portfolio_state['realized_pnl'] = realized
    return realized

# ---------- Derived per-trade ----------

def position_value_from_position(position, new_quantity, price):
    """
    Calculate Position Value PV from position type and quantity.
    
    Formula:
        - Long position: PV = quantity * price
        - Short position: PV = abs(quantity) * price (shown as positive for display)
        - Hold position: PV = 0 * price = 0
    
    Args:
        position (str): Position type ('long', 'short', 'hold')
        new_quantity (float): Quantity after trade (positive for long, negative for short)
        price (float): Current price
    
    Returns:
        float: Position Value PV
    
    Edge cases:
        - Short positions return positive PV for display purposes
        - Zero quantity returns 0
    """
    pos = str(position).lower()
    if pos == 'short':
        # Show positive PV for shorts (uses absolute value)
        return abs(new_quantity) * price
    # long/hold → normal signed PV
    return new_quantity * price

# Update pnl_unrealized_components function (around line 600):

def pnl_unrealized_components(new_quantity, price, avg_price, current_ticker, current_price):
    """
    Calculate unrealized PnL components for long and short positions.
    
    Formulas:
        - Long Unrealized PnL: (current_price - avg_entry_price) * quantity (when quantity > 0)
        - Short Unrealized PnL: (avg_entry_price - current_price) * abs(quantity) (when quantity < 0)
        - Total Unrealized PnL for current ticker: long_unrealized + short_unrealized
        - Total Unrealized PnL for all tickers: sum of unrealized PnL across all open positions
    
    Args:
        new_quantity (float): Current quantity (positive for long, negative for short)
        price (float): Current price
        avg_price (float): Average entry price
        current_ticker (str): Ticker being traded in current row
        current_price (float): Current price for current ticker
    
    Returns:
        tuple: (long_unrealized, short_unrealized, total_unrealized_current_ticker, total_unrealized_all_tickers)
    
    Edge cases:
        - Returns 0.0 for unrealized PnL if avg_price <= 0 or quantity == 0
        - Only calculates long unrealized when quantity > 0
        - Only calculates short unrealized when quantity < 0
    """
    # Long position unrealized PnL for current ticker
    # Formula: (price - avg_price) * quantity
    if new_quantity > 0 and avg_price > 0:
        long_u = (price - avg_price) * new_quantity
    else:
        long_u = 0.0

    # Short position unrealized PnL for current ticker
    # Formula: (avg_price - price) * abs(quantity)
    if new_quantity < 0 and avg_price > 0:
        short_u = (avg_price - price) * abs(new_quantity)
    else:
        short_u = 0.0

    # Total unrealized PnL for current ticker
    total_unrealized_current_ticker = long_u + short_u
    
    # Calculate total unrealized PnL across ALL tickers
    total_unrealized_all_tickers = 0.0
    for t, q in portfolio_state['quantities'].items():
        if q != 0:
            avg = portfolio_state['avg_price'][t]
            
            # Determine current price for this ticker
            if t == current_ticker:
                p = current_price
            else:
                p = portfolio_state['last_price'].get(t, 0.0)
            
            # Calculate unrealized PnL for this ticker
            if q > 0 and avg > 0:
                # Long: (price - avg_price) * quantity
                ticker_unrealized = (p - avg) * q
            elif q < 0 and avg > 0:
                # Short: (avg_price - price) * abs(quantity)
                ticker_unrealized = (avg - p) * abs(q)
            else:
                ticker_unrealized = 0.0
            
            total_unrealized_all_tickers += ticker_unrealized

    return long_u, short_u, total_unrealized_current_ticker, total_unrealized_all_tickers

def open_positions_str():
    """
    Generate string of all open positions.
    
    Format: "TICKER1 quantity1, TICKER2 quantity2, ..."
    
    Returns:
        str: Comma-separated list of tickers with quantities, or "None" if no positions
    
    Edge cases:
        - Only includes tickers where quantity != 0
        - Returns "None" if no open positions
    """
    parts = []
    for t, q in portfolio_state['quantities'].items():
        if q != 0:
            ticker_upper = str(t).upper()
            parts.append(f"{ticker_upper} {q}")
    return ", ".join(parts) if parts else "None"

def open_pv_str(current_ticker, current_price, current_position):
    """
    Generate string of all open position values (PV) for each ticker.
    
    Formula for each ticker: PV = Cost Basis + Unrealized PnL
        - Long: PV = cost_basis + (current_price - avg_price) * quantity
        - Short: PV = cost_basis + (avg_price - current_price) * abs(quantity)
    
    Format: "TICKER1 PV1, TICKER2 PV2, ..."
    
    Args:
        current_ticker (str): Ticker being traded in current row
        current_price (float): Current price for current ticker
        current_position (str): Current position type
    
    Returns:
        str: Comma-separated list of tickers with PV values, or "None" if no positions
    
    Edge cases:
        - Uses current_price for current ticker, last_price for others
        - Returns 0.0 PV if avg_price <= 0
    """
    parts = []
    for t, q in portfolio_state['quantities'].items():
        if q != 0:
            cb = portfolio_state['cost_basis'][t]
            avg = portfolio_state['avg_price'][t]
            
            # Determine current price for this ticker
            if t == current_ticker:
                p = current_price
            else:
                p = portfolio_state['last_price'].get(t, 0.0)
            
            # Calculate PV = Cost Basis + Unrealized PnL
            if q > 0 and avg > 0:
                # Long position: PV = cost_basis + (price - avg) * quantity
                long_u = (p - avg) * q
                ticker_pv = cb + long_u
            elif q < 0 and avg > 0:
                # Short position: PV = cost_basis + (avg - price) * abs(quantity)
                short_u = (avg - p) * abs(q)
                ticker_pv = cb + short_u
            else:
                ticker_pv = 0.0
            
            ticker_upper = str(t).upper()
            parts.append(f"{ticker_upper} {ticker_pv}")
    return ", ".join(parts) if parts else "None"

def calculate_pv_for_current_ticker(current_price, current_position, new_q, avg_p, cb):
    """
    Calculate PV (Long) and PV (Short) ONLY for the current ticker being traded.
    
    Formula: PV = Cost Basis + Unrealized PnL
        - Long PV: cost_basis + (current_price - avg_price) * quantity
        - Short PV: cost_basis + (avg_price - current_price) * abs(quantity)
    
    Args:
        current_ticker (str): Ticker being traded
        current_price (float): Current price
        current_position (str): Current position type ('long', 'short', 'hold')
        new_q (float): New quantity after trade
        avg_p (float): Average entry price
        cb (float): Cost basis
    
    Returns:
        tuple: (pv_long, pv_short) - PV for long and short positions (only one is non-zero)
    
    Edge cases:
        - Returns (0.0, 0.0) if position is 'hold' or avg_price <= 0
        - Only calculates long PV when position is 'long' and quantity > 0
        - Only calculates short PV when position is 'short' and quantity < 0
    """
    pos = str(current_position).lower()
    
    if pos == 'long' and new_q > 0 and avg_p > 0:
        # Long position: PV = cost_basis + (price - avg_price) * quantity
        long_u = (current_price - avg_p) * new_q
        pv_long = cb + long_u
        return pv_long, 0.0
    elif pos == 'short' and new_q < 0 and avg_p > 0:
        # Short position: PV = cost_basis + (avg_price - price) * abs(quantity)
        short_u = (avg_p - current_price) * abs(new_q)
        pv_short = cb + short_u
        return 0.0, pv_short
    else:
        return 0.0, 0.0

def calculate_total_pv_all_tickers(current_ticker, current_price):
    """
    Calculate Total PV = sum of all long PVs + sum of all short PVs across all tickers.
    
    Also returns a dictionary of ticker -> PV for reuse in other functions.
    
    Formula: Total PV = Σ(PV for each ticker) where PV = Cost Basis + Unrealized PnL
    
    For each ticker:
        - Long: PV = cost_basis + (price - avg_price) * quantity
        - Short: PV = cost_basis + (avg_price - price) * abs(quantity)
    
    Args:
        current_ticker (str): Ticker being traded in current row
        current_price (float): Current price for current ticker
    
    Returns:
        tuple: (total_pv, ticker_pv_dict) where:
            - total_pv (float): Total PV across all tickers
            - ticker_pv_dict (dict): {ticker: pv} for each ticker with position
    """
    total_long_pv = 0.0
    total_short_pv = 0.0
    ticker_pv_dict = {}  # Store PV for each ticker
    
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
                # Formula: PV = cost_basis + (price - avg_price) * quantity
                long_u = (p - avg) * q
                ticker_pv_long = cb + long_u
                total_long_pv += ticker_pv_long
                ticker_pv_dict[t] = ticker_pv_long
            elif q < 0 and avg > 0:  # Short position
                # Formula: PV = cost_basis + (avg_price - price) * abs(quantity)
                short_u = (avg - p) * abs(q)
                ticker_pv_short = cb + short_u
                total_short_pv += ticker_pv_short
                ticker_pv_dict[t] = ticker_pv_short
            else:
                ticker_pv_dict[t] = 0.0
    
    total_pv = total_long_pv + total_short_pv
    return total_pv, ticker_pv_dict

def open_pnl_unrealized_str(current_ticker, current_price):
    """
    Generate string of unrealized PnL for all open positions.
    
    Formula for each ticker:
        - Long: unrealized = (current_price - avg_price) * quantity
        - Short: unrealized = (avg_price - current_price) * abs(quantity)
    
    Format: "TICKER1 unrealized1, TICKER2 unrealized2, ..."
    
    Args:
        current_ticker (str): Ticker being traded in current row
        current_price (float): Current price for current ticker
        current_position (str): Current position type
    
    Returns:
        str: Comma-separated list of tickers with unrealized PnL, or "None" if no positions
    
    Edge cases:
        - Uses current_price for current ticker, last_price for others
        - Returns 0.0 if avg_price <= 0 or quantity == 0
    """
    parts = []
    for t, q in portfolio_state['quantities'].items():
        if q != 0:
            avg = portfolio_state['avg_price'][t]
            
            if t == current_ticker:
                p = current_price
            else:
                p = portfolio_state['last_price'].get(t, 0.0)
            
            # Calculate unrealized PnL
            if q > 0 and avg > 0:
                # Long: (price - avg_price) * quantity
                ticker_unrealized = (p - avg) * q
            elif q < 0 and avg > 0:
                # Short: (avg_price - price) * abs(quantity)
                ticker_unrealized = (avg - p) * abs(q)
            else:
                ticker_unrealized = 0.0
            
            ticker_upper = str(t).upper()
            parts.append(f"{ticker_upper} {ticker_unrealized}")
    
    return ", ".join(parts) if parts else "None"

def calculate_liquidation_price(current_position, new_q, avg_p):
    """
    Calculate liquidation price based on current position at that point in time.
    
    Formulas:
        - Long position: liquidation_price = 0 (if stock price goes to 0, position is liquidated)
        - Short position: liquidation_price = 2 * avg_price (if price reaches 2x entry, 100% loss = liquidated)
            Example: Short at 100, if price goes to 200 = 100% loss = liquidated
    
    Args:
        current_position (str): Current position type ('long', 'short', 'hold')
        new_q (float): New quantity after trade (quantity must be != 0)
        avg_p (float): Average entry price
    
    Returns:
        float or None: Liquidation price, or None if no position (quantity = 0)
    
    Edge cases:
        - Returns None if quantity == 0 (no position to liquidate)
        - Returns None if avg_price <= 0 (invalid entry price)
        - Returns None if position is 'hold'
    """
    pos = str(current_position).lower()
    
    # Check that we actually have a position (quantity != 0)
    if new_q == 0:
        return None
    
    if pos == 'long' and avg_p > 0:
        # Long position: liquidation at price = 0
        return 0.0
    elif pos == 'short' and avg_p > 0:
        # Short position: liquidation at price = 2 * avg_price
        # (e.g., short at 100, if price goes to 200 = 100% loss = liquidated)
        return 2.0 * avg_p
    else:
        # Hold position or invalid avg_p
        return None

def calculate_take_profit(current_position, new_q, avg_p, take_profit_pct=0.20):
    """
    Calculate Take Profit price based on current position.
    
    Formulas:
        - Long position: take_profit = avg_price * (1 + take_profit_pct)
        - Short position: take_profit = avg_price * (1 - take_profit_pct)
    
    Default take_profit_pct = 20% (0.20)
    
    Args:
        current_position (str): Current position type ('long', 'short', 'hold')
        new_q (float): New quantity after trade
        avg_p (float): Average entry price
        take_profit_pct (float): Take profit percentage (default 0.20 = 20%)
    
    Returns:
        float or None: Take profit price, or None if no position or invalid avg_price
    
    Edge cases:
        - Returns None if quantity == 0 or avg_price <= 0
        - Returns None if position is 'hold'
    """
    if new_q == 0 or avg_p <= 0:
        return None
    
    pos = str(current_position).lower()
    
    if pos == 'long':
        # Long: Take Profit = Avg Price * (1 + Percentage)
        # Example: Entry at 100, 20% TP = 100 * 1.20 = 120
        return avg_p * (1 + take_profit_pct)
    elif pos == 'short':
        # Short: Take Profit = Avg Price * (1 - Percentage)
        # Example: Entry at 100, 20% TP = 100 * 0.80 = 80
        return avg_p * (1 - take_profit_pct)
    else:
        return None

def calculate_stop_loss(current_position, new_q, avg_p, stop_loss_pct=0.10):
    """
    Calculate Stop Loss price based on current position.
    
    Formulas:
        - Long position: stop_loss = avg_price * (1 - stop_loss_pct)
            Unrealized loss = -cost_basis * percentage when hit
        - Short position: stop_loss = avg_price * (1 + stop_loss_pct)
            Vice versa of Take Profit
    
    Default stop_loss_pct = 10% (0.10)
    
    Args:
        current_position (str): Current position type ('long', 'short', 'hold')
        new_q (float): New quantity after trade
        avg_p (float): Average entry price
        stop_loss_pct (float): Stop loss percentage (default 0.10 = 10%)
    
    Returns:
        float or None: Stop loss price, or None if no position or invalid avg_price
    
    Edge cases:
        - Returns None if quantity == 0 or avg_price <= 0
        - Returns None if position is 'hold'
    """
    if new_q == 0 or avg_p <= 0:
        return None
    
    pos = str(current_position).lower()
    
    if pos == 'long':
        # Long: Stop Loss = Avg Price * (1 - Percentage)
        # Example: Entry at 100, 10% SL = 100 * 0.90 = 90
        return avg_p * (1 - stop_loss_pct)
    elif pos == 'short':
        # Short: Stop Loss = Avg Price * (1 + Percentage) [vice versa of TP]
        # Example: Entry at 100, 10% SL = 100 * 1.10 = 110
        return avg_p * (1 + stop_loss_pct)
    else:
        return None

def calculate_trade_win_loss(trade_string, realized_pnl_at_point):
    """
    Determine if a closed trade is a win or loss.
    
    Logic:
        - Win: realized PnL at point of time > 0
        - Loss: realized PnL at point of time <= 0
        - Only applies when trade closes (trade_string contains "- Close")
    
    Args:
        trade_string (str): Formatted trade string (e.g., "Long - Sell - #1 Trade - 0 - Close")
        realized_pnl_at_point (float or None): Realized PnL for this specific closing trade
    
    Returns:
        str or None: "Win" if profitable, "Loss" if not profitable, None if trade not closed
    
    Edge cases:
        - Returns None if trade_string is "No Buy/Sell" or doesn't contain "- Close"
        - Returns None if realized_pnl_at_point is None
        - Uses the independent point-of-time realized PnL calculation
    """
    # Check if trade is closing based on trade string format
    if trade_string == "No Buy/Sell" or "- Close" not in trade_string:
        return None
    
    # Trade is closing, check the realized PnL at point of time
    # Win if PnL > 0, Loss if PnL <= 0
    if realized_pnl_at_point > 0:
        return "Win"
    elif realized_pnl_at_point <= 0:
        return "Loss"
    else:
        return None

def calculate_win_rate(previous_df, current_win_loss):
    """
    Calculate win rate at this point in time.
    
    Formula: Win Rate = (Win values / Total non-None values) * 100
    
    Counts all previous Win/Loss values plus current one.
    
    Args:
        previous_df (pd.DataFrame): Previous rows of portfolio DataFrame
        current_win_loss (str or None): Current row's Win/Loss value
    
    Returns:
        float or None: Win rate percentage, or None if no closed trades
    
    Edge cases:
        - Returns None if no non-None Win/Loss values exist
        - Only counts closed trades (non-None values)
    """
    # Get all previous Win/Loss values
    if len(previous_df) > 0:
        previous_win_loss = previous_df['Win/Loss'].tolist()
        # Add current win_loss to the list
        all_win_loss = previous_win_loss + [current_win_loss]
    else:
        all_win_loss = [current_win_loss]
    
    # Count non-None values (only closed trades have Win/Loss)
    non_none_values = [v for v in all_win_loss if v is not None]
    total_non_none = len(non_none_values)
    
    # Count Win values
    win_count = sum(1 for v in non_none_values if v == "Win")
    
    # Calculate win rate
    if total_non_none > 0:
        # Formula: (wins / total_closed_trades) * 100
        win_rate = (win_count / total_non_none) * 100
        return win_rate
    else:
        return None

def calculate_win_loss_ratio(previous_df, current_win_loss):
    """
    Calculate win:loss ratio at this point in time.
    
    Formula: Win:Loss Ratio = "win_count:loss_count"
    
    Examples: "1:0", "2:1", "0:1", "3:2"
    
    Args:
        previous_df (pd.DataFrame): Previous rows of portfolio DataFrame
        current_win_loss (str or None): Current row's Win/Loss value
    
    Returns:
        str: Win:loss ratio in "win_count:loss_count" format
    
    Edge cases:
        - Returns "0:0" if no closed trades
        - Only counts non-None Win/Loss values (closed trades)
    """
    # Get all previous Win/Loss values
    if len(previous_df) > 0:
        previous_win_loss = previous_df['Win/Loss'].tolist()
        # Add current win_loss to the list
        all_win_loss = previous_win_loss + [current_win_loss]
    else:
        all_win_loss = [current_win_loss]
    
    # Count non-None values (only closed trades)
    non_none_values = [v for v in all_win_loss if v is not None]
    
    # Count Win and Loss values
    win_count = sum(1 for v in non_none_values if v == "Win")
    loss_count = sum(1 for v in non_none_values if v == "Loss")
    
    # Return ratio in "win_count:loss_count" format
    return f"{win_count}:{loss_count}"

def calculate_trades_per_month(previous_df, current_date, current_trade_string):
    """
    Calculate number of open trades in the current month.
    
    Logic:
        - A trade counts in a month if it's open and continues counting in subsequent months until it closes
        - Tracks open trades by extracting trade numbers from trade strings
        - Removes trade from count when trade closes (contains "- close")
    
    Returns format: "count (Month Name)" (e.g., "3 (October)")
    
    Args:
        previous_df (pd.DataFrame): Previous rows of portfolio DataFrame
        current_date (str or datetime): Current trade date
        current_trade_string (str): Current row's trade string
    
    Returns:
        str or None: Count and month name, or None if date parsing fails
    
    Edge cases:
        - Returns None if current_date is None
        - Handles multiple date formats
        - Only counts trades in the current month
        - Tracks trades that span multiple months
    """
    from datetime import datetime
    
    if current_date is None:
        return None
    
    # Parse current date and get month name
    try:
        if isinstance(current_date, str):
            # Try common date formats
            for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y']:
                try:
                    current_dt = datetime.strptime(current_date, fmt)
                    break
                except:
                    continue
            else:
                return None
        else:
            current_dt = pd.to_datetime(current_date)
        
        month_name = current_dt.strftime('%B')  # Full month name (e.g., "October")
        current_month_year = (current_dt.year, current_dt.month)
    except:
        return None
    
    # Get all rows including current
    all_data = []
    if len(previous_df) > 0:
        for _, row in previous_df.iterrows():
            all_data.append({
                'Date': row.get('Date'),
                'Trade String': row.get('Trade No. (Position - Trade no. - Current Quantity)', '')
            })
    all_data.append({
        'Date': current_date,
        'Trade String': current_trade_string
    })
    
    # Find open trades in current month
    open_trade_nums = set()
    
    for row_data in all_data:
        row_date = row_data['Date']
        trade_str = row_data['Trade String']
        
        if row_date is None or trade_str == "No Buy/Sell":
            continue
        
        # Check if row is in current month
        try:
            if isinstance(row_date, str):
                for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y']:
                    try:
                        row_dt = datetime.strptime(row_date, fmt)
                        break
                    except:
                        continue
                else:
                    continue
            else:
                row_dt = pd.to_datetime(row_date)
            
            row_month_year = (row_dt.year, row_dt.month)
            if row_month_year != current_month_year:
                continue
            
            # Extract trade number from trade string (e.g., "#1" from "Long - Buy - #1 Trade - 10")
            import re
            match = re.search(r'#(\d+)', str(trade_str))
            if match:
                trade_num = int(match.group(1))
                # Check if trade is closed
                if "- close" not in str(trade_str):
                    # Trade is open, add to set
                    open_trade_nums.add(trade_num)
                else:
                    # Trade is closed, remove from set
                    open_trade_nums.discard(trade_num)
        except:
            continue
    
    count = len(open_trade_nums)
    return f"{count} ({month_name})"

def calculate_most_least_traded(previous_df, current_ticker, current_qty_buy):
    """
    Calculate most and least traded symbols based on cumulative absolute quantity counts.
    
    Formula: For each ticker, sum absolute values from "Quantity Buy" column
        cumulative_quantity[ticker] = Σ|quantity_buy| for all trades
    
    Returns:
        - Absolute Quantity Counts: "TICKER1 quantity1, TICKER2 quantity2, ..." (ordered descending)
        - Most Traded Symbol: All tickers ordered by quantity descending
        - Least Traded: All tickers ordered by quantity ascending
    
    Args:
        previous_df (pd.DataFrame): Previous rows of portfolio DataFrame
        current_ticker (str): Current ticker being traded
        current_qty_buy (float): Current quantity buy value
    
    Returns:
        tuple: (absolute_quantity_counts_str, most_traded, least_traded)
    
    Edge cases:
        - Returns ("None", "None", "None") if no trades
        - Handles invalid quantity values gracefully
        - Sorts by quantity first, then alphabetically for ties
    """
    from collections import defaultdict
    
    # Track cumulative absolute quantities for each ticker from "Quantity Buy" column
    ticker_quantities = defaultdict(float)
    
    # Process previous rows - sum absolute quantities from "Quantity Buy" column
    if len(previous_df) > 0:
        for _, row in previous_df.iterrows():
            row_ticker = str(row.get('Ticker', '')).upper()
            row_qty_buy = row.get('Quantity Buy', 0)
            if row_ticker and row_qty_buy != 0:
                try:
                    qty = float(row_qty_buy)
                    # Sum absolute quantities (treats buys and sells the same)
                    ticker_quantities[row_ticker] += abs(qty)
                except:
                    pass
    
    # Add current quantity buy
    if current_ticker and current_qty_buy != 0:
        try:
            ticker_quantities[current_ticker] += abs(float(current_qty_buy))
        except:
            pass
    
    if not ticker_quantities:
        return None, "None", "None"
    
    # Sort tickers by quantity (descending), then alphabetically for ties
    sorted_tickers_desc = sorted(ticker_quantities.items(), key=lambda x: (-x[1], x[0]))
    
    # Calculate absolute quantity counts string (ordered by quantity descending)
    abs_counts = [f"{ticker} {int(qty)}" for ticker, qty in sorted_tickers_desc]
    abs_counts_str = ", ".join(abs_counts) if abs_counts else "None"
    
    # Most Traded: All tickers ordered by quantity descending
    most_traded_list = [ticker for ticker, qty in sorted_tickers_desc]
    most_traded_str = ", ".join(most_traded_list) if most_traded_list else "None"
    
    # Least Traded: All tickers ordered by quantity ascending
    sorted_tickers_asc = sorted(ticker_quantities.items(), key=lambda x: (x[1], x[0]))
    least_traded_list = [ticker for ticker, qty in sorted_tickers_asc]
    least_traded_str = ", ".join(least_traded_list) if least_traded_list else "None"
    
    return abs_counts_str, most_traded_str, least_traded_str

def calculate_avg_losing_winning_pnl(previous_df, current_realized_pnl_at_point):
    """
    Calculate average losing and winning PnL from realized PnL at point of time.
    
    Formulas:
        - Avg Losing PnL = Average of all realized PnL at point of time where PnL < 0
        - Avg Winning PnL = Average of all realized PnL at point of time where PnL > 0
    
    Formula: avg = sum(pnl_values) / count(pnl_values)
    
    Args:
        previous_df (pd.DataFrame): Previous rows of portfolio DataFrame
        current_realized_pnl_at_point (float or None): Current row's realized PnL at point of time
    
    Returns:
        tuple: (avg_losing_pnl, avg_winning_pnl)
    
    Edge cases:
        - Returns 0.0 if no losing/winning trades exist
        - Filters out None values
        - Only considers realized PnL at point of time (not cumulative)
    """
    # Get all previous realized PnL at point of time values
    if len(previous_df) > 0:
        previous_pnl_values = previous_df['PnL Realized at Point of Time'].tolist()
        # Add current value to the list
        all_pnl_values = previous_pnl_values + [current_realized_pnl_at_point]
    else:
        all_pnl_values = [current_realized_pnl_at_point]
    
    # Filter for losing PnL (< 0) and winning PnL (> 0)
    losing_pnl = [pnl for pnl in all_pnl_values if pnl is not None and pnl < 0]
    winning_pnl = [pnl for pnl in all_pnl_values if pnl is not None and pnl > 0]
    
    # Calculate averages
    # Formula: avg = sum(values) / count(values)
    avg_losing_pnl = sum(losing_pnl) / len(losing_pnl) if len(losing_pnl) > 0 else 0.0
    avg_winning_pnl = sum(winning_pnl) / len(winning_pnl) if len(winning_pnl) > 0 else 0.0
    
    return avg_losing_pnl, avg_winning_pnl

def calculate_most_least_profitable(previous_df, current_ticker, current_realized_pnl_at_point):
    """
    Calculate most and least profitable tickers based on realized PnL at point of time.
    
    Formulas:
        - Most Profitable: Ticker where Max(value where realized pnl > 0)
        - Least Profitable: Ticker where Min(value where realized pnl > 0)
    
    Only considers winning trades (PnL > 0).
    
    Returns format: "TICKER PnL_Value" or "TICKER1 PnL1, TICKER2 PnL2" if multiple tickers tie
    
    Args:
        previous_df (pd.DataFrame): Previous rows of portfolio DataFrame
        current_ticker (str): Current ticker being traded
        current_realized_pnl_at_point (float or None): Current row's realized PnL at point of time
    
    Returns:
        tuple: (most_profitable, least_profitable) - Formatted strings or "None"
    
    Edge cases:
        - Returns ("None", "None") if no winning trades exist
        - Handles multiple tickers with same max/min PnL
        - Only considers PnL > 0 (winning trades)
    """
    # Track all winning trades (ticker, pnl) pairs where pnl > 0
    winning_trades = []
    
    # Process previous rows
    if len(previous_df) > 0:
        for _, row in previous_df.iterrows():
            row_ticker = str(row.get('Ticker', '')).upper()
            row_pnl = row.get('PnL Realized at Point of Time', 0)
            if row_ticker and row_pnl is not None and row_pnl > 0:
                winning_trades.append((row_ticker, row_pnl))
    
    # Add current PnL if it's a winning trade
    if current_ticker and current_realized_pnl_at_point is not None and current_realized_pnl_at_point > 0:
        winning_trades.append((current_ticker, current_realized_pnl_at_point))
    
    if not winning_trades:
        return "None", "None"
    
    # Find overall max and min PnL values across all winning trades
    max_pnl = max(pnl for _, pnl in winning_trades)
    min_pnl = min(pnl for _, pnl in winning_trades)
    
    # Find tickers with max PnL (most profitable)
    most_profitable_trades = [(ticker, pnl) for ticker, pnl in winning_trades if pnl == max_pnl]
    most_profitable_list = [f"{ticker} {pnl}" for ticker, pnl in sorted(most_profitable_trades)]
    most_profitable = ", ".join(most_profitable_list) if most_profitable_list else "None"
    
    # Find tickers with min PnL (least profitable among winners)
    least_profitable_trades = [(ticker, pnl) for ticker, pnl in winning_trades if pnl == min_pnl]
    least_profitable_list = [f"{ticker} {pnl}" for ticker, pnl in sorted(least_profitable_trades)]
    least_profitable = ", ".join(least_profitable_list) if least_profitable_list else "None"
    
    return most_profitable, least_profitable

def calculate_max_drawdown(current_total_pv):
    """
    Calculate Max Drawdown from row 2 to current row.
    
    Formula: Max Drawdown = (MAX(Total PV from row 2 to current) - MIN(Total PV from row 2 to current)) / MAX(Total PV from row 2 to current)
    
    Calculation:
        - Row 1: Returns 0 (no calculation)
        - Row 2: (MAX(Total PV row 2) - MIN(Total PV row 2)) / MAX(Total PV row 2) = 0
        - Row 3: (MAX(Total PV rows 2-3) - MIN(Total PV rows 2-3)) / MAX(Total PV rows 2-3)
        - Row 4: (MAX(Total PV rows 2-4) - MIN(Total PV rows 2-4)) / MAX(Total PV rows 2-4)
        - etc.
    
    Note: Calculation starts from row 2 (skips first row).
    
    Args:
        current_total_pv (float): Current Total PV value for this row
    
    Returns:
        float: Max Drawdown as decimal (0 for first row, calculated value for subsequent rows)
    
    Edge cases:
        - Returns 0.0 for first row (no calculation)
        - Returns 0.0 if max_pv == 0 (avoid division by zero)
        - Only considers rows from row 2 onwards
    """
    # Check if this is the first row (portfolio_df is empty)
    if portfolio_state['portfolio_df'] is None or portfolio_state['portfolio_df'].empty:
        return 0.0
    
    # Get Total PV values starting from row 2 (skip first row)
    # We need to get Total PV from row 2 onwards, plus the current row
    total_pv_values = []
    
    # Get Total PV values from previous rows (starting from row 2, skipping row 1)
    if 'Total PV' in portfolio_state['portfolio_df'].columns:
        # Get all Total PV values starting from row 2 (index 1 onwards)
        all_pv_values = portfolio_state['portfolio_df']['Total PV'].tolist()
        if len(all_pv_values) > 0:
            # Skip first row (index 0), take from row 2 onwards (index 1 onwards)
            total_pv_values.extend(all_pv_values[1:])
    
    # Add current row's Total PV (this will be row 2 or later)
    total_pv_values.append(current_total_pv)
    
    # Calculate MAX and MIN
    if not total_pv_values:
        return 0.0
    
    max_pv = max(total_pv_values)
    min_pv = min(total_pv_values)
    
    if max_pv == 0:
        return 0.0
    
    # Calculate Max Drawdown: (MAX - MIN) / MAX
    max_drawdown = (max_pv - min_pv) / max_pv
    
    return max_drawdown

def update_max_investment_history(ticker, price, quantity_buy, action, old_quantity):
    """
    Update the historical maximum investment for a ticker.
    
    Formula: investment_value = price * quantity_buy
    
    Logic:
        - Tracks maximum investment value (price * quantity) for each ticker
        - Only updates when opening/expanding positions (entry points)
        - Tracks both 'buy' actions (long positions) and 'sell' actions (short positions)
        - Does NOT update when closing positions
    
    Args:
        ticker (str): Ticker symbol
        price (float): Price of the trade
        quantity_buy (float): Quantity in the trade (positive for buy, positive for sell)
        action (str): Trade action ('buy', 'sell', 'hold')
        old_quantity (float): Quantity before this trade
    
    Edge cases:
        - Only tracks 'buy' actions for long positions
        - Only tracks 'sell' actions when old_quantity <= 0 (opening short)
        - Does not track when closing long positions (old_quantity > 0 and selling)
        - Does not update if investment_value is not greater than current max
    """
    action_lower = str(action).lower()
    qty = abs(quantity_buy) if quantity_buy < 0 else quantity_buy

    # Track maximum investment for:
    # 1. Buy actions (opening/expanding long positions)
    # 2. Sell actions that open/expand short positions (when old_quantity <= 0)
    if action_lower == 'buy':
        # Buy action: opening/expanding long position
        # Formula: investment_value = price * quantity
        investment_value = price * qty
        current_max = portfolio_state['max_investment_history'].get(ticker, 0.0)
        if investment_value > current_max:
            portfolio_state['max_investment_history'][ticker] = investment_value

    elif action_lower == 'sell':
        # Sell action: check if it's opening/expanding a short position
        # Short position opens when old_quantity <= 0 and we're selling
        if old_quantity <= 0:
            # This sell opens or expands a short position (entry point for short)
            # Formula: investment_value = price * quantity
            investment_value = price * qty
            current_max = portfolio_state['max_investment_history'].get(ticker, 0.0)
            if investment_value > current_max:
                portfolio_state['max_investment_history'][ticker] = investment_value
        # If old_quantity > 0, this is closing a long position, not opening a short
        # So we don't track it as an entry point


def calculate_biggest_investment():
    """
    Calculate Biggest Investment = max(price * quantity_buy) historically for all symbols.
    
    Formula: For each ticker, max_investment = max(all investment_values)
    
    Returns:
        - All tickers that have ever had positions (including closed positions)
        - Ordered by maximum investment descending (biggest first)
        - Format: "SYMBOL: MAX_INVESTMENT, SYMBOL: MAX_INVESTMENT"
    
    Returns:
        str: Formatted string of tickers and their max investments, or "None" if no investments
    
    Edge cases:
        - Returns "None" if no historical investments exist
        - Includes tickers with closed positions (historical tracking)
        - Only includes tickers where max_investment > 0
        - Sorts by investment value descending, then formats as integers
    """
    # Get all tickers that have historical investment records (including closed positions)
    positions = []
    for ticker, max_investment in portfolio_state['max_investment_history'].items():
        if max_investment > 0:  # Only include tickers with historical max > 0
            positions.append({
                'ticker': ticker.upper(),
                'max_investment': max_investment
            })
    
    if not positions:
        return "None"
    
    # Sort by historical maximum investment (descending) - biggest first
    positions.sort(key=lambda x: x['max_investment'], reverse=True)
    
    # Format as "SYMBOL: MAX_INVESTMENT, SYMBOL: MAX_INVESTMENT"
    parts = [f"{pos['ticker']}: {int(pos['max_investment'])}" for pos in positions]
    
    return ", ".join(parts)

def calculate_average_position(current_df, current_action, current_position_value_pv):
    """
    Calculate Average Position = average of Position Value PV when Buy/Sell == "buy".
    
    Formula: IF(Buy/Sell == "buy", AVERAGEIF(Buy/Sell == "buy" up to current row, Position Value PV), "0")
    
    Logic:
        - Only calculates when current action is 'buy'
        - Averages all Position Value PV values from previous 'buy' actions plus current 'buy'
        - Returns "0" if current action is not 'buy'
    
    Args:
        current_df (pd.DataFrame): DataFrame up to current row (excluding current row)
        current_action (str): Current row's Buy/Sell action
        current_position_value_pv (float): Current row's Position Value PV
    
    Returns:
        float or str: Average Position Value PV (rounded to 4 decimals), or "0" string if not buy
    
    Edge cases:
        - Returns "0" if current action is not 'buy'
        - Returns "0" if no previous 'buy' actions exist
        - Includes current row's PV in calculation if it's a 'buy'
    """
    action_lower = str(current_action).lower()

    if action_lower != 'buy':
        return "0"

    # Get all Position Value PV values where Buy/Sell == "buy" up to and including current row
    # We need to include the current row's PV in the calculation
    buy_pvs = []

    # Add PVs from previous rows where Buy/Sell == "buy"
    if not current_df.empty and 'Buy/Sell' in current_df.columns and 'Position Value PV' in current_df.columns:
        buy_rows = current_df[current_df['Buy/Sell'].str.lower() == 'buy']
        if not buy_rows.empty:
            buy_pvs.extend(buy_rows['Position Value PV'].tolist())

    # Add current row's PV if it's a buy
    if action_lower == 'buy':
        buy_pvs.append(current_position_value_pv)

    # Calculate average
    # Formula: avg = sum(pv_values) / count(pv_values)
    if buy_pvs:
        avg = sum(buy_pvs) / len(buy_pvs)
        return round(avg, 4)  # Match the format shown (4 decimal places)
    else:
        return "0"
    
def calculate_holdings():
    """
    Calculate Holdings = Count of unique tickers with non-zero current quantity.
    
    Formula: Holdings = COUNTA(UNIQUE Tickers, where Current Quantity ≠ 0)
    
    Logic:
        - Counts distinct tickers that have open positions (quantity != 0)
        - Includes both long positions (quantity > 0) and short positions (quantity < 0)
        - Each unique ticker counts as 1, regardless of position size
    
    Returns:
        int: Number of unique tickers with non-zero quantity
    
    Edge cases:
        - Returns 0 if no open positions
        - Counts each ticker only once (uses set to track unique tickers)
        - Includes both long and short positions in count
    """
    unique_tickers = set()
    for ticker, qty in portfolio_state['quantities'].items():
        if qty != 0:
            unique_tickers.add(ticker.upper())
    return len(unique_tickers)

def update_traded_volume_history(price, quantity_buy, action):
    """
    Update the historical highest and lowest traded volume.
    
    Formula: traded_volume = quantity_buy * price
    
    Logic:
        - Tracks maximum and minimum traded volume across all trades
        - Updates when action is 'buy' or 'sell' (not 'hold')
        - Tracks based on the action (buy/sell), not the resulting position
        - This ensures we track trades even if they close positions (resulting in 'hold')
    
    Args:
        price (float): Price of the trade
        quantity_buy (float): Quantity in the trade
        action (str): Trade action ('buy', 'sell', 'hold')
        current_position (str): Current position after the trade ('long', 'short', 'hold')
    
    Edge cases:
        - Does not update if action is 'hold'
        - Uses absolute value of quantity_buy to handle negative inputs
        - Initializes to None, then sets to first traded volume
    """
    action_lower = str(action).lower()
    
    # Track if action is 'buy' or 'sell' (not 'hold')
    # This ensures we track trades even if they close positions (resulting in 'hold')
    if action_lower in ['buy', 'sell']:
        qty = abs(quantity_buy) if quantity_buy < 0 else quantity_buy
        # Formula: traded_volume = quantity_buy * price
        traded_volume = price * qty
        
        # Update highest traded volume
        if portfolio_state['highest_traded_volume'] is None:
            portfolio_state['highest_traded_volume'] = traded_volume
        else:
            if traded_volume > portfolio_state['highest_traded_volume']:
                portfolio_state['highest_traded_volume'] = traded_volume
        
        # Update lowest traded volume
        if portfolio_state['lowest_traded_volume'] is None:
            portfolio_state['lowest_traded_volume'] = traded_volume
        else:
            if traded_volume < portfolio_state['lowest_traded_volume']:
                portfolio_state['lowest_traded_volume'] = traded_volume
    # If action is 'hold', don't update - keep last known values

def get_highest_traded_volume():
    """
    Get Highest Traded Volume = max(quantity * price) across all trades.
    
    Returns the historical maximum traded volume across all trades.
    
    Returns:
        int: Highest traded volume, or 0 if no trades have been recorded
    
    Edge cases:
        - Returns 0 if no trades recorded (highest_traded_volume is None)
        - Converts to integer for display
    """
    if portfolio_state['highest_traded_volume'] is None:
        return 0
    return int(portfolio_state['highest_traded_volume'])

def get_lowest_traded_volume():
    """
    Get Lowest Traded Volume = min(quantity * price) across all trades.
    
    Returns the historical minimum traded volume across all trades.
    
    Returns:
        int: Lowest traded volume, or 0 if no trades have been recorded
    
    Edge cases:
        - Returns 0 if no trades recorded (lowest_traded_volume is None)
        - Converts to integer for display
    """
    if portfolio_state['lowest_traded_volume'] is None:
        return 0
    return int(portfolio_state['lowest_traded_volume'])

# ---------- Average Holding Days Calculation ----------

def track_position_opening(ticker, current_period):
    """
    Track when a position is first opened for a ticker.
    
    Logic:
        - Records the period/row number when a position is first opened
        - Only tracks if position is being opened (not already tracked)
        - Used later to calculate holding period when position closes
    
    Args:
        ticker (str): Ticker symbol
        current_period (int): Current period/row number
    
    Edge cases:
        - Only tracks if ticker not already in position_open_period
        - Prevents overwriting existing tracking
    """
    if ticker not in portfolio_state['position_open_period']:
        portfolio_state['position_open_period'][ticker] = current_period

def detect_closed_positions(old_quantities, new_quantities, current_period):
    """
    Detect positions that were closed (quantity went from non-zero to zero).
    
    Logic:
        - Compares old_quantities and new_quantities to find positions that closed
        - Position closed if: old_quantity != 0 and new_quantity == 0
        - Calculates holding period for closed positions
    
    Formula: holding_period = current_period - open_period + 1
    
    Args:
        old_quantities (dict): Quantities before trade {ticker: quantity}
        new_quantities (dict): Quantities after trade {ticker: quantity}
        current_period (int): Current period/row number
    
    Returns:
        list: List of dicts with 'ticker' and 'holding_period' for closed positions
    
    Edge cases:
        - Checks all tickers in both old and new quantities
        - Removes ticker from tracking when position closes
        - Returns empty list if no positions closed
    """
    closed_positions = []
    
    # Check all tickers that were in old_quantities
    for ticker in set(list(old_quantities.keys()) + list(new_quantities.keys())):
        old_qty = old_quantities.get(ticker, 0)
        new_qty = new_quantities.get(ticker, 0)
        
        # Position closed if it went from non-zero to zero
        if old_qty != 0 and new_qty == 0:
            # Calculate holding period
            # Formula: holding_period = current_period - open_period + 1
            if ticker in portfolio_state['position_open_period']:
                open_period = portfolio_state['position_open_period'][ticker]
                holding_period = current_period - open_period + 1
                closed_positions.append({
                    'ticker': ticker,
                    'holding_period': holding_period
                })
                # Remove from tracking since it's closed
                del portfolio_state['position_open_period'][ticker]
    
    return closed_positions

def update_average_holding_days(closed_positions):
    """
    Update cumulative average holding days when positions are closed.
    
    Logic:
        - Adds holding period to cumulative sum
        - Increments count of closed positions
        - Used later to calculate average holding days
    
    Formula:
        - cumulative_holding_sum += holding_period
        - closed_positions_count += 1
    
    Args:
        closed_positions (list): List of dicts with 'ticker' and 'holding_period'
    
    Edge cases:
        - Does nothing if closed_positions is empty
        - Processes multiple closed positions in one call
    """
    if closed_positions:
        for closed in closed_positions:
            holding_period = closed['holding_period']
            # Add to cumulative sum
            portfolio_state['cumulative_holding_sum'] += holding_period
            # Increment count of closed positions
            portfolio_state['closed_positions_count'] += 1

def calculate_average_holding_days(is_closing_trade=False, previous_df=None):
    """
    Calculate Average Holding Days for all closed positions.
    
    Formula: Average Holding Days = (sum of all holding periods) / (number of closed positions)
    
    Formula: avg = cumulative_holding_sum / closed_positions_count
    
    Args:
        is_closing_trade (bool): Whether this trade row is closing a position (contains "- Close" in trade string)
        previous_df (pd.DataFrame): Previous rows of portfolio DataFrame (not used, kept for compatibility)
    
    Returns:
        float or None: Average holding days (rounded to 3 decimals), or None if no positions closed or trade not closing
    
    Edge cases:
        - Returns None if no positions closed yet (closed_positions_count == 0)
        - Returns None if is_closing_trade is False (only show value when trade closes)
        - Rounds to 3 decimal places for display
    """
    # Only calculate and return average when a trade is actually closing
    if not is_closing_trade:
        return None  # Don't show value for non-closing trades
    
    # Trade is closing, calculate average
    if portfolio_state['closed_positions_count'] == 0:
        return None  # No positions closed yet
    
    # Formula: avg = cumulative_sum / count
    avg = portfolio_state['cumulative_holding_sum'] / portfolio_state['closed_positions_count']
    return round(avg, 3)  # Round to 3 decimal places

def calculate_asset_count():
    """
    Calculate Asset Count = Count of distinct tickers per asset type where Current Quantity != 0.
    
    Formula: For each asset type, count distinct tickers where quantity != 0
    
    Returns format: "AssetType1: Count1, AssetType2: Count2, ..."
    
    Returns:
        str: Comma-separated list of asset types and their counts, or "None" if no positions
    
    Edge cases:
        - Returns "None" if no open positions exist
        - Only counts tickers where quantity != 0
        - Sorted alphabetically by asset type name
    """
    from collections import defaultdict
    
    # Count tickers per asset type where quantity != 0
    asset_type_counts = defaultdict(int)
    
    for ticker, qty in portfolio_state['quantities'].items():
        if qty != 0:
            # Get asset type for this ticker
            ticker_asset_type = portfolio_state['asset_types'].get(ticker)
            if ticker_asset_type:
                asset_type_counts[ticker_asset_type] += 1
    
    # Format as "AssetType1: Count1, AssetType2: Count2" ordered by asset type name
    if not asset_type_counts:
        return "None"
    
    # Sort by asset type name
    sorted_asset_types = sorted(asset_type_counts.items())
    parts = [f"{asset_type}: {count}" for asset_type, count in sorted_asset_types]
    return ", ".join(parts)

def calculate_investment_count():
    """
    Calculate Investment Count = Total cumulative count of positions opened.
    
    Logic:
        - Investment Count increments when opening a position:
            - Long position: when action='buy' and position='long'
            - Short position: when action='sell' and position='short'
        - Does NOT increment when closing positions
        - Tracks cumulative count across all trades
    
    Formula: investment_count is incremented in process_trade() when:
        (position == 'long' and action == 'buy') OR (position == 'short' and action == 'sell')
    
    Returns:
        int: Cumulative investment count (number of positions opened)
    
    Edge cases:
        - Returns 0 if no positions have been opened
        - Only counts opening actions, not closing actions
    """
    global investment_count
    return investment_count

def calculate_ytd_pnl(previous_df, current_date, current_realized_pnl_cumulative):
    """
    Calculate YTD PnL (Year-to-Date PnL) = Cumulative realized PnL for the current year.
    
    Formula: YTD PnL = Current Cumulative Realized PnL - Cumulative Realized PnL on Jan 1 of current year
    
    Logic:
        - Finds the cumulative realized PnL from the last row before the current year
        - Subtracts it from current cumulative realized PnL
        - This gives realized PnL from January 1st to current date
    
    Args:
        previous_df (pd.DataFrame): Previous rows of portfolio DataFrame
        current_date (str or datetime): Current trade date
        current_realized_pnl_cumulative (float): Current cumulative realized PnL
    
    Returns:
        float or None: YTD PnL, or None if date parsing fails
    
    Edge cases:
        - Returns None if current_date is None
        - Assumes date format '%m/%d/%Y' for string dates
        - Returns 0.0 if portfolio started in current year (no previous year data)
        - Uses last row from previous year as baseline
    """
    from datetime import datetime
    
    if current_date is None:
        return None
    
    # Parse current date to get current year
    if isinstance(current_date, str):
        current_dt = datetime.strptime(current_date, '%m/%d/%Y')
    else:
        current_dt = pd.to_datetime(current_date)
    
    current_year = current_dt.year
    
    # Find cumulative realized PnL on Jan 1 of current year
    # Look for the last row from before Jan 1 of current year
    start_of_year_cumulative = 0.0
    
    if len(previous_df) > 0:
        # Go through rows from end to beginning (reverse order)
        for i in range(len(previous_df) - 1, -1, -1):
            row = previous_df.iloc[i]
            row_date = row.get('Date')
            
            if row_date is None:
                continue
            
            if isinstance(row_date, str):
                row_dt = datetime.strptime(row_date, '%m/%d/%Y')
            else:
                row_dt = pd.to_datetime(row_date)
            
            # If row is from previous year, use its cumulative PnL
            if row_dt.year < current_year:
                start_of_year_cumulative = row.get('PnL Realized Cummulative', 0.0)
                break
    
    # Formula: YTD PnL = Current Cumulative - Cumulative on Jan 1
    ytd_pnl = current_realized_pnl_cumulative - start_of_year_cumulative
    
    return ytd_pnl

def calculate_diversification(total_pv, ticker_pv_dict):
    """
    Calculate Distribution and Distribution in % based on asset type.
    
    Uses already-calculated PV values from calculate_total_pv_all_tickers.
    
    Formulas:
        - For each asset type: total_pv_per_type = Σ(PV for each ticker with that asset type)
        - Total PV across all assets = total_pv (already calculated)
        - Distribution = "AssetType1: TotalPV1, AssetType2: TotalPV2, ..."
        - Distribution in % = "AssetType1: (TotalPV1/total_pv)*100%, AssetType2: (TotalPV2/total_pv)*100%, ..."
    
    Args:
        current_ticker (str): Current ticker being traded
        current_price (float): Current price for current ticker
        total_pv (float): Already calculated Total PV across all tickers
        ticker_pv_dict (dict): Dictionary of {ticker: pv} from calculate_total_pv_all_tickers
    
    Returns:
        tuple: (distribution, distribution_pct) - Formatted strings, or ("None", "None") if no positions
    """
    from collections import defaultdict
    
    # Calculate total PV per asset type using already-calculated PV values
    asset_type_totals = defaultdict(float)
    
    for ticker, pv in ticker_pv_dict.items():
        # Get asset type for this ticker
        ticker_asset_type = portfolio_state['asset_types'].get(ticker)
        if ticker_asset_type:
            asset_type_totals[ticker_asset_type] += pv
    
    if not asset_type_totals:
        return "None", "None"
    
    # Use already-calculated total_pv
    if total_pv == 0:
        return "None", "None"
    
    # Format Distribution: "AssetType1: TotalPV1, AssetType2: TotalPV2, ..."
    sorted_asset_types = sorted(asset_type_totals.items())
    distribution_parts = [f"{asset_type}: {total:.2f}" for asset_type, total in sorted_asset_types]
    distribution = ", ".join(distribution_parts)
    
    # Format Distribution in %: "AssetType1: (TotalPV1/total_pv)*100%, AssetType2: (TotalPV2/total_pv)*100%, ..."
    distribution_pct_parts = [
        f"{asset_type}: {(total/total_pv)*100:.2f}%" 
        for asset_type, total in sorted_asset_types
    ]
    distribution_pct = ", ".join(distribution_pct_parts)
    
    return distribution, distribution_pct

# ---------- Equity Distribution ----------

EQUITY_METADATA = { # To be dynamically mapped
    # Market Cap, Industry, Sector mappings
    'AAPL': {'market_cap': 'High', 'industry': 'Software', 'sector': 'Technology'},
    'MSFT': {'market_cap': 'High', 'industry': 'Software', 'sector': 'Technology'},
    'NVDA': {'market_cap': 'High', 'industry': 'Software', 'sector': 'Technology'},
    'TSLA': {'market_cap': 'High', 'industry': 'Auto Manufacturers', 'sector': 'Consumer Cyclical'},
    'JPM': {'market_cap': 'High', 'industry': 'Credit Services', 'sector': 'Financial Services'},
    'V': {'market_cap': 'High', 'industry': 'Credit Services', 'sector': 'Financial Services'},
    'JNJ': {'market_cap': 'High', 'industry': 'Drug Manufacturers', 'sector': 'Healthcare'},
    'SQ': {'market_cap': 'Mid', 'industry': 'Software', 'sector': 'Technology'},
    'PLTR': {'market_cap': 'Mid', 'industry': 'Software', 'sector': 'Technology'},
    'DOCU': {'market_cap': 'Mid', 'industry': 'Software', 'sector': 'Technology'},
    'PFE': {'market_cap': 'Mid', 'industry': 'Drug Manufacturers', 'sector': 'Healthcare'},
    'F': {'market_cap': 'Mid', 'industry': 'Auto Manufacturers', 'sector': 'Consumer Cyclical'},
    'GM': {'market_cap': 'Mid', 'industry': 'Auto Manufacturers', 'sector': 'Consumer Cyclical'},
    'COIN': {'market_cap': 'Mid', 'industry': 'Credit Services', 'sector': 'Financial Services'},
    'SOFI': {'market_cap': 'Low', 'industry': 'Credit Services', 'sector': 'Financial Services'},
    'LCID': {'market_cap': 'Low', 'industry': 'Auto Manufacturers', 'sector': 'Consumer Cyclical'},
    'RIVN': {'market_cap': 'Low', 'industry': 'Auto Manufacturers', 'sector': 'Consumer Cyclical'},
    'HOOD': {'market_cap': 'Low', 'industry': 'Credit Services', 'sector': 'Financial Services'},
    'PATH': {'market_cap': 'Low', 'industry': 'Software', 'sector': 'Technology'},
}

def calculate_equity_distribution_market_cap(ticker_pv_dict):
    """
    Calculate Equity Distribution (Market Cap) for Equity asset types only.
    
    Uses already-calculated PV values from calculate_total_pv_all_tickers.
    
    Formulas:
        - For each market cap category: total_pv_per_category = Σ(PV for each Equity ticker with that market cap)
        - Total PV across all Equity positions = sum of all Equity ticker PVs
        - Distribution in % = "MarketCap1: (TotalPV1/TotalPV_Equity)*100%, MarketCap2: (TotalPV2/TotalPV_Equity)*100%, ..."
    
    Only includes tickers where Asset Type == 'Equity'.
    
    Args:
        ticker_pv_dict (dict): Dictionary of {ticker: pv} from calculate_total_pv_all_tickers
    
    Returns:
        str: Formatted string of market cap distribution, or "None" if no Equity positions
    """
    from collections import defaultdict
    
    # Calculate total PV per market cap category (only for Equity asset types)
    market_cap_totals = defaultdict(float)
    
    for ticker, pv in ticker_pv_dict.items():
        # Check if this ticker is Equity asset type
        ticker_asset_type = portfolio_state['asset_types'].get(ticker, '').lower()
        if ticker_asset_type != 'equity':
            continue
        
        # Get market cap category for this ticker
        market_cap_category = portfolio_state['market_cap'].get(ticker)
        if market_cap_category:
            market_cap_totals[market_cap_category] += pv
    
    if not market_cap_totals:
        return "None"
    
    # Calculate total PV across all Equity market cap categories
    total_pv_equity = sum(market_cap_totals.values())
    
    if total_pv_equity == 0:
        return "None"
    
    # Format Distribution in %: "MarketCap1: (TotalPV1/TotalPV_Equity)*100%, MarketCap2: (TotalPV2/TotalPV_Equity)*100%, ..."
    sorted_market_caps = sorted(market_cap_totals.items())
    distribution_pct_parts = [
        f"{market_cap}: {(total/total_pv_equity)*100:.2f}%" 
        for market_cap, total in sorted_market_caps
    ]
    distribution_pct = ", ".join(distribution_pct_parts)
    
    return distribution_pct


def calculate_equity_distribution_industry(ticker_pv_dict):
    """
    Calculate Equity Distribution (Industry) for Equity asset types only.
    
    Uses already-calculated PV values from calculate_total_pv_all_tickers.
    
    Formulas:
        - For each industry: total_pv_per_industry = Σ(PV for each Equity ticker with that industry)
        - Total PV across all Equity positions = sum of all Equity ticker PVs
        - Distribution in % = "Industry1: (TotalPV1/TotalPV_Equity)*100%, Industry2: (TotalPV2/TotalPV_Equity)*100%, ..."
    
    Only includes tickers where Asset Type == 'Equity'.
    
    Args:
        ticker_pv_dict (dict): Dictionary of {ticker: pv} from calculate_total_pv_all_tickers
    
    Returns:
        str: Formatted string of industry distribution, or "None" if no Equity positions
    """
    from collections import defaultdict
    
    # Calculate total PV per industry (only for Equity asset types)
    industry_totals = defaultdict(float)
    
    for ticker, pv in ticker_pv_dict.items():
        # Check if this ticker is Equity asset type
        ticker_asset_type = portfolio_state['asset_types'].get(ticker, '').lower()
        if ticker_asset_type != 'equity':
            continue
        
        # Get industry for this ticker
        industry = portfolio_state['industry'].get(ticker)
        if industry:
            industry_totals[industry] += pv
    
    if not industry_totals:
        return "None"
    
    # Calculate total PV across all Equity industries
    total_pv_equity = sum(industry_totals.values())
    
    if total_pv_equity == 0:
        return "None"
    
    # Format Distribution in %: "Industry1: (TotalPV1/TotalPV_Equity)*100%, Industry2: (TotalPV2/TotalPV_Equity)*100%, ..."
    sorted_industries = sorted(industry_totals.items())
    distribution_pct_parts = [
        f"{industry}: {(total/total_pv_equity)*100:.2f}%" 
        for industry, total in sorted_industries
    ]
    distribution_pct = ", ".join(distribution_pct_parts)
    
    return distribution_pct

def calculate_equity_distribution_sector(ticker_pv_dict):
    """
    Calculate Equity Distribution (Sector) for Equity asset types only.
    
    Uses already-calculated PV values from calculate_total_pv_all_tickers.
    
    Formulas:
        - For each sector: total_pv_per_sector = Σ(PV for each Equity ticker with that sector)
        - Total PV across all Equity positions = sum of all Equity ticker PVs
        - Distribution in % = "Sector1: (TotalPV1/TotalPV_Equity)*100%, Sector2: (TotalPV2/TotalPV_Equity)*100%, ..."
    
    Only includes tickers where Asset Type == 'Equity'.
    
    Args:
        ticker_pv_dict (dict): Dictionary of {ticker: pv} from calculate_total_pv_all_tickers
    
    Returns:
        str: Formatted string of sector distribution, or "None" if no Equity positions
    """
    from collections import defaultdict
    
    # Calculate total PV per sector (only for Equity asset types)
    sector_totals = defaultdict(float)
    
    for ticker, pv in ticker_pv_dict.items():
        # Check if this ticker is Equity asset type
        ticker_asset_type = portfolio_state['asset_types'].get(ticker, '').lower()
        if ticker_asset_type != 'equity':
            continue
        
        # Get sector for this ticker
        sector = portfolio_state['sector'].get(ticker)
        if sector:
            sector_totals[sector] += pv
    
    if not sector_totals:
        return "None"
    
    # Calculate total PV across all Equity sectors
    total_pv_equity = sum(sector_totals.values())
    
    if total_pv_equity == 0:
        return "None"
    
    # Format Distribution in %: "Sector1: (TotalPV1/TotalPV_Equity)*100%, Sector2: (TotalPV2/TotalPV_Equity)*100%, ..."
    sorted_sectors = sorted(sector_totals.items())
    distribution_pct_parts = [
        f"{sector}: {(total/total_pv_equity)*100:.2f}%" 
        for sector, total in sorted_sectors
    ]
    distribution_pct = ", ".join(distribution_pct_parts)
    
    return distribution_pct

# ---------- Main entry per trade ----------

def process_trade(ticker, asset_type, action, position, price, quantity_buy, date=None, take_profit_pct=0.20, stop_loss_pct=0.10):
    """
    Main function to process a single trade and update portfolio state.
    
    This function orchestrates all calculations for a trade:
        1. Updates quantities, cost basis, and average prices
        2. Calculates realized and unrealized PnL
        3. Calculates position values and equity
        4. Calculates all derived metrics (win rate, drawdown, etc.)
        5. Updates portfolio DataFrame
    
    Formulas calculated:
        - Total PnL Overall = Equity - Initial Cash
        - Daily PnL = Today's Total PnL Overall - Yesterday's Total PnL Overall
        - Daily % = (Today's Equity - Yesterday's Equity) / Yesterday's Equity * 100
        - Cumulative % = ((Equity / Initial Cash) - 1) * 100
        - Average Gain = Total Gain / Total Trades
    
    Args:
        ticker (str): Ticker symbol
        asset_type (str): Asset type (e.g., 'Stock', 'Crypto', 'ETF')
        action (str): Trade action ('buy', 'sell', 'hold')
        position (str): Position type ('long', 'short', 'hold')
        price (float): Trade price
        quantity_buy (float or str): Quantity to trade (can be numeric or string like "-(-10)")
        date (str or datetime, optional): Trade date
        take_profit_pct (float, optional): Take profit percentage (default 0.20 = 20%)
        stop_loss_pct (float, optional): Stop loss percentage (default 0.10 = 10%)
    
    Returns:
        dict: Row dictionary with all calculated values
    
    Edge cases:
        - Handles position flips (long to short, short to long)
        - Handles partial position closes
        - Handles first trade (no previous data)
        - Normalizes quantity input (handles string formats)
        - Updates global state (quantities, cost_basis, avg_price, etc.)
    """
    ticker = str(ticker).strip().upper()
    
    # Normalize quantity input (handles various formats)
    q_in = normalize_quantity(quantity_buy)

    # Get previous state
    old_q = portfolio_state['quantities'][ticker]
    old_cb = portfolio_state['cost_basis'][ticker]

    # Calculate cash and remaining
    cash = calculate_cash_single()
    new_remaining = calculate_remaining_single(action, price, q_in, old_q, old_cb)

    # Update quantity and determine current position
    new_q = calculate_current_quantity_single(ticker, action, q_in, old_q)
    
    # Track position opening and closing for average holding days
    # Get current period (row number) - 1-indexed
    previous_df = portfolio_state['portfolio_df']
    current_period = len(previous_df) + 1  # Current row number
    
    # Track position opening if this is a new position (old_q == 0 and new_q != 0)
    if old_q == 0 and new_q != 0:
        track_position_opening(ticker, current_period)
    
    # Detect closed positions (quantity went from non-zero to zero)
    # Build old_quantities_dict: copy all current quantities, but use old_q for current ticker
    old_quantities_dict = {}
    for t in portfolio_state['quantities'].keys():
        if t == ticker:
            old_quantities_dict[t] = old_q  # Use old quantity for current ticker
        else:
            old_quantities_dict[t] = portfolio_state['quantities'][t]
    
    # Build new_quantities_dict: current state (already has new_q for ticker)
    new_quantities_dict = dict(portfolio_state['quantities'])
    
    # Detect closed positions
    closed_positions = detect_closed_positions(
        old_quantities_dict,
        new_quantities_dict,
        current_period
    )
    
    # Update average holding days when positions close
    if closed_positions:
        update_average_holding_days(closed_positions)
    
    # Determine current position based on new quantity
    # Formula: short if q < 0, long if q > 0, hold if q == 0
    current_position = 'short' if new_q < 0 else 'long'
    
    # Calculate realized PnL at point of time (independent calculation)
    realized_pnl_at_point = calculate_realized_pnl_at_point_of_time(
        ticker, action, position, price, q_in, old_q
    )
    
    # Calculate cumulative realized PnL (updates global state)
    realized_pnl_cumulative = calculate_realized_pnl_cumulative(
        ticker, action, position, price, q_in, old_q
    )
    
    # Calculate average price and cost basis
    avg_p, cb = calculate_avg_price_and_cost_basis_single(
        ticker, action, price, q_in, old_q, new_q, old_cb  
    )
    # Calculate buyable/sellable shares
    # Formula: buyable_sellable = prev_remaining_cash / price
    previous_remaining = portfolio_state['remaining']
    buyable_sellable = (previous_remaining / price) if price > 0 else 0.0
    
    # Calculate position value and unrealized PnL components
    pv = position_value_from_position(current_position, new_q, price)
    long_unrealized, short_unrealized, total_current_ticker_unrealized, total_unrealized_all_tickers = pnl_unrealized_components(new_q, price, avg_p, ticker, price)
    
    # Generate strings for open positions
    open_pos = open_positions_str()
    open_pv = open_pv_str(ticker, price, current_position)
    open_unrealized_pnl = open_pnl_unrealized_str(ticker, price) 

    # Calculate PV for current ticker only
    pv_long_current, pv_short_current = calculate_pv_for_current_ticker(price, current_position, new_q, avg_p, cb)
    
    # Calculate total PV across all tickers
    total_pv, ticker_pv_dict = calculate_total_pv_all_tickers(ticker, price)
    
    # Calculate equity
    # Formula: equity = total_pv + remaining_cash
    total_pv_equity = total_pv + new_remaining
    
    # Calculate Total PnL Overall
    # Formula: total_pnl_overall = equity - initial_cash
    total_pnl_overall = total_pv_equity - cash
    
    # Calculate Daily PnL = Today's Total PnL Overall - Yesterday's Total PnL Overall
    # Get previous row's Total PnL Overall if it exists, otherwise 0 (first trade)
    if len(previous_df) > 0:
        previous_total_pnl_overall = previous_df.iloc[-1]['Total PnL Overall (Unrealized+Realized)']
        # Get yesterday's equity for Daily % calculation
        yesterday_equity = previous_df.iloc[-1]['Equity: Total PV + Remaining']
    else:
        previous_total_pnl_overall = 0.0
        yesterday_equity = cash  # Use initial cash as baseline for first trade
    
    # Formula: daily_pnl = today_total_pnl - yesterday_total_pnl
    daily_pnl = total_pnl_overall - previous_total_pnl_overall
    
    # Calculate Daily % = (Today's Equity - Yesterday's Equity) / Yesterday's Equity * 100
    # Formula: daily_pct = ((today_equity - yesterday_equity) / yesterday_equity) * 100
    if yesterday_equity > 0:
        daily_pct = ((total_pv_equity - yesterday_equity) / yesterday_equity) * 100
    else:
        daily_pct = None  # Avoid division by zero
    
    # Calculate Cumulative % = ((Equity / Initial Cash) - 1) * 100
    # Formula: cumulative_pct = ((equity / initial_cash) - 1) * 100
    if cash > 0:
        cumulative_pct = ((total_pv_equity / cash) - 1) * 100
    else:
        cumulative_pct = None  # Avoid division by zero

    # Performance is the same as Cumulative %
    performance = cumulative_pct
    
    # Calculate Number of Trades = Track trades per ticker
    # Get trade number for this ticker
    trade_number = get_or_create_trade_number(ticker, old_q, new_q, action)
    
    # Format trade string
    trade_string = format_trade_string(action, current_position, trade_number, new_q)

    # Calculate Average Holding Days - only calculate when trade closes
    # Check if trade_string contains "- Close" to determine if this is a closing trade
    is_closing_trade = "- Close" in trade_string
    average_holding_days = calculate_average_holding_days(is_closing_trade=is_closing_trade, previous_df=previous_df)
    
    # Calculate total trades (max trade number that has been assigned)
    if trade_string == "No Buy/Sell":
        # Use previous total if available
        if len(previous_df) > 0:
            last_total = previous_df.iloc[-1].get('Total Trades', "No Buy/Sell")
            if isinstance(last_total, str) and "Trades" in last_total:
                total_trades_str = last_total
            else:
                total_trades_str = "No Buy/Sell"
        else:
            total_trades_str = "No Buy/Sell"
    else:
        # Total trades is the highest trade number opened so far
        # Formula: total_trades = next_trade_number - 1
        total_trades_str = f"{next_trade_number - 1} Trades"
    
    # Calculate Liquidation Price based on current position and quantity
    liquidation_price = calculate_liquidation_price(current_position, new_q, avg_p)
    
    # Calculate Take Profit and Stop Loss
    take_profit = calculate_take_profit(current_position, new_q, avg_p, take_profit_pct)
    stop_loss = calculate_stop_loss(current_position, new_q, avg_p, stop_loss_pct)
    
    # Calculate Win/Loss for closed trades (using trade_string to check if trade closed)
    win_loss = calculate_trade_win_loss(trade_string, realized_pnl_at_point)
    
    # Calculate Win Rate at this point
    win_rate = calculate_win_rate(previous_df, win_loss)
    
    # Calculate Win:Loss Ratio at this point
    win_loss_ratio = calculate_win_loss_ratio(previous_df, win_loss)
    
    # Calculate Trades/Month
    trades_per_month = calculate_trades_per_month(previous_df, date, trade_string)
    
    # Calculate Most/Least Traded
    abs_quantity_counts, most_traded_symbol, least_traded_symbol = calculate_most_least_traded(
        previous_df, ticker, q_in
    )
    
    # Calculate Avg Losing PnL and Avg Winning PnL
    avg_losing_pnl, avg_winning_pnl = calculate_avg_losing_winning_pnl(
        previous_df, realized_pnl_at_point
    )
    
    # Calculate Most/Least Profitable
    most_profitable, least_profitable = calculate_most_least_profitable(
        previous_df, ticker, realized_pnl_at_point
    )

    # Calculate Max Drawdown
    max_drawdown = calculate_max_drawdown(total_pv)

    # Calculate Total Gain (same as PnL Realized Cummulative)
    # Formula: total_gain = realized_pnl_cumulative
    total_gain = realized_pnl_cumulative
    
    # Calculate Average Gain = Total Gain / Total Trades
    # Formula: avg_gain = total_gain / total_trades_count
    # Total Trades = next_trade_number - 1 (number of trades that have been opened)
    total_trades_count = next_trade_number - 1
    if total_trades_count > 0:
        avg_gain = total_gain / total_trades_count
    else:
        avg_gain = None  # No trades opened yet
    
    # Update historical maximum investment
    update_max_investment_history(ticker, price, q_in, action, old_q)
    
    # Calculate Biggest Investment
    biggest_investment = calculate_biggest_investment()

    # Calculate new columns: Average Position, Holdings, Assets
    current_df = portfolio_state['portfolio_df'].copy() if portfolio_state['portfolio_df'] is not None else pd.DataFrame()
    avg_position = calculate_average_position(current_df, action, pv)
    holdings = calculate_holdings()

    # Update historical traded volume 
    update_traded_volume_history(price, q_in, action)

    # Calculate Highest/Lowest Traded Volume
    highest_traded_volume = get_highest_traded_volume()
    lowest_traded_volume = get_lowest_traded_volume()
    
    # Store asset type for this ticker
    if asset_type:
        portfolio_state['asset_types'][ticker] = asset_type

    # Calculate Asset Count
    asset_count = calculate_asset_count()
    
    # Increment Investment Count when opening a position:
    # - Long position: when action='buy' and position='long'
    # - Short position: when action='sell' and position='short'
    global investment_count
    action_lower = str(action).lower()
    position_lower = str(position).lower()
    
    # If Opening a Position (increments investment_count)
    if (position_lower == 'long' and action_lower == 'buy') or \
        (position_lower == 'short' and action_lower == 'sell'):
        investment_count += 1
    
    # Calculate Investment Count
    investment_count_value = calculate_investment_count()

    # Calculate YTD PnL (using cumulative realized PnL)
    ytd_pnl = calculate_ytd_pnl(previous_df, date, realized_pnl_cumulative)

    # Calculate Distribution and Distribution in %
    distribution, distribution_pct = calculate_diversification(total_pv, ticker_pv_dict)

    # Calculate Equity Distribution 
    # Store equity metadata (market cap, industry, sector) if Equity asset type
    if asset_type and asset_type.lower() == 'equity':
        # Get metadata from hardcoded mapping (or default to None if not found)
        metadata = EQUITY_METADATA.get(ticker, {})
        if metadata.get('market_cap'):
            portfolio_state['market_cap'][ticker] = metadata['market_cap']
        if metadata.get('industry'):
            portfolio_state['industry'][ticker] = metadata['industry']
        if metadata.get('sector'):
            portfolio_state['sector'][ticker] = metadata['sector']

    # Calculate Equity Distributions (only for Equity asset types)
    equity_dist_market_cap = calculate_equity_distribution_market_cap(ticker_pv_dict)
    equity_dist_industry = calculate_equity_distribution_industry(ticker_pv_dict)
    equity_dist_sector = calculate_equity_distribution_sector(ticker_pv_dict)

    # Build row dictionary with all calculated values
    row = {
        'Date': date,
        'Ticker': ticker,
        'Asset Type': asset_type.capitalize(),
        'Buy/Sell': action.capitalize(),
        'Position Taken': position.capitalize(),
        'Current Position': current_position.capitalize(),
        'Cash': cash,
        'Buyable/Sellable': buyable_sellable,
        'Quantity Buy': q_in,
        'Remaining': new_remaining,
        'Current Quantity': new_q,
        'Price': price,
        'Avg Price': avg_p,
        'Cost Basis': cb,
        'Position Value PV': pv,
        'PnL (Long) Unrealized': long_unrealized,
        'PnL (Short) Unrealized': short_unrealized,
        'Pnl Unrealized':open_unrealized_pnl,
        'PnL Unrealized Total Value for Current Ticker': total_current_ticker_unrealized,
        'Total Unrealized PnL': total_unrealized_all_tickers,
        'PV (Long)': pv_long_current,
        'PV (Short)': pv_short_current,
        'Open Position': open_pos,
        'Open PV': open_pv,
        'Total PV': total_pv,
        'Equity: Total PV + Remaining': total_pv_equity,
        'PnL Realized at Point of Time': realized_pnl_at_point,
        'PnL Realized Cummulative': realized_pnl_cumulative,
        'Total PnL Overall (Unrealized+Realized)': total_pnl_overall,
        'Daily PnL (Unrealized+Realized)': daily_pnl,
        'Liquidation Price': liquidation_price,
        'Take Profit': take_profit,
        'Stop Loss': stop_loss,
        'Last Day Pnl / Daily $': daily_pnl,
        'Daily %': daily_pct,
        'Cumulative %': cumulative_pct,
        'Investment Count': investment_count_value,
        'Performance': performance,
        'Asset Count': asset_count,
        'Trade No. (Position - Trade no. - Current Quantity)': trade_string,
        'Total Trades': total_trades_str,
        'Win/Loss': win_loss,
        'Win Rate': win_rate,
        'Win:Loss Ratio': win_loss_ratio,
        'Trades/Month': trades_per_month,
        'Absolute Quantity Counts': abs_quantity_counts,
        'Most Traded Symbol': most_traded_symbol,
        'Least Traded': least_traded_symbol,
        'Avg Losing PnL': avg_losing_pnl,
        'Avg Winning PnL': avg_winning_pnl,
        'Most Profitable': most_profitable,
        'Least Profitable': least_profitable,
        'Max Drawdown': max_drawdown,
        'Total Gain': total_gain,
        'Average Gain': avg_gain,
        'Biggest Investment': biggest_investment,
        'Average Position': avg_position,
        'Holdings': holdings,
        'YTD PnL': ytd_pnl,
        'Highest Traded Volume': highest_traded_volume,
        'Lowest Traded Volume': lowest_traded_volume,
        'Average Holding Days': average_holding_days,
        'Distribution': distribution,
        'Distribution in %': distribution_pct,
        'Equity Distribution (Market Cap)': equity_dist_market_cap, 
        'Equity Distribution (Industry)': equity_dist_industry,     
        'Equity Distribution (Sector)': equity_dist_sector,     
    }

    # Update global state
    portfolio_state['remaining'] = new_remaining
    portfolio_state['last_price'][ticker] = price

    # Append row to portfolio DataFrame
    df_row = pd.DataFrame([row], columns=COLUMNS)
    portfolio_state['portfolio_df'] = pd.concat(
        [portfolio_state['portfolio_df'], df_row],
        ignore_index=True
    )

    return row

    
def add_trade(ticker, asset_type, action, position, price, quantity_buy, date=None, take_profit_pct=0.20, stop_loss_pct=0.10):
    """
    Add a trade to the portfolio and return the updated DataFrame.
    
    This is a convenience wrapper around process_trade() that also returns the DataFrame.
    
    Args:
        ticker (str): Ticker symbol
        asset_type (str): Asset type (e.g., 'Stock', 'Crypto', 'ETF')
        action (str): Trade action ('buy', 'sell', 'hold')
        position (str): Position type ('long', 'short', 'hold')
        price (float): Trade price
        quantity_buy (float or str): Quantity to trade
        date (str or datetime, optional): Trade date
        take_profit_pct (float, optional): Take profit percentage (default 0.20 = 20%)
        stop_loss_pct (float, optional): Stop loss percentage (default 0.10 = 10%)
    
    Returns:
        pd.DataFrame: Updated portfolio DataFrame with all trades
    """
    process_trade(ticker, asset_type, action, position, price, quantity_buy, date, take_profit_pct, stop_loss_pct)
    return get_portfolio_df()