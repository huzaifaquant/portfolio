import io
from collections import defaultdict
from typing import Any, Optional
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, send_file

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
    'Date','Ticker', 'Asset Type','Side','Direction','Quantity Buy','Initial Balance','Buyable/Sellable',
    'Available Balance','Current Quantity','Price',
    'Avg Price','Cost Basis','Equity',
    'PnL (Long) Unrealized','PnL (Short) Unrealized','Pnl Unrealized','PnL Unrealized Total Value for Current Ticker','PnL realized Total Value for Current Ticker',
    'PnL Realized at Point of Time','PnL Unrealized at Point of Time',
    'Equity (Long)','Equity (Short)','Open Position','Open Equity',
    'Total Equity','Account Value','Realized PnL at Point of Time (Portfolio)','Unrealized PnL at Point of Time (Portfolio)','Total PnL Overall (Unrealized+Realized)',
    'Daily PnL (Unrealized+Realized)','Liquidation Price','Take Profit','Stop Loss', 
    'Last Day Pnl / Daily $', 'Daily %', 'Cumulative %', 'Investment Count', 'Performance', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Asset Count',
    'Trade No. (Position - Trade no. - Current Quantity)', 'Total Trades', 'Win/Loss', 'Win Rate', 'Win:Loss Ratio', 
    'Trades/Month', 'Absolute Quantity Counts', 'Most Traded Symbol', 'Most Bought', 'Least Traded',
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
        - Aggregators for Average Position Size (APS) and Average Holding Period (AHP)
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
        'sector': {},          # For Equity Distribution
        'max_investment_history': defaultdict(float),
        'highest_traded_volume': None,
        'lowest_traded_volume': None,
        'position_open_period': {},
        'position_open_date': {},          # For AHP: entry date per ticker
        'cumulative_holding_sum': 0.0,     # For AHP: Σ(holdingPeriod_i × closeQty_i)
        'closed_positions_count': 0,       # For AHP: Σ closeQty_i
        'current_period': 0,
        'previous_realized_pnl': 0.0,
        'aps_open_qty_sum': 0.0,           # For APS: Σ|openQty_i|
        'aps_trade_count': 0,              # For APS: N trades (positions opened)
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


# ---------- Date helper ----------

def normalize_trade_date(value: Any) -> Optional[str]:
    """
    Normalize any incoming date to a single string format '%m/%d/%Y'.

    - Accepts strings like '2025-01-02', '01/02/2025', '02/01/2025', '01-02-2025', etc.
    - Also accepts pandas Timestamps or other datetime-like values.
    - Returns None if the value is missing or cannot be parsed.
    """
    if value is None:
        return None

    # Try pandas first (handles many formats)
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if not pd.isna(dt):
            return dt.strftime("%m/%d/%Y")
    except Exception:
        pass

    from datetime import datetime

    s = str(value).strip()
    if not s:
        return None

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%m/%d/%Y")
        except ValueError:
            continue

    return None

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

def get_or_create_trade_number(ticker, old_q, new_q, side):
    """
    Determine trade number for a ticker based on position state.
    
    Trade numbering logic:
        - If old_q == 0 and new_q != 0: Opens new trade (assign new trade number)
        - If old_q != 0 and new_q != 0: Continues existing trade (keep same trade number)
        - If old_q != 0 and new_q == 0: Closes existing trade (keep same trade number, mark as closed)
        - If old_q == 0 and new_q == 0: No trade
        - If side == 'hold': Returns existing trade number if position exists, else None
    
    Args:
        ticker (str): Ticker symbol
        old_q (float): Quantity before this trade
        new_q (float): Quantity after this trade
        side (str): Trade side ('buy', 'sell', 'hold')
    
    Returns:
        int or None: Trade number if trade exists, None otherwise
    
    Edge cases:
        - Handles position flips (long to short, short to long)
        - Handles hold sides (maintains existing trade number)
    """
    global next_trade_number
    ticker = str(ticker).upper()
    
    side_lower = str(side).lower()
    
    # If side is 'hold', check if position exists
    if side_lower == 'hold':
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

def format_trade_string(side, current_direction, trade_number, new_q):
    """
    Format trade string for display.
    
    Format: "Direction - side - #TradeNo Trade - Quantity" or "... - 0 - Close"
    
    Args:
        side (str): Trade side ('buy', 'sell', 'hold')
        current_direction (str): Current direction ('long', 'short', 'hold')
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
    
    side_str = str(side)
    dir_str = str(current_direction)
    
    # Format quantity - use absolute value since direction already explains it
    if new_q == 0:
        quantity_str = "0 - Close"
    else:
        # Use absolute value for quantity
        quantity_str = str(int(abs(new_q)))
    
    # Include Buy/Sell in the output
    return f" {dir_str.capitalize()} - {side_str.capitalize()} - #{trade_number} Trade - {quantity_str}"

# ---------- Core Single-Trade Calculations ----------

def calculate_cash_single():
    """
    Get the initial cash amount (constant after initialization).
    
    Formula: Cash = Initial Cash (does not change)
    
    Returns:
        float: Initial cash amount
    """
    return portfolio_state['cash']

def calculate_remaining_single(side, price, q_in, old_quantity, old_cost_basis):
    """
    Calculate remaining cash after a trade.
    
    Remaining cash update formulas:
        - Buy Long:   remaining = previous_remaining - price * quantity
        - Sell Long:  remaining = previous_remaining + price * quantity
        - Sell Short: remaining = previous_remaining - price * quantity
        - Buy Short (Cover): remaining = previous_remaining + [initial + (initial - final)]
                              where initial = avg_price * close_qty, final = price * close_qty
    
    What is "Cover"?
        - Covering a short position means buying back the shares you borrowed and sold short
        - When you short sell, you borrow shares and sell them (you receive cash)
        - When you cover, you buy back those shares to close the short position (you pay cash)
        - The "cover" quantity is the number of shares you're buying back to close the short
        - Example: Short 10 shares at $100 → receive $1,000. Cover 10 shares at $90 → pay $900.
                   Net cash flow: $1,000 + ($1,000 - $900) = $1,100 (profit of $100)
    
    Position flips automatically when quantity crosses zero:
        - Long selling more than owned: closes long, opens short with excess
        - Short buying more than owed: covers short, opens long with excess
    
    Args:
        side (str): Trade side ('buy', 'sell', 'hold')
        price (float): Trade price
        q_in (float): Quantity traded
        old_quantity (float): Quantity before trade (positive for long, negative for short)
        old_cost_basis (float): Cost basis before trade
    
    Returns:
        float: Remaining cash after trade
    
    Edge cases:
        - Handles position flips (long to short, short to long)
        - Handles partial closes (closing some shares, keeping rest)
        - Uses previous avg price for closing calculations
        - When covering a short: only the shares actually owed are "covered"
          (if buying more than owed, excess opens a new long position)
    """
    rem = portfolio_state['remaining']
    a = str(side).lower()
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

def calculate_current_quantity_single(ticker, side, q_in, old_quantity):
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
        side (str): Trade side ('buy', 'sell', 'hold')
        position (str): Position type ('long', 'short', 'hold')
        q_in (float): Quantity traded
        old_quantity (float): Quantity before trade
    
    Returns:
        float: New quantity after trade (positive for long, negative for short)
    
    Edge cases:
        - Handles position flips naturally through arithmetic
        - Updates global quantities dictionary
    """
    a = str(side).lower()
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

def calculate_avg_price_and_cost_basis_single(ticker, side, price, q_in, old_quantity, new_quantity, old_cost_basis):
    """
    Calculate average price and cost basis for net position using segment-based logic.
    
    Segment Logic:
        - NEW Segment (=1): Opens when old_quantity == 0 and new_quantity != 0
        - SAME Segment (NO): Continues while adding to position (avg cost changes)
        - END Segment (YES): Closes when new_quantity == 0
        - FLIP (YES): Old segment ends, new segment starts with opening price
    
    Cost Basis Formulas by Situation:
        1. Open Long (NEW):     cb = price × qty
        2. Open Short (NEW):    cb = price × qty
        3. Add to Long (NO):    cb = ((old_cost + new_cost) / total_qty) × new_qty
        4. Add to Short (NO):   cb = ((old_cost + new_cost) / total_qty) × abs(new_qty)
        5. Full Close Long (YES):  cb = 0
        6. Full Close Short (YES): cb = 0
        7. Flip Long→Short (YES):  cb = new_short_qty × new_price
        8. Flip Short→Long (YES):  cb = new_long_qty × new_price
    
    Note: Partial closes are handled proportionally (not shown in table)
    
    Args:
        ticker (str): Ticker symbol
        side (str): Trade side ('buy', 'sell', 'hold')
        price (float): Trade price
        q_in (float): Quantity traded
        old_quantity (float): Quantity before trade
        new_quantity (float): Quantity after trade
        old_cost_basis (float): Cost basis before trade
    
    Returns:
        tuple: (avg_price, cost_basis) - Average price and cost basis after trade
    """
    a = str(side).lower()
    qty = abs(q_in) if q_in < 0 else q_in
    cb = old_cost_basis

    if a == 'hold':
        # No change
        pass
    
    elif a == 'buy':
        if old_quantity == 0:
            # Case 1: Open Long (NEW segment = 1)
            # Formula: cb = price × qty
            cb = price * qty
            
        elif old_quantity > 0:
            # Case 3: Add to Long (SAME segment, NO change in segment)
            # Formula: cb = ((old_cost + new_cost) / total_qty) × new_qty
            old_cost = old_cost_basis
            new_cost = price * qty
            total_qty = old_quantity + qty
            cb = ((old_cost + new_cost) / total_qty) * new_quantity
            
        elif old_quantity < 0:
            # Buying when short (covering)
            if new_quantity > 0:
                # Case 8: Flip Short→Long (NEW segment, YES change)
                # Formula: cb = new_long_qty × new_price
                new_long_qty = new_quantity
                cb = new_long_qty * price
                
            elif new_quantity == 0:
                # Case 6: Full Close Short (END segment, YES)
                # Formula: cb = 0
                cb = 0.0
                
            else:
                # Still short (partial cover) - proportional reduction
                to_cover = min(qty, abs(old_quantity))
                remaining_short = abs(old_quantity) - to_cover
                if remaining_short > 0:
                    cb = cb * (remaining_short / abs(old_quantity))
                else:
                    cb = 0.0

    elif a == 'sell':
        if old_quantity == 0:
            # Case 2: Open Short (NEW segment = 1)
            # Formula: cb = price × qty
            cb = price * qty
            
        elif old_quantity < 0:
            # Case 4: Add to Short (SAME segment, NO change)
            # Formula: cb = ((old_cost + new_cost) / total_qty) × abs(new_qty)
            old_cost = old_cost_basis
            new_cost = price * qty
            total_qty = abs(old_quantity) + qty
            cb = ((old_cost + new_cost) / total_qty) * abs(new_quantity)
            
        elif old_quantity > 0:
            # Selling when long
            if new_quantity < 0:
                # Case 7: Flip Long→Short (NEW segment, YES change)
                # Formula: cb = new_short_qty × new_price
                new_short_qty = abs(new_quantity)
                cb = new_short_qty * price
                
            elif new_quantity == 0:
                # Case 5: Full Close Long (END segment, YES)
                # Formula: cb = 0
                cb = 0.0
                
            else:
                # Still long (partial close) - proportional reduction
                sell_qty = qty
                remaining_long = old_quantity - sell_qty
                if remaining_long > 0:
                    cb = cb * (remaining_long / old_quantity)
                else:
                    cb = 0.0

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

def calculate_realized_pnl_at_point_of_time(ticker, side, position, price, q_in, old_quantity):
    """
    Calculate realized PnL at point of time for a specific closing side.
    
    Formula for Long positions (when selling):
        realized_pnl = (sell_price - avg_entry_price) * shares_closed
    
    Formula for Short positions (when buying/covering):
        realized_pnl = (avg_entry_price - cover_price) * shares_closed
    
    This is independent and dynamic, not dependent on cumulative calculations.
    Returns PnL for this specific closing side only.
    
    Args:
        ticker (str): Ticker symbol
        side (str): Trade side ('buy', 'sell', 'hold')
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
    a = str(side).lower()
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

def calculate_realized_pnl_cumulative(ticker, side, position, price, q_in, old_quantity):
    """
    Calculate cumulative realized PnL across ALL tickers.
    
    Formulas:
        - Closing long by selling: (sell_price - avg_entry) * shares_closed
        - Covering short by buying: (avg_entry - cover_price) * shares_closed
    
    Uses previous avg price from state (reads directly before updating).
    Updates global portfolio_state['realized_pnl'].
    
    Args:
        ticker (str): Ticker symbol
        side (str): Trade side ('buy', 'sell', 'hold')
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
    a = str(side).lower()
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
            - total_pv (float): Total Equity across all tickers
            - ticker_pv_dict (dict): {ticker: equity} for each ticker with position
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
    
    # Check if realized_pnl_at_point is None (can happen when position flips or no closing occurs)
    if realized_pnl_at_point is None:
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

def calculate_most_least_traded(previous_df, current_ticker, current_qty_buy, current_trade_string=None):
    """
    Calculate:
        - Absolute Quantity Counts per symbol (volume-based), and
        - Most/Least Traded symbols **by number of completed trades**.

    Definitions:
        - A trade is one full story of a position in a symbol: from opening (qty moves 0→≠0)
          until it is fully closed (qty goes back to 0).
        - We count a trade for a symbol when that trade **closes** (trade string contains "- Close").

    Logic:
        1) Absolute Quantity Counts (unchanged):
            cumulative_quantity[ticker] = Σ|Quantity Buy| across all rows (buys and sells).
        2) Trade counts per ticker:
            trade_count[ticker] = number of closed trades for that ticker
            (rows where Trade String includes "- Close").
        3) Most Traded Symbol: tickers ordered by trade_count descending.
        4) Least Traded: same trade_count logic, but ordered ascending.

    Args:
        previous_df (pd.DataFrame): Previous rows of portfolio DataFrame
        current_ticker (str): Current ticker being traded
        current_qty_buy (float): Current quantity buy value
        current_trade_string (str or None): Trade string for the current row

    Returns:
        tuple: (absolute_quantity_counts_str, most_traded, least_traded)

    Edge cases:
        - Returns ("None", "None", "None") if no trades
        - Handles invalid quantity values gracefully
        - Sorts by quantity (for Absolute Quantity Counts) and by trade count for Most/Least Traded
    """
    from collections import defaultdict

    # ---------- Absolute Quantity Counts (volume-based) ----------
    ticker_quantities = defaultdict(float)

    # Process previous rows - sum absolute quantities from "Quantity Buy" column
    if len(previous_df) > 0:
        for _, row in previous_df.iterrows():
            row_ticker = str(row.get('Ticker', '')).upper()
            row_qty_buy = row.get('Quantity Buy', 0)
            if row_ticker and row_qty_buy != 0:
                try:
                    qty = float(row_qty_buy)
                    ticker_quantities[row_ticker] += abs(qty)
                except Exception:
                    pass

    # Add current quantity buy
    if current_ticker and current_qty_buy != 0:
        try:
            ticker_quantities[current_ticker.upper()] += abs(float(current_qty_buy))
        except Exception:
            pass

    if not ticker_quantities:
        return None, "None", "None"

    # Sort tickers by quantity (descending), then alphabetically for ties
    sorted_tickers_desc = sorted(ticker_quantities.items(), key=lambda x: (-x[1], x[0]))

    # Absolute Quantity Counts string (ordered by quantity descending)
    abs_counts = [f"{ticker} {int(qty)}" for ticker, qty in sorted_tickers_desc]
    abs_counts_str = ", ".join(abs_counts) if abs_counts else "None"

    # ---------- Trade counts per ticker (Most/Least Traded) ----------
    # We count trades by distinct trade numbers per symbol, including
    # both open and closed trades.
    from collections import defaultdict
    import re

    trades_per_ticker = defaultdict(set)  # {ticker: {trade_numbers}}

    # Count distinct trades in previous rows
    if len(previous_df) > 0:
        for _, row in previous_df.iterrows():
            t = str(row.get('Ticker', '')).upper()
            trade_str = str(row.get('Trade No. (Position - Trade no. - Current Quantity)', ''))
            if not t or trade_str == "No Buy/Sell":
                continue
            m = re.search(r"#(\d+)", trade_str)
            if m:
                trades_per_ticker[t].add(int(m.group(1)))

    # Include current row (open or close) if it has a trade number
    if current_ticker and current_trade_string and current_trade_string != "No Buy/Sell":
        m = re.search(r"#(\d+)", str(current_trade_string))
        if m:
            trades_per_ticker[current_ticker.upper()].add(int(m.group(1)))

    trade_counts = {t: len(nums) for t, nums in trades_per_ticker.items() if nums}

    if not trade_counts:
        most_traded_str = "None"
        least_traded_str = "None"
    else:
        # Most Traded: order by trade count descending, then ticker
        sorted_by_most = sorted(trade_counts.items(), key=lambda x: (-x[1], x[0]))
        most_traded_list = [f"{ticker} {count}" for ticker, count in sorted_by_most]
        most_traded_str = ", ".join(most_traded_list) if most_traded_list else "None"

        # Least Traded: opposite order (ascending trade count)
        sorted_by_least = sorted(trade_counts.items(), key=lambda x: (x[1], x[0]))
        least_traded_list = [f"{ticker} {count}" for ticker, count in sorted_by_least]
        least_traded_str = ", ".join(least_traded_list) if least_traded_list else "None"

    return abs_counts_str, most_traded_str, least_traded_str


def calculate_most_bought(previous_df, current_ticker, current_direction, current_side, current_qty_buy):
    """
    Calculate Most Bought = stock/coin with the largest total quantity **opened**.

    We only count opening positions:
        - LONG & BUY   → opening a long
        - SHORT & SELL → opening a short

    Ignore closing positions:
        - LONG & SELL
        - SHORT & BUY

    Formula:
        total_open_qty[ticker] = Σ ABS(quantity) where
            (direction == LONG and side == BUY) OR
            (direction == SHORT and side == SELL)

    Args:
        previous_df (pd.DataFrame): Previous rows of portfolio DataFrame
        current_ticker (str): Current ticker being traded
        current_direction (str): Current direction argument ("long" / "short" / ...)
        current_side (str): Current Side / side ("buy" / "sell" / ...)
        current_qty_buy (float): Current quantity value

    Returns:
        str: "TICKER qty, ..." for the ticker(s) with the largest total opened quantity,
             or "None" if no opening trades exist.
    """
    from collections import defaultdict

    total_open_qty = defaultdict(float)

    # Helper to decide if a row is an opening trade
    def _is_opening(direction_val, side_val):
        d = str(direction_val).strip().lower()
        s = str(side_val).strip().lower()
        return (d == 'long' and s == 'buy') or (d == 'short' and s == 'sell')

    # Process previous rows
    if len(previous_df) > 0:
        for _, row in previous_df.iterrows():
            row_ticker = str(row.get('Ticker', '')).upper()
            row_direction = row.get('Direction', '')
            row_side = row.get('Side', '')
            row_qty = row.get('Quantity Buy', 0)

            if not row_ticker or row_qty == 0:
                continue

            if not _is_opening(row_direction, row_side):
                continue

            try:
                qty = abs(float(row_qty))
                total_open_qty[row_ticker] += qty
            except Exception:
                continue

    # Add current row if it is an opening trade
    if current_ticker and current_qty_buy != 0 and _is_opening(current_direction, current_side):
        try:
            qty = abs(float(current_qty_buy))
            total_open_qty[current_ticker.upper()] += qty
        except Exception:
            pass

    if not total_open_qty:
        return "None"

    # Find the maximum opened quantity
    max_qty = max(total_open_qty.values())
    # All tickers that tie for max
    winners = sorted([t for t, q in total_open_qty.items() if q == max_qty])

    parts = [f"{ticker} {int(max_qty)}" for ticker in winners]
    return ", ".join(parts) if parts else "None"

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
    # Get all previous realized PnL at point of time values (per-trade, irrespective of ticker)
    if len(previous_df) > 0 and 'PnL Realized at Point of Time' in previous_df.columns:
        previous_pnl_values = previous_df['PnL Realized at Point of Time'].tolist()
    elif len(previous_df) > 0 and 'Realized PnL at Point of Time (Portfolio)' in previous_df.columns:
        # Backward compatibility: older sheets may only have the cumulative portfolio column
        previous_pnl_values = previous_df['Realized PnL at Point of Time (Portfolio)'].tolist()
    else:
        previous_pnl_values = []

    # Add current value to the list
    if previous_pnl_values:
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
    
    # Get Total Equity values starting from row 2 (skip first row)
    # We need to get Total Equity from row 2 onwards, plus the current row
    total_pv_values = []
    
    # Get Total Equity values from previous rows (starting from row 2, skipping row 1)
    if 'Total Equity' in portfolio_state['portfolio_df'].columns:
        # Get all Total Equity values starting from row 2 (index 1 onwards)
        all_pv_values = portfolio_state['portfolio_df']['Total Equity'].tolist()
        if len(all_pv_values) > 0:
            # Skip first row (index 0), take from row 2 onwards (index 1 onwards)
            total_pv_values.extend(all_pv_values[1:])
    
    # Add current row's Total Equity (this will be row 2 or later)
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

def update_max_investment_history(ticker, price, quantity_buy, side, old_quantity):
    """
    Update the historical maximum investment for a ticker.
    
    Formula: investment_value = price * quantity_buy
    
    Logic:
        - Tracks maximum investment value (price * quantity) for each ticker
        - Only updates when opening/expanding positions (entry points)
        - Tracks both 'buy' sides (long positions) and 'sell' sides (short positions)
        - Does NOT update when closing positions
    
    Args:
        ticker (str): Ticker symbol
        price (float): Price of the trade
        quantity_buy (float): Quantity in the trade (positive for buy, positive for sell)
        side (str): Trade side ('buy', 'sell', 'hold')
        old_quantity (float): Quantity before this trade
    
    Edge cases:
        - Only tracks 'buy' sides for long positions
        - Only tracks 'sell' sides when old_quantity <= 0 (opening short)
        - Does not track when closing long positions (old_quantity > 0 and selling)
        - Does not update if investment_value is not greater than current max
    """
    side_lower = str(side).lower()
    qty = abs(quantity_buy) if quantity_buy < 0 else quantity_buy

    # Track maximum investment for:
    # 1. Buy sides (opening/expanding long positions)
    # 2. Sell sides that open/expand short positions (when old_quantity <= 0)
    if side_lower == 'buy':
        # Buy side: opening/expanding long position
        # Formula: investment_value = price * quantity
        investment_value = price * qty
        current_max = portfolio_state['max_investment_history'].get(ticker, 0.0)
        if investment_value > current_max:
            portfolio_state['max_investment_history'][ticker] = investment_value

    elif side_lower == 'sell':
        # Sell side: check if it's opening/expanding a short position
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

def calculate_average_position(is_opening_trade, open_quantity):
    """
    Calculate Average Position Size (APS) using open quantities.

    New definition:
        APS = (Σ|openQty_i|) / N

    Where:
        - openQty_i is the absolute open size (units) for trade i when the position was opened
        - N is the number of trades (summaries) included so far (number of position openings)

    Args:
        is_opening_trade (bool): Whether the current row opens a new position (old quantity == 0 and new quantity != 0)
        open_quantity (float or None): Absolute open quantity for the current trade (openQty_i), or None if not opening

    Returns:
        float or None: Current APS value, or None if no trades have been opened yet

    Edge cases:
        - Ignores rows that do not open a new position
        - Uses absolute value for all quantities
    """
    # Update aggregators when a new position is opened
    if is_opening_trade and open_quantity is not None:
        qty = abs(float(open_quantity))
        if qty > 0:
            portfolio_state['aps_open_qty_sum'] += qty
            portfolio_state['aps_trade_count'] += 1

    count = portfolio_state.get('aps_trade_count', 0)
    if count <= 0:
        return None

    return portfolio_state['aps_open_qty_sum'] / count
    
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

def update_traded_volume_history(price, quantity_buy, side):
    """
    Update the historical highest and lowest traded volume.
    
    Formula: traded_volume = quantity_buy * price
    
    Logic:
        - Tracks maximum and minimum traded volume across all trades
        - Updates when side is 'buy' or 'sell' (not 'hold')
        - Tracks based on the side (buy/sell), not the resulting position
        - This ensures we track trades even if they close positions (resulting in 'hold')
    
    Args:
        price (float): Price of the trade
        quantity_buy (float): Quantity in the trade
        side (str): Trade side ('buy', 'sell', 'hold')
        current_position (str): Current position after the trade ('long', 'short', 'hold')
    
    Edge cases:
        - Does not update if side is 'hold'
        - Uses absolute value of quantity_buy to handle negative inputs
        - Initializes to None, then sets to first traded volume
    """
    side_lower = str(side).lower()
    
    # Track if side is 'buy' or 'sell' (not 'hold')
    # This ensures we track trades even if they close positions (resulting in 'hold')
    if side_lower in ['buy', 'sell']:
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
    # If side is 'hold', don't update - keep last known values

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

def track_position_opening(ticker, current_period, open_date):
    """
    Track when a position is first opened for a ticker.

    Logic:
        - Records the period/row number when a position is first opened
        - Records the entry date for holding-period calculations
        - Only tracks if position is being opened (not already tracked)
        - Used later to calculate holding period when position closes

    Args:
        ticker (str): Ticker symbol
        current_period (int): Current period/row number (1-based)
        open_date (str or datetime): Trade date for the opening trade

    Edge cases:
        - Only tracks if ticker not already in position_open_period
        - Prevents overwriting existing tracking
        - If date parsing fails, period-based holding period is used as a fallback
    """
    if ticker not in portfolio_state['position_open_period']:
        portfolio_state['position_open_period'][ticker] = current_period
        if open_date is not None:
            try:
                portfolio_state['position_open_date'][ticker] = pd.to_datetime(open_date)
            except Exception:
                # If date parsing fails, we still track the period, but skip date-based AHP for this position
                pass

def detect_closed_positions(old_quantities, new_quantities, current_period, current_date):
    """
    Detect positions that were closed (quantity went from non-zero to zero).

    Logic:
        - Compares old_quantities and new_quantities to find positions that closed
        - Position closed if: old_quantity != 0 and new_quantity == 0
        - Calculates holding period in days for closed positions using entry/exit dates
        - Falls back to period-based holding period if dates are unavailable

    Formulas:
        - holding_period_days = (exit_date - entry_date) in days
        - closeQty_i = absolute quantity closed for this summary (|old_quantity|)

    Args:
        old_quantities (dict): Quantities before trade {ticker: quantity}
        new_quantities (dict): Quantities after trade {ticker: quantity}
        current_period (int): Current period/row number
        current_date (str or datetime): Trade date of the closing trade

    Returns:
        list: List of dicts with keys 'ticker', 'holding_period', 'closed_qty' for closed positions

    Edge cases:
        - Checks all tickers in both old and new quantities
        - Removes ticker from tracking when position closes
        - Returns empty list if no positions closed
        - If date parsing fails, uses period-based holding period
    """
    closed_positions = []

    # Try to parse current_date once
    exit_dt = None
    if current_date is not None:
        try:
            exit_dt = pd.to_datetime(current_date)
        except Exception:
            exit_dt = None

    # Check all tickers that were in old_quantities or new_quantities
    for ticker in set(list(old_quantities.keys()) + list(new_quantities.keys())):
        old_qty = old_quantities.get(ticker, 0)
        new_qty = new_quantities.get(ticker, 0)

        # Position closed if it went from non-zero to zero
        if old_qty != 0 and new_qty == 0:
            holding_period = None

            # Prefer date-based holding period if we have both entry and exit dates
            open_dt = portfolio_state.get('position_open_date', {}).get(ticker)
            if open_dt is not None and exit_dt is not None:
                holding_period = (exit_dt - open_dt).days
            else:
                # Fallback: period-based holding period (same as previous implementation)
                if ticker in portfolio_state['position_open_period']:
                    open_period = portfolio_state['position_open_period'][ticker]
                    holding_period = current_period - open_period + 1

            if holding_period is None:
                continue

            closed_qty = abs(old_qty)
            if closed_qty <= 0:
                continue

            closed_positions.append(
                {
                    'ticker': ticker,
                    'holding_period': holding_period,
                    'closed_qty': closed_qty,
                }
            )

            # Remove from tracking since it's closed
            portfolio_state['position_open_period'].pop(ticker, None)
            portfolio_state.get('position_open_date', {}).pop(ticker, None)

    return closed_positions

def update_average_holding_days(closed_positions):
    """
    Update accumulators used for Average Holding Period (AHP) when positions are closed.

    Logic:
        - For each closed summary trade i, updates:
            cumulative_holding_sum += holdingPeriod_i × closeQty_i
            closed_positions_count += closeQty_i
        - Used later to calculate quantity-weighted average holding period:
            AHP = Σ(holdingPeriod_i × closeQty_i) / Σ closeQty_i

    Args:
        closed_positions (list): List of dicts with 'ticker', 'holding_period', and 'closed_qty'

    Edge cases:
        - Does nothing if closed_positions is empty
        - Ignores entries with non-positive closed_qty
    """
    if closed_positions:
        for closed in closed_positions:
            holding_period = closed.get('holding_period')
            closed_qty = closed.get('closed_qty', 0)

            if holding_period is None or closed_qty <= 0:
                continue

            # Weighted sum: holdingPeriod_i × closeQty_i
            portfolio_state['cumulative_holding_sum'] += holding_period * closed_qty
            # Total quantity closed across all trades
            portfolio_state['closed_positions_count'] += closed_qty

def calculate_average_holding_days(is_closing_trade=False, previous_df=None):
    """
    Calculate Average Holding Period (AHP) for all closed trades.

    New definition:
        - For each closed summary trade i:
            holdingPeriod_i = (weighted exit date_i − weighted entry date_i) in days
            closeQty_i      = quantity closed in that summary
        - AHP = (Σ(holdingPeriod_i × closeQty_i)) / (Σ closeQty_i)

    Args:
        is_closing_trade (bool): Whether this trade row is closing a position (contains "- Close" in trade string)
        previous_df (pd.DataFrame): Previous rows (unused, kept for compatibility)

    Returns:
        float or None: Average holding period in days (rounded to 3 decimals), or None if insufficient data

    Edge cases:
        - Returns None if no quantity has been closed yet
        - Returns None if is_closing_trade is False (only show value when trade closes)
    """
    # Only calculate and return average when a trade is actually closing
    if not is_closing_trade:
        return None  # Don't show value for non-closing trades

    total_closed_qty = portfolio_state.get('closed_positions_count', 0)
    if total_closed_qty <= 0:
        return None  # No quantity closed yet

    # Formula: AHP = Σ(holdingPeriod_i × closeQty_i) / Σ closeQty_i
    avg = portfolio_state['cumulative_holding_sum'] / total_closed_qty
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
            - Long position: when side='buy' and position='long'
            - Short position: when side='sell' and position='short'
        - Does NOT increment when closing positions
        - Tracks cumulative count across all trades
    
    Formula: investment_count is incremented in process_trade() when:
        (position == 'long' and side == 'buy') OR (position == 'short' and side == 'sell')
    
    Returns:
        int: Cumulative investment count (number of positions opened)
    
    Edge cases:
        - Returns 0 if no positions have been opened
        - Only counts opening sides, not closing sides
    """
    global investment_count
    return investment_count


def calculate_sharpe_ratio(previous_df, current_realized_pnl_at_point, current_realized_pnl_cumulative, initial_balance, risk_free_rate=0.0):
    """Calculate Sharpe Ratio at this point in time.

    Formula:
        - Portfolio Return = Realized PnL at Point of Time (Portfolio) / Initial Balance
        - Risk Free Rate (default 0.0, configurable via parameter)
        - Denominator (Portfolio Return Std) = Std Dev of all valid 'PnL Realized at Point of Time' values
          from the start up to and including the current row.

        Sharpe Ratio = (Portfolio Return - Risk Free Rate) / StdDev(PnL Realized at Point of Time)

    Notes:
        - Ignores None and NaN values in 'PnL Realized at Point of Time'
        - Requires at least 2 valid observations to return a Sharpe value

    Args:
        previous_df (pd.DataFrame): Previous portfolio rows
        current_realized_pnl_at_point (float or None): Current row's realized PnL at point of time
        current_realized_pnl_cumulative (float): Current cumulative realized PnL (portfolio-level)
        initial_balance (float): Initial portfolio balance
        risk_free_rate (float, optional): Risk-free rate to subtract from portfolio return (default 0.0)

    Returns:
        float or None: Sharpe ratio value, or None if insufficient data.
    """

    # Need a valid initial balance
    if initial_balance is None or initial_balance == 0:
        return None

    # Build list of per-trade realized PnL values up to current row (drop None/NaN)
    values = []
    if previous_df is not None and len(previous_df) > 0 and 'PnL Realized at Point of Time' in previous_df.columns:
        ser = previous_df['PnL Realized at Point of Time']
        values.extend([float(v) for v in ser if pd.notnull(v)])

    # Include current value if valid
    if current_realized_pnl_at_point is not None and not (isinstance(current_realized_pnl_at_point, float) and np.isnan(current_realized_pnl_at_point)):
        values.append(float(current_realized_pnl_at_point))

    # Need at least 2 observations for a meaningful std dev
    if len(values) < 2:
        return None

    # Std dev of per-trade realized PnL (population std)
    std_val = float(np.std(values))
    if std_val == 0 or np.isnan(std_val):
        return None
    
    # User uses PnL in absolute terms; scale std down to be comparable with return
    std_val = std_val / 100.0

    # Portfolio return = cumulative realized PnL / initial balance
    portfolio_return = float(current_realized_pnl_cumulative) / float(initial_balance)

    # Excess return over risk free rate
    excess_return = portfolio_return - float(risk_free_rate)

    return excess_return / std_val


def calculate_sortino_ratio(previous_df, current_realized_pnl_at_point, current_realized_pnl_cumulative, initial_balance, risk_free_rate=0.0):
    """Calculate Sortino Ratio at this point in time.

    Same numerator as Sharpe, but denominator is **downside deviation** (std of negative returns).

    Implementation here mirrors the Sharpe ratio setup:
        - We use per-trade "returns" implicitly in % units by
          taking raw 'PnL Realized at Point of Time' and scaling std by 100.
        - Downside std is computed only from negative PnL values.

    If there is portfolio return but **no negative PnL values yet**, the Sortino Ratio
    is conceptually infinite, so this function returns ``float('inf')``.

    Args:
        previous_df (pd.DataFrame): Previous portfolio rows
        current_realized_pnl_at_point (float or None): Current row's realized PnL at point of time
        current_realized_pnl_cumulative (float): Current cumulative realized PnL (portfolio-level)
        initial_balance (float): Initial portfolio balance
        risk_free_rate (float, optional): Risk-free rate to subtract from portfolio return (default 0.0)

    Returns:
        float or None: Sortino ratio value, ``float('inf')`` if no downside and positive return,
                       or None if insufficient data.
    """

    # Need a valid initial balance
    if initial_balance is None or initial_balance == 0:
        return None

    # Build list of per-trade realized PnL values up to current row
    values = []
    if previous_df is not None and len(previous_df) > 0 and 'PnL Realized at Point of Time' in previous_df.columns:
        ser = previous_df['PnL Realized at Point of Time']
        values.extend([float(v) for v in ser if pd.notnull(v)])

    if current_realized_pnl_at_point is not None and not (isinstance(current_realized_pnl_at_point, float) and np.isnan(current_realized_pnl_at_point)):
        values.append(float(current_realized_pnl_at_point))

    # Extract downside values (negative PnL only)
    downside = [v for v in values if v < 0]

    # Portfolio return (same as Sharpe numerator)
    portfolio_return = float(current_realized_pnl_cumulative) / float(initial_balance)
    excess_return = portfolio_return - float(risk_free_rate)

    # If we have return but no downside values yet -> infinite Sortino
    if len(downside) == 0:
        return float('inf') if excess_return != 0 else None

    if len(downside) < 2:
        # Not enough downside observations for a stable std
        return None

    # Downside deviation (std of negative PnL values)
    downside_std = float(np.std(downside))
    if downside_std == 0 or np.isnan(downside_std):
        return None

    # Keep scaling consistent with Sharpe (Pnl-based, scaled down by 100)
    downside_std = downside_std / 100.0

    return excess_return / downside_std


def calculate_calmar_ratio(previous_df, current_account_value, current_date, initial_balance, debug=False):
    """Calculate Calmar Ratio using Account Value, maximum drawdown (MDD), and annualized return (AAR).

    Calmar = ARR / MDD

    Where:
        - ARR (here) = Annualized Account Return based on Account Value and Date
          ARR = (Account Value / Initial Balance) ** (365 / days) - 1
          where days is the number of days from first trade date to current date.
        - MDD is computed from the history of Account Value using peak/trough logic:
            * Peak updates when account value makes a new high.
            * Trough updates when price falls below the last peak and makes a new low
              before the next peak.

    Args:
        previous_df (pd.DataFrame): Previous portfolio rows
        current_account_value (float): Current Account Value
        current_date (str or datetime): Date of the current row
        initial_balance (float): Initial portfolio balance

    Returns:
        float or None: Calmar ratio, ``float('inf')`` if positive ARR but MDD is 0,
                       or 0.0 if no drawdown and no gain/loss, or None if no data.
    """

    if initial_balance is None or initial_balance == 0:
        if debug:
            print("Calmar debug - invalid initial_balance", initial_balance)
        return None

    # Build full history of valid (non-NaN) account values up to current row
    values = []
    if previous_df is not None and len(previous_df) > 0 and 'Account Value' in previous_df.columns:
        ser_val = previous_df['Account Value']
        values = [float(v) for v in ser_val if pd.notnull(v)]

    if current_account_value is not None and not (isinstance(current_account_value, float) and np.isnan(current_account_value)):
        values.append(float(current_account_value))

    if debug:
        print("Calmar debug - account values (history):", values)

    if not values:
        if debug:
            print("Calmar debug - no valid account values")
        return None

    # Build date history to compute days between first trade and current date
    date_list = []
    if previous_df is not None and len(previous_df) > 0 and 'Date' in previous_df.columns:
        for d in previous_df['Date']:
            if pd.notnull(d):
                try:
                    date_list.append(pd.to_datetime(d))
                except Exception:
                    continue

    if current_date is not None:
        try:
            date_list.append(pd.to_datetime(current_date))
        except Exception:
            pass

    if date_list:
        start_date = min(date_list)
        end_date = max(date_list)
        days = (end_date - start_date).days
    else:
        days = None

    if debug:
        print("Calmar debug - dates (history):", date_list)
        print("Calmar debug - days:", days)

    # Peak/trough traversal for MDD (using Account Value history)
    peak = values[0]
    trough = values[0]
    max_drawdown = 0.0

    for v in values[1:]:
        if v > peak:
            peak = v
            trough = v
        elif v < trough:
            trough = v
            if peak > 0:
                dd = (peak - trough) / peak
                if dd > max_drawdown:
                    max_drawdown = dd

    if debug:
        print("Calmar debug - peak:", peak)
        print("Calmar debug - trough:", trough)
        print("Calmar debug - max_drawdown:", max_drawdown)

    # Ratio of current account value to initial
    ratio = float(values[-1]) / float(initial_balance)

    # Annualized Account Return (ARR)
    if days is not None and days > 0:
        try:
            arr = ratio ** (365.0 / float(days)) - 1.0
        except Exception:
            arr = ratio - 1.0
    else:
        # Fallback: simple return if we cannot compute days
        arr = ratio - 1.0

    if debug:
        print("Calmar debug - ratio:", ratio)
        print("Calmar debug - arr (annualized):", arr)

    if max_drawdown == 0:
        # No drawdown yet: infinite if ARR>0 (profit), 0.0 if ARR==0 (flat), None if loss
        if arr > 0.0:
            if debug:
                print("Calmar debug - result: inf (no drawdown, positive ARR)")
            return float('inf')
        elif arr == 0.0:
            if debug:
                print("Calmar debug - result: 0.0 (no drawdown, flat ARR)")
            return 0.0
        else:
            if debug:
                print("Calmar debug - result: None (no drawdown, negative ARR)")
            return None

    result = arr / max_drawdown
    if debug:
        print("Calmar debug - result (Calmar):", result)

    return result


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
                start_of_year_cumulative = row.get('Realized PnL at Point of Time (Portfolio)', 0.0)
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

# Automatic asset type mapping for common tickers (extend as needed)
ASSET_TYPE_MAP = {
    # Large-cap U.S. equities / major S&P 500 names
    'AAPL': 'Equity', 'MSFT': 'Equity', 'GOOGL': 'Equity', 'GOOG': 'Equity',
    'AMZN': 'Equity', 'META': 'Equity', 'NVDA': 'Equity', 'TSLA': 'Equity',
    'BRK.B': 'Equity', 'BRK.A': 'Equity', 'JPM': 'Equity', 'V': 'Equity',
    'JNJ': 'Equity', 'PG': 'Equity', 'XOM': 'Equity', 'CVX': 'Equity',
    'HD': 'Equity', 'MA': 'Equity', 'UNH': 'Equity', 'PFE': 'Equity',
    'DIS': 'Equity', 'KO': 'Equity', 'PEP': 'Equity', 'INTC': 'Equity',
    'NFLX': 'Equity', 'ADBE': 'Equity', 'ORCL': 'Equity', 'CSCO': 'Equity',

    # Broad index / ETF tickers (treated as Equity asset type here)
    'SPY': 'Equity', 'IVV': 'Equity', 'VOO': 'Equity', 'QQQ': 'Equity',

    # Popular crypto tickers
    'BTC': 'Crypto', 'BTCUSD': 'Crypto', 'BTC-USDT': 'Crypto',
    'ETH': 'Crypto', 'ETHUSD': 'Crypto', 'ETH-USDT': 'Crypto',
    'SOL': 'Crypto', 'SOLUSD': 'Crypto',
    'BNB': 'Crypto', 'XRP': 'Crypto', 'ADA': 'Crypto',
    'DOGE': 'Crypto', 'DOGEUSD': 'Crypto',
}

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

def process_trade(ticker, asset_type, side, price, quantity_buy, date=None, take_profit_pct=0.20, stop_loss_pct=0.10):
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
        side (str): Trade side ('buy', 'sell', 'hold')
        direction (str): Direction type ('long', 'short', 'hold')
        price (float): Trade price
        quantity_buy (float or str): Quantity to trade (can be numeric or string like "-(-10)")
        date (str or datetime, optional): Trade date
        take_profit_pct (float, optional): Take profit percentage (default 0.20 = 20%)
        stop_loss_pct (float, optional): Stop loss percentage (default 0.10 = 10%)
    
    Returns:
        dict: Row dictionary with all calculated values
    
    Raises:
        ValueError: If attempting to open a different direction type (long vs short) on the same ticker
                   when a position already exists. Must close current direction first.
    
    Edge cases:
        - Handles direction flips (long to short, short to long)
        - Handles partial direction closes
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
    
    # Calculate what the new quantity would be to check for natural flips
    # We need this to distinguish between natural flips and explicit opposite position opening
    a = str(side).lower()
    qty = abs(q_in) if q_in < 0 else q_in
    
    if a == 'hold':
        new_q_calc = old_q
    elif a == 'buy':
        new_q_calc = old_q + qty
    elif a == 'sell':
        new_q_calc = old_q - qty
    else:
        new_q_calc = old_q
    

    # Calculate cash and remaining
    cash = calculate_cash_single()
    new_remaining = calculate_remaining_single(side, price, q_in, old_q, old_cb)

    # Update quantity and determine current position
    new_q = calculate_current_quantity_single(ticker, side, q_in, old_q)
    
    # Track position opening and closing for AHP / APS
    # Get current period (row number) - 1-indexed
    previous_df = portfolio_state['portfolio_df']
    current_period = len(previous_df) + 1  # Current row number

    # Determine if this trade opens a new position (summary trade)
    is_opening_trade = (old_q == 0 and new_q != 0)
    
    # Track position opening if this is a new position (old_q == 0 and new_q != 0)
    if is_opening_trade:
        track_position_opening(ticker, current_period, date)
    
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
    
    # Detect closed positions (to feed AHP)
    closed_positions = detect_closed_positions(
        old_quantities_dict,
        new_quantities_dict,
        current_period,
        date
    )
    
    # Update Average Holding Period aggregators when positions close
    if closed_positions:
        update_average_holding_days(closed_positions)
    
    # Determine current direction based on new quantity
    # Rule: if current quantity >= 0 → long, else → short
    if new_q < 0:
        current_direction = 'short'
    else:
        current_direction = 'long'

    # Determine previous position based on quantity before trade (for realized PnL)
    if old_q < 0:
        prev_position = 'short'
    elif old_q > 0:
        prev_position = 'long'
    else:
        prev_position = 'hold'

    # Calculate realized PnL at point of time (independent calculation)
    realized_pnl_at_point = calculate_realized_pnl_at_point_of_time(
        ticker, side, prev_position, price, q_in, old_q
    )
    
    # Calculate cumulative realized PnL (updates global state)
    realized_pnl_cumulative = calculate_realized_pnl_cumulative(
        ticker, side, prev_position, price, q_in, old_q
    )
    
    # Calculate average price and cost basis
    avg_p, cb = calculate_avg_price_and_cost_basis_single(
        ticker, side, price, q_in, old_q, new_q, old_cb  
    )
    # Calculate buyable/sellable shares
    # Formula: buyable_sellable = prev_remaining_cash / price
    previous_remaining = portfolio_state['remaining']
    buyable_sellable = (previous_remaining / price) if price > 0 else 0.0
    
    # Calculate position value and unrealized PnL components
    pv = position_value_from_position(current_direction, new_q, price)
    long_unrealized, short_unrealized, total_current_ticker_unrealized, total_unrealized_all_tickers = pnl_unrealized_components(new_q, price, avg_p, ticker, price)

    # Calculate realized total value for current ticker (cumulative realized PnL per ticker)
    if len(previous_df) > 0 and 'PnL realized Total Value for Current Ticker' in previous_df.columns:
        prev_ticker_rows = previous_df[previous_df['Ticker'].str.upper() == ticker]
        if not prev_ticker_rows.empty:
            prev_realized_total = prev_ticker_rows.iloc[-1]['PnL realized Total Value for Current Ticker'] or 0.0
        else:
            prev_realized_total = 0.0
    else:
        prev_realized_total = 0.0
    realized_pnl_total_current_ticker = prev_realized_total + (realized_pnl_at_point or 0.0)
    
    # Generate strings for open positions
    open_pos = open_positions_str()
    open_pv = open_pv_str(ticker, price, current_direction)
    open_unrealized_pnl = open_pnl_unrealized_str(ticker, price) 

    # Calculate PV for current ticker only
    pv_long_current, pv_short_current = calculate_pv_for_current_ticker(price, current_direction, new_q, avg_p, cb)
    
    # Calculate total PV across all tickers
    total_pv, ticker_pv_dict = calculate_total_pv_all_tickers(ticker, price)
    
    # Calculate account value
    # Formula: account_value = total_equity + available_balance
    total_pv_equity = total_pv + new_remaining
    
    # Calculate Total PnL Overall
    # Formula: total_pnl_overall = equity - initial_cash
    total_pnl_overall = total_pv_equity - cash
    
    # Calculate Daily PnL = Today's Total PnL Overall - Yesterday's Total PnL Overall
    # Get previous row's Total PnL Overall if it exists, otherwise 0 (first trade)
    if len(previous_df) > 0:
        previous_total_pnl_overall = previous_df.iloc[-1]['Total PnL Overall (Unrealized+Realized)']
        # Get yesterday's account value for Daily % calculation
        yesterday_equity = previous_df.iloc[-1]['Account Value']
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

    # Sharpe Ratio based on cumulative realized PnL and per-trade realized PnL history
    # risk_free_rate kept at 0.0 by default (can be changed by passing a different value)
    sharpe_ratio = calculate_sharpe_ratio(
        previous_df,
        realized_pnl_at_point,
        realized_pnl_cumulative,
        cash,
        risk_free_rate=0.0
    )

    # Sortino Ratio using downside deviation of negative PnL values
    sortino_ratio = calculate_sortino_ratio(
        previous_df,
        realized_pnl_at_point,
        realized_pnl_cumulative,
        cash,
        risk_free_rate=0.0
    )

    # Calmar Ratio using annualized account return and MDD
    calmar_ratio = calculate_calmar_ratio(
        previous_df,
        total_pv_equity,
        date,
        cash
    )
    
    # Calculate Number of Trades = Track trades per ticker
    # Get trade number for this ticker
    trade_number = get_or_create_trade_number(ticker, old_q, new_q, side)
    
    # Format trade string
    trade_string = format_trade_string(side, current_direction, trade_number, new_q)

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
    
    # Calculate Liquidation Price based on current direction and quantity
    liquidation_price = calculate_liquidation_price(current_direction, new_q, avg_p)
    
    # Calculate Take Profit and Stop Loss
    take_profit = calculate_take_profit(current_direction, new_q, avg_p, take_profit_pct)
    stop_loss = calculate_stop_loss(current_direction, new_q, avg_p, stop_loss_pct)
    
    # Calculate Win/Loss for closed trades (using trade_string to check if trade closed)
    win_loss = calculate_trade_win_loss(trade_string, realized_pnl_at_point)
    
    # Calculate Win Rate at this point
    win_rate = calculate_win_rate(previous_df, win_loss)
    
    # Calculate Win:Loss Ratio at this point
    win_loss_ratio = calculate_win_loss_ratio(previous_df, win_loss)
    
    # Calculate Trades/Month
    trades_per_month = calculate_trades_per_month(previous_df, date, trade_string)
    
    # Calculate Most/Least Traded (by number of completed trades per symbol)
    abs_quantity_counts, most_traded_symbol, least_traded_symbol = calculate_most_least_traded(
        previous_df, ticker, q_in, trade_string
    )

    # Calculate Most Bought (only opening positions: LONG+BUY, SHORT+SELL)
    most_bought = calculate_most_bought(previous_df, ticker, current_direction, side, q_in)
    
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

    # Calculate Total Gain using realized + unrealized PnL at this point in time
    # Formula: total_gain = PnL Realized Cummulative + PnL Unrealized at Point of Time (all tickers)
    total_gain = realized_pnl_cumulative + total_unrealized_all_tickers
    
    # Calculate Average Gain = Total Gain / Total Trades
    # Formula: avg_gain = total_gain / total_trades_count
    # Total Trades = next_trade_number - 1 (number of trades that have been opened)
    total_trades_count = next_trade_number - 1
    if total_trades_count > 0:
        avg_gain = total_gain / total_trades_count
    else:
        avg_gain = None  # No trades opened yet
    
    # Update historical maximum investment
    update_max_investment_history(ticker, price, q_in, side, old_q)
    
    # Calculate Biggest Investment
    biggest_investment = calculate_biggest_investment()

    # Calculate new columns: Average Position Size (APS), Holdings, Assets
    open_quantity = abs(new_q) if is_opening_trade else None
    avg_position = calculate_average_position(is_opening_trade, open_quantity)
    holdings = calculate_holdings()

    # Update historical traded volume 
    update_traded_volume_history(price, q_in, side)

    # Calculate Highest/Lowest Traded Volume
    highest_traded_volume = get_highest_traded_volume()
    lowest_traded_volume = get_lowest_traded_volume()
    
    # Store asset type for this ticker (auto-map if not provided)
    inferred_asset_type = asset_type
    if not inferred_asset_type:
        inferred_asset_type = ASSET_TYPE_MAP.get(ticker)

    if inferred_asset_type:
        portfolio_state['asset_types'][ticker] = inferred_asset_type

    # Calculate Asset Count
    asset_count = calculate_asset_count()
    
    # Increment Investment Count when opening a direction:
    # - Long direction: when side='buy' and direction='long'
    # - Short direction: when side='sell' and direction='short'
    global investment_count
    side_lower = str(side).lower()
    direction_lower = str(current_direction).lower()
    
    # If Opening a Direction (increments investment_count)
    if (direction_lower == 'long' and side_lower == 'buy') or \
        (direction_lower == 'short' and side_lower == 'sell'):
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
        'Asset Type': (inferred_asset_type or asset_type or '').capitalize(),
        'Side': side.capitalize(),
        'Direction': current_direction.capitalize(),
        'Initial Balance': cash,
        'Buyable/Sellable': buyable_sellable,
        'Quantity Buy': q_in,
        'Available Balance': new_remaining,
        'Current Quantity': new_q,
        'Price': price,
        'Avg Price': avg_p,
        'Cost Basis': cb,
        'Equity': pv,
        'PnL (Long) Unrealized': long_unrealized,
        'PnL (Short) Unrealized': short_unrealized,
        'Pnl Unrealized':open_unrealized_pnl,
        'PnL Unrealized Total Value for Current Ticker': total_current_ticker_unrealized,
        'PnL realized Total Value for Current Ticker': realized_pnl_total_current_ticker,
        'PnL Realized at Point of Time': realized_pnl_at_point,
        'PnL Unrealized at Point of Time': total_unrealized_all_tickers,
        'Equity (Long)': pv_long_current,
        'Equity (Short)': pv_short_current,
        'Open Position': open_pos,
        'Open Equity': open_pv,
        'Total Equity': total_pv,
        'Account Value': total_pv_equity,
        'Realized PnL at Point of Time (Portfolio)': realized_pnl_cumulative,
        'Unrealized PnL at Point of Time (Portfolio)': total_unrealized_all_tickers,
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
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Asset Count': asset_count,
        'Trade No. (Position - Trade no. - Current Quantity)': trade_string,
        'Total Trades': total_trades_str,
        'Win/Loss': win_loss,
        'Win Rate': win_rate,
        'Win:Loss Ratio': win_loss_ratio,
        'Trades/Month': trades_per_month,
        'Absolute Quantity Counts': abs_quantity_counts,
        'Most Traded Symbol': most_traded_symbol,
        'Most Bought': most_bought,
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

    
def add_trade(ticker, asset_type=None, side='buy', price=0.0, quantity_buy=0.0, date=None, take_profit_pct=0.20, stop_loss_pct=0.10):
    """
    Add a trade to the portfolio and return the updated DataFrame.
    
    This is a convenience wrapper around process_trade() that also returns the DataFrame.
    
    Args:
        ticker (str): Ticker symbol
        asset_type (str): Asset type (e.g., 'Stock', 'Crypto', 'ETF')
        side (str): Trade side ('buy', 'sell', 'hold')
        direction (str): Direction type ('long', 'short', 'hold')
        price (float): Trade price
        quantity_buy (float or str): Quantity to trade
        date (str or datetime, optional): Trade date
        take_profit_pct (float, optional): Take profit percentage (default 0.20 = 20%)
        stop_loss_pct (float, optional): Stop loss percentage (default 0.10 = 10%)
    
    Returns:
        pd.DataFrame: Updated portfolio DataFrame with all trades
    """
    process_trade(ticker, asset_type, side, price, quantity_buy, date, take_profit_pct, stop_loss_pct)
    return get_portfolio_df()

# ---------- Formulas ----------

def get_formulas_dict():
    """
    Return a dictionary mapping column names to their formulas.
    
    Returns:
        dict: {column_name: formula_string} for all columns
    """
    return {
        'Date': 'Trade Date',
        'Ticker': 'Ticker Symbol',
        'Asset Type': 'Asset Type (Equity, Crypto, etc.)',
        'Side': 'Trade side (Buy, Sell, Hold)',
        'Direction': 'Derived Direction: Long if Qty>=0, Short if Qty<0',
        'Initial Balance': 'Initial Balance (Constant)',
        'Buyable/Sellable': 'Previous Available Balance / Price',
        'Quantity Buy': 'Quantity Traded (Absolute Value)',
        'Available Balance': 'Previous Available Balance + Trade Cash Flow',
        'Current Quantity': 'Previous Qty + Trade Qty (Buy adds, Sell subtracts)',
        'Price': 'Trade Price',
        'Avg Price': 'Cost Basis / |Quantity| (when Qty≠0)',
        'Cost Basis': 'Sum of Entry Prices * Quantities',
        'Equity': 'Long: Qty*Price, Short: |Qty|*Price',
        'PnL (Long) Unrealized': '(Price - Avg Price) * Quantity (when Qty>0)',
        'PnL (Short) Unrealized': '(Avg Price - Price) * |Quantity| (when Qty<0)',
        'Pnl Unrealized': 'Long + Short for Current Ticker',
        'PnL Unrealized Total Value for Current Ticker': 'Long + Short Unrealized PnL',
        'PnL realized Total Value for Current Ticker': 'Sum of PnL Realized at Point of Time for this ticker up to current row',
        'PnL Realized at Point of Time': 'Realized PnL for this specific closing trade (portfolio-level, irrespective of ticker)',
        'PnL Unrealized at Point of Time': 'Sum of Unrealized PnL across ALL tickers at this point in time (portfolio-level, irrespective of ticker)',
        'PV (Long)': 'Cost Basis + (Price - Avg Price) * Qty',
        'PV (Short)': 'Cost Basis + (Avg Price - Price) * |Qty|',
        'Open Position': 'String of All Open Positions',
        'Open Equity': 'String of All Open Position Equities',
        'Total Equity': 'Σ(Equity for each ticker) where Equity = Cost Basis + Unrealized PnL',
        'Account Value': 'Total Equity + Available Balance',
        'Realized PnL at Point of Time (Portfolio)': 'Sum of All Realized PnL up to this point (portfolio-level)',
        'Unrealized PnL at Point of Time (Portfolio)': 'Sum of Unrealized PnL across ALL tickers at this point in time (portfolio-level)',
        'Total PnL Overall (Unrealized+Realized)': 'Equity - Initial Balance',
        'Daily PnL (Unrealized+Realized)': 'Today Total PnL Overall - Yesterday Total PnL Overall',
        'Liquidation Price': 'Long: 0, Short: 2 * Avg Price',
        'Take Profit': 'Long: Avg Price * (1 + %), Short: Avg Price * (1 - %)',
        'Stop Loss': 'Long: Avg Price * (1 - %), Short: Avg Price * (1 + %)',
        'Last Day Pnl / Daily $': 'Same as Daily PnL',
        'Daily %': '((Today Equity - Yesterday Equity) / Yesterday Equity) * 100',
        'Cumulative %': '((Equity / Initial Balance) - 1) * 100',
        'Investment Count': 'Cumulative Count of Positions Opened',
        'Performance': 'Same as Cumulative %',
        'Sharpe Ratio': 'Sharpe = ((Realized PnL at Point of Time (Portfolio) / Initial Balance) - Risk Free Rate) / StdDev(PnL Realized at Point of Time)',
        'Sortino Ratio': 'Sortino = ((Realized PnL at Point of Time (Portfolio) / Initial Balance) - Risk Free Rate) / DownsideDeviation(negative PnL Realized at Point of Time)',
        'Asset Count': 'Count of Distinct Tickers per Asset Type where Qty≠0',
        'Trade No. (Position - Trade no. - Current Quantity)': 'Formatted Trade String',
        'Total Trades': 'Max Trade Number Assigned',
        'Win/Loss': 'Win if Realized PnL>0, Loss if ≤0, None if No Close',
        'Win Rate': '(Wins / Total Closed Trades) * 100',
        'Win:Loss Ratio': 'Win Count : Loss Count',
        'Trades/Month': 'Count of Open Trades in Current Month',
        'Absolute Quantity Counts': 'Σ|Quantity Buy| per Ticker',
        'Most Traded Symbol': 'Symbol(s) with the highest number of trades (distinct trade numbers, open or closed)',
        'Most Bought': 'Ticker(s) with largest ΣABS(quantity) over opening trades (LONG+BUY or SHORT+SELL)',
        'Least Traded': 'Symbol(s) with the lowest number of trades (same logic as Most Traded, opposite order)',
        'Avg Losing PnL': 'Sum(Losing PnL) / Count (where PnL<0)',
        'Avg Winning PnL': 'Sum(Winning PnL) / Count (where PnL>0)',
        'Most Profitable': 'Ticker with Highest Realized PnL',
        'Least Profitable': 'Ticker with Lowest Realized PnL',
        'Max Drawdown': '(MAX(Total Equity from row 2 to current) - MIN(Total Equity from row 2 to current)) / MAX(Total Equity from row 2 to current)',
        'Total Gain': 'PnL Realized Cummulative + PnL Unrealized at Point of Time (all tickers)',
        'Average Gain': 'Total Gain / Total Trades',
        'Biggest Investment': 'Max(Price * Quantity Buy) Historically per Ticker',
        'Average Position': 'Average Position Size (APS) = Σ|openQty_i| / N (number of opened trades)',
        'Holdings': 'Count of Unique Tickers with Non-Zero Current Quantity',
        'YTD PnL': 'Current Cumulative Realized PnL - Cumulative Realized PnL on Jan 1',
        'Highest Traded Volume': 'Max(Price * |Quantity Buy|) Historically',
        'Lowest Traded Volume': 'Min(Price * |Quantity Buy|) Historically',
        'Average Holding Days': 'Average Holding Period (AHP) = Σ(holdingPeriod_i × closeQty_i) / Σ closeQty_i',
        'Distribution': 'AssetType1: TotalPV1, AssetType2: TotalPV2, ...',
        'Distribution in %': 'AssetType1: (TotalPV1/TotalPV)*100%, AssetType2: (TotalPV2/TotalPV)*100%, ...',
        'Equity Distribution (Market Cap)': '(PV in Market Cap Category / Total PV) * 100',
        'Equity Distribution (Industry)': '(PV in Industry / Total PV) * 100',
        'Equity Distribution (Sector)': '(PV in Sector / Total PV) * 100'
    }

def add_formulas_row_to_df(df=None):
    """
    Add a formulas row to the portfolio DataFrame.
    
    This function inserts a row containing formulas for each column at the top of the DataFrame
    (right after column headers, before actual data rows). This is for display purposes only
    and should be called after all trades are processed.
    
    Args:
        df (pd.DataFrame, optional): Portfolio DataFrame. If None, uses current portfolio_df from state.
    
    Returns:
        pd.DataFrame: DataFrame with formulas row inserted at index 0
    
    Note:
        - This function does NOT modify the global portfolio_state['portfolio_df']
        - It returns a new DataFrame with the formulas row added
        - Call this function when you want to display the DataFrame with formulas
    """
    if df is None:
        df = portfolio_state['portfolio_df'].copy()
    
    # Get formulas dictionary
    formulas_dict = get_formulas_dict()
    
    # Create formulas row - map each column to its formula
    formulas_row = {col: formulas_dict.get(col, '') for col in COLUMNS}
    
    # Create DataFrame with formulas row
    formulas_df = pd.DataFrame([formulas_row], columns=COLUMNS)
    
    # Insert formulas row at the top (index 0)
    # Concatenate formulas row with existing data
    df_with_formulas = pd.concat([formulas_df, df], ignore_index=True)
    
    return df_with_formulas

def get_portfolio_df_with_formulas():
    """
    Get portfolio DataFrame with formulas row added at the top.
    
    This is a convenience function that calls get_portfolio_df() and adds the formulas row.
    
    Returns:
        pd.DataFrame: Portfolio DataFrame with formulas row at index 0
    """
    df = get_portfolio_df()
    return add_formulas_row_to_df(df)

def generate_trades(csv_file, portfolio_type='simple'):
    """
    Generate trade code from CSV.

    Args:
        csv_file: Path to CSV file. It must have:
            - one of: ticker, symbol, tvId, tv_Id
            - side, price, quantity columns
            - optional date/cts/mts column for trade date
        portfolio_type: 'simple' or 'full'
    """
    df = pd.read_csv(csv_file)

    # Build case-insensitive column lookup
    cols = {str(c).lower(): c for c in df.columns}

    def get_col(candidates, required=True, label=None):
        """Return the first matching column name from candidates (case-insensitive)."""
        label = label or candidates
        for cand in candidates:
            key = str(cand).lower()
            if key in cols:
                return cols[key]
        if required:
            raise ValueError(
                f"CSV must contain column(s) {candidates} for {label} (case-insensitive). "
                f"Found columns: {list(df.columns)}"
            )
        return None

    # Core columns
    ticker_col   = get_col(["ticker", "symbol", "tvid", "tv_id"], label="ticker")
    side_col     = get_col(["side"], label="side")
    price_col    = get_col(["price"], label="price")
    quantity_col = get_col(["quantity"], label="quantity")
    date_col     = get_col(["date", "cts", "mts"], required=False, label="date")

    print("# Reset portfolio")
    print("reset_portfolio()\n")

    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).lower()
        side   = str(row[side_col]).lower()
        price  = float(row[price_col])
        qty    = abs(float(row[quantity_col]))

        # Optional date column; normalize if present
        trade_date = normalize_trade_date(row[date_col]) if date_col is not None else None

        if portfolio_type == 'simple':
            if trade_date is not None:
                print(
                    f"add_trade(ticker='{ticker}', side='{side}', "
                    f"price={price}, quantity_buy={qty}, date='{trade_date}')"
                )
            else:
                print(
                    f"add_trade(ticker='{ticker}', side='{side}', "
                    f"price={price}, quantity_buy={qty})"
                )
        else:  # full
            if trade_date is not None:
                print(
                    f"add_trade(ticker='{ticker}', asset_type='Equity', side='{side}', "
                    f"price={price}, quantity_buy={qty}, date='{trade_date}')"
                )
            else:
                print(
                    f"add_trade(ticker='{ticker}', asset_type='Equity', side='{side}', "
                    f"price={price}, quantity_buy={qty})"
                )

    print("\n# Display")
    print("df = get_portfolio_df()")
    print("df")


# ---------- Web Frontend (Flask) ----------

app = Flask(__name__)


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Portfolio CSV Frontend</title>
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
      rel="stylesheet"
    />
    <link
      href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css"
      rel="stylesheet"
    />
    <style>
      body {
        min-height: 100vh;
        background: radial-gradient(circle at top, #1f2937 0, #020617 55%, #000 100%);
        color: #e5e7eb;
      }
      .app-shell {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem 3rem;
      }
      .card-glass {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.7);
      }
      .card-glass-header {
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
      }
      .hero-title {
        letter-spacing: 0.05em;
      }
      .subtitle {
        color: #9ca3af;
      }
      .table-wrapper {
        position: relative;
        max-height: 70vh;
        overflow-x: auto;
        overflow-y: auto;
        border-radius: 0.75rem;
        background: #020617;
      }
      .table-wrapper table {
        margin: 0 !important; /* keep first column flush to the left when scrolling */
      }
      table.dataTable thead th {
        position: sticky !important;
        top: 0;
        background: #020617;
        color: #e5e7eb;
        z-index: 5;
      }
      table.dataTable tbody tr:hover {
        background-color: #0b1120;
      }
      table.dataTable thead th,
      table.dataTable tbody td {
        padding: 0.3rem 0.5rem;
        font-size: 0.8rem;
        white-space: nowrap;
      }
      .form-text {
        color: #9ca3af;
      }
    </style>
  </head>
  <body>
    <div class="app-shell">
      <header class="mb-4 text-center">
        <h1 class="hero-title display-6 fw-semibold text-light mb-2">
          Portfolio Validator
        </h1>
        <p class="subtitle mb-0">
          Upload a trades CSV and explore full portfolio metrics, PnL, and risk in one view.
        </p>
      </header>

      <section class="mb-4">
        <div class="card-glass">
          <div class="card-body p-4">
            <form method="post" enctype="multipart/form-data" class="row g-3 align-items-end">
              <div class="col-md-5 col-lg-4">
                <label for="csv_file" class="form-label text-light">Trades CSV</label>
                <input
                  class="form-control"
                  type="file"
                  id="csv_file"
                  name="csv_file"
                  accept=".csv"
                  required
                />
              </div>
              <div class="col-md-3 col-lg-2">
                <label for="initial_cash" class="form-label text-light">Initial Balance</label>
                <input
                  type="text"
                  class="form-control"
                  id="initial_cash"
                  name="initial_cash"
                  value="{{ initial_cash }}"
                />
              </div>
              <div class="col-md-3 col-lg-2">
                <label class="form-label text-light d-block">&nbsp;</label>
                <button class="btn btn-primary w-100" type="submit">
                  Run Portfolio
                </button>
              </div>
              <div class="col-12">
                <div class="form-text mt-1">
                  Required (case-insensitive): ticker/symbol/tvId/tv_Id, side, price, quantity.
                  Optional: date/cts/mts.
                </div>
              </div>
            </form>
          </div>
        </div>
      </section>

      {% if error %}
        <section class="mb-3">
          <div class="alert alert-danger shadow-sm mb-0" role="alert">
            {{ error }}
          </div>
        </section>
      {% endif %}

      {% if df_html %}
        <section class="mt-3">
          <div class="card-glass">
            <div
              class="card-glass-header d-flex justify-content-between align-items-center px-4 py-3"
            >
              <div>
                <h2 class="h5 mb-0 text-light">Portfolio Results</h2>
                <small class="subtitle">Sortable, searchable trade and metric table</small>
              </div>
              <a href="{{ url_for('download_csv') }}" class="btn btn-outline-light btn-sm">
                Download CSV
              </a>
            </div>
            <div class="table-wrapper p-3">
              {{ df_html | safe }}
            </div>
          </div>
        </section>
      {% endif %}
    </div>

    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    <script>
      document.addEventListener('DOMContentLoaded', function () {
        // Live format initial balance with commas while typing
        const cashInput = document.getElementById('initial_cash');
        if (cashInput) {
          const formatWithCommas = (val) => {
            if (val === '' || val === null || val === undefined) return '';
            const str = String(val).replace(/,/g, '');
            if (!str) return '';
            const parts = str.split('.');
            const intPart = parts[0].replace(/[^0-9-]/g, '');
            const decPart = parts[1] ? parts[1].replace(/[^0-9]/g, '') : '';
            if (!intPart) return decPart ? '0.' + decPart : '';
            const intNum = Number(intPart);
            if (Number.isNaN(intNum)) return val;
            const formattedInt = intNum.toLocaleString('en-US');
            return decPart ? formattedInt + '.' + decPart : formattedInt;
          };

          cashInput.addEventListener('input', function (e) {
            const cursorPos = this.selectionStart;
            const before = this.value.slice(0, cursorPos);
            const formattedBefore = formatWithCommas(before);
            const formattedFull = formatWithCommas(this.value);
            this.value = formattedFull;
            // Simple caret reposition: put it at end for now (good enough UX)
            this.selectionStart = this.selectionEnd = this.value.length;
          });
        }

        // Helper to freeze first N columns horizontally while allowing sideways scroll
        function freezeColumns(tableEl, count) {
          if (!tableEl || !tableEl.tHead) return;
          const headerCells = tableEl.tHead.rows[0].cells;
          const max = Math.min(count, headerCells.length);

          // Compute left offsets based on rendered widths
          const leftOffsets = [];
          let left = 0;
          for (let i = 0; i < max; i++) {
            leftOffsets[i] = left;
            left += headerCells[i].offsetWidth;
          }

          for (let i = 0; i < max; i++) {
            const selector = `#${tableEl.id} thead th:nth-child(${i + 1}), #${tableEl.id} tbody td:nth-child(${i + 1})`;
            tableEl.querySelectorAll(selector).forEach((cell) => {
              cell.style.position = 'sticky';
              cell.style.left = leftOffsets[i] + 'px';
              // Do NOT override background so color stays consistent with other cells
              cell.style.zIndex = cell.tagName === 'TH' ? 6 : 4;
            });
          }
        }

        const table = document.getElementById('results-table');
        if (table) {
          new DataTable(table, {
            paging: false,   // show all rows on a single page
            info: false,     // hide "showing X of Y" text
            ordering: true,
            searching: true
          });

          // Freeze the first 6 columns: Date, Ticker, Asset Type, Side, Direction, Quantity Buy
          freezeColumns(table, 6);
        }
      });
    </script>
  </body>
  </html>
"""


def _run_portfolio_on_dataframe(
    trades: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    """
    Core driver: given a trades DataFrame, feed it into the portfolio engine.
    Column detection is case-insensitive and supports multiple aliases:
      - ticker: one of ticker/symbol/tvId/tv_Id
      - date: one of date/cts/mts (optional)
    """
    # Build case-insensitive column lookup
    cols = {str(c).lower(): c for c in trades.columns}

    def get_col(candidates, required: bool = True, label: Optional[str] = None):
        """Return the first matching column name from candidates (case-insensitive)."""
        label = label or ",".join(candidates)
        for cand in candidates:
            key = str(cand).lower()
            if key in cols:
                return cols[key]
        if required:
            raise ValueError(
                f"CSV must contain column(s) {candidates} for {label} (case-insensitive). "
                f"Found columns: {list(trades.columns)}"
            )
        return None

    # Core columns
    ticker_col = get_col(["ticker", "symbol", "tvid", "tv_id"], label="ticker")
    side_col = get_col(["side"], label="side")
    price_col = get_col(["price"], label="price")
    quantity_col = get_col(["quantity"], label="quantity")
    date_col = get_col(["date", "cts", "mts"], required=False, label="date")

    reset_portfolio(initial_cash)

    for _, row in trades.iterrows():
        ticker = str(row[ticker_col]).upper()
        side_raw = str(row[side_col]).strip().lower()
        if side_raw not in {"buy", "sell"}:
            continue

        price = float(row[price_col])
        qty = float(row[quantity_col])

        # Optional date column (case-insensitive) – normalize to one format
        trade_date = None
        if date_col is not None:
            trade_date = normalize_trade_date(row[date_col])

        # Let engine infer asset_type from ASSET_TYPE_MAP if None
        add_trade(
            ticker=ticker,
            asset_type=None,
            side=side_raw,
            price=price,
            quantity_buy=qty,
            date=trade_date,
        )

    # Store last result for CSV export
    last_result_df = get_portfolio_df()
    return last_result_df


@app.route("/", methods=["GET", "POST"])
def index():
    df_html = None
    error: Optional[str] = None
    default_initial_cash = 200.0
    global last_result_df

    if request.method == "POST":
        file = request.files.get("csv_file")
        initial_cash_raw = request.form.get("initial_cash", "").strip()

        try:
            # Strip commas for numeric parsing (allows inputs like "10,000")
            cleaned = initial_cash_raw.replace(",", "")
            initial_cash = float(cleaned) if cleaned else default_initial_cash
        except ValueError:
            initial_cash = default_initial_cash

        if not file or file.filename == "":
            error = "Please select a CSV file."
        else:
            try:
                content = file.read()
                trades_df = pd.read_csv(io.BytesIO(content))
                result_df = _run_portfolio_on_dataframe(trades_df, initial_cash)
                last_result_df = result_df
                df_html = result_df.to_html(
                    classes="table table-striped table-sm",
                    border=0,
                    table_id="results-table",
                )
            except Exception as exc:  # pragma: no cover - user-facing error
                error = f"Error processing file: {exc}"

    return render_template_string(
        HTML_TEMPLATE,
        df_html=df_html,
        error=error,
        initial_cash=default_initial_cash,
    )


@app.route("/download", methods=["GET"])
def download_csv():
    """
    Download the last computed portfolio results as a CSV file.
    """
    global last_result_df
    if last_result_df is None or last_result_df.empty:
        return "No portfolio results to download. Please run the portfolio first.", 400

    # Convert DataFrame to CSV in memory
    csv_buffer = io.StringIO()
    last_result_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="portfolio_results.csv",
    )


def create_app() -> Flask:
    """
    Factory for creating the Flask app (useful if you want `flask run`).
    """
    return app


if __name__ == "__main__":
    # Run on localhost:5000 by default
    app.run(host="127.0.0.1", port=5000, debug=True)