"""
Portfolio Engine — Comprehensive Backtesting Portfolio & Metrics Calculator
============================================================================

OVERVIEW:
---------
This is the SINGLE SOURCE OF TRUTH for all portfolio metrics in the backtester.
It maintains a state-based portfolio that tracks positions, calculates P&L, and 
computes 78+ performance metrics including Sharpe ratio, Sortino ratio, drawdown,
win rate, and more.

HOW IT WORKS:
-------------
1. Reset the portfolio once with initial cash: reset_portfolio(10000)
2. Add trades one by one as they execute: add_trade(ticker, side, price, quantity, date)
3. Get the complete portfolio DataFrame: get_portfolio_df()

Each trade (buy, sell, or hold) runs through the same pipeline:
  • Portfolio state is updated (cash, positions, P&L)
  • Metrics are calculated (returns, ratios, drawdown, etc.)
  • One row is appended to the output DataFrame

IMPORTANT CONCEPTS:
-------------------
• Hold Trades: Use side='hold' with quantity=0 to create a row without changing positions.
  Useful for tracking daily portfolio value even when no trades occur.

• Trade Counting: The 'Total Trades' metric counts POSITIONS (buy+sell pairs), not 
  individual buy/sell actions. For example:
  - 3 buys + 3 sells = 3 positions = "3 Trades"
  - NOT 6 trades

• State-Based: No "previous DataFrame" lookups. All metrics use the internal 
  portfolio_state dictionary for maximum performance.

TYPICAL USAGE FLOW:
-------------------
```python
from portfolio import reset_portfolio, add_trade, get_portfolio_df

# 1. Initialize
reset_portfolio(initial_cash=10000)

# 2. Add trades as they execute
add_trade(ticker='AAPL', side='buy', price=150.0, quantity=10, date='2024-01-01')
add_trade(ticker='AAPL', side='sell', price=155.0, quantity=10, date='2024-01-15')

# 3. Get results
portfolio_df = get_portfolio_df()
print(portfolio_df[['Date', 'Side', 'Account Value', 'Total PnL', 'Sharpe Ratio']])
```

METRICS CALCULATED (78+ columns):
----------------------------------
• Position Tracking: Current Quantity, Avg Price, Cost Basis, Equity
• P&L: Realized PnL, Unrealized PnL, Total PnL, Daily PnL
• Returns: Total Return %, Daily %, Cumulative %, Asset Performance %
• Risk Metrics: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown
• Trade Stats: Total Trades, Win Rate, Win:Loss Ratio, Avg Winning/Losing PnL
• Portfolio: Account Value, Available Balance, Investment Count, Holdings
• And many more...

For detailed formulas, see get_formulas_dict() function at the end of this file.
"""
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# ---------- Display (optional) ----------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.max_colwidth', None)

# Output column order (one row per trade/hold)
COLUMNS = [
    'Date','Ticker', 'Asset Type','Side','Direction','Initial Balance','Buyable/Sellable',
    'Quantity Buy','Available Balance','Current Quantity','Price',
    'Avg Price','Cost Basis','Equity',
    'PnL (Long) Unrealized','PnL (Short) Unrealized','Pnl Unrealized',
    'PnL Unrealized Total Value for Current Ticker','PnL realized Total Value for Current Ticker',
    'PnL Realized at Point of Time','PnL Unrealized at Point of Time',
    'Equity (Long)','Equity (Short)','Open Position','Open Equity',
    'Total Equity','Account Value',
    'Realized PnL at Point of Time (Portfolio)','Unrealized PnL at Point of Time (Portfolio)',
    'Total PnL Overall (Unrealized+Realized)',
    'Daily PnL (Unrealized+Realized)','Liquidation Price','Take Profit','Stop Loss', 
    'Last Day Pnl / Daily $', 'Daily %', 'Cumulative %', 'Investment Count',
    'Performance',
    'Backtester Net Performance %',
    'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Asset Count', 'Asset Performance %',
    'Trade No. (Position - Trade no. - Current Quantity)', 'Total Trades', 'Win/Loss',
    'Win Rate', 'Win:Loss Ratio',
    'Backtester Reward/Risk Ratio', 'Backtester Expectancy',
    'Backtester Avg Winning PnL %', 'Backtester Avg Losing PnL %', 'Backtester Avg PnL %',
    'Backtester Max Drawdown',
    'Trades/Month', 'Absolute Quantity Counts', 'Most Traded Symbol', 'Most Bought', 'Least Traded',
    'Avg Losing PnL', 'Avg Winning PnL', 'Most Profitable', 'Least Profitable', 'Max Drawdown',
    'Total Gain', 'Average Gain', 'Biggest Investment', 'Average Position', 'Holdings','YTD PnL',
    'Highest Traded Volume', 'Lowest Traded Volume', 'Average Holding Days',
    'Distribution', 'Distribution in %',
    'Equity Distribution (Market Cap)', 'Equity Distribution (Industry)', 'Equity Distribution (Sector)'
]


def _initial_portfolio_state(initial_cash):
    """Single source of truth for portfolio state. Used at module load and by reset_portfolio."""
    return {
        'cash': initial_cash,
        'remaining': initial_cash,
        'quantities': defaultdict(int),
        'cost_basis': defaultdict(float),
        'avg_price': defaultdict(float),
        'realized_pnl': 0.0,
        'trade_returns': [],
        'last_price': {},
        'asset_types': {},
        'market_cap': {},
        'industry': {},
        'sector': {},
        'max_investment_history': defaultdict(float),
        'highest_traded_volume': None,
        'lowest_traded_volume': None,
        'position_open_period': {},
        'position_open_date': {},
        'cumulative_holding_sum': 0.0,
        'closed_positions_count': 0,
        'current_period': 0,
        'previous_realized_pnl': 0.0,
        'aps_open_qty_sum': 0.0,
        'aps_trade_count': 0,
        'portfolio_df': pd.DataFrame(columns=pd.Index(COLUMNS)),
        'rows': [],
        'win_loss_counts': {'win': 0, 'loss': 0},
        'open_trade_win_loss_by_ticker': {},  # ticker -> "Win"|"Loss" for current open trade (one slot per trade; flip on hold)
        'ticker_quantity_totals': defaultdict(float),
        'trades_per_ticker': defaultdict(set),
        'total_open_qty': defaultdict(float),
        'realized_pnl_by_ticker': defaultdict(float),
        'ticker_unrealized': defaultdict(float),
        'ticker_pv': defaultdict(float),
        'total_unrealized': 0.0,
        'total_pv': 0.0,
        'avg_losing_sum': 0.0,
        'avg_losing_count': 0,
        'avg_winning_sum': 0.0,
        'avg_winning_count': 0,
        'pnl_stats': {'n': 0, 'mean': 0.0, 'M2': 0.0},
        'downside_stats': {'n': 0, 'mean': 0.0, 'M2': 0.0},
        'first_trade_date': None,
        'calmar_peak': None,
        'calmar_max_drawdown': 0.0,
        'peak_equity': None,
        'winning_trades_max': None,
        'winning_trades_min': None,
        'winning_trades_max_tickers': [],
        'winning_trades_min_tickers': [],
        'current_month_year': None,
        'trades_current_month': set(),
        'trades_month_carried': 0,       # trades open at start of this month (carried from previous)
        'trades_opened_this_month': set(),  # trade numbers that opened in this month
        'last_known_open_trades': set(),   # snapshot of open trade numbers at end of last row (for carried count)
        'calmar_min_date': None,
        'calmar_max_date': None,
        'last_trade_date': None,
        'last_total_pnl_overall': 0.0,
        'last_account_value': float(initial_cash),
        'last_realized_pnl_cumulative': 0.0,
        'last_total_trades_str': "No Buy/Sell",
        'ytd_year': None,
        'ytd_start_realized_pnl': 0.0,
        'bnp_completed_returns': [],
        'bnp_entry_price': {},
        'bnp_direction': {},
        'bnp_last_price': {},
        'ticker_start_price': {}
    }


# Trade numbering (per-ticker) and investment count
trade_tracker = {}
next_trade_number = 1
investment_count = 0

portfolio_state = _initial_portfolio_state(200)


# ---------- Lifecycle: reset → add trades → get DataFrame ----------

def reset_portfolio(initial_cash=200):
    """
    Reset portfolio to initial state. Call once before adding trades.

    Clears all state: positions, cash, running metrics (win rate, Sharpe, etc.),
    trade counters, and accumulated rows. Next get_portfolio_df() will be empty
    until add_trade is called again.

    Args:
        initial_cash (float): Starting cash. Defaults to 200.
    """
    global portfolio_state, trade_tracker, next_trade_number, investment_count
    portfolio_state = _initial_portfolio_state(initial_cash)
    trade_tracker = {}
    next_trade_number = 1
    investment_count = 0

def get_portfolio_df():
    """
    Build and return the portfolio table from accumulated state.
    
    Rows are stored in state as each trade is processed; this builds the DataFrame
    from those rows (or returns a cached copy). Call after adding all trades.
    
    Returns:
        pd.DataFrame: One row per trade (and per hold, if used), all metric columns.
    
    Example:
        >>> reset_portfolio(10000)
        >>> for order in orders:
        ...     add_trade(ticker, asset_type, side, price, quantity_buy, date)
        >>> df = get_portfolio_df()
    """

    rows = portfolio_state.get('rows', [])
    df = portfolio_state.get('portfolio_df')
    if df is None or len(df) != len(rows):
        if rows:
            df = pd.DataFrame(rows, columns=pd.Index(COLUMNS))
        else:
            df = pd.DataFrame(columns=pd.Index(COLUMNS))
        portfolio_state['portfolio_df'] = df
    return df.copy()

# ---------- Helpers (quantity, dates, trade strings, PnL stats) ----------

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

def _parse_date_value(value):
    """Parse a date value into datetime; returns None if parsing fails."""
    if value is None:
        return None
    from datetime import datetime
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:
            pass
    s = str(value).strip()
    if not s:
        return None
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _update_pnl_stats(current_realized_pnl_at_point):
    """Update running PnL statistics for Sharpe/Sortino."""
    if current_realized_pnl_at_point is None:
        return
    if isinstance(current_realized_pnl_at_point, float) and np.isnan(current_realized_pnl_at_point):
        return
    value = float(current_realized_pnl_at_point)
    stats = portfolio_state.get('pnl_stats')
    if stats is None:
        stats = {'n': 0, 'mean': 0.0, 'M2': 0.0}
        portfolio_state['pnl_stats'] = stats
    stats['n'] += 1
    delta = value - stats['mean']
    stats['mean'] += delta / stats['n']
    stats['M2'] += delta * (value - stats['mean'])

    if value < 0:
        downside = portfolio_state.get('downside_stats')
        if downside is None:
            downside = {'n': 0, 'mean': 0.0, 'M2': 0.0}
            portfolio_state['downside_stats'] = downside
        downside['n'] += 1
        d_delta = value - downside['mean']
        downside['mean'] += d_delta / downside['n']
        downside['M2'] += d_delta * (value - downside['mean'])

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
        # Detect flip: position direction reversed (long→short or short→long)
        is_flip = (old_q > 0 and new_q < 0) or (old_q < 0 and new_q > 0)
        if is_flip:
            # The flip row closes the old trade — return the old trade number for this row.
            old_trade_number = trade_tracker.get(ticker)
            # Assign a fresh trade number for the new position (used on subsequent rows).
            new_trade_number = next_trade_number
            trade_tracker[ticker] = new_trade_number
            next_trade_number += 1
            return old_trade_number
        # Continue existing trade (partial fill, add-on, etc.)
        return trade_tracker.get(ticker)
    
    # No position (old_q == 0, new_q == 0)
    return None

def format_trade_string(side, current_direction, trade_number, new_q, is_flip=False):
    """
    Format trade string for display.
    
    Format: "Direction - side - #TradeNo Trade - Quantity" or "... - 0.0 - Close" / "... - 10.0 - Flipped"
    
    Args:
        side (str): Trade side ('buy', 'sell', 'hold')
        current_direction (str or None): Current direction ('long', 'short', 'hold'), or None if no buy/sell yet
        trade_number (int or None): Trade number
        new_q (float): New quantity after trade
        is_flip (bool): Whether this trade is a position reversal (flip)
    
    Returns:
        str: Formatted trade string or "No Buy/Sell" if no trade
    
    Examples:
        - "Long - Buy - #1 Trade - 10.0"
        - "Long - Sell - #1 Trade - 0.0 - Close"
        - "Short - Sell - #2 Trade - 10.0 - Flipped"
    """
    if trade_number is None:
        return "No Buy/Sell"
    
    side_str = str(side)
    dir_str = 'Null' if current_direction is None else str(current_direction)
    
    # Format quantity using floats; label close and flip events explicitly
    if new_q == 0:
        quantity_str = f"{abs(new_q)} - Close"
    elif is_flip:
        quantity_str = f"{abs(new_q)} - Flipped"
    else:
        quantity_str = str(abs(new_q))
    
    # Include Buy/Sell in the output
    return f" {dir_str.capitalize()} - {side_str.capitalize()} - #{trade_number} Trade - {quantity_str}"

def is_close_trade_string(trade_string):
    """
    Return True if a trade string represents a closing trade (full close or flip).
    Uses case-insensitive matching for "- Close" and "- Flipped" suffixes.
    """
    if not trade_string or trade_string == "No Buy/Sell":
        return False
    lower = str(trade_string).lower()
    return "- close" in lower or "- flipped" in lower

# ---------- Core single-trade calculations (cash, quantity, cost basis, PnL) ----------

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
    
    # Initialize trade_returns if not exists
    if 'trade_returns' not in portfolio_state:
        portfolio_state['trade_returns'] = []

    # Long position: calculate realized PnL when selling
    if pos == 'long' and a == 'sell' and old_quantity > 0:
        qty = abs(q_in) if q_in < 0 else q_in
        # Only realize on shares actually closed (min of qty and owned)
        # If selling more than owned, only the owned shares generate realized PnL
        closed = min(qty, old_quantity)
        if closed > 0 and prev_avg_price > 0:
            # Formula: (price - avg_entry) * closed
            realized += (price - prev_avg_price) * closed
            
            # Track trade return for Backtester Net Performance
            # Return = (exit_price - entry_price) / entry_price
            trade_return = (price - prev_avg_price) / prev_avg_price
            portfolio_state['trade_returns'].append(trade_return)

    # Short position: calculate realized PnL when buying/covering
    if pos == 'short' and a == 'buy' and old_quantity < 0:
        qty = abs(q_in) if q_in < 0 else q_in
        # Only realize on shares actually covered (min of qty and owed)
        # If buying more than owed, only the owed shares generate realized PnL
        closed = min(qty, abs(old_quantity))
        if closed > 0 and prev_avg_price > 0:
            # Formula: (avg_entry - price) * closed
            realized += (prev_avg_price - price) * closed
            
            # Track trade return for Backtester Net Performance
            # Return = (entry_price - exit_price) / entry_price
            trade_return = (prev_avg_price - price) / prev_avg_price
            portfolio_state['trade_returns'].append(trade_return)

    portfolio_state['realized_pnl'] = realized
    return realized


# ---------- Backtester Net Performance % (self-contained; uses only bnp_* state) ----------

def _bnp_update_on_trade(ticker, price, old_q, new_q):
    """
    Update BNP-only state for one trade. Uses only bnp_* keys in portfolio_state.
    On open: store entry price and direction. On close: append price-only R, clear entry.
    Always: store this ticker's price as bnp_last_price.
    """
    state = portfolio_state
    state.setdefault('bnp_completed_returns', [])
    state.setdefault('bnp_entry_price', {})
    state.setdefault('bnp_direction', {})
    state.setdefault('bnp_last_price', {})

    state['bnp_last_price'][ticker] = price

    if old_q == 0 and new_q != 0:
        state['bnp_entry_price'][ticker] = price
        state['bnp_direction'][ticker] = 'long' if new_q > 0 else 'short'
        return
    if new_q == 0 and old_q != 0:
        entry = state['bnp_entry_price'].get(ticker)
        if entry is not None and entry > 0:
            if old_q > 0:
                r = (price - entry) / entry
            else:
                r = (entry - price) / entry
            state['bnp_completed_returns'].append(r)
        state['bnp_entry_price'].pop(ticker, None)
        state['bnp_direction'].pop(ticker, None)


def calculate_backtester_net_performance(current_ticker=None, current_price=None):
    """
    Backtester Net Performance % from BNP-only state. No other metrics used.
    Price-only returns; one R per trade (finalized on close); open trades use current price.
    Formula: [(1 + R1) × (1 + R2) × ... × (1 + Rn) - 1] × 100
    """
    completed = portfolio_state.get('bnp_completed_returns', [])
    entry_price = portfolio_state.get('bnp_entry_price', {})
    direction = portfolio_state.get('bnp_direction', {})
    last_price = portfolio_state.get('bnp_last_price', {})

    compound = 1.0
    for r in completed:
        compound *= (1.0 + r)
    for ticker in entry_price:
        entry = entry_price.get(ticker) or 0.0
        if entry <= 0:
            continue
        current_p = (current_price if ticker == current_ticker else last_price.get(ticker)) or entry
        if direction.get(ticker) == 'long':
            r_open = (current_p - entry) / entry
        else:
            r_open = (entry - current_p) / entry
        compound *= (1.0 + r_open)
    return (compound - 1.0) * 100.0

def calculate_backtester_avg_winning_pnl_pct():
    """
    Calculate Backtester Average Winning PnL % from trade returns.
    
    Formula: Average of all positive trade returns × 100
    
    Returns:
        float: Average winning PnL % (or None if no winning trades)
    """
    trade_returns = portfolio_state.get('trade_returns', [])
    
    if len(trade_returns) == 0:
        return None
    
    # Filter only winning trades (positive returns)
    winning_returns = [r for r in trade_returns if r > 0]
    
    if len(winning_returns) == 0:
        return None
    
    # Calculate average and convert to percentage
    avg_winning_return = sum(winning_returns) / len(winning_returns)
    return avg_winning_return * 100.0

def calculate_backtester_avg_losing_pnl_pct():
    """
    Calculate Backtester Average Losing PnL % from trade returns.
    
    Formula: Average of all negative trade returns × 100
    
    Returns:
        float: Average losing PnL % (or None if no losing trades)
    """
    trade_returns = portfolio_state.get('trade_returns', [])
    
    if len(trade_returns) == 0:
        return None
    
    # Filter only losing trades (negative returns)
    losing_returns = [r for r in trade_returns if r < 0]
    
    if len(losing_returns) == 0:
        return None
    
    # Calculate average and convert to percentage
    avg_losing_return = sum(losing_returns) / len(losing_returns)
    return avg_losing_return * 100.0

def calculate_backtester_avg_pnl_pct():
    """
    Calculate Backtester Average PnL % from all trade returns.
    
    Formula: Average of all trade returns × 100
    
    Returns:
        float: Average PnL % (or None if no trades)
    """
    trade_returns = portfolio_state.get('trade_returns', [])
    
    if len(trade_returns) == 0:
        return None
    
    # Calculate average of all returns and convert to percentage
    avg_return = sum(trade_returns) / len(trade_returns)
    return avg_return * 100.0

def calculate_backtester_reward_risk_and_expectancy(win_rate):
    """
    Calculate Backtester Reward/Risk Ratio and Backtester Expectancy.
    
    Uses backtester avg winning/losing PnL % (from trade returns) instead of portfolio-based metrics.

    Definitions:
        - Backtester Reward/Risk Ratio = Backtester Avg Winning PnL % / |Backtester Avg Losing PnL %|
        - Backtester Expectancy = (rrRatio * Win Ratio) - Loss Ratio

    Where:
        - Win Ratio  = Win Rate / 100  (Win Rate is stored as percentage 0–100)
        - Loss Ratio = 1 - Win Ratio

    Args:
        win_rate (float or None): Win Rate in percent (0–100), or None if not available

    Returns:
        tuple: (backtester_reward_risk_ratio, backtester_expectancy)
            backtester_reward_risk_ratio (float or None): AvgWin / |AvgLoss|, or None if not defined
            backtester_expectancy (float or None): (rrRatio * WinRatio) - LossRatio, or None if not defined

    Edge cases:
        - Returns (None, None) if not enough data (no winners/losers or no win_rate).
        - Handles avg_losing_pnl >= 0 (invalid) by returning None for both.
    """
    # Get backtester avg winning and losing PnL %
    avg_winning_pnl_pct = calculate_backtester_avg_winning_pnl_pct()
    avg_losing_pnl_pct = calculate_backtester_avg_losing_pnl_pct()
    
    # ---------- Backtester Reward/Risk Ratio ----------
    # If we have winning trades but no losing trades, reward/risk ratio is infinity
    if avg_winning_pnl_pct is not None and avg_winning_pnl_pct > 0:
        if avg_losing_pnl_pct is None:
            # All wins, no losses = infinite reward/risk ratio
            backtester_reward_risk_ratio = float('inf')
        elif avg_losing_pnl_pct < 0:
            # Normal case: both wins and losses
            backtester_reward_risk_ratio = avg_winning_pnl_pct / abs(avg_losing_pnl_pct)
        else:
            # Invalid case: avg_losing_pnl_pct >= 0
            backtester_reward_risk_ratio = None
    else:
        # No winning trades
        backtester_reward_risk_ratio = None

    # ---------- Backtester Expectancy ----------
    # Expectancy = (rrRatio * Win Ratio) - Loss Ratio
    # Win Ratio = Win Rate (fraction), Loss Ratio = 1 - Win Ratio
    if win_rate is not None and backtester_reward_risk_ratio is not None:
        # win_rate is stored as a percentage (0–100), convert to 0–1
        win_ratio = win_rate / 100.0
        loss_ratio = 1.0 - win_ratio
        backtester_expectancy = (backtester_reward_risk_ratio * win_ratio) - loss_ratio
    else:
        backtester_expectancy = None

    return backtester_reward_risk_ratio, backtester_expectancy

def calculate_backtester_max_drawdown():
    """
    Calculate Backtester Max Drawdown from cumulative trade returns (TrendSpider style).
    
    This calculates the maximum peak-to-trough decline in the cumulative equity curve
    built from compounding trade returns, similar to how TrendSpider calculates it.
    
    Formula:
        1. Build equity curve: equity[i] = equity[i-1] * (1 + return[i])
        2. Track running peak
        3. Calculate drawdown at each point: (equity - peak) / peak (negative value)
        4. Return maximum drawdown as percentage (most negative value)
    
    Returns:
        float: Maximum drawdown % (negative value on -100 to 0 scale), or None if no trades
    """
    trade_returns = portfolio_state.get('trade_returns', [])
    
    if len(trade_returns) == 0:
        return None
    
    # Build equity curve starting at 100 (representing 100% or initial capital)
    equity_curve = [100.0]
    for trade_return in trade_returns:
        new_equity = equity_curve[-1] * (1.0 + trade_return)
        equity_curve.append(new_equity)
    
    # Calculate max drawdown using peak-to-trough (will be negative)
    peak = equity_curve[0]
    max_drawdown = 0.0  # Start at 0 (no drawdown)
    
    for equity in equity_curve[1:]:
        if equity > peak:
            peak = equity
        else:
            # Calculate drawdown from peak (equity - peak gives negative value)
            drawdown = (equity - peak) / peak
            # Keep the most negative (largest drawdown)
            if drawdown < max_drawdown:
                max_drawdown = drawdown
    
    # Return as percentage (-100 to 0 scale, negative values)
    return max_drawdown * 100.0

# ---------- Position value and unrealized PnL (per ticker / all tickers) ----------

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

def pnl_unrealized_components(new_quantity, price, avg_price, current_ticker, current_price, old_quantity=0):
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
        old_quantity (float): Previous quantity before this trade (default 0)
    
    Returns:
        tuple: (long_unrealized, short_unrealized, total_unrealized_current_ticker, total_unrealized_all_tickers)
    
    Edge cases:
        - Returns 0.0 for unrealized PnL if avg_price <= 0 or quantity == 0
        - Only calculates long unrealized when quantity > 0
        - Only calculates short unrealized when quantity < 0
        - CRITICAL: When opening a new position (old_quantity == 0), unrealized PnL is forced to 0.0
          to prevent floating-point precision errors.
    """
    # When opening a new position (old_quantity == 0), unrealized PnL is exactly 0.0 by definition
    # since entry price equals current price. We enforce this directly to avoid floating-point
    # precision artifacts from avg_price and cost_basis calculations.
    if old_quantity == 0 and new_quantity != 0:
        long_u = 0.0
        short_u = 0.0
        total_unrealized_current_ticker = 0.0
    else:
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
    
    # Update cached unrealized PnL for current ticker and total
    ticker_unrealized = portfolio_state.setdefault('ticker_unrealized', defaultdict(float))
    prev_ticker_unrealized = ticker_unrealized.get(current_ticker, 0.0)
    ticker_unrealized[current_ticker] = total_unrealized_current_ticker

    prev_total_unrealized = portfolio_state.get('total_unrealized', 0.0)
    total_unrealized_all_tickers = prev_total_unrealized + (total_unrealized_current_ticker - prev_ticker_unrealized)
    portfolio_state['total_unrealized'] = total_unrealized_all_tickers

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

def open_pv_str():
    """
    Generate string of all open position values (PV) for each ticker.
    
    Formula for each ticker: PV = Cost Basis + Unrealized PnL
        - Long: PV = cost_basis + (current_price - avg_price) * quantity
        - Short: PV = cost_basis + (avg_price - current_price) * abs(quantity)
    
    Format: "TICKER1 PV1, TICKER2 PV2, ..."
    
    Returns:
        str: Comma-separated list of tickers with PV values, or "None" if no positions
    
    Edge cases:
        - Returns 0.0 PV if avg_price <= 0
    """
    parts = []
    ticker_pv_cache = portfolio_state.get('ticker_pv', {})
    for t, q in portfolio_state['quantities'].items():
        if q != 0:
            ticker_pv = ticker_pv_cache.get(t, 0.0)
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
    ticker_pv_cache = portfolio_state.setdefault('ticker_pv', defaultdict(float))
    prev_ticker_pv = ticker_pv_cache.get(current_ticker, 0.0)

    # Calculate PV for the current ticker only (others use cached values)
    q = portfolio_state['quantities'].get(current_ticker, 0)
    cb = portfolio_state['cost_basis'].get(current_ticker, 0.0)
    avg = portfolio_state['avg_price'].get(current_ticker, 0.0)

    if q > 0 and avg > 0:
        new_ticker_pv = cb + (current_price - avg) * q
    elif q < 0 and avg > 0:
        new_ticker_pv = cb + (avg - current_price) * abs(q)
    else:
        new_ticker_pv = 0.0

    ticker_pv_cache[current_ticker] = new_ticker_pv

    prev_total_pv = portfolio_state.get('total_pv', 0.0)
    total_pv = prev_total_pv + (new_ticker_pv - prev_ticker_pv)
    portfolio_state['total_pv'] = total_pv

    # Build PV dict from cached values (only for open positions)
    ticker_pv_dict = {}
    for t, qty in portfolio_state['quantities'].items():
        if qty != 0:
            ticker_pv_dict[t] = ticker_pv_cache.get(t, 0.0)

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
    ticker_unrealized_cache = portfolio_state.get('ticker_unrealized', {})
    for t, q in portfolio_state['quantities'].items():
        if q != 0:
            ticker_unrealized = ticker_unrealized_cache.get(t, 0.0)
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

def calculate_trade_win_loss(trade_string, realized_pnl_at_point, unrealized_pnl_at_point=None, side=None):
    """
    Determine if a trade is a win or loss from combined trade PnL (realized + unrealized).
    Returns None when there is no trade or combined PnL is zero (neither).

    Logic:
        - No trade ("No Buy/Sell") -> None.
        - For both hold and close: total trade PnL = realized at point + unrealized at point.
        - Win if total > 0, Loss if total < 0, None if total == 0 (neither).
    """
    if not trade_string or trade_string == "No Buy/Sell":
        return None

    realized = realized_pnl_at_point if realized_pnl_at_point is not None else 0.0
    unrealized = unrealized_pnl_at_point if unrealized_pnl_at_point is not None else 0.0
    total_pnl = realized + unrealized

    if total_pnl > 0:
        return "Win"
    if total_pnl < 0:
        return "Loss"
    return None  # total == 0: neither (e.g. trade just started or flat)

def update_win_loss_counts_from_trade_pnl(ticker, is_closing_trade, realized_pnl_at_close, current_ticker_unrealized_pnl, new_quantity):
    """
    One slot per trade: update win/loss from trade PnL (realized at close, unrealized while open).
    - On full close (new_quantity == 0): remove open slot, count realized PnL as win or loss.
    - On flip (old_quantity and new_quantity have opposite signs): count realized PnL from the
      closed portion, then fall through to set the new position's open slot.
    - While open (new_quantity != 0, no flip): update open slot from current unrealized PnL.
    Keeps hold-run and no-hold-run counts aligned (same closes; open slot included in displayed rate/ratio).
    """
    counts = portfolio_state.setdefault('win_loss_counts', {'win': 0, 'loss': 0})
    open_outcomes = portfolio_state.setdefault('open_trade_win_loss_by_ticker', {})

    if is_closing_trade:
        open_outcomes.pop(ticker, None)
        if realized_pnl_at_close is not None:
            if realized_pnl_at_close > 0:
                counts['win'] += 1
            else:
                counts['loss'] += 1
        # Full close: nothing new to track
        if new_quantity == 0:
            return
        # Flip: don't return — fall through to set the new position's open slot below.

    if new_quantity == 0:
        open_outcomes.pop(ticker, None)
        return

    # Only set open slot when we have a definite Win or Loss (not when unrealized is 0 / trade just started)
    u = current_ticker_unrealized_pnl if current_ticker_unrealized_pnl is not None else 0
    if u > 0:
        open_outcomes[ticker] = "Win"
    elif u < 0:
        open_outcomes[ticker] = "Loss"
    else:
        open_outcomes.pop(ticker, None)  # neither: don't default to Loss

def calculate_win_rate(ticker):
    """
    Win rate including current ticker's open outcome (one slot per trade).
    So hold and no-hold show same denominator and comparable rate at same (Date, Ticker).
    """
    counts = portfolio_state.get('win_loss_counts', {'win': 0, 'loss': 0})
    open_outcomes = portfolio_state.get('open_trade_win_loss_by_ticker', {})
    w = counts['win'] + (1 if open_outcomes.get(ticker) == 'Win' else 0)
    l = counts['loss'] + (1 if open_outcomes.get(ticker) == 'Loss' else 0)
    total = w + l
    return (w / total * 100) if total > 0 else None

def calculate_win_loss_ratio(ticker):
    """
    Win:loss ratio including current ticker's open outcome (one slot per trade).
    """
    counts = portfolio_state.get('win_loss_counts', {'win': 0, 'loss': 0})
    open_outcomes = portfolio_state.get('open_trade_win_loss_by_ticker', {})
    w = counts['win'] + (1 if open_outcomes.get(ticker) == 'Win' else 0)
    l = counts['loss'] + (1 if open_outcomes.get(ticker) == 'Loss' else 0)
    return f"{w}:{l}"

def calculate_trades_per_month(current_date, is_opening_trade=False, trade_number=None):
    """
    Calculate number of trades in the current month (calendar-based, same for hold and trades-only).

    Logic:
        - Trades that opened in this month count.
        - Trades that were open at the start of this month (carried from previous month) also count.
        - When month changes: carried = number of trades currently open; opened_this_month = empty.
        - When we open a trade: add its number to opened_this_month.
        - count = carried + len(opened_this_month)

    Returns format: "count (Month Name)" (e.g., "3 (October)")
    """
    dt = _parse_date_value(current_date)
    if dt is None:
        return None

    current_month_year = (dt.year, dt.month)
    if portfolio_state.get('current_month_year') != current_month_year:
        portfolio_state['current_month_year'] = current_month_year
        # Carried = open trades at end of *previous* month (before this row); avoid undercount when first row of month is a close
        portfolio_state['trades_month_carried'] = len(portfolio_state.get('last_known_open_trades', set()))
        portfolio_state['trades_opened_this_month'] = set()

    if is_opening_trade and trade_number is not None:
        portfolio_state.setdefault('trades_opened_this_month', set()).add(trade_number)

    carried = portfolio_state.get('trades_month_carried', 0)
    opened = portfolio_state.get('trades_opened_this_month', set())
    count = carried + len(opened)
    month_name = dt.strftime('%B')
    return f"{count} ({month_name})"

def calculate_most_least_traded(current_ticker, current_qty_buy, current_trade_string=None):
    """
    Calculate:
        - Absolute Quantity Counts per symbol (volume-based), and
        - Most/Least Traded symbols **by number of completed trades**.
    State-based: uses portfolio_state['ticker_quantity_totals'], ['trades_per_ticker'].

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
    ticker_quantities = portfolio_state.setdefault('ticker_quantity_totals', defaultdict(float))
    if current_ticker and current_qty_buy != 0:
        try:
            ticker_quantities[current_ticker.upper()] += abs(float(current_qty_buy))
        except Exception:
            pass

    if not ticker_quantities:
        return None, "None", "None"

    sorted_tickers_desc = sorted(ticker_quantities.items(), key=lambda x: (-x[1], x[0]))
    abs_counts = [f"{ticker} {int(qty)}" for ticker, qty in sorted_tickers_desc]
    abs_counts_str = ", ".join(abs_counts) if abs_counts else "None"

    trades_per_ticker = portfolio_state.setdefault('trades_per_ticker', defaultdict(set))
    if current_ticker and current_trade_string and current_trade_string != "No Buy/Sell":
        m = re.search(r"#(\d+)", str(current_trade_string))
        if m:
            trades_per_ticker[current_ticker.upper()].add(int(m.group(1)))

    trade_counts = {t: len(nums) for t, nums in trades_per_ticker.items() if nums}
    if not trade_counts:
        return abs_counts_str, "None", "None"

    sorted_by_most = sorted(trade_counts.items(), key=lambda x: (-x[1], x[0]))
    most_traded_list = [f"{ticker} {count}" for ticker, count in sorted_by_most]
    most_traded_str = ", ".join(most_traded_list) if most_traded_list else "None"

    sorted_by_least = sorted(trade_counts.items(), key=lambda x: (x[1], x[0]))
    least_traded_list = [f"{ticker} {count}" for ticker, count in sorted_by_least]
    least_traded_str = ", ".join(least_traded_list) if least_traded_list else "None"

    return abs_counts_str, most_traded_str, least_traded_str


def calculate_most_bought(current_ticker, current_direction, current_side, current_qty_buy):
    """
    Calculate Most Bought = stock/coin with the largest total quantity **opened**.
    State-based: uses portfolio_state['total_open_qty'].

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
        current_ticker (str): Current ticker being traded
        current_direction (str): Current Direction argument ("long" / "short" / ...)
        current_side (str): Current Side / side ("buy" / "sell" / ...)
        current_qty_buy (float): Current quantity value

    Returns:
        str: "TICKER qty, ..." for the ticker(s) with the largest total opened quantity,
             or "None" if no opening trades exist.
    """
    def _is_opening(direction_val, side_val):
        d = str(direction_val).strip().lower()
        s = str(side_val).strip().lower()
        return (d == 'long' and s == 'buy') or (d == 'short' and s == 'sell')

    total_open_qty = portfolio_state.setdefault('total_open_qty', defaultdict(float))
    if current_ticker and current_qty_buy != 0 and _is_opening(current_direction, current_side):
        try:
            qty = abs(float(current_qty_buy))
            total_open_qty[current_ticker.upper()] += qty
        except Exception:
            pass
    if not total_open_qty:
        return "None"
    max_qty = max(total_open_qty.values())
    winners = sorted([t for t, q in total_open_qty.items() if q == max_qty])
    parts = [f"{ticker} {int(max_qty)}" for ticker in winners]
    return ", ".join(parts) if parts else "None"

def calculate_avg_losing_winning_pnl(current_realized_pnl_at_point):
    """
    Calculate average losing and winning PnL from realized PnL at point of time.
    State-based: uses portfolio_state['avg_losing_sum'], ['avg_winning_sum'], counts.
    
    Formulas:
        - Avg Losing PnL = Average of all realized PnL at point of time where PnL < 0
        - Avg Winning PnL = Average of all realized PnL at point of time where PnL > 0
    
    Formula: avg = sum(pnl_values) / count(pnl_values)
    
    Args:
        current_realized_pnl_at_point (float or None): Current row's realized PnL at point of time
    
    Returns:
        tuple: (avg_losing_pnl, avg_winning_pnl)
    
    Edge cases:
        - Returns 0.0 if no losing/winning trades exist
        - Filters out None values
        - Only considers realized PnL at point of time (not cumulative)
    """
    if current_realized_pnl_at_point is not None:
        if current_realized_pnl_at_point < 0:
            portfolio_state['avg_losing_sum'] += current_realized_pnl_at_point
            portfolio_state['avg_losing_count'] += 1
        elif current_realized_pnl_at_point > 0:
            portfolio_state['avg_winning_sum'] += current_realized_pnl_at_point
            portfolio_state['avg_winning_count'] += 1
    losing_count = portfolio_state.get('avg_losing_count', 0)
    winning_count = portfolio_state.get('avg_winning_count', 0)
    avg_losing_pnl = (portfolio_state.get('avg_losing_sum', 0.0) / losing_count) if losing_count > 0 else 0.0
    avg_winning_pnl = (portfolio_state.get('avg_winning_sum', 0.0) / winning_count) if winning_count > 0 else 0.0
    return avg_losing_pnl, avg_winning_pnl

def calculate_most_least_profitable(current_ticker, current_realized_pnl_at_point):
    """
    Calculate most and least profitable tickers based on realized PnL at point of time.
    State-based: uses portfolio_state['winning_trades_max'], ['winning_trades_min'], etc.
    
    Formulas:
        - Most Profitable: Ticker where Max(value where realized pnl > 0)
        - Least Profitable: Ticker where Min(value where realized pnl > 0)
    
    Only considers winning trades (PnL > 0).
    
    Returns format: "TICKER PnL_Value" or "TICKER1 PnL1, TICKER2 PnL2" if multiple tickers tie
    
    Args:
        current_ticker (str): Current ticker being traded
        current_realized_pnl_at_point (float or None): Current row's realized PnL at point of time
    
    Returns:
        tuple: (most_profitable, least_profitable) - Formatted strings or "None"
    
    Edge cases:
        - Returns ("None", "None") if no winning trades exist
        - Handles multiple tickers with same max/min PnL
        - Only considers PnL > 0 (winning trades)
    """
    max_pnl = portfolio_state.get('winning_trades_max')
    min_pnl = portfolio_state.get('winning_trades_min')
    max_tickers = portfolio_state.setdefault('winning_trades_max_tickers', [])
    min_tickers = portfolio_state.setdefault('winning_trades_min_tickers', [])

    if current_ticker and current_realized_pnl_at_point is not None and current_realized_pnl_at_point > 0:
        pnl = float(current_realized_pnl_at_point)
        t = current_ticker.upper()
        if max_pnl is None or pnl > max_pnl:
            max_pnl = pnl
            max_tickers = [t]
        elif pnl == max_pnl:
            max_tickers.append(t)

        if min_pnl is None or pnl < min_pnl:
            min_pnl = pnl
            min_tickers = [t]
        elif pnl == min_pnl:
            min_tickers.append(t)

        portfolio_state['winning_trades_max'] = max_pnl
        portfolio_state['winning_trades_min'] = min_pnl
        portfolio_state['winning_trades_max_tickers'] = max_tickers
        portfolio_state['winning_trades_min_tickers'] = min_tickers

    if max_pnl is None or min_pnl is None:
        return "None", "None"

    most_profitable_list = [f"{ticker} {max_pnl}" for ticker in sorted(max_tickers)]
    least_profitable_list = [f"{ticker} {min_pnl}" for ticker in sorted(min_tickers)]
    most_profitable = ", ".join(most_profitable_list) if most_profitable_list else "None"
    least_profitable = ", ".join(least_profitable_list) if least_profitable_list else "None"
    return most_profitable, least_profitable

def calculate_max_drawdown(current_total_pv):
    """
    Calculate Max Drawdown using peak-to-trough method.
    
    Formula: Max Drawdown = max((peak - current) / peak) for all points
    
    This is the standard definition of maximum drawdown:
    - Track the running peak (highest equity seen so far)
    - At each point, calculate drawdown from peak: (peak - current) / peak
    - Return the maximum drawdown encountered
    
    Args:
        current_total_pv (float): Current Total PV value for this row
    
    Returns:
        float: Max Drawdown as percentage (0-100 scale)
    
    Edge cases:
        - Returns 0.0 for first row (no calculation)
        - Returns 0.0 if no previous data
    """
    peak = portfolio_state.get('peak_equity')
    max_dd = portfolio_state.get('max_drawdown', 0.0)
    if peak is None:
        peak = current_total_pv
        max_dd = 0.0
    if current_total_pv > peak:
        peak = current_total_pv
    elif peak > 0:
        drawdown = (peak - current_total_pv) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    portfolio_state['peak_equity'] = peak
    portfolio_state['max_drawdown'] = max_dd
    return max_dd * 100

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
        float: Highest traded volume rounded to 2 decimal places, or 0 if no trades have been recorded
    
    Edge cases:
        - Returns 0 if no trades recorded (highest_traded_volume is None)
        - Rounds to 2 decimal places to preserve sub-dollar volumes (e.g. crypto/penny stocks)
    """
    if portfolio_state['highest_traded_volume'] is None:
        return 0
    return round(portfolio_state['highest_traded_volume'], 2)

def get_lowest_traded_volume():
    """
    Get Lowest Traded Volume = min(quantity * price) across all trades.
    
    Returns the historical minimum traded volume across all trades.
    
    Returns:
        float: Lowest traded volume rounded to 2 decimal places, or 0 if no trades have been recorded
    
    Edge cases:
        - Returns 0 if no trades recorded (lowest_traded_volume is None)
        - Rounds to 2 decimal places to preserve sub-dollar volumes (e.g. crypto/penny stocks)
    """
    if portfolio_state['lowest_traded_volume'] is None:
        return 0
    return round(portfolio_state['lowest_traded_volume'], 2)

# ---------- Average holding days (track opens, detect closes, update AHP) ----------

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

def detect_closed_positions(old_quantities, new_quantities, current_period, current_date, exit_dt=None):
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
        exit_dt (datetime, optional): Pre-parsed current_date; if None, current_date is parsed here

    Returns:
        list: List of dicts with keys 'ticker', 'holding_period', 'closed_qty' for closed positions

    Edge cases:
        - Checks all tickers in both old and new quantities
        - Removes ticker from tracking when position closes
        - Returns empty list if no positions closed
        - If date parsing fails, uses period-based holding period
    """
    closed_positions = []

    if exit_dt is None and current_date is not None:
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

def calculate_average_holding_days(is_closing_trade=False):
    """
    Calculate Average Holding Period (AHP) for all closed trades.
    State-based: uses portfolio_state['cumulative_holding_sum'], ['closed_positions_count'].

    New definition:
        - For each closed summary trade i:
            holdingPeriod_i = (weighted exit date_i − weighted entry date_i) in days
            closeQty_i      = quantity closed in that summary
        - AHP = (Σ(holdingPeriod_i × closeQty_i)) / (Σ closeQty_i)

    Args:
        is_closing_trade (bool): Whether this trade row is closing a position (contains "- Close" in trade string)

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


def calculate_sharpe_ratio(current_realized_pnl_at_point, current_realized_pnl_cumulative, initial_balance, risk_free_rate=0.0):
    """Calculate Sharpe Ratio at this point in time.
    State-based: uses portfolio_state['pnl_stats'] (Welford).

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
        current_realized_pnl_at_point (float or None): Current row's realized PnL at point of time
        current_realized_pnl_cumulative (float): Current cumulative realized PnL (portfolio-level)
        initial_balance (float): Initial portfolio balance
        risk_free_rate (float, optional): Risk-free rate to subtract from portfolio return (default 0.0)

    Returns:
        float or None: Sharpe ratio value, or None if insufficient data.
    """

    if initial_balance is None or initial_balance == 0:
        return None

    stats = portfolio_state.get('pnl_stats', {'n': 0, 'mean': 0.0, 'M2': 0.0})
    if stats['n'] < 2:
        return None
    variance = stats['M2'] / stats['n']
    std_val = float(np.sqrt(variance))
    if std_val == 0 or np.isnan(std_val):
        return None
    std_val = std_val / 100.0
    portfolio_return = float(current_realized_pnl_cumulative) / float(initial_balance)
    excess_return = portfolio_return - float(risk_free_rate)
    return (excess_return / std_val) * 100


def calculate_sortino_ratio(current_realized_pnl_at_point, current_realized_pnl_cumulative, initial_balance, risk_free_rate=0.0):
    """Calculate Sortino Ratio at this point in time.
    State-based: uses portfolio_state['downside_stats'].

    Same numerator as Sharpe, but denominator is **downside deviation** (std of negative returns).

    Implementation here mirrors the Sharpe ratio setup:
        - We use per-trade "returns" implicitly in % units by
          taking raw 'PnL Realized at Point of Time' and scaling std by 100.
        - Downside std is computed only from negative PnL values.

    If there is portfolio return but **no negative PnL values yet**, the Sortino Ratio
    is conceptually infinite, so this function returns ``float('inf')``.

    Args:
        current_realized_pnl_at_point (float or None): Current row's realized PnL at point of time
        current_realized_pnl_cumulative (float): Current cumulative realized PnL (portfolio-level)
        initial_balance (float): Initial portfolio balance
        risk_free_rate (float, optional): Risk-free rate to subtract from portfolio return (default 0.0)

    Returns:
        float or None: Sortino ratio value, ``float('inf')`` if no downside and positive return,
                       or None if insufficient data.
    """

    if initial_balance is None or initial_balance == 0:
        return None

    portfolio_return = float(current_realized_pnl_cumulative) / float(initial_balance)
    excess_return = portfolio_return - float(risk_free_rate)

    downside = portfolio_state.get('downside_stats', {'n': 0, 'mean': 0.0, 'M2': 0.0})
    if downside['n'] == 0:
        return float('inf') if excess_return != 0 else None
    if downside['n'] < 2:
        return None
    variance = downside['M2'] / downside['n']
    downside_std = float(np.sqrt(variance))
    if downside_std == 0 or np.isnan(downside_std):
        return None
    downside_std = downside_std / 100.0
    return (excess_return / downside_std) * 100


def calculate_calmar_ratio(current_account_value, current_date, initial_balance):
    """Calculate Calmar Ratio using Account Value, maximum drawdown (MDD), and annualized return (AAR).
    State-based: uses portfolio_state for calmar_min_date, calmar_max_date, calmar_peak, calmar_max_drawdown.

    Calmar = ARR / MDD

    Where:
        - ARR (here) = Annualized Account Return based on Account Value and Date
          ARR = (Account Value / Initial Balance) ** (365 / days) - 1
          where days is the number of days from first trade date to current date.
        - MDD is computed from the history of Account Value using peak/trough logic in state.

    Args:
        current_account_value (float): Current Account Value
        current_date (str or datetime): Date of the current row
        initial_balance (float): Initial portfolio balance

    Returns:
        float or None: Calmar ratio, ``float('inf')`` if positive ARR but MDD is 0,
                       or 0.0 if no drawdown and no gain/loss, or None if no data.
    """

    if initial_balance is None or initial_balance == 0:
        return None

    dt = _parse_date_value(current_date)
    if dt is None:
        return None

    min_date = portfolio_state.get('calmar_min_date')
    max_date = portfolio_state.get('calmar_max_date')
    if min_date is None or dt < min_date:
        min_date = dt
    if max_date is None or dt > max_date:
        max_date = dt
    portfolio_state['calmar_min_date'] = min_date
    portfolio_state['calmar_max_date'] = max_date

    days = (max_date - min_date).days
    ratio = float(current_account_value) / float(initial_balance)
    if days <= 0:
        # Fallback: simple return when days cannot be computed
        arr = ratio - 1.0
    else:
        try:
            arr = ratio ** (365.0 / days) - 1.0
        except Exception:
            arr = ratio - 1.0

    peak = portfolio_state.get('calmar_peak')
    max_drawdown = portfolio_state.get('calmar_max_drawdown', 0.0)
    if peak is None:
        peak = float(current_account_value)
    if current_account_value > peak:
        peak = float(current_account_value)
    elif peak > 0:
        dd = (peak - float(current_account_value)) / peak
        if dd > max_drawdown:
            max_drawdown = dd

    MIN_DRAWDOWN_THRESHOLD = 0.0001
    if max_drawdown < MIN_DRAWDOWN_THRESHOLD:
        max_drawdown = 0.0

    portfolio_state['calmar_peak'] = peak
    portfolio_state['calmar_max_drawdown'] = max_drawdown

    if max_drawdown == 0:
        if arr > 0.0:
            return float('inf')
        elif arr == 0.0:
            return 0.0
        return None

    return arr / max_drawdown




def calculate_ytd_pnl(current_date, current_realized_pnl_cumulative):
    """
    Calculate YTD PnL (Year-to-Date PnL) = Cumulative realized PnL for the current year.
    State-based: uses portfolio_state['ytd_year'], ['ytd_start_realized_pnl'].
    
    Formula: YTD PnL = Current Cumulative Realized PnL - Cumulative Realized PnL on Jan 1 of current year
    
    Logic:
        - Finds the cumulative realized PnL from the last row before the current year
        - Subtracts it from current cumulative realized PnL
        - This gives realized PnL from January 1st to current date
    
    Args:
        current_date (str or datetime): Current trade date
        current_realized_pnl_cumulative (float): Current cumulative realized PnL
    
    Returns:
        float or None: YTD PnL, or None if date parsing fails
    
    Edge cases:
        - Returns None if current_date is None
        - Uses pd.to_datetime() for flexible date parsing (supports various formats)
        - Returns 0.0 if portfolio started in current year (no previous year data)
        - Uses last row from previous year as baseline
    """
    dt = _parse_date_value(current_date)
    if dt is None:
        return None
    current_year = dt.year
    ytd_year = portfolio_state.get('ytd_year')
    if ytd_year != current_year:
        portfolio_state['ytd_year'] = current_year
        portfolio_state['ytd_start_realized_pnl'] = portfolio_state.get('last_realized_pnl_cumulative', 0.0)
    start_of_year_cumulative = portfolio_state.get('ytd_start_realized_pnl', 0.0)
    return current_realized_pnl_cumulative - start_of_year_cumulative

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

# ---------- Main entry: one trade (or hold) → one row, state updated ----------

def process_trade(ticker, asset_type, side, price, quantity_buy, date=None, take_profit_pct=0.20, stop_loss_pct=0.10, market_cap=None, industry=None, sector=None):
    """
    Process one trade (or hold) and append one row. All metrics come from state.
    
    Same pipeline for buy, sell, and hold. For side='hold', quantity is forced to 0
    so position and cash stay unchanged but equity and metrics still update from price.
    
    Flow: normalize quantity (and force 0 if hold) → update state (qty, cash, cost basis)
    → realized/unrealized PnL → equity and derived metrics (win rate, Sharpe, daily PnL, etc.)
    → build row dict → append to state['rows'], update last_* for next row.
    
    Args:
        ticker (str): Ticker symbol
        asset_type (str): Asset type (e.g., 'Stock', 'Crypto', 'ETF')
        side (str): 'buy', 'sell', or 'hold' (hold forces quantity to 0)
        price (float): Trade price (used for mark-to-market on hold)
        quantity_buy (float or str): Quantity; ignored if side is 'hold'
        date (str or datetime, optional): Trade date
        take_profit_pct (float, optional): Take profit % (default 0.20)
        stop_loss_pct (float, optional): Stop loss % (default 0.10)
    
    Returns:
        dict: One row of portfolio columns (also appended to state['rows']).
    """
    ticker = str(ticker).strip().upper()
    
    q_in = normalize_quantity(quantity_buy)
    if str(side).lower() == 'hold':
        q_in = 0  # hold → same pipeline as other trades, with quantity 0

    # Current state for this ticker
    old_q = portfolio_state['quantities'][ticker]
    old_cb = portfolio_state['cost_basis'][ticker]
    
    # Calculate cash and remaining
    cash = calculate_cash_single()
    new_remaining = calculate_remaining_single(side, price, q_in, old_q, old_cb)

    # Update quantity and determine current position
    new_q = calculate_current_quantity_single(ticker, side, q_in, old_q)
    
    # Row number (1-based) for AHP / period tracking; from state
    current_period = portfolio_state.get('current_period', 0) + 1

    # Detect position flip (long→short or short→long) — needed early for AHP and trades/month
    _is_flip = (old_q > 0 and new_q < 0) or (old_q < 0 and new_q > 0)

    # Determine if this trade opens a new position (fresh open or flip into opposite side)
    is_opening_trade = (old_q == 0 and new_q != 0)

    # A flip = close old position + open new one on the same bar.
    # Treat the close side exactly like a regular close for AHP, then re-open.
    if _is_flip:
        open_dt   = portfolio_state.get('position_open_date', {}).get(ticker)
        exit_dt   = _parse_date_value(date)
        open_period = portfolio_state.get('position_open_period', {}).get(ticker)
        if open_dt is not None and exit_dt is not None:
            holding_period = (exit_dt - open_dt).days
        elif open_period is not None:
            holding_period = current_period - open_period + 1
        else:
            holding_period = None
        if holding_period is not None:
            update_average_holding_days([{'ticker': ticker, 'holding_period': holding_period, 'closed_qty': abs(old_q)}])
        portfolio_state['position_open_period'].pop(ticker, None)
        portfolio_state['position_open_date'].pop(ticker, None)
        track_position_opening(ticker, current_period, date)
    elif is_opening_trade:
        track_position_opening(ticker, current_period, date)
    
    # Backtester Net Performance % (independent state only)
    _bnp_update_on_trade(ticker, price, old_q, new_q)
    
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
    exit_dt = _parse_date_value(date)
    closed_positions = detect_closed_positions(
        old_quantities_dict,
        new_quantities_dict,
        current_period,
        date,
        exit_dt=exit_dt
    )
    
    # Update Average Holding Period aggregators when positions close
    if closed_positions:
        update_average_holding_days(closed_positions)
    
    # Determine current direction: null until first buy/sell; then from quantity; when flat, from previous
    if new_q < 0:
        current_direction = 'short'
    elif new_q > 0:
        current_direction = 'long'
    else:
        # new_q == 0 (flat): use old quantity to know which direction we came from, or null if no trade yet
        if old_q < 0:
            current_direction = 'short'
        elif old_q > 0:
            current_direction = 'long'
        else:
            current_direction = None  # no buy/sell trade yet, direction remains null

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
    long_unrealized, short_unrealized, total_current_ticker_unrealized, total_unrealized_all_tickers = pnl_unrealized_components(new_q, price, avg_p, ticker, price, old_q)

    # Calculate realized total value for current ticker (cumulative realized PnL per ticker)
    prev_realized_total = portfolio_state.get('realized_pnl_by_ticker', {}).get(ticker, 0.0)
    realized_pnl_total_current_ticker = prev_realized_total + (realized_pnl_at_point or 0.0)
    portfolio_state.setdefault('realized_pnl_by_ticker', {})[ticker] = realized_pnl_total_current_ticker
    
    # Calculate PV for current ticker only
    pv_long_current, pv_short_current = calculate_pv_for_current_ticker(price, current_direction, new_q, avg_p, cb)
    
    # Calculate total PV across all tickers
    total_pv, ticker_pv_dict = calculate_total_pv_all_tickers(ticker, price)

    # Generate strings for open positions
    open_pos = open_positions_str()
    open_pv = open_pv_str()
    open_unrealized_pnl = open_pnl_unrealized_str(ticker, price) 
    
    # Calculate account value
    # Formula: account_value = total_equity + available_balance
    total_pv_equity = total_pv + new_remaining
    
    # Calculate Total PnL Overall
    # Formula: total_pnl_overall = equity - initial_cash
    total_pnl_overall = total_pv_equity - cash
    
    # Calculate Daily PnL = Today's Total PnL Overall - Yesterday's Total PnL Overall
    previous_total_pnl_overall = portfolio_state.get('last_total_pnl_overall', 0.0)
    yesterday_equity = portfolio_state.get('last_account_value', cash)
    
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

    # Update running PnL stats for Sharpe/Sortino
    _update_pnl_stats(realized_pnl_at_point)

    # Sharpe Ratio based on cumulative realized PnL and per-trade realized PnL history
    # risk_free_rate kept at 0.0 by default (can be changed by passing a different value)
    sharpe_ratio = calculate_sharpe_ratio(
        realized_pnl_at_point,
        realized_pnl_cumulative,
        cash,
        risk_free_rate=0.0
    )

    # Sortino Ratio using downside deviation of negative PnL values
    sortino_ratio = calculate_sortino_ratio(
        realized_pnl_at_point,
        realized_pnl_cumulative,
        cash,
        risk_free_rate=0.0
    )

    # Calmar Ratio using annualized account return and MDD
    calmar_ratio = calculate_calmar_ratio(
        total_pv_equity,
        date,
        cash
    )

    
    # Calculate Number of Trades = Track trades per ticker
    # Get trade number for this ticker
    trade_number = get_or_create_trade_number(ticker, old_q, new_q, side)
    
    # Format trade string (_is_flip already computed above alongside track_position_opening)
    trade_string = format_trade_string(side, current_direction, trade_number, new_q, is_flip=_is_flip)

    # Calculate Average Holding Days - only calculate when trade closes
    # Check if trade_string contains "- Close" to determine if this is a closing trade
    is_closing_trade = is_close_trade_string(trade_string)
    average_holding_days = calculate_average_holding_days(is_closing_trade=is_closing_trade)
    
    # Calculate total trades (max trade number that has been assigned)
    if trade_string == "No Buy/Sell":
        total_trades_str = portfolio_state.get('last_total_trades_str', "No Buy/Sell")
    else:
        total_trades_str = f"{next_trade_number - 1} Trades"
    portfolio_state['last_total_trades_str'] = total_trades_str
    
    # Calculate Liquidation Price based on current direction and quantity
    liquidation_price = calculate_liquidation_price(current_direction, new_q, avg_p)
    
    # Calculate Take Profit and Stop Loss
    take_profit = calculate_take_profit(current_direction, new_q, avg_p, take_profit_pct)
    stop_loss = calculate_stop_loss(current_direction, new_q, avg_p, stop_loss_pct)
    
    # Win/Loss: per-trade PnL (realized + unrealized for this ticker only), so hold vs trades-only match
    win_loss = calculate_trade_win_loss(trade_string, realized_pnl_at_point, total_current_ticker_unrealized, side)
    update_win_loss_counts_from_trade_pnl(
        ticker, is_closing_trade, realized_pnl_at_point, total_current_ticker_unrealized, new_q
    )
    win_rate = calculate_win_rate(ticker)
    win_loss_ratio = calculate_win_loss_ratio(ticker)
    
    # Calculate Trades/Month — flip counts as opening a new trade.
    # On a flip trade_number is the old (closing) number; the new one is in trade_tracker.
    is_opening_trade = (old_q == 0 and new_q != 0) or _is_flip
    new_trade_num = trade_tracker.get(ticker) if _is_flip else trade_number
    trades_per_month = calculate_trades_per_month(date, is_opening_trade=is_opening_trade, trade_number=new_trade_num)
    
    # Calculate Most/Least Traded (by number of completed trades per symbol)
    abs_quantity_counts, most_traded_symbol, least_traded_symbol = calculate_most_least_traded(
        ticker, q_in, trade_string
    )

    # Calculate Most Bought (only opening positions: LONG+BUY, SHORT+SELL)
    most_bought = calculate_most_bought(ticker, current_direction, side, q_in)
    
    # Calculate Avg Losing PnL and Avg Winning PnL
    avg_losing_pnl, avg_winning_pnl = calculate_avg_losing_winning_pnl(
        realized_pnl_at_point
    )
    
    # Calculate Most/Least Profitable
    most_profitable, least_profitable = calculate_most_least_profitable(
        ticker, realized_pnl_at_point
    )

    # ---------- Backtester Reward/Risk Ratio & Backtester Expectancy ----------
    # Use backtester version that calculates from trade returns
    reward_risk_ratio, expectancy = calculate_backtester_reward_risk_and_expectancy(
        win_rate
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
    direction_lower = str(current_direction or '').lower()
    
    # If Opening a Direction (increments investment_count)
    if (direction_lower == 'long' and side_lower == 'buy') or \
        (direction_lower == 'short' and side_lower == 'sell'):
        investment_count += 1
    
    # Calculate Investment Count
    investment_count_value = calculate_investment_count()

    # Calculate YTD PnL (using cumulative realized PnL)
    ytd_pnl = calculate_ytd_pnl(date, realized_pnl_cumulative)

    # Calculate Distribution and Distribution in %
    distribution, distribution_pct = calculate_diversification(total_pv, ticker_pv_dict)

    # Calculate Equity Distribution 
    # Store equity metadata (market cap, industry, sector) if Equity asset type
    # Priority: explicit argument > hardcoded EQUITY_METADATA > existing stored value
    if inferred_asset_type and inferred_asset_type.lower() == 'equity':
        existing_mc  = portfolio_state['market_cap'].get(ticker)
        existing_ind = portfolio_state['industry'].get(ticker)
        existing_sec = portfolio_state['sector'].get(ticker)

        metadata = EQUITY_METADATA.get(ticker, {})

        mc_value  = market_cap  or metadata.get('market_cap')  or existing_mc
        ind_value = industry    or metadata.get('industry')     or existing_ind
        sec_value = sector      or metadata.get('sector')       or existing_sec

        if mc_value:
            portfolio_state['market_cap'][ticker] = mc_value
        if ind_value:
            portfolio_state['industry'][ticker] = ind_value
        if sec_value:
            portfolio_state['sector'][ticker] = sec_value

    # Calculate Equity Distributions (only for Equity asset types)
    equity_dist_market_cap = calculate_equity_distribution_market_cap(ticker_pv_dict)
    equity_dist_industry = calculate_equity_distribution_industry(ticker_pv_dict)
    equity_dist_sector = calculate_equity_distribution_sector(ticker_pv_dict)

    # Asset Performance %: (current price - starting price) / starting price at every row (buy/sell/hold)
    portfolio_state.setdefault('ticker_start_price', {})
    if ticker not in portfolio_state['ticker_start_price']:
        portfolio_state['ticker_start_price'][ticker] = float(price)
    start_p = portfolio_state['ticker_start_price'][ticker]
    asset_performance_pct = ((float(price) - start_p) / start_p * 100.0) if start_p and start_p > 0 else None

    # Build row dictionary with all calculated values
    row = {
        'Date': date,
        'Ticker': ticker,
        'Asset Type': (inferred_asset_type or asset_type or '').capitalize(),
        'Side': side.capitalize(),
        'Direction': current_direction.capitalize() if current_direction else None,
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
        'Backtester Net Performance %': calculate_backtester_net_performance(current_ticker=ticker, current_price=price),
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Asset Count': asset_count,
        'Asset Performance %': asset_performance_pct,
        'Trade No. (Position - Trade no. - Current Quantity)': trade_string,
        'Total Trades': total_trades_str,
        'Win/Loss': win_loss,
        'Win Rate': win_rate,
        'Win:Loss Ratio': win_loss_ratio,
        'Backtester Reward/Risk Ratio': reward_risk_ratio,
        'Backtester Expectancy': expectancy,
        'Backtester Avg Winning PnL %': calculate_backtester_avg_winning_pnl_pct(),
        'Backtester Avg Losing PnL %': calculate_backtester_avg_losing_pnl_pct(),
        'Backtester Avg PnL %': calculate_backtester_avg_pnl_pct(),
        'Backtester Max Drawdown': calculate_backtester_max_drawdown(),
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
    portfolio_state['last_known_open_trades'] = set(trade_tracker.values())  # for Trades/Month carried count next month
    portfolio_state['remaining'] = new_remaining
    portfolio_state['last_price'][ticker] = price
    portfolio_state['last_total_pnl_overall'] = total_pnl_overall
    portfolio_state['last_account_value'] = total_pv_equity
    portfolio_state['last_realized_pnl_cumulative'] = realized_pnl_cumulative
    portfolio_state['current_period'] = current_period
    portfolio_state['last_trade_date'] = date
    portfolio_state['rows'].append(row)

    return row

    
def add_trade(ticker, asset_type=None, side='buy', price=0.0, quantity_buy=0.0, date=None, take_profit_pct=0.20, stop_loss_pct=0.10, market_cap=None, industry=None, sector=None):
    """
    Add one trade (or hold) to the portfolio. Updates state and appends one row.
    
    This is the PRIMARY FUNCTION for adding trades to the portfolio engine.
    Each call processes the trade, updates positions/cash/metrics, and adds a row to the output.
    
    TRADE COUNTING NOTE:
    -------------------
    The 'Total Trades' metric counts POSITIONS (complete trade cycles), not individual actions:
    • 1 buy + 1 sell = 1 position = "1 Trade"
    • 3 buys + 3 sells = 3 positions = "3 Trades" (NOT 6)
    
    This matches standard trading terminology where a "trade" means opening and closing a position.
    
    Args:
        ticker (str): Ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        asset_type (str, optional): Asset type (e.g., 'Equity', 'Crypto', 'ETF'). Defaults to None.
        side (str): Trade direction - 'buy', 'sell', or 'hold'
                   • 'buy' or 'sell': Normal trade execution
                   • 'hold': Creates a row without changing positions (quantity is forced to 0)
        price (float): Execution price for the trade
        quantity_buy (float or str): Quantity traded
                                     • For 'buy'/'sell': actual quantity
                                     • For 'hold': ignored (forced to 0)
        date (str or datetime, optional): Trade execution date/time. 
                                         Format: 'YYYY-MM-DD HH:MM:SS' or datetime object
        take_profit_pct (float, optional): Take profit percentage for display (default 0.20 = 20%)
        stop_loss_pct (float, optional): Stop loss percentage for display (default 0.10 = 10%)
        market_cap (str, optional): Market cap category for equity distribution (e.g., 'High', 'Mid', 'Low').
        industry (str, optional): Industry for equity distribution (e.g., 'Software', 'Healthcare').
        sector (str, optional): Sector for equity distribution (e.g., 'Technology', 'Financials').
    
    Returns:
        None. The trade is processed and stored internally. 
        Call get_portfolio_df() to retrieve the complete portfolio DataFrame with all metrics.
    
    Example:
        >>> reset_portfolio(10000)
        >>> add_trade('AAPL', 'Equity', 'buy', 150.0, 10, '2024-01-01')
        >>> add_trade('AAPL', 'Equity', 'sell', 155.0, 10, '2024-01-15')
        >>> df = get_portfolio_df()
        >>> print(df[['Date', 'Side', 'Current Quantity', 'Account Value', 'Total Trades']])
        # Total Trades will show "1 Trades" (1 position: buy+sell pair)
    """
    process_trade(ticker, asset_type, side, price, quantity_buy, date, take_profit_pct, stop_loss_pct, market_cap=market_cap, industry=industry, sector=sector)


# ---------- Formula reference (for docs / export) ----------

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
        'Equity (Long)': 'Cost Basis + (Price - Avg Price) * Qty (long positions only)',
        'Equity (Short)': 'Cost Basis + (Avg Price - Price) * |Qty| (short positions only)',
        'Open Position': 'String of All Open Positions',
        'Open Equity': 'String of All Open Position Equities',
        'Total Equity': 'Σ(Equity for each ticker) where Equity = Cost Basis + Unrealized PnL',
        'Account Value': 'Total Equity + Available Balance',
        'Realized PnL at Point of Time (Portfolio)': 'Sum of All Realized PnL up to this point (portfolio-level)',
        'Unrealized PnL at Point of Time (Portfolio)': 'Sum of Unrealized PnL across ALL tickers at this point in time (portfolio-level)',
        'Total PnL Overall (Unrealized+Realized)': 'Account Value - Initial Balance',
        'Daily PnL (Unrealized+Realized)': 'Today Total PnL Overall - Yesterday Total PnL Overall',
        'Liquidation Price': 'Long: 0, Short: 2 * Avg Price',
        'Take Profit': 'Long: Avg Price * (1 + %), Short: Avg Price * (1 - %)',
        'Stop Loss': 'Long: Avg Price * (1 - %), Short: Avg Price * (1 + %)',
        'Last Day Pnl / Daily $': 'Same as Daily PnL',
        'Daily %': '((Today Account Value - Yesterday Account Value) / Yesterday Account Value) * 100',
        'Cumulative %': '((Account Value / Initial Balance) - 1) * 100',
        'Investment Count': 'Cumulative Count of Positions Opened',
        'Performance': 'Same as Cumulative %',
        'Backtester Net Performance %': 'Compound returns: [(1 + R1) × (1 + R2) × ... × (1 + Rn) - 1] × 100, where each R is trade return (assumes 100% capital reinvestment per trade)',
        'Sharpe Ratio': '(Excess Return / StdDev(PnL Realized at Point of Time)); excess = (Realized PnL Portfolio / Initial Balance) - Risk Free Rate',
        'Sortino Ratio': '(Excess Return / DownsideDeviation(negative PnL Realized at Point of Time)); excess = (Realized PnL Portfolio / Initial Balance) - Risk Free Rate',
        'Calmar Ratio': 'Annualized Account Return (ARR) / Max Drawdown (MDD); ARR = (Account Value / Initial Balance)^(365/days) - 1; MDD from peak/trough of Account Value',
        'Asset Count': 'Count of Distinct Tickers per Asset Type where Qty≠0',
        'Asset Performance %': '(Price - Ticker Start Price) / Ticker Start Price × 100; start price = first price seen for that ticker; computed at every row (buy/sell/hold)',
        'Trade No. (Position - Trade no. - Current Quantity)': 'Formatted Trade String',
        'Total Trades': 'Max Trade Number Assigned',
        'Win/Loss': 'Win if Realized PnL>0, Loss if ≤0, None if No Close',
        'Win Rate': '(Wins / Total Closed Trades) * 100',
        'Win:Loss Ratio': 'Win Count : Loss Count',
        'Backtester Reward/Risk Ratio': 'Backtester Avg Winning PnL % / |Backtester Avg Losing PnL %| (from trade returns)',
        'Backtester Expectancy': '(Reward/Risk Ratio × Win Ratio) - Loss Ratio; Win Ratio = Win Rate/100, Loss Ratio = 1 - Win Ratio',
        'Backtester Avg Winning PnL %': 'Average of all positive trade returns × 100',
        'Backtester Avg Losing PnL %': 'Average of all negative trade returns × 100',
        'Backtester Avg PnL %': 'Average of all trade returns × 100',
        'Backtester Max Drawdown': 'Max peak-to-trough decline % on equity curve from compounding trade returns: equity[i] = equity[i-1]×(1+return[i]); drawdown = (equity - peak)/peak',
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
        'Total Gain': 'PnL Realized Cumulative + PnL Unrealized at Point of Time (all tickers)',
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
        df = get_portfolio_df()
    
    # Get formulas dictionary
    formulas_dict = get_formulas_dict()
    
    # Create formulas row - map each column to its formula
    formulas_row = {col: formulas_dict.get(col, '') for col in COLUMNS}
    
    # Create DataFrame with formulas row
    formulas_df = pd.DataFrame([formulas_row], columns=pd.Index(COLUMNS))
    
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


# ---------- Test / demo (runs only when this file is executed directly) ----------

if __name__ == "__main__":
    reset_portfolio(200)
    add_trade(ticker='aapl', side='hold', price=10, quantity_buy=0, date='1/1/2025')
    add_trade(ticker='aapl', side='buy', price=10, quantity_buy=10, date='1/2/2025')
    add_trade(ticker='aapl', side='hold', price=11, quantity_buy=0, date='1/3/2025')
    add_trade(ticker='aapl', side='hold', price=12, quantity_buy=0, date='1/4/2025')
    add_trade(ticker='aapl', side='sell', price=12, quantity_buy=1, date='1/5/2025')
    add_trade(ticker='aapl', side='hold', price=13, quantity_buy=0, date='1/6/2025')
    add_trade(ticker='msft', side='buy', price=5, quantity_buy=5, date='1/7/2025')
    add_trade(ticker='msft', side='hold', price=6, quantity_buy=0, date='2/5/2025')
    add_trade(ticker='aapl', side='buy', price=13, quantity_buy=2, date='2/6/2025')
    add_trade(ticker='aapl', side='sell', price=13, quantity_buy=11, date='2/7/2025')
    add_trade(ticker='aapl', side='sell', price=12, quantity_buy=10, date='2/8/2025') 
    add_trade(ticker='msft', side='sell', price=8, quantity_buy=1, date='2/9/2025')
    add_trade(ticker='msft', side='buy', price=9, quantity_buy=2, date='3/2/2025')
    add_trade(ticker='msft', side='sell', price=9, quantity_buy=6, date='3/3/2025')
    add_trade(ticker='tsla', side='sell', price=20, quantity_buy=5, date='3/4/2025') 
    add_trade(ticker='tsla', side='hold', price=22, quantity_buy=0, date='3/5/2025')
    add_trade(ticker='tsla', side='buy', price=21, quantity_buy=4, date='3/6/2025')  
    add_trade(ticker='tsla', side='sell', price=22, quantity_buy=1, date='3/7/2025')  
    add_trade(ticker='tsla', side='buy', price=24, quantity_buy=2, date='3/8/2025') 

    df = get_portfolio_df()
    df_with_formulas = add_formulas_row_to_df(df)
    print(df_with_formulas)  # portfolio output


"""
============================================================================
KEY METRICS EXPLAINED
============================================================================

This section explains the most important metrics calculated by portfolio.py.

TRADE COUNTING:
--------------
• Total Trades: Counts POSITIONS (complete trade cycles), not individual buy/sell actions
  Example: 3 buys + 3 sells = 3 positions = "3 Trades"
  
• Trade No.: Format "Long - Buy - #1 Trade - 5" or "Long - Sell - #1 Trade - 0 - Close"
  Shows position side, action, trade number, and current quantity

PROFIT & LOSS:
--------------
• PnL Realized: Profit/loss from closed positions only
• PnL Unrealized: Profit/loss from open positions (mark-to-market)
• Total PnL: Realized + Unrealized combined
• Daily PnL: Change in total P&L from previous row

RETURNS:
--------
• Backtester Net Performance %: Overall return on initial capital
  Formula: (Account Value - Initial Balance) / Initial Balance * 100
  
• Asset Performance %: Buy-and-hold return of the underlying asset
  Formula: (Current Price - First Price) / First Price * 100
  
• Daily %: Daily return percentage
• Cumulative %: Running cumulative return

RISK-ADJUSTED RETURNS:
---------------------
• Sharpe Ratio: Return per unit of total risk (volatility)
  Higher is better. >1.0 is good, >2.0 is very good
  Formula: (Portfolio Return - Risk Free Rate) / StdDev of Returns
  
• Sortino Ratio: Return per unit of downside risk only
  Similar to Sharpe but only penalizes downside volatility
  Higher is better
  
• Calmar Ratio: Return per unit of maximum drawdown
  Measures return relative to worst-case scenario loss
  Higher is better

TRADE QUALITY:
-------------
• Win Rate: Percentage of winning trades
  Formula: Winning Trades / Total Trades * 100
  
• Win:Loss Ratio: Ratio of winning to losing trades
  Example: "2:1" means 2 wins for every 1 loss
  
• Reward/Risk Ratio: Average win size vs average loss size
  Formula: Avg Winning PnL / Avg Losing PnL
  
• Expectancy: Expected profit per trade
  Formula: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
  Positive expectancy means profitable system

DRAWDOWN:
---------
• Max Drawdown: Largest peak-to-trough decline
  Shows worst-case loss from highest portfolio value
  Always shown as negative percentage
  
• Backtester Max Drawdown: Maximum drawdown for the strategy

POSITION METRICS:
----------------
• Current Quantity: Number of shares/units currently held
• Avg Price: Average entry price for current position
• Cost Basis: Total cost of current position
• Equity: Current market value of position
• Open Position: List of tickers with open positions

PORTFOLIO METRICS:
-----------------
• Account Value: Total portfolio value (cash + positions)
• Available Balance: Cash available for new trades
• Investment Count: Number of different positions held
• Asset Count: Number of unique tickers traded

ADDITIONAL METRICS:
------------------
• Liquidation Price: Price where position would be liquidated (for shorts)
• Take Profit: Target price for taking profit
• Stop Loss: Price where stop loss would trigger
• Trades/Month: Average monthly trade frequency
• Average Holding Days: Average days positions are held
• Holdings: Text representation of current positions

For complete formula documentation, call get_formulas_dict() or 
use get_portfolio_df_with_formulas() to get a DataFrame with formulas as first row.
"""
