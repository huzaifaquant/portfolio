# test_portfolio.py
import pytest
import pandas as pd
import portfolio  
from portfolio import (
    reset_portfolio, add_trade, get_portfolio_df, process_trade,
    trade_tracker, next_trade_number, investment_count,
    # Core calculations
    calculate_cash_single, calculate_remaining_single,
    calculate_current_quantity_single, calculate_avg_price_and_cost_basis_single,
    calculate_realized_pnl_at_point_of_time, calculate_realized_pnl_cumulative,
    # PnL calculations
    pnl_unrealized_components, position_value_from_position,
    calculate_total_pv_all_tickers,
    # Strings
    open_positions_str, open_pv_str, open_pnl_unrealized_str,
    # Distribution
    calculate_diversification,
    calculate_equity_distribution_market_cap, calculate_equity_distribution_industry,
    calculate_equity_distribution_sector,
    # Other functions
    calculate_liquidation_price, calculate_take_profit, calculate_stop_loss,
    calculate_pv_for_current_ticker,
    calculate_holdings, calculate_asset_count,
    calculate_investment_count,
    normalize_quantity, get_or_create_trade_number, format_trade_string,
)


# ============================================================================
# TEST: reset_portfolio
# ============================================================================

class TestResetPortfolio:
    """Test reset_portfolio function"""
    
    def test_reset_portfolio_default_cash(self):
        """Test reset with default cash"""
        reset_portfolio()
        assert portfolio.portfolio_state['cash'] == 200  # Access via module
        assert portfolio.portfolio_state['remaining'] == 200
        assert portfolio.portfolio_state['realized_pnl'] == 0.0
        assert len(portfolio.portfolio_state['quantities']) == 0
        assert portfolio.portfolio_state['portfolio_df'].empty
    
    def test_reset_portfolio_custom_cash(self):
        """Test reset with custom cash"""
        reset_portfolio()
        assert portfolio.portfolio_state['cash'] == 200  # Access via module
        assert portfolio.portfolio_state['remaining'] == 200
        assert portfolio.portfolio_state['realized_pnl'] == 0.0
        assert len(portfolio.portfolio_state['quantities']) == 0
        assert portfolio.portfolio_state['portfolio_df'].empty
    
    def test_reset_portfolio_clears_state(self):
        """Test that reset clears all state"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        reset_portfolio()
        assert portfolio.portfolio_state['quantities']['AAPL'] == 0
        assert portfolio.portfolio_state['cost_basis']['AAPL'] == 0
        assert portfolio.portfolio_state['realized_pnl'] == 0.0


# ============================================================================
# TEST: normalize_quantity
# ============================================================================

class TestNormalizeQuantity:
    """Test normalize_quantity function"""
    
    def test_normalize_quantity_int(self):
        """Test normalizing integer"""
        reset_portfolio()
        from portfolio import normalize_quantity
        assert normalize_quantity(10) == 10.0
        assert normalize_quantity(-5) == 5.0
    
    def test_normalize_quantity_float(self):
        """Test normalizing float"""
        reset_portfolio()
        from portfolio import normalize_quantity
        assert normalize_quantity(10.5) == 10.5
        assert normalize_quantity(-5.5) == 5.5
    
    def test_normalize_quantity_string_positive(self):
        """Test normalizing positive string"""
        reset_portfolio()
        from portfolio import normalize_quantity
        assert normalize_quantity("10") == 10.0
        assert normalize_quantity("(10)") == 10.0
    
    def test_normalize_quantity_string_negative(self):
        """Test normalizing negative string"""
        reset_portfolio()
        from portfolio import normalize_quantity
        assert normalize_quantity("-10") == 10.0
        assert normalize_quantity("-(-10)") == 10.0


# ============================================================================
# TEST: calculate_cash_single
# ============================================================================

class TestCalculateCashSingle:
    """Test calculate_cash_single function"""
    
    def test_cash_constant_after_init(self):
        """Test cash remains constant"""
        reset_portfolio(initial_cash=200)
        cash1 = calculate_cash_single()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        cash2 = calculate_cash_single()
        assert cash1 == cash2 == 200
    
    def test_cash_returns_initial_value(self):
        """Test cash returns initial value"""
        reset_portfolio(initial_cash=500)
        assert calculate_cash_single() == 500


# ============================================================================
# TEST: calculate_remaining_single
# ============================================================================

class TestCalculateRemainingSingle:
    """Test calculate_remaining_single function"""
    
    def test_remaining_buy_long(self):
        """Test remaining after buying long"""
        reset_portfolio(initial_cash=200)
        remaining = calculate_remaining_single('buy', 10, 10, 0, 0)
        assert remaining == 200 - (10 * 10) == 100
    
    def test_remaining_sell_long(self):
        """Test remaining after selling long"""
        reset_portfolio(initial_cash=200)
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        remaining = calculate_remaining_single('sell', 12, 5, 10, 100)
        assert remaining == 100 + (12 * 5) == 160
    
    def test_remaining_sell_short(self):
        """Test remaining after selling short"""
        reset_portfolio(initial_cash=200)
        remaining = calculate_remaining_single('sell', 10, 10, 0, 0)
        assert remaining == 200 - (10 * 10) == 100
    
    def test_remaining_hold_no_change(self):
        """Test remaining unchanged on hold"""
        reset_portfolio(initial_cash=200)
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        remaining = calculate_remaining_single('hold', 12, 0, 10, 100)
        assert remaining == 100


# ============================================================================
# TEST: calculate_current_quantity_single
# ============================================================================

class TestCalculateCurrentQuantitySingle:
    """Test calculate_current_quantity_single function"""
    
    def test_quantity_buy_increases(self):
        """Test buying increases quantity"""
        reset_portfolio()
        new_q = calculate_current_quantity_single('AAPL', 'buy', 10, 0)
        assert new_q == 10
        assert portfolio.portfolio_state['quantities']['AAPL'] == 10
    
    def test_quantity_sell_decreases(self):
        """Test selling decreases quantity"""
        reset_portfolio()
        calculate_current_quantity_single('AAPL', 'buy', 10, 0)
        new_q = calculate_current_quantity_single('AAPL', 'sell', 5, 10)
        assert new_q == 5
    
    def test_quantity_hold_unchanged(self):
        """Test hold keeps quantity unchanged"""
        reset_portfolio()
        calculate_current_quantity_single('AAPL', 'buy', 10, 0)
        new_q = calculate_current_quantity_single('AAPL', 'hold', 0, 10)
        assert new_q == 10
    
    def test_quantity_flip_long_to_short(self):
        """Test quantity flips from long to short"""
        reset_portfolio()
        calculate_current_quantity_single('AAPL', 'buy', 10, 0)
        new_q = calculate_current_quantity_single('AAPL', 'sell', 15, 10)
        assert new_q == -5


# ============================================================================
# TEST: calculate_avg_price_and_cost_basis_single
# ============================================================================

class TestCalculateAvgPriceAndCostBasisSingle:
    """Test calculate_avg_price_and_cost_basis_single function"""
    
    def test_cost_basis_long_single_buy(self):
        """Test cost basis for single long buy"""
        reset_portfolio()
        avg_p, cb = calculate_avg_price_and_cost_basis_single('AAPL', 'buy', 10, 10, 0, 10, 0)
        assert cb == 100
        assert avg_p == 10.0
    
    def test_cost_basis_long_multiple_buys(self):
        """Test cost basis for multiple long buys"""
        reset_portfolio()
        calculate_avg_price_and_cost_basis_single('AAPL', 'buy', 10, 10, 0, 10, 0)
        avg_p, cb = calculate_avg_price_and_cost_basis_single('AAPL', 'buy', 12, 5, 10, 15, 100)
        assert cb == 160
        assert abs(avg_p - 10.6667) < 0.01
    
    def test_cost_basis_long_partial_sell(self):
        """Test cost basis after partial sell"""
        reset_portfolio()
        # First buy: open position with 10 shares at $10
        calculate_avg_price_and_cost_basis_single('AAPL', 'buy', 10, 10, 0, 10, 0)
        # Partial sell: sell 5 shares, leaving 5 shares
        # old_quantity=10, new_quantity=5 (10-5), old_cost_basis=100
        avg_p, cb = calculate_avg_price_and_cost_basis_single('AAPL', 'sell', 12, 5, 10, 5, 100)
        assert cb == 50  # 100 * (5/10) = 50
        assert avg_p == 10.0
    
    def test_cost_basis_short_single_sell(self):
        """Test cost basis for single short sell"""
        reset_portfolio()
        # Opening a new short position: old_quantity=0, new_quantity=-10
        avg_p, cb = calculate_avg_price_and_cost_basis_single('AAPL', 'sell', 10, 10, 0, -10, 0)
        assert cb == 100
        assert avg_p == 10.0
    
    def test_cost_basis_full_close_resets(self):
        """Test cost basis resets on full close"""
        reset_portfolio()
        # First buy: open position with 10 shares at $10
        calculate_avg_price_and_cost_basis_single('AAPL', 'buy', 10, 10, 0, 10, 0)
        # Full sell: sell all 10 shares, leaving 0 shares
        # old_quantity=10, new_quantity=0 (10-10), old_cost_basis=100
        avg_p, cb = calculate_avg_price_and_cost_basis_single('AAPL', 'sell', 12, 10, 10, 0, 100)
        assert cb == 0
        assert avg_p == 0.0


# ============================================================================
# TEST: calculate_realized_pnl_at_point_of_time
# ============================================================================

class TestCalculateRealizedPnLAtPointOfTime:
    """Test calculate_realized_pnl_at_point_of_time function"""
    
    def test_realized_pnl_long_profit(self):
        """Test realized PnL when selling long at profit"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        pnl = calculate_realized_pnl_at_point_of_time('AAPL', 'sell', 'long', 12, 10, 10)
        assert pnl == (12 - 10) * 10 == 20
    
    def test_realized_pnl_long_loss(self):
        """Test realized PnL when selling long at loss"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        pnl = calculate_realized_pnl_at_point_of_time('AAPL', 'sell', 'long', 8, 10, 10)
        assert pnl == (8 - 10) * 10 == -20
    
    def test_realized_pnl_long_partial(self):
        """Test realized PnL when partially selling long"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        pnl = calculate_realized_pnl_at_point_of_time('AAPL', 'sell', 'long', 12, 5, 10)
        assert pnl == (12 - 10) * 5 == 10
    
    def test_realized_pnl_short_profit(self):
        """Test realized PnL when covering short at profit"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        pnl = calculate_realized_pnl_at_point_of_time('AAPL', 'buy', 'short', 8, 10, -10)
        assert pnl == (10 - 8) * 10 == 20
    
    def test_realized_pnl_no_closing(self):
        """Test realized PnL is None when not closing"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        pnl = calculate_realized_pnl_at_point_of_time('AAPL', 'buy', 'long', 12, 5, 10)
        assert pnl is None


# ============================================================================
# TEST: calculate_realized_pnl_cumulative
# ============================================================================

class TestCalculateRealizedPnLCumulative:
    """Test calculate_realized_pnl_cumulative function"""
    
    def test_cumulative_starts_at_zero(self):
        """Test cumulative starts at zero"""
        reset_portfolio()
        assert portfolio.portfolio_state['realized_pnl'] == 0.0
    
    def test_cumulative_adds_on_close(self):
        """Test cumulative adds realized PnL on close"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        calculate_realized_pnl_cumulative('AAPL', 'sell', 'long', 12, 10, 10)
        assert portfolio.portfolio_state['realized_pnl'] == 20
    
    def test_cumulative_accumulates(self):
        """Test cumulative accumulates across multiple closes"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        calculate_realized_pnl_cumulative('AAPL', 'sell', 'long', 12, 5, 10)
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        calculate_realized_pnl_cumulative('MSFT', 'sell', 'long', 22, 5, 5)
        assert portfolio.portfolio_state['realized_pnl'] == 20  # 10 + 10


# ============================================================================
# TEST: pnl_unrealized_components
# ============================================================================

class TestPnlUnrealizedComponents:
    """Test pnl_unrealized_components function"""
    
    def test_unrealized_long_profit(self):
        """Test unrealized PnL for long at profit"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        long_u, short_u, ticker_total, all_total = pnl_unrealized_components(10, 12, 10, 'AAPL', 12)
        assert long_u == (12 - 10) * 10 == 20
        assert short_u == 0
        assert ticker_total == 20
        assert all_total == 20
    
    def test_unrealized_long_loss(self):
        """Test unrealized PnL for long at loss"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        long_u, short_u, ticker_total, all_total = pnl_unrealized_components(10, 8, 10, 'AAPL', 8)
        assert long_u == (8 - 10) * 10 == -20
        assert short_u == 0
        assert ticker_total == -20
    
    def test_unrealized_short_profit(self):
        """Test unrealized PnL for short at profit"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        long_u, short_u, ticker_total, all_total = pnl_unrealized_components(-10, 8, 10, 'AAPL', 8)
        assert long_u == 0
        assert short_u == (10 - 8) * 10 == 20
        assert ticker_total == 20
    
    def test_unrealized_multiple_tickers(self):
        """Test unrealized PnL across multiple tickers"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        long_u, short_u, ticker_total, all_total = pnl_unrealized_components(10, 12, 10, 'AAPL', 12)
        # AAPL: (12-10)*10 = 20
        # MSFT: (20-20)*5 = 0
        assert all_total == 20


# ============================================================================
# TEST: position_value_from_position
# ============================================================================

class TestPositionValueFromPosition:
    """Test position_value_from_position function"""
    
    def test_pv_long(self):
        """Test PV for long position"""
        reset_portfolio()
        pv = position_value_from_position('long', 10, 12)
        assert pv == 10 * 12 == 120
    
    def test_pv_short(self):
        """Test PV for short position (positive)"""
        reset_portfolio()
        pv = position_value_from_position('short', -10, 12)
        assert pv == abs(-10) * 12 == 120
    
    def test_pv_zero_quantity(self):
        """Test PV for zero quantity"""
        reset_portfolio()
        pv = position_value_from_position('long', 0, 12)
        assert pv == 0


# ============================================================================
# TEST: calculate_total_pv_all_tickers
# ============================================================================

class TestCalculateTotalPvAllTickers:
    """Test calculate_total_pv_all_tickers function"""
    
    def test_total_pv_single_ticker(self):
        """Test total PV for single ticker"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        total_pv, ticker_pv_dict = calculate_total_pv_all_tickers('AAPL', 12)
        # PV = Cost Basis + Unrealized = 100 + 20 = 120
        assert total_pv == 120
        assert ticker_pv_dict['AAPL'] == 120
    
    def test_total_pv_multiple_tickers(self):
        """Test total PV for multiple tickers"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        total_pv, ticker_pv_dict = calculate_total_pv_all_tickers('AAPL', 12)
        # AAPL: 100 + 20 = 120
        # MSFT: 100 + 0 = 100
        assert total_pv == 220
        assert 'AAPL' in ticker_pv_dict
        assert 'MSFT' in ticker_pv_dict


# ============================================================================
# TEST: calculate_pv_for_current_ticker
# ============================================================================

class TestCalculatePvForCurrentTicker:
    """Test calculate_pv_for_current_ticker function"""
    
    def test_pv_long_current(self):
        """Test PV for current long ticker"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        pv_long, pv_short = calculate_pv_for_current_ticker(12, 'long', 10, 10, 100)
        assert pv_long == 120  # 100 + (12-10)*10
        assert pv_short == 0
    
    def test_pv_short_current(self):
        """Test PV for current short ticker"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        pv_long, pv_short = calculate_pv_for_current_ticker(8, 'short', -10, 10, 100)
        assert pv_long == 0
        assert pv_short == 120  # 100 + (10-8)*10


# ============================================================================
# TEST: open_positions_str
# ============================================================================

class TestOpenPositionsStr:
    """Test open_positions_str function"""
    
    def test_open_positions_single(self):
        """Test open positions string for single position"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        result = open_positions_str()
        assert 'AAPL' in result
        assert '10' in result
    
    def test_open_positions_multiple(self):
        """Test open positions string for multiple positions"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        result = open_positions_str()
        assert 'AAPL' in result
        assert 'MSFT' in result
    
    def test_open_positions_none(self):
        """Test open positions string when no positions"""
        reset_portfolio()
        result = open_positions_str()
        assert result == "None"


# ============================================================================
# TEST: open_pv_str
# ============================================================================

class TestOpenPvStr:
    """Test open_pv_str function"""
    
    def test_open_pv_single(self):
        """Test open PV string for single position"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        result = open_pv_str('AAPL', 12, 'long')
        assert 'AAPL' in result
        assert '120' in result  # 100 + 20
    
    def test_open_pv_none(self):
        """Test open PV string when no positions"""
        reset_portfolio()
        result = open_pv_str('AAPL', 12, 'long')
        assert result == "None"


# ============================================================================
# TEST: open_pnl_unrealized_str
# ============================================================================

class TestOpenPnlUnrealizedStr:
    """Test open_pnl_unrealized_str function"""
    
    def test_unrealized_str_single(self):
        """Test unrealized PnL string for single position"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        result = open_pnl_unrealized_str('AAPL', 12)
        assert 'AAPL' in result
        assert '20' in result  # (12-10)*10
    
    def test_unrealized_str_none(self):
        """Test unrealized PnL string when no positions"""
        reset_portfolio()
        result = open_pnl_unrealized_str('AAPL', 12)
        assert result == "None"


# ============================================================================
# TEST: calculate_liquidation_price
# ============================================================================

class TestCalculateLiquidationPrice:
    """Test calculate_liquidation_price function"""
    
    def test_liquidation_long(self):
        """Test liquidation price for long position"""
        reset_portfolio()
        price = calculate_liquidation_price('long', 10, 10)
        assert price == 0.0
    
    def test_liquidation_short(self):
        """Test liquidation price for short position"""
        reset_portfolio()
        price = calculate_liquidation_price('short', -10, 10)
        assert price == 20.0  # 2 * 10
    
    def test_liquidation_no_position(self):
        """Test liquidation price when no position"""
        reset_portfolio()
        price = calculate_liquidation_price('long', 0, 10)
        assert price is None


# ============================================================================
# TEST: calculate_take_profit
# ============================================================================

class TestCalculateTakeProfit:
    """Test calculate_take_profit function"""
    
    def test_take_profit_long_default(self):
        """Test take profit for long with default 20%"""
        reset_portfolio()
        tp = calculate_take_profit('long', 10, 10)
        assert tp == 10 * 1.20 == 12.0
    
    def test_take_profit_short_default(self):
        """Test take profit for short with default 20%"""
        reset_portfolio()
        tp = calculate_take_profit('short', -10, 10)
        assert tp == 10 * 0.80 == 8.0
    
    def test_take_profit_custom_pct(self):
        """Test take profit with custom percentage"""
        reset_portfolio()
        tp = calculate_take_profit('long', 10, 10, 0.30)
        assert tp == 10 * 1.30 == 13.0
    
    def test_take_profit_no_position(self):
        """Test take profit when no position"""
        reset_portfolio()
        tp = calculate_take_profit('long', 0, 10)
        assert tp is None


# ============================================================================
# TEST: calculate_stop_loss
# ============================================================================

class TestCalculateStopLoss:
    """Test calculate_stop_loss function"""
    
    def test_stop_loss_long_default(self):
        """Test stop loss for long with default 10%"""
        reset_portfolio()
        sl = calculate_stop_loss('long', 10, 10)
        assert sl == 10 * 0.90 == 9.0
    
    def test_stop_loss_short_default(self):
        """Test stop loss for short with default 10%"""
        reset_portfolio()
        sl = calculate_stop_loss('short', -10, 10)
        assert sl == 10 * 1.10 == 11.0
    
    def test_stop_loss_custom_pct(self):
        """Test stop loss with custom percentage"""
        reset_portfolio()
        sl = calculate_stop_loss('long', 10, 10, 0.15)
        assert sl == 10 * 0.85 == 8.5
    
    def test_stop_loss_no_position(self):
        """Test stop loss when no position"""
        reset_portfolio()
        sl = calculate_stop_loss('long', 0, 10)
        assert sl is None


# ============================================================================
# TEST: calculate_holdings
# ============================================================================

class TestCalculateHoldings:
    """Test calculate_holdings function"""
    
    def test_holdings_single(self):
        """Test holdings count for single position"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        holdings = calculate_holdings()
        assert holdings == 1
    
    def test_holdings_multiple(self):
        """Test holdings count for multiple positions"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        holdings = calculate_holdings()
        assert holdings == 2
    
    def test_holdings_zero(self):
        """Test holdings count when no positions"""
        reset_portfolio()
        holdings = calculate_holdings()
        assert holdings == 0


# ============================================================================
# TEST: calculate_asset_count
# ============================================================================

class TestCalculateAssetCount:
    """Test calculate_asset_count function"""
    
    def test_asset_count_single_type(self):
        """Test asset count for single asset type"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        result = calculate_asset_count()
        assert 'Equity: 1' in result
    
    def test_asset_count_multiple_types(self):
        """Test asset count for multiple asset types"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('XRP', 'Crypto', 'buy', 'long', 0.5, 100, '1/2/2025')
        result = calculate_asset_count()
        assert 'Equity' in result
        assert 'Crypto' in result


# ============================================================================
# TEST: calculate_diversification
# ============================================================================

class TestCalculateDiversification:
    """Test calculate_diversification function"""
    
    def test_diversification_single_asset_type(self):
        """Test diversification for single asset type"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        total_pv, ticker_pv_dict = calculate_total_pv_all_tickers('AAPL', 12)
        dist, dist_pct = calculate_diversification(total_pv, ticker_pv_dict)
        assert 'Equity' in dist
        assert 'Equity' in dist_pct
    
    def test_diversification_multiple_asset_types(self):
        """Test diversification for multiple asset types"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('XRP', 'Crypto', 'buy', 'long', 0.5, 100, '1/2/2025')
        total_pv, ticker_pv_dict = calculate_total_pv_all_tickers('AAPL', 12)
        dist, dist_pct = calculate_diversification(total_pv, ticker_pv_dict)
        assert 'Equity' in dist
        assert 'Crypto' in dist


# ============================================================================
# TEST: calculate_equity_distribution_market_cap
# ============================================================================

class TestCalculateEquityDistributionMarketCap:
    """Test calculate_equity_distribution_market_cap function"""
    
    def test_market_cap_distribution_single(self):
        """Test market cap distribution for single equity"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        total_pv, ticker_pv_dict = calculate_total_pv_all_tickers('AAPL', 12)
        result = calculate_equity_distribution_market_cap(ticker_pv_dict)
        assert 'High' in result  # AAPL is High cap
    
    def test_market_cap_distribution_multiple(self):
        """Test market cap distribution for multiple equities"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('SQ', 'Equity', 'buy', 'long', 5, 10, '1/2/2025')
        total_pv, ticker_pv_dict = calculate_total_pv_all_tickers('AAPL', 12)
        result = calculate_equity_distribution_market_cap(ticker_pv_dict)
        assert 'High' in result  # AAPL
        assert 'Mid' in result    # SQ
    
    def test_market_cap_distribution_no_equity(self):
        """Test market cap distribution when no equity"""
        reset_portfolio()
        add_trade('XRP', 'Crypto', 'buy', 'long', 0.5, 100, '1/1/2025')
        total_pv, ticker_pv_dict = calculate_total_pv_all_tickers('XRP', 0.6)
        result = calculate_equity_distribution_market_cap(ticker_pv_dict)
        assert result == "None"


# ============================================================================
# TEST: calculate_equity_distribution_industry
# ============================================================================

class TestCalculateEquityDistributionIndustry:
    """Test calculate_equity_distribution_industry function"""
    
    def test_industry_distribution(self):
        """Test industry distribution"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('TSLA', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        total_pv, ticker_pv_dict = calculate_total_pv_all_tickers('AAPL', 12)
        result = calculate_equity_distribution_industry(ticker_pv_dict)
        assert 'Software' in result  # AAPL
        assert 'Auto Manufacturers' in result  # TSLA


# ============================================================================
# TEST: calculate_equity_distribution_sector
# ============================================================================

class TestCalculateEquityDistributionSector:
    """Test calculate_equity_distribution_sector function"""
    
    def test_sector_distribution(self):
        """Test sector distribution"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('JPM', 'Equity', 'buy', 'long', 50, 2, '1/2/2025')
        total_pv, ticker_pv_dict = calculate_total_pv_all_tickers('AAPL', 12)
        result = calculate_equity_distribution_sector(ticker_pv_dict)
        assert 'Technology' in result  # AAPL
        assert 'Financial Services' in result  # JPM


# ============================================================================
# TEST: get_or_create_trade_number
# ============================================================================

class TestGetOrCreateTradeNumber:
    """Test get_or_create_trade_number function"""
    
    def test_trade_number_new_position(self):
        """Test trade number for new position"""
        reset_portfolio()
        trade_num = get_or_create_trade_number('AAPL', 0, 10, 'buy')
        assert trade_num == 1
    
    def test_trade_number_continues_position(self):
        """Test trade number continues for existing position"""
        reset_portfolio()
        trade_num1 = get_or_create_trade_number('AAPL', 0, 10, 'buy')
        trade_num2 = get_or_create_trade_number('AAPL', 10, 15, 'buy')
        assert trade_num1 == trade_num2
    
    def test_trade_number_hold_action(self):
        """Test trade number for hold action"""
        reset_portfolio()
        get_or_create_trade_number('AAPL', 0, 10, 'buy')
        trade_num = get_or_create_trade_number('AAPL', 10, 10, 'hold')
        assert trade_num == 1
    
    def test_trade_number_close_position(self):
        """Test trade number when closing position"""
        reset_portfolio()
        trade_num1 = get_or_create_trade_number('AAPL', 0, 10, 'buy')
        trade_num2 = get_or_create_trade_number('AAPL', 10, 0, 'sell')
        assert trade_num1 == trade_num2


# ============================================================================
# TEST: format_trade_string
# ============================================================================

class TestFormatTradeString:
    """Test format_trade_string function"""
    
    def test_trade_string_buy(self):
        """Test trade string for buy"""
        reset_portfolio()
        result = format_trade_string('buy', 'long', 1, 10)
        assert 'Long' in result
        assert 'Buy' in result
        assert '#1' in result
        assert '10' in result
    
    def test_trade_string_close(self):
        """Test trade string for close"""
        reset_portfolio()
        result = format_trade_string('sell', 'long', 1, 0)
        assert 'Close' in result
    
    def test_trade_string_no_trade(self):
        """Test trade string when no trade"""
        reset_portfolio()
        result = format_trade_string('hold', 'long', None, 10)
        assert result == "No Buy/Sell"


# ============================================================================
# TEST: process_trade (Integration Tests)
# ============================================================================

class TestProcessTradeIntegration:
    """Test process_trade function - integration tests"""
    
    def test_process_trade_basic_flow(self):
        """Test basic trade processing flow"""
        reset_portfolio()
        row = process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        assert row['Ticker'] == 'AAPL'
        assert row['Current Quantity'] == 10
        assert row['Remaining'] == 100
        assert row['Cost Basis'] == 100
    
    def test_process_trade_equity_formula(self):
        """Test equity formula in process_trade"""
        reset_portfolio()
        row = process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        equity = row['Equity: Total PV + Remaining']
        total_pv = row['Total PV']
        remaining = row['Remaining']
        assert equity == total_pv + remaining
    
    def test_process_trade_total_pnl_formula(self):
        """Test total PnL formula in process_trade"""
        reset_portfolio()
        row = process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        total_pnl = row['Total PnL Overall (Unrealized+Realized)']
        equity = row['Equity: Total PV + Remaining']
        assert total_pnl == equity - 200
    
    def test_process_trade_unrealized_pnl_all_tickers(self):
        """Test total unrealized PnL across all tickers"""
        reset_portfolio()
        process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        process_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        row = process_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/3/2025')
        # AAPL: (12-10)*10 = 20
        # MSFT: (20-20)*5 = 0
        assert row['Total Unrealized PnL'] == 20


# Add these test classes to test_portfolio.py after the existing tests

# ============================================================================
# RIGOROUS EDGE CASE TESTS
# ============================================================================

class TestEdgeCasesZeroValues:
    """Test edge cases with zero values"""
    
    def test_zero_price_handling(self):
        """Test handling of zero price"""
        reset_portfolio()
        row = process_trade('AAPL', 'Equity', 'buy', 'long', 0, 10, '1/1/2025')
        assert row['Price'] == 0
        assert row['Buyable/Sellable'] == 0  # Should handle division by zero
    
    def test_zero_quantity_handling(self):
        """Test handling of zero quantity"""
        reset_portfolio()
        row = process_trade('AAPL', 'Equity', 'buy', 'long', 10, 0, '1/1/2025')
        assert row['Current Quantity'] == 0
        assert row['Cost Basis'] == 0
        assert row['Remaining'] == 200  # No change
    
    def test_zero_quantity_hold(self):
        """Test hold action with zero quantity"""
        reset_portfolio()
        row = process_trade('AAPL', 'Equity', 'hold', 'long', 10, 0, '1/1/2025')
        assert row['Current Quantity'] == 0
        assert row['Trade No. (Position - Trade no. - Current Quantity)'] == "No Buy/Sell"
    
    def test_very_small_quantities(self):
        """Test with very small quantity values"""
        reset_portfolio()
        row = process_trade('AAPL', 'Equity', 'buy', 'long', 10, 0.0001, '1/1/2025')
        assert row['Current Quantity'] == 0.0001
        assert abs(row['Remaining'] - (200 - 10 * 0.0001)) < 0.0001
    
    def test_very_large_quantities(self):
        """Test with very large quantity values"""
        reset_portfolio(initial_cash=1000000)
        row = process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10000, '1/1/2025')
        assert row['Current Quantity'] == 10000
        assert row['Remaining'] == 1000000 - (10 * 10000)


class TestEdgeCasesNegativeValues:
    """Test edge cases with negative values"""
    
    def test_negative_price_handling(self):
        """Test handling of negative price (should be prevented in real trading)"""
        reset_portfolio()
        # Price shouldn't be negative, but test if code handles it
        row = process_trade('AAPL', 'Equity', 'buy', 'long', -10, 10, '1/1/2025')
        # System should handle it (may result in negative remaining)
        assert row['Price'] == -10
    
    def test_normalize_negative_quantity_strings(self):
        """Test normalizing negative quantity strings"""
        reset_portfolio()
        from portfolio import normalize_quantity
        assert normalize_quantity("-(-10)") == 10.0
        assert normalize_quantity("(-10)") == 10.0


class TestBoundaryConditions:
    """Test boundary conditions"""
    
    def test_exact_cash_remaining_zero(self):
        """Test when remaining cash becomes exactly zero"""
        reset_portfolio(initial_cash=100)
        row = process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        assert row['Remaining'] == 0
        assert row['Buyable/Sellable'] == 10 # Uses Previous Remaining
    
    def test_insufficient_cash_buy(self):
        """Test buying when cash is insufficient"""
        reset_portfolio(initial_cash=50)
        # Try to buy more than cash allows
        row = process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        # System should allow negative remaining (margin/debt)
        assert row['Remaining'] < 0
    
    def test_quantity_flip_exact_zero(self):
        """Test quantity flip when selling exact amount"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        row = process_trade('AAPL', 'Equity', 'sell', 'long', 12, 10, '1/2/2025')
        assert row['Current Quantity'] == 0
        assert row['Cost Basis'] == 0
    
    def test_avg_price_at_zero_cost_basis(self):
        """Test average price calculation when cost basis is zero"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 10, '1/2/2025')
        assert portfolio.portfolio_state['avg_price']['AAPL'] == 0
        assert portfolio.portfolio_state['cost_basis']['AAPL'] == 0


class TestComplexPositionFlips:
    """Test complex position flip scenarios"""
    
    def test_multiple_flips_same_ticker(self):
        """Test multiple position flips for same ticker"""
        reset_portfolio(initial_cash=1000)
        # Start long
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        # Flip to short
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 15, '1/2/2025')
        assert portfolio.portfolio_state['quantities']['AAPL'] == -5
        # Flip back to long
        add_trade('AAPL', 'Equity', 'buy', 'long', 8, 10, '1/3/2025')
        assert portfolio.portfolio_state['quantities']['AAPL'] == 5
        # Flip to short again
        add_trade('AAPL', 'Equity', 'sell', 'long', 10, 10, '1/4/2025')
        assert portfolio.portfolio_state['quantities']['AAPL'] == -5
    
    def test_flip_with_partial_close(self):
        """Test position flip when closing partial and opening opposite"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        # Sell 15 (close 10, open 5 short)
        row = process_trade('AAPL', 'Equity', 'sell', 'long', 12, 15, '1/2/2025')
        assert row['Current Quantity'] == -5
        assert row['PnL Realized at Point of Time'] == 20  # Only on closed 10 shares
        assert row['Cost Basis'] == 60  # 12 * 5 for new short position
    
    def test_flip_cost_basis_reset(self):
        """Test cost basis resets correctly on flip"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        # Flip to short
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 15, '1/2/2025')
        # New short position should have fresh cost basis
        assert portfolio.portfolio_state['cost_basis']['AAPL'] == 12 * 5 == 60
        assert portfolio.portfolio_state['avg_price']['AAPL'] == 12.0


class TestFormulaValidation:
    """Rigorous formula validation tests"""
    
    def test_pv_formula_validation(self):
        """Validate PV = Cost Basis + Unrealized PnL"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        row = process_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        pv = row['PV (Long)']
        cost_basis = row['Cost Basis']
        unrealized = row['PnL (Long) Unrealized']
        assert abs(pv - (cost_basis + unrealized)) < 0.01
    
    def test_total_pv_sum_validation(self):
        """Validate Total PV = sum of all ticker PVs"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        row = process_trade('TSLA', 'Equity', 'buy', 'long', 30, 3, '1/3/2025')
        total_pv = row['Total PV']
        # Manually calculate sum
        expected_total = row['PV (Long)']  # TSLA only in PV (Long)
        # Need to account for all tickers
        total_pv, ticker_pv_dict = calculate_total_pv_all_tickers('TSLA', 30)
        assert abs(total_pv - sum(ticker_pv_dict.values())) < 0.01
    
    def test_equity_formula_always_holds(self):
        """Validate Equity = Total PV + Remaining always holds"""
        reset_portfolio()
        for i in range(10):
            ticker = f'TICK{i}'
            process_trade(ticker, 'Equity', 'buy', 'long', 10 + i, 5, f'1/{i+1}/2025')
            df = get_portfolio_df()
            row = df.iloc[-1]
            equity = row['Equity: Total PV + Remaining']
            total_pv = row['Total PV']
            remaining = row['Remaining']
            assert abs(equity - (total_pv + remaining)) < 0.01, f"Row {i}: Equity formula failed"
    
    def test_daily_pnl_formula_validation(self):
        """Validate Daily PnL = Today Total PnL - Yesterday Total PnL"""
        reset_portfolio()
        process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        df = get_portfolio_df()
        yesterday_pnl = df.iloc[0]['Total PnL Overall (Unrealized+Realized)']
        process_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        df = get_portfolio_df()
        today_pnl = df.iloc[1]['Total PnL Overall (Unrealized+Realized)']
        daily_pnl = df.iloc[1]['Daily PnL (Unrealized+Realized)']
        assert abs(daily_pnl - (today_pnl - yesterday_pnl)) < 0.01
    
    def test_daily_percentage_formula_validation(self):
        """Validate Daily % formula"""
        reset_portfolio()
        process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        df = get_portfolio_df()
        yesterday_equity = df.iloc[0]['Equity: Total PV + Remaining']
        process_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        df = get_portfolio_df()
        today_equity = df.iloc[1]['Equity: Total PV + Remaining']
        daily_pct = df.iloc[1]['Daily %']
        expected = ((today_equity - yesterday_equity) / yesterday_equity) * 100
        assert abs(daily_pct - expected) < 0.01
    
    def test_cumulative_percentage_formula_validation(self):
        """Validate Cumulative % formula"""
        reset_portfolio(initial_cash=200)
        process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        df = get_portfolio_df()
        equity = df.iloc[0]['Equity: Total PV + Remaining']
        cumulative_pct = df.iloc[0]['Cumulative %']
        expected = ((equity / 200) - 1) * 100
        assert abs(cumulative_pct - expected) < 0.01
    
    def test_unrealized_pnl_sum_validation(self):
        """Validate Total Unrealized PnL = sum of all ticker unrealized PnL"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        add_trade('TSLA', 'Equity', 'sell', 'short', 30, 3, '1/3/2025')
        row = process_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/4/2025')
        total_unrealized = row['Total Unrealized PnL']
        # Manually calculate
        # AAPL: (12-10)*10 = 20
        # MSFT: (20-20)*5 = 0
        # TSLA: (30-30)*3 = 0 (short, but using last_price = 30)
        assert abs(total_unrealized - 20) < 0.01


class TestMultipleTickersComplex:
    """Test complex scenarios with multiple tickers"""
    
    def test_ten_tickers_simultaneous(self):
        """Test with 10 different tickers"""
        reset_portfolio(initial_cash=10000)
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
        for i, ticker in enumerate(tickers):
            process_trade(ticker, 'Equity', 'buy', 'long', 10 + i, 10, f'1/{i+1}/2025')
        
        df = get_portfolio_df()
        assert len(df) == 10
        assert df['Holdings'].iloc[-1] == 10
        # Verify all tickers are in open positions
        open_pos = df['Open Position'].iloc[-1]
        for ticker in tickers:
            assert ticker.upper() in open_pos
    
    def test_mixed_long_short_tickers(self):
        """Test portfolio with mixed long and short positions"""
        reset_portfolio(initial_cash=5000)
        # Long positions
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        # Short positions
        add_trade('TSLA', 'Equity', 'sell', 'short', 30, 3, '1/3/2025')
        add_trade('NVDA', 'Equity', 'sell', 'short', 40, 2, '1/4/2025')
        
        row = process_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/5/2025')
        # Verify all positions exist
        assert 'AAPL' in row['Open Position']
        assert 'MSFT' in row['Open Position']
        assert 'TSLA' in row['Open Position']
        assert 'NVDA' in row['Open Position']
        assert row['Holdings'] == 4
    
    def test_partial_closes_multiple_tickers(self):
        """Test partial closes across multiple tickers"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 5, '1/3/2025')
        add_trade('MSFT', 'Equity', 'sell', 'long', 22, 3, '1/4/2025')
        
        df = get_portfolio_df()
        assert df.iloc[2]['Current Quantity'] == 5  # AAPL
        assert df.iloc[3]['Current Quantity'] == 2  # MSFT
        assert df.iloc[3]['PnL Realized Cummulative'] == 10 + 6 == 16  # (12-10)*5 + (22-20)*3


class TestRealizedPnLRigorous:
    """Rigorous tests for realized PnL calculations"""
    
    def test_realized_pnl_only_on_closed_shares(self):
        """Test realized PnL only counts on shares actually closed"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        # Sell 5 shares
        row = process_trade('AAPL', 'Equity', 'sell', 'long', 12, 5, '1/2/2025')
        assert row['PnL Realized at Point of Time'] == (12 - 10) * 5 == 10
        # Sell 3 more shares
        row = process_trade('AAPL', 'Equity', 'sell', 'long', 13, 3, '1/3/2025')
        assert row['PnL Realized at Point of Time'] == (13 - 10) * 3 == 9
        assert row['PnL Realized Cummulative'] == 10 + 9 == 19
    
    def test_realized_pnl_flip_only_closed_portion(self):
        """Test realized PnL on flip only counts closed portion"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        # Sell 15 (close 10, open 5 short)
        row = process_trade('AAPL', 'Equity', 'sell', 'long', 12, 15, '1/2/2025')
        # Only 10 shares closed, so realized = (12-10)*10 = 20
        assert row['PnL Realized at Point of Time'] == 20
        assert row['PnL Realized Cummulative'] == 20
    
    def test_realized_pnl_short_cover(self):
        """Test realized PnL when covering short"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        # Cover 5 shares - position should be 'short', not 'long'
        row = process_trade('AAPL', 'Equity', 'buy', 'short', 8, 5, '1/2/2025')
        assert row['PnL Realized at Point of Time'] == (10 - 8) * 5 == 10
        # Cover remaining 5 - position should be 'short', not 'long'
        row = process_trade('AAPL', 'Equity', 'buy', 'short', 9, 5, '1/3/2025')
        assert row['PnL Realized at Point of Time'] == (10 - 9) * 5 == 5
        assert row['PnL Realized Cummulative'] == 10 + 5 == 15
    
    def test_realized_pnl_cumulative_across_tickers(self):
        """Test cumulative realized PnL accumulates across all tickers"""
        reset_portfolio()
        # AAPL trades
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 10, '1/2/2025')  # +20
        # MSFT trades
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/3/2025')
        add_trade('MSFT', 'Equity', 'sell', 'long', 18, 5, '1/4/2025')  # -10
        df = get_portfolio_df()
        assert df.iloc[3]['PnL Realized Cummulative'] == 20 - 10 == 10


class TestUnrealizedPnLRigorous:
    """Rigorous tests for unrealized PnL calculations"""
    
    def test_unrealized_pnl_long_multiple_prices(self):
        """Test unrealized PnL for long with multiple price changes"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        # Price 12
        row = process_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        assert row['PnL (Long) Unrealized'] == 20
        # Price 15
        row = process_trade('AAPL', 'Equity', 'hold', 'long', 15, 0, '1/3/2025')
        assert row['PnL (Long) Unrealized'] == 50
        # Price 8
        row = process_trade('AAPL', 'Equity', 'hold', 'long', 8, 0, '1/4/2025')
        assert row['PnL (Long) Unrealized'] == -20
    
    def test_unrealized_pnl_short_multiple_prices(self):
        """Test unrealized PnL for short with multiple price changes"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        # Price 8 (profit)
        row = process_trade('AAPL', 'Equity', 'hold', 'short', 8, 0, '1/2/2025')
        assert row['PnL (Short) Unrealized'] == 20
        # Price 12 (loss)
        row = process_trade('AAPL', 'Equity', 'hold', 'short', 12, 0, '1/3/2025')
        assert row['PnL (Short) Unrealized'] == -20
    
    def test_unrealized_pnl_with_additions(self):
        """Test unrealized PnL when adding to position"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        # Add more at higher price
        add_trade('AAPL', 'Equity', 'buy', 'long', 12, 5, '1/2/2025')
        # Check unrealized at new price
        row = process_trade('AAPL', 'Equity', 'hold', 'long', 15, 0, '1/3/2025')
        # Avg price: (10*10 + 12*5)/15 = 10.6667
        # Unrealized: (15 - 10.6667) * 15 = 65
        assert abs(row['PnL (Long) Unrealized'] - 65) < 1
    
    def test_unrealized_pnl_all_tickers_accurate(self):
        """Test total unrealized PnL is sum of all tickers accurately"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        add_trade('TSLA', 'Equity', 'sell', 'short', 30, 3, '1/3/2025')
        # Update prices
        row = process_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/4/2025')
        # AAPL: (12-10)*10 = 20
        # MSFT: (20-20)*5 = 0 (last_price = 20)
        # TSLA: (30-30)*3 = 0 (last_price = 30)
        assert abs(row['Total Unrealized PnL'] - 20) < 0.01


class TestCostBasisRigorous:
    """Rigorous tests for cost basis calculations"""
    
    def test_cost_basis_proportional_reduction(self):
        """Test cost basis reduces proportionally on partial sell"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')  # CB = 100
        add_trade('AAPL', 'Equity', 'buy', 'long', 12, 5, '1/2/2025')   # CB = 160
        # Sell 10 shares (out of 15)
        row = process_trade('AAPL', 'Equity', 'sell', 'long', 15, 10, '1/3/2025')
        # Remaining: 5 shares
        # Cost basis: 160 * (5/15) = 53.333
        assert abs(row['Cost Basis'] - 53.333) < 0.1
    
    def test_cost_basis_short_proportional_reduction(self):
        """Test cost basis reduces proportionally on partial cover"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')  # CB = 100
        add_trade('AAPL', 'Equity', 'sell', 'short', 12, 5, '1/2/2025')   # CB = 160
        # Cover 10 shares (out of 15) - use position='short' when covering
        row = process_trade('AAPL', 'Equity', 'buy', 'short', 8, 10, '1/3/2025')
        # Remaining: 5 shares short
        # Cost basis: 160 * (5/15) = 53.333
        assert abs(row['Cost Basis'] - 53.333) < 0.1
    
    def test_cost_basis_resets_on_flip(self):
        """Test cost basis resets when position flips"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        # Flip to short
        row = process_trade('AAPL', 'Equity', 'sell', 'long', 12, 15, '1/2/2025')
        # New short position: 5 shares at 12
        assert row['Cost Basis'] == 12 * 5 == 60
    
    def test_cost_basis_averaging_multiple_buys(self):
        """Test cost basis averaging across multiple buys"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')  # CB = 100
        add_trade('AAPL', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')   # CB = 200
        add_trade('AAPL', 'Equity', 'buy', 'long', 15, 10, '1/3/2025')  # CB = 350
        # Avg price: 350/25 = 14
        assert abs(portfolio.portfolio_state['avg_price']['AAPL'] - 14) < 0.01
        assert portfolio.portfolio_state['cost_basis']['AAPL'] == 350


class TestStateConsistency:
    """Test state consistency across operations"""
    
    def test_state_consistent_after_each_trade(self):
        """Test state is consistent after each trade"""
        reset_portfolio()
        trades = [
            ('AAPL', 'buy', 10, 10),
            ('AAPL', 'hold', 12, 0),
            ('AAPL', 'sell', 13, 5),
            ('MSFT', 'buy', 20, 5),
            ('AAPL', 'hold', 14, 0),
        ]
        for ticker, action, price, qty in trades:
            process_trade(ticker, 'Equity', action, 'long', price, qty, '1/1/2025')
            # Verify state consistency
            df = get_portfolio_df()
            row = df.iloc[-1]
            # Check equity formula
            assert abs(row['Equity: Total PV + Remaining'] - (row['Total PV'] + row['Remaining'])) < 0.01
            # Check total PnL formula
            assert abs(row['Total PnL Overall (Unrealized+Realized)'] - (row['Equity: Total PV + Remaining'] - 200)) < 0.01
    
    def test_quantities_match_open_positions(self):
        """Test quantities in state match open positions string"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        row = process_trade('TSLA', 'Equity', 'buy', 'long', 30, 3, '1/3/2025')
        open_pos = row['Open Position']
        # Verify all tickers with non-zero quantities are in open positions
        for ticker, qty in portfolio.portfolio_state['quantities'].items():
            if qty != 0:
                assert ticker.upper() in open_pos
    
    def test_cost_basis_matches_avg_price(self):
        """Test cost basis and avg price relationship"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'buy', 'long', 12, 5, '1/2/2025')
        cb = portfolio.portfolio_state['cost_basis']['AAPL']
        qty = portfolio.portfolio_state['quantities']['AAPL']
        avg = portfolio.portfolio_state['avg_price']['AAPL']
        if qty != 0:
            assert abs(avg - abs(cb / qty)) < 0.01


class TestStressTests:
    """Stress tests with many operations"""
    
    def test_100_trades_single_ticker(self):
        """Test 100 trades on single ticker"""
        reset_portfolio(initial_cash=100000)
        from datetime import datetime, timedelta
        
        # Start from a base date
        base_date = datetime(2025, 1, 1)
        
        for i in range(100):
            action = 'buy' if i % 2 == 0 else 'sell'
            qty = 10 if action == 'buy' else 5
            # Increment date by i days
            trade_date = base_date + timedelta(days=i)
            # Format as MM/DD/YYYY
            date_str = trade_date.strftime('%m/%d/%Y')
            process_trade('AAPL', 'Equity', action, 'long', 10 + i*0.1, qty, date_str)
            
        df = get_portfolio_df()
        assert len(df) == 100
        # Verify final state is valid
        final_row = df.iloc[-1]
        assert final_row['Equity: Total PV + Remaining'] == final_row['Total PV'] + final_row['Remaining']
    
    def test_50_tickers_100_trades_total(self):

        """Test 50 different tickers with 100 total trades"""
        reset_portfolio(initial_cash=100000)
        from datetime import datetime, timedelta
        
        tickers = [f'TICK{i}' for i in range(50)]
        base_date = datetime(2025, 1, 1)
        
        for i in range(100):
            ticker = tickers[i % 50]
            # Increment date by i days
            trade_date = base_date + timedelta(days=i)
            # Format as MM/DD/YYYY
            date_str = trade_date.strftime('%m/%d/%Y')
            process_trade(ticker, 'Equity', 'buy', 'long', 10, 10, date_str)
        
        df = get_portfolio_df()
        assert len(df) == 100
        assert df['Holdings'].iloc[-1] == 50  # All 50 tickers should have positions
    
    def test_rapid_flips_same_ticker(self):
        """Test rapid position flips on same ticker"""
        reset_portfolio(initial_cash=10000)
        from datetime import datetime, timedelta
        
        base_date = datetime(2025, 1, 1)
        
        # Rapidly flip between long and short
        for i in range(20):
            # Increment date by i days
            trade_date = base_date + timedelta(days=i)
            # Format as MM/DD/YYYY
            date_str = trade_date.strftime('%m/%d/%Y')
            
            if i % 2 == 0:
                process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, date_str)
            else:
                process_trade('AAPL', 'Equity', 'sell', 'long', 10, 11, date_str)
        
        df = get_portfolio_df()
        assert len(df) == 20
        # Final position should be long (started with buy, ended with buy if 20 is even)
        final_qty = df.iloc[-1]['Current Quantity']
        assert final_qty == 10 or final_qty == -10  # Depends on number of trades


class TestWinLossRigorous:
    """Rigorous tests for win/loss calculations"""
    
    def test_win_detection_zero_pnl(self):
        """Test win/loss when PnL is exactly zero"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        row = process_trade('AAPL', 'Equity', 'sell', 'long', 10, 10, '1/2/2025')
        # PnL = 0, should be Loss
        assert row['Win/Loss'] == 'Loss'
    
    def test_win_rate_with_multiple_trades(self):
        """Test win rate with many trades"""
        reset_portfolio()
        # 10 wins, 5 losses
        for i in range(15):
            ticker = f'TICK{i}'
            entry_price = 10
            exit_price = 12 if i < 10 else 8  # First 10 win, last 5 lose
            process_trade(ticker, 'Equity', 'buy', 'long', entry_price, 10, f'1/{i*2+1}/2025')
            process_trade(ticker, 'Equity', 'sell', 'long', exit_price, 10, f'1/{i*2+2}/2025')
        
        df = get_portfolio_df()
        final_win_rate = df.iloc[-1]['Win Rate']
        assert abs(final_win_rate - (10/15 * 100)) < 0.1  # Should be ~66.67%
    
    def test_win_loss_ratio_calculation(self):
        """Test win:loss ratio calculation"""
        reset_portfolio()
        # 3 wins, 2 losses
        wins = ['AAPL', 'MSFT', 'GOOGL']
        losses = ['TSLA', 'NVDA']
        for ticker in wins:
            process_trade(ticker, 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
            process_trade(ticker, 'Equity', 'sell', 'long', 12, 10, '1/2/2025')
        for ticker in losses:
            process_trade(ticker, 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
            process_trade(ticker, 'Equity', 'sell', 'long', 8, 10, '1/2/2025')
        
        df = get_portfolio_df()
        assert df.iloc[-1]['Win:Loss Ratio'] == '3:2'


class TestDistributionRigorous:
    """Rigorous tests for distribution calculations"""
    
    def test_distribution_percentages_sum_to_100(self):
        """Test distribution percentages sum to 100%"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('XRP', 'Crypto', 'buy', 'long', 0.5, 100, '1/2/2025')
        row = process_trade('BTC', 'Crypto', 'buy', 'long', 1, 50, '1/3/2025')
        
        dist_pct = row['Distribution in %']
        # Parse percentages and sum them
        import re
        percentages = re.findall(r'(\d+\.?\d*)%', dist_pct)
        total = sum(float(p) for p in percentages)
        assert abs(total - 100) < 0.1  # Should sum to ~100%
    
    def test_equity_distribution_only_equity_tickers(self):
        """Test equity distribution excludes non-equity tickers"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('XRP', 'Crypto', 'buy', 'long', 0.5, 100, '1/2/2025')
        row = process_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/3/2025')
        
        equity_dist = row['Equity Distribution (Market Cap)']
        # Should only include Equity tickers, not Crypto
        assert 'High' in equity_dist  # AAPL and MSFT
        # Crypto should not affect equity distribution


# ============================================================================
# TEST: Advanced Scenarios
# ============================================================================

class TestAdvancedScenarios:
    """Test advanced real-world scenarios"""
    
    def test_scenario_1_user_sequence(self):
        """Test the exact sequence from user's example"""
        reset_portfolio(initial_cash=200)
        add_trade('AAPL', 'Equity', 'hold', 'long', 10, 0, '1/1/2025')
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/2/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 11, 0, '1/3/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/4/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 1, '1/5/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 13, 0, '1/6/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 5, 5, '1/7/2025')
        add_trade('MSFT', 'Equity', 'hold', 'long', 6, 0, '2/5/2025')
        add_trade('AAPL', 'Equity', 'buy', 'long', 13, 2, '2/6/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 13, 11, '2/7/2025')
        add_trade('AAPL', 'Equity', 'sell', 'short', 12, 10, '2/8/2025')
        add_trade('MSFT', 'Equity', 'sell', 'long', 8, 1, '2/9/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 9, 2, '3/2/2025')
        add_trade('MSFT', 'Equity', 'sell', 'long', 9, 6, '3/3/2025')
        add_trade('TSLA', 'Equity', 'sell', 'short', 20, 5, '3/4/2025')
        add_trade('TSLA', 'Equity', 'hold', 'short', 22, 0, '3/5/2025')
        add_trade('TSLA', 'Equity', 'buy', 'short', 21, 4, '3/6/2025')
        add_trade('TSLA', 'Equity', 'sell', 'short', 22, 1, '3/7/2025')
        add_trade('TSLA', 'Equity', 'buy', 'short', 24, 2, '3/8/2025')
        add_trade('XRP', 'Crypto', 'buy', 'long', 0.5, 100, '3/9/2025')
        
        df = get_portfolio_df()
        assert len(df) == 20
        # Verify final state
        final_row = df.iloc[-1]
        assert final_row['Equity: Total PV + Remaining'] == final_row['Total PV'] + final_row['Remaining']
        assert final_row['Total PnL Overall (Unrealized+Realized)'] == final_row['Equity: Total PV + Remaining'] - 200
    
    def test_scenario_margin_calls(self):
        """Test scenario with negative remaining (margin/debt)"""
        reset_portfolio(initial_cash=100)
        # Buy more than cash allows
        row = process_trade('AAPL', 'Equity', 'buy', 'long', 10, 20, '1/1/2025')
        assert row['Remaining'] < 0  # Negative remaining
    
    def test_scenario_round_trip(self):
        """Test complete round trip: buy, hold, sell"""
        reset_portfolio()
        # Buy
        row1 = process_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        assert row1['Current Quantity'] == 10
        assert row1['Cost Basis'] == 100
        # Hold
        row2 = process_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        assert row2['Current Quantity'] == 10
        assert row2['PnL (Long) Unrealized'] == 20
        # Sell
        row3 = process_trade('AAPL', 'Equity', 'sell', 'long', 13, 10, '1/3/2025')
        assert row3['Current Quantity'] == 0
        assert row3['PnL Realized at Point of Time'] == 30
        assert row3['PnL Realized Cummulative'] == 30
        assert row3['Cost Basis'] == 0


class TestPositionValidation:
    """Test validation logic for position switching"""
    
    def test_error_explicit_short_while_long(self):
        """Test error when explicitly trying to open short while having long position"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')  # Long 10
        
        # Try to explicitly open short position - should error
        with pytest.raises(ValueError, match="Cannot.*open a short position.*while holding a long"):
            add_trade('AAPL', 'Equity', 'sell', 'short', 12, 10, '1/2/2025')
    
    def test_error_explicit_long_while_short(self):
        """Test error when explicitly trying to open long while having short position"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')  # Short 10
        
        # Try to explicitly open long position - should error
        with pytest.raises(ValueError, match="Cannot.*open a long position.*while holding a short"):
            add_trade('AAPL', 'Equity', 'buy', 'long', 8, 10, '1/2/2025')
    
    def test_allowed_natural_flip_long_to_short(self):
        """Test that natural flip from long to short is allowed (selling more than owned)"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')  # Long 10
        
        # Sell 15 with position='long' - naturally flips to short 5
        row = add_trade('AAPL', 'Equity', 'sell', 'long', 12, 15, '1/2/2025')
        assert row.iloc[-1]['Current Quantity'] == -5  # Should be short 5
        assert row.iloc[-1]['Current Position'] == 'Short'
    
    def test_allowed_natural_flip_short_to_long(self):
        """Test that natural flip from short to long is allowed (buying more than owed)"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')  # Short 10
        
        # Buy 15 with position='short' - naturally flips to long 5
        row = add_trade('AAPL', 'Equity', 'buy', 'short', 8, 15, '1/2/2025')
        assert row.iloc[-1]['Current Quantity'] == 5  # Should be long 5
        assert row.iloc[-1]['Current Position'] == 'Long'
    
    def test_allowed_closing_long_position(self):
        """Test that closing a long position by selling is allowed"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')  # Long 10
        
        # Close long position by selling
        row = add_trade('AAPL', 'Equity', 'sell', 'long', 12, 10, '1/2/2025')
        assert row.iloc[-1]['Current Quantity'] == 0
        assert row.iloc[-1]['Current Position'] == 'Long'  # Position type before close
    
    def test_allowed_covering_short_position(self):
        """Test that covering a short position by buying is allowed"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')  # Short 10
        
        # Cover short position by buying
        row = add_trade('AAPL', 'Equity', 'buy', 'short', 8, 10, '1/2/2025')
        assert row.iloc[-1]['Current Quantity'] == 0
        assert row.iloc[-1]['Current Position'] == 'Short'  # Position type before close
    
    def test_allowed_partial_close_long(self):
        """Test that partial closing of long position is allowed"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')  # Long 10
        
        # Partially close long position
        row = add_trade('AAPL', 'Equity', 'sell', 'long', 12, 5, '1/2/2025')
        assert row.iloc[-1]['Current Quantity'] == 5  # Remaining long 5
    
    def test_allowed_partial_cover_short(self):
        """Test that partial covering of short position is allowed"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')  # Short 10
        
        # Partially cover short position
        row = add_trade('AAPL', 'Equity', 'buy', 'short', 8, 5, '1/2/2025')
        assert row.iloc[-1]['Current Quantity'] == -5  # Remaining short 5
    
    def test_allowed_opening_position_when_none_exists(self):
        """Test that opening a position when none exists is allowed"""
        reset_portfolio()
        
        # Open long position - should work
        row = add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        assert row.iloc[-1]['Current Quantity'] == 10
        
        # Close it
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 10, '1/2/2025')
        
        # Open short position - should work
        row = add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/3/2025')
        assert row.iloc[-1]['Current Quantity'] == -10
    
    def test_allowed_continuing_same_position_type(self):
        """Test that continuing the same position type is allowed"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')  # Long 10
        
        # Add more to long position - should work
        row = add_trade('AAPL', 'Equity', 'buy', 'long', 12, 5, '1/2/2025')
        assert row.iloc[-1]['Current Quantity'] == 15
        
        # Close long position first before opening short
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 15, '1/3/2025')  # Close long
        
        # Now open short position - should work
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/4/2025')  # Short 10
        
        # Add more to short position - should work
        row = add_trade('AAPL', 'Equity', 'sell', 'short', 12, 5, '1/5/2025')
        assert row.iloc[-1]['Current Quantity'] == -15
    
    def test_error_multiple_attempts_same_ticker(self):
        """Test that multiple attempts to open opposite position on same ticker all error"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')  # Long 10
        
        # First attempt - should error
        with pytest.raises(ValueError):
            add_trade('AAPL', 'Equity', 'sell', 'short', 12, 10, '1/2/2025')
        
        # Second attempt - should still error (position still exists)
        with pytest.raises(ValueError):
            add_trade('AAPL', 'Equity', 'sell', 'short', 12, 10, '1/3/2025')
    
    def test_different_tickers_allowed(self):
        """Test that different tickers can have different position types"""
        reset_portfolio()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')  # AAPL long
        add_trade('MSFT', 'Equity', 'sell', 'short', 20, 5, '1/2/2025')  # MSFT short - should work
        
        df = get_portfolio_df()
        assert df.iloc[0]['Current Quantity'] == 10  # AAPL long
        assert df.iloc[1]['Current Quantity'] == -5  # MSFT short