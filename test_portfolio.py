# test_portfolio.py
import pytest
import pandas as pd
from portfolio import (
    reset_portfolio, add_trade, get_portfolio_df,
    portfolio_state, calculate_cash_single, calculate_remaining_single,
    calculate_current_quantity_single, calculate_avg_price_and_cost_basis_single,
    calculate_realized_pnl_at_point_of_time, calculate_realized_pnl_cumulative,
    pnl_unrealized_components, position_value_from_position,
    calculate_total_pv_all_tickers, calculate_diversification,
    calculate_equity_distribution_market_cap, calculate_equity_distribution_industry,
    calculate_equity_distribution_sector
)


class TestBasicCalculations:
    """Test basic calculation functions"""
    
    def setup_method(self):
        """Reset portfolio before each test"""
        reset_portfolio(initial_cash=200)
    
    def test_cash_constant(self):
        """Test that cash remains constant after initialization"""
        cash1 = calculate_cash_single()
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        cash2 = calculate_cash_single()
        assert cash1 == cash2 == 200
    
    def test_remaining_cash_buy_long(self):
        """Test remaining cash after buying long"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        df = get_portfolio_df()
        assert df.iloc[0]['Remaining'] == 200 - (10 * 10) == 100
    
    def test_remaining_cash_sell_long(self):
        """Test remaining cash after selling long"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 5, '1/2/2025')
        df = get_portfolio_df()
        assert df.iloc[1]['Remaining'] == 100 + (12 * 5) == 160
    
    def test_remaining_cash_sell_short(self):
        """Test remaining cash after selling short"""
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        df = get_portfolio_df()
        assert df.iloc[0]['Remaining'] == 200 - (10 * 10) == 100
    
    def test_quantity_buy_increases_long(self):
        """Test that buying increases quantity"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        assert portfolio_state['quantities']['AAPL'] == 10
    
    def test_quantity_sell_decreases_long(self):
        """Test that selling decreases quantity"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 5, '1/2/2025')
        assert portfolio_state['quantities']['AAPL'] == 5
    
    def test_quantity_flip_long_to_short(self):
        """Test quantity flips from long to short when selling more than owned"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 15, '1/2/2025')
        assert portfolio_state['quantities']['AAPL'] == -5  # Flipped to short
    
    def test_quantity_flip_short_to_long(self):
        """Test quantity flips from short to long when buying more than owed"""
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'buy', 'long', 12, 15, '1/2/2025')
        assert portfolio_state['quantities']['AAPL'] == 5  # Flipped to long


class TestCostBasisAndAvgPrice:
    """Test cost basis and average price calculations"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_cost_basis_long_single_buy(self):
        """Test cost basis for single long buy"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        assert portfolio_state['cost_basis']['AAPL'] == 10 * 10 == 100
        assert portfolio_state['avg_price']['AAPL'] == 10.0
    
    def test_cost_basis_long_multiple_buys(self):
        """Test cost basis for multiple long buys (averaging)"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'buy', 'long', 12, 5, '1/2/2025')
        # Cost basis: 100 + 60 = 160
        # Quantity: 15
        # Avg price: 160/15 = 10.6667
        assert portfolio_state['cost_basis']['AAPL'] == 160
        assert abs(portfolio_state['avg_price']['AAPL'] - 10.6667) < 0.01
    
    def test_cost_basis_long_partial_sell(self):
        """Test cost basis after partial sell of long position"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 5, '1/2/2025')
        # Remaining: 5 shares
        # Cost basis: 100 * (5/10) = 50
        assert portfolio_state['cost_basis']['AAPL'] == 50
        assert portfolio_state['avg_price']['AAPL'] == 10.0
    
    def test_cost_basis_short_single_sell(self):
        """Test cost basis for single short sell"""
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        assert portfolio_state['cost_basis']['AAPL'] == 10 * 10 == 100
        assert portfolio_state['avg_price']['AAPL'] == 10.0
    
    def test_cost_basis_short_multiple_sells(self):
        """Test cost basis for multiple short sells"""
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'short', 12, 5, '1/2/2025')
        # Cost basis: 100 + 60 = 160
        assert portfolio_state['cost_basis']['AAPL'] == 160
        assert abs(portfolio_state['avg_price']['AAPL'] - 10.6667) < 0.01
    
    def test_cost_basis_flip_long_to_short(self):
        """Test cost basis resets when flipping from long to short"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 15, '1/2/2025')
        # After flip: short 5 shares at 12
        assert portfolio_state['cost_basis']['AAPL'] == 12 * 5 == 60
        assert portfolio_state['avg_price']['AAPL'] == 12.0


class TestRealizedPnL:
    """Test realized PnL calculations"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_realized_pnl_long_profit(self):
        """Test realized PnL when selling long at profit"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 10, '1/2/2025')
        df = get_portfolio_df()
        # Profit: (12 - 10) * 10 = 20
        assert df.iloc[1]['PnL Realized at Point of Time'] == 20
        assert df.iloc[1]['PnL Realized Cummulative'] == 20
    
    def test_realized_pnl_long_loss(self):
        """Test realized PnL when selling long at loss"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 8, 10, '1/2/2025')
        df = get_portfolio_df()
        # Loss: (8 - 10) * 10 = -20
        assert df.iloc[1]['PnL Realized at Point of Time'] == -20
        assert df.iloc[1]['PnL Realized Cummulative'] == -20
    
    def test_realized_pnl_long_partial(self):
        """Test realized PnL when partially selling long"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 5, '1/2/2025')
        df = get_portfolio_df()
        # Profit: (12 - 10) * 5 = 10
        assert df.iloc[1]['PnL Realized at Point of Time'] == 10
        assert df.iloc[1]['PnL Realized Cummulative'] == 10
    
    def test_realized_pnl_short_profit(self):
        """Test realized PnL when covering short at profit"""
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'buy', 'long', 8, 10, '1/2/2025')
        df = get_portfolio_df()
        # Profit: (10 - 8) * 10 = 20
        assert df.iloc[1]['PnL Realized at Point of Time'] == 20
        assert df.iloc[1]['PnL Realized Cummulative'] == 20
    
    def test_realized_pnl_short_loss(self):
        """Test realized PnL when covering short at loss"""
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'buy', 'long', 12, 10, '1/2/2025')
        df = get_portfolio_df()
        # Loss: (10 - 12) * 10 = -20
        assert df.iloc[1]['PnL Realized at Point of Time'] == -20
        assert df.iloc[1]['PnL Realized Cummulative'] == -20
    
    def test_realized_pnl_cumulative_multiple_trades(self):
        """Test cumulative realized PnL across multiple trades"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 5, '1/2/2025')  # +10
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/3/2025')
        add_trade('MSFT', 'Equity', 'sell', 'long', 22, 5, '1/4/2025')  # +10
        df = get_portfolio_df()
        assert df.iloc[3]['PnL Realized Cummulative'] == 20
    
    def test_realized_pnl_no_closing(self):
        """Test that realized PnL is None when not closing position"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'buy', 'long', 12, 5, '1/2/2025')
        df = get_portfolio_df()
        assert df.iloc[1]['PnL Realized at Point of Time'] is None


class TestUnrealizedPnL:
    """Test unrealized PnL calculations"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_unrealized_pnl_long_profit(self):
        """Test unrealized PnL for long position at profit"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        df = get_portfolio_df()
        # Unrealized: (12 - 10) * 10 = 20
        assert df.iloc[1]['PnL (Long) Unrealized'] == 20
        assert df.iloc[1]['PnL Unrealized Total Value for Current Ticker'] == 20
        assert df.iloc[1]['Total Unrealized PnL'] == 20
    
    def test_unrealized_pnl_long_loss(self):
        """Test unrealized PnL for long position at loss"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 8, 0, '1/2/2025')
        df = get_portfolio_df()
        # Unrealized: (8 - 10) * 10 = -20
        assert df.iloc[1]['PnL (Long) Unrealized'] == -20
        assert df.iloc[1]['PnL Unrealized Total Value for Current Ticker'] == -20
    
    def test_unrealized_pnl_short_profit(self):
        """Test unrealized PnL for short position at profit"""
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'short', 8, 0, '1/2/2025')
        df = get_portfolio_df()
        # Unrealized: (10 - 8) * 10 = 20
        assert df.iloc[1]['PnL (Short) Unrealized'] == 20
        assert df.iloc[1]['PnL Unrealized Total Value for Current Ticker'] == 20
    
    def test_unrealized_pnl_short_loss(self):
        """Test unrealized PnL for short position at loss"""
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'short', 12, 0, '1/2/2025')
        df = get_portfolio_df()
        # Unrealized: (10 - 12) * 10 = -20
        assert df.iloc[1]['PnL (Short) Unrealized'] == -20
        assert df.iloc[1]['PnL Unrealized Total Value for Current Ticker'] == -20
    
    def test_unrealized_pnl_multiple_tickers(self):
        """Test total unrealized PnL across multiple tickers"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/3/2025')
        df = get_portfolio_df()
        # AAPL: (12 - 10) * 10 = 20
        # MSFT: (20 - 20) * 5 = 0 (using last_price = 20)
        assert df.iloc[2]['Total Unrealized PnL'] == 20


class TestPositionValue:
    """Test Position Value (PV) calculations"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_pv_long(self):
        """Test PV for long position"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        df = get_portfolio_df()
        # PV = Cost Basis + Unrealized PnL
        # Cost Basis: 100
        # Unrealized: (12 - 10) * 10 = 20
        # PV: 100 + 20 = 120
        assert df.iloc[1]['PV (Long)'] == 120
        assert df.iloc[1]['Position Value PV'] == 120
    
    def test_pv_short(self):
        """Test PV for short position"""
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'short', 8, 0, '1/2/2025')
        df = get_portfolio_df()
        # PV = Cost Basis + Unrealized PnL
        # Cost Basis: 100
        # Unrealized: (10 - 8) * 10 = 20
        # PV: 100 + 20 = 120
        assert df.iloc[1]['PV (Short)'] == 120
    
    def test_total_pv_multiple_tickers(self):
        """Test Total PV across multiple tickers"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/3/2025')
        df = get_portfolio_df()
        # AAPL PV: 100 + 20 = 120
        # MSFT PV: 100 + 0 = 100
        # Total: 220
        assert df.iloc[2]['Total PV'] == 220


class TestEquityAndTotalPnL:
    """Test Equity and Total PnL calculations"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_equity_formula(self):
        """Test Equity = Total PV + Remaining"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        df = get_portfolio_df()
        equity = df.iloc[1]['Equity: Total PV + Remaining']
        total_pv = df.iloc[1]['Total PV']
        remaining = df.iloc[1]['Remaining']
        assert equity == total_pv + remaining
    
    def test_total_pnl_overall(self):
        """Test Total PnL Overall = Equity - Initial Cash"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        df = get_portfolio_df()
        total_pnl = df.iloc[1]['Total PnL Overall (Unrealized+Realized)']
        equity = df.iloc[1]['Equity: Total PV + Remaining']
        assert total_pnl == equity - 200
    
    def test_daily_pnl(self):
        """Test Daily PnL calculation"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        df = get_portfolio_df()
        daily_pnl = df.iloc[1]['Daily PnL (Unrealized+Realized)']
        total_pnl_today = df.iloc[1]['Total PnL Overall (Unrealized+Realized)']
        total_pnl_yesterday = df.iloc[0]['Total PnL Overall (Unrealized+Realized)']
        assert abs(daily_pnl - (total_pnl_today - total_pnl_yesterday)) < 0.01
    
    def test_daily_percentage(self):
        """Test Daily % calculation"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        df = get_portfolio_df()
        daily_pct = df.iloc[1]['Daily %']
        equity_today = df.iloc[1]['Equity: Total PV + Remaining']
        equity_yesterday = df.iloc[0]['Equity: Total PV + Remaining']
        expected = ((equity_today - equity_yesterday) / equity_yesterday) * 100
        assert abs(daily_pct - expected) < 0.01
    
    def test_cumulative_percentage(self):
        """Test Cumulative % calculation"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        df = get_portfolio_df()
        cumulative_pct = df.iloc[1]['Cumulative %']
        equity = df.iloc[1]['Equity: Total PV + Remaining']
        expected = ((equity / 200) - 1) * 100
        assert abs(cumulative_pct - expected) < 0.01


class TestWinLoss:
    """Test Win/Loss tracking"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_win_detection(self):
        """Test win detection when closing at profit"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 10, '1/2/2025')
        df = get_portfolio_df()
        assert df.iloc[1]['Win/Loss'] == 'Win'
    
    def test_loss_detection(self):
        """Test loss detection when closing at loss"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 8, 10, '1/2/2025')
        df = get_portfolio_df()
        assert df.iloc[1]['Win/Loss'] == 'Loss'
    
    def test_win_rate_calculation(self):
        """Test win rate calculation"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 10, '1/2/2025')  # Win
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/3/2025')
        add_trade('MSFT', 'Equity', 'sell', 'long', 18, 5, '1/4/2025')  # Loss
        df = get_portfolio_df()
        # Win rate: 1 win / 2 trades = 50%
        assert df.iloc[3]['Win Rate'] == 50.0
    
    def test_win_loss_ratio(self):
        """Test win:loss ratio calculation"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 10, '1/2/2025')  # Win
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/3/2025')
        add_trade('MSFT', 'Equity', 'sell', 'long', 18, 5, '1/4/2025')  # Loss
        df = get_portfolio_df()
        assert df.iloc[3]['Win:Loss Ratio'] == '1:1'


class TestDistribution:
    """Test distribution calculations"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_asset_type_distribution(self):
        """Test asset type distribution"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('XRP', 'Crypto', 'buy', 'long', 0.5, 100, '1/2/2025')
        df = get_portfolio_df()
        distribution = df.iloc[1]['Distribution']
        assert 'Equity' in distribution
        assert 'Crypto' in distribution
    
    def test_equity_distribution_market_cap(self):
        """Test equity distribution by market cap"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('SQ', 'Equity', 'buy', 'long', 5, 10, '1/2/2025')
        df = get_portfolio_df()
        equity_dist = df.iloc[1]['Equity Distribution (Market Cap)']
        assert 'High' in equity_dist  # AAPL
        assert 'Mid' in equity_dist   # SQ
    
    def test_equity_distribution_industry(self):
        """Test equity distribution by industry"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('TSLA', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        df = get_portfolio_df()
        equity_dist = df.iloc[1]['Equity Distribution (Industry)']
        assert 'Software' in equity_dist  # AAPL
        assert 'Auto Manufacturers' in equity_dist  # TSLA
    
    def test_equity_distribution_sector(self):
        """Test equity distribution by sector"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('JPM', 'Equity', 'buy', 'long', 50, 2, '1/2/2025')
        df = get_portfolio_df()
        equity_dist = df.iloc[1]['Equity Distribution (Sector)']
        assert 'Technology' in equity_dist  # AAPL
        assert 'Financial Services' in equity_dist  # JPM


class TestEdgeCases:
    """Test edge cases and complex scenarios"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_position_flip_long_to_short(self):
        """Test position flip from long to short"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 15, '1/2/2025')
        df = get_portfolio_df()
        # Should close 10 shares (realized PnL) and open short 5 shares
        assert df.iloc[1]['Current Position'] == 'Short'
        assert df.iloc[1]['Current Quantity'] == -5
        assert df.iloc[1]['PnL Realized at Point of Time'] == 20  # (12-10)*10
    
    def test_position_flip_short_to_long(self):
        """Test position flip from short to long"""
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'buy', 'long', 8, 15, '1/2/2025')
        df = get_portfolio_df()
        # Should cover 10 shares (realized PnL) and open long 5 shares
        assert df.iloc[1]['Current Position'] == 'Long'
        assert df.iloc[1]['Current Quantity'] == 5
        assert df.iloc[1]['PnL Realized at Point of Time'] == 20  # (10-8)*10
    
    def test_multiple_tickers_unrealized_pnl(self):
        """Test unrealized PnL calculation across multiple tickers"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        add_trade('TSLA', 'Equity', 'buy', 'long', 30, 3, '1/3/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/4/2025')
        df = get_portfolio_df()
        # AAPL: (12-10)*10 = 20
        # MSFT: (20-20)*5 = 0
        # TSLA: (30-30)*3 = 0
        assert df.iloc[3]['Total Unrealized PnL'] == 20
    
    def test_hold_action_maintains_position(self):
        """Test that hold action maintains position"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 12, 0, '1/2/2025')
        df = get_portfolio_df()
        assert df.iloc[1]['Current Quantity'] == 10
        assert df.iloc[1]['Current Position'] == 'Long'
    
    def test_full_close_resets_cost_basis(self):
        """Test that full close resets cost basis"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 10, '1/2/2025')
        assert portfolio_state['cost_basis']['AAPL'] == 0
        assert portfolio_state['avg_price']['AAPL'] == 0


class TestComplexScenarios:
    """Test complex trading scenarios"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_complex_sequence_1(self):
        """Test complex sequence: buy, hold, sell partial, hold, sell rest"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 11, 0, '1/2/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 5, '1/3/2025')
        add_trade('AAPL', 'Equity', 'hold', 'long', 13, 0, '1/4/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 13, 5, '1/5/2025')
        df = get_portfolio_df()
        
        # After first sell: 5 shares remaining, cost basis = 50
        assert df.iloc[2]['Current Quantity'] == 5
        assert df.iloc[2]['Cost Basis'] == 50
        assert df.iloc[2]['PnL Realized at Point of Time'] == 10  # (12-10)*5
        
        # After second sell: 0 shares, cost basis = 0
        assert df.iloc[4]['Current Quantity'] == 0
        assert df.iloc[4]['Cost Basis'] == 0
        assert df.iloc[4]['PnL Realized at Point of Time'] == 15  # (13-10)*5
    
    def test_short_sequence(self):
        """Test short position sequence"""
        add_trade('AAPL', 'Equity', 'sell', 'short', 10, 10, '1/1/2025')
        add_trade('AAPL', 'Equity', 'hold', 'short', 9, 0, '1/2/2025')
        add_trade('AAPL', 'Equity', 'buy', 'long', 8, 10, '1/3/2025')
        df = get_portfolio_df()
        
        # After cover: realized PnL = (10-8)*10 = 20
        assert df.iloc[2]['PnL Realized at Point of Time'] == 20
        assert df.iloc[2]['Current Quantity'] == 0
    
    def test_multiple_tickers_complex(self):
        """Test multiple tickers with complex interactions"""
        add_trade('AAPL', 'Equity', 'buy', 'long', 10, 10, '1/1/2025')
        add_trade('MSFT', 'Equity', 'buy', 'long', 20, 5, '1/2/2025')
        add_trade('AAPL', 'Equity', 'sell', 'long', 12, 5, '1/3/2025')
        add_trade('MSFT', 'Equity', 'hold', 'long', 22, 0, '1/4/2025')
        df = get_portfolio_df()
        
        # Check cumulative realized PnL
        assert df.iloc[3]['PnL Realized Cummulative'] == 10  # (12-10)*5
        
        # Check unrealized PnL for MSFT
        # MSFT: (22-20)*5 = 10
        # AAPL: (12-10)*5 = 10 (using last_price)
        assert df.iloc[3]['Total Unrealized PnL'] == 20


# Run tests with: pytest test_portfolio.py -v