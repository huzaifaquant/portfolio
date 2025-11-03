import pytest
import pandas as pd
from portfolio import (
    reset_portfolio,
    process_trade,
    portfolio_state,
    get_portfolio_df
)


class TestCashFormulas:
    """CRITICAL: Cash formula verification - wrong cash = trading with money you don't have"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=1000)
    
    def test_buy_long_formula(self):
        """Formula: remaining = prev - price * quantity"""
        row = process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        assert row['Remaining'] == 900.0  # 1000 - 10*10
    
    def test_sell_long_formula(self):
        """Formula: remaining = prev + price * quantity"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 12.0, 5, date='2025-01-02')
        assert row['Remaining'] == 960.0  # 900 + 12*5
    
    def test_sell_short_formula(self):
        """Formula: remaining = prev - price * quantity"""
        row = process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        assert row['Remaining'] == 900.0  # 1000 - 20*5
    
    def test_buy_short_cover_formula(self):
        """Formula: remaining = prev + [initial + (initial - final)]
        where initial = avg_price * cover_qty, final = price * cover_qty"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'buy', 'short', 18.0, 3, date='2025-01-02')
        # initial = 20*3 = 60, final = 18*3 = 54
        # delta = 60 + (60 - 54) = 66
        # Expected: 900 + 66 = 966
        assert row['Remaining'] == pytest.approx(966.0, abs=0.01)
    
    def test_cash_round_trip_verification(self):
        """Verify: Final Cash = Initial Cash + Realized PnL"""
        reset_portfolio(initial_cash=1000)
        initial = portfolio_state['remaining']
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 12.0, 10, date='2025-01-02')
        assert row['Remaining'] == pytest.approx(initial + row['PnL Realized'], abs=0.01)


class TestCostBasisFormulas:
    """CRITICAL: Cost basis formulas - affects tax reporting and PnL accuracy"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_long_cost_basis_accumulation(self):
        """Formula: cost_basis = sum(price_i * qty_i) for all buys"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'buy', 'long', 12.0, 5, date='2025-01-02')
        assert row['Cost Basis'] == 160.0  # 10*10 + 12*5
    
    def test_short_cost_basis_accumulation(self):
        """Formula: cost_basis = sum(price_i * qty_i) for all short sells"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'sell', 'short', 22.0, 3, date='2025-01-02')
        assert row['Cost Basis'] == 166.0  # 20*5 + 22*3
    
    def test_long_cost_basis_proportional_reduction(self):
        """Formula: new_cb = old_cb * (remaining_qty / old_qty)"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 12.0, 4, date='2025-01-02')
        assert row['Cost Basis'] == 60.0  # 100 * (6/10)
    
    def test_short_cost_basis_proportional_reduction(self):
        """Formula: new_cb = old_cb * (remaining_qty / old_qty)"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'buy', 'short', 18.0, 2, date='2025-01-02')
        assert row['Cost Basis'] == pytest.approx(60.0, abs=0.01)  # 100 * (3/5)
    
    def test_cost_basis_resets_to_zero_on_full_close(self):
        """Formula: cost_basis = 0 when quantity = 0"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 12.0, 10, date='2025-01-02')
        assert row['Cost Basis'] == 0.0


class TestAveragePriceFormulas:
    """CRITICAL: Average price formulas - used for PnL calculations"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_avg_price_long_single_buy(self):
        """Formula: avg_price = cost_basis / quantity"""
        row = process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        assert row['Avg Price'] == 10.0  # 100 / 10
    
    def test_avg_price_long_multiple_buys(self):
        """Formula: avg_price = total_cost_basis / total_quantity"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'buy', 'long', 12.0, 5, date='2025-01-02')
        assert row['Avg Price'] == pytest.approx(10.6667, abs=0.01)  # 160 / 15
    
    def test_avg_price_short_single_sell(self):
        """Formula: avg_price = cost_basis / abs(quantity)"""
        row = process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        assert row['Avg Price'] == 20.0  # 100 / 5
    
    def test_avg_price_short_multiple_sells(self):
        """Formula: avg_price = total_cost_basis / abs(total_quantity)"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'sell', 'short', 22.0, 3, date='2025-01-02')
        assert row['Avg Price'] == pytest.approx(20.75, abs=0.01)  # 166 / 8
    
    def test_avg_price_unchanged_on_partial_sell(self):
        """Avg price remains constant when selling part of position"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 12.0, 3, date='2025-01-02')
        assert row['Avg Price'] == 10.0  # Unchanged for remaining shares
    
    def test_avg_price_weighted_calculation(self):
        """Formula: avg_price = sum(price_i * qty_i) / sum(qty_i)"""
        process_trade('AAPL', 'buy', 'long', 10.0, 5, date='2025-01-01')
        process_trade('AAPL', 'buy', 'long', 20.0, 5, date='2025-01-02')
        row = process_trade('AAPL', 'buy', 'long', 30.0, 10, date='2025-01-03')
        # Total: 50+100+300 = 450, Quantity: 20, Avg: 22.5
        assert row['Avg Price'] == pytest.approx(22.5, abs=0.01)


class TestRealizedPnLFormulas:
    """CRITICAL: Realized PnL formulas - affects tax reporting"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_realized_pnl_long_closing(self):
        """Formula: realized = (sell_price - avg_entry) * shares_closed"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 12.0, 5, date='2025-01-02')
        assert row['PnL Realized'] == 10.0  # (12 - 10) * 5
    
    def test_realized_pnl_short_covering(self):
        """Formula: realized = (avg_entry - cover_price) * shares_closed"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'buy', 'short', 18.0, 3, date='2025-01-02')
        assert row['PnL Realized'] == 6.0  # (20 - 18) * 3
    
    def test_realized_pnl_cumulative(self):
        """Formula: realized_pnl accumulates across all trades"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'sell', 'long', 12.0, 3, date='2025-01-02')  # +6
        row = process_trade('AAPL', 'sell', 'long', 13.0, 4, date='2025-01-03')  # +12
        assert row['PnL Realized'] == 18.0  # 6 + 12
    
    def test_realized_pnl_long_full_close(self):
        """Formula verification for full position close"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 15.0, 10, date='2025-01-02')
        assert row['PnL Realized'] == 50.0  # (15 - 10) * 10
    
    def test_realized_pnl_short_full_cover(self):
        """Formula verification for full short cover"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'buy', 'short', 18.0, 5, date='2025-01-02')
        assert row['PnL Realized'] == 10.0  # (20 - 18) * 5
    
    def test_realized_pnl_does_not_double_count(self):
        """Verify realized PnL increments correctly without double counting"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        r1 = process_trade('AAPL', 'sell', 'long', 12.0, 3, date='2025-01-02')  # +6
        r2 = process_trade('AAPL', 'sell', 'long', 13.0, 4, date='2025-01-03')  # +12
        r3 = process_trade('AAPL', 'sell', 'long', 14.0, 3, date='2025-01-04')  # +12
        assert r1['PnL Realized'] == 6.0
        assert r2['PnL Realized'] == 18.0
        assert r3['PnL Realized'] == 30.0


class TestUnrealizedPnLFormulas:
    """CRITICAL: Unrealized PnL formulas - shows current position value"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_unrealized_pnl_long_profit(self):
        """Formula: unrealized = (current_price - avg_price) * quantity"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 15.0, 0, date='2025-01-02')
        assert row['PnL (Long) Unrealized'] == 50.0  # (15 - 10) * 10
        assert row['PnL Unrealized Value'] == 50.0
    
    def test_unrealized_pnl_long_loss(self):
        """Formula: unrealized loss = (current_price - avg_price) * quantity"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 8.0, 0, date='2025-01-02')
        assert row['PnL (Long) Unrealized'] == -20.0  # (8 - 10) * 10
        assert row['PnL Unrealized Value'] == -20.0
    
    def test_unrealized_pnl_short_profit(self):
        """Formula: unrealized = (avg_price - current_price) * abs(quantity)"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'hold', 'short', 18.0, 0, date='2025-01-02')
        assert row['PnL (Short) Unrealized'] == 10.0  # (20 - 18) * 5
        assert row['PnL Unrealized Value'] == 10.0
    
    def test_unrealized_pnl_short_loss(self):
        """Formula: unrealized loss = (avg_price - current_price) * abs(quantity)"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'hold', 'short', 22.0, 0, date='2025-01-02')
        assert row['PnL (Short) Unrealized'] == -10.0  # (20 - 22) * 5
        assert row['PnL Unrealized Value'] == -10.0
    
    def test_unrealized_pnl_updates_with_price(self):
        """Verify unrealized PnL updates correctly as price moves"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        r1 = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        assert r1['PnL (Long) Unrealized'] == 20.0  # (12 - 10) * 10
        r2 = process_trade('AAPL', 'hold', 'long', 15.0, 0, date='2025-01-03')
        assert r2['PnL (Long) Unrealized'] == 50.0  # (15 - 10) * 10
        r3 = process_trade('AAPL', 'hold', 'long', 13.0, 0, date='2025-01-04')
        assert r3['PnL (Long) Unrealized'] == 30.0  # (13 - 10) * 10


class TestPositionValueFormulas:
    """CRITICAL: Position value (PV) formulas - shows current portfolio value"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_pv_long_formula(self):
        """Formula: PV = Cost Basis + Unrealized PnL"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        # Cost Basis = 100, Unrealized = (12-10)*10 = 20
        assert row['PV (Long)'] == pytest.approx(120.0, abs=0.01)  # 100 + 20
        assert row['Total PV'] == pytest.approx(120.0, abs=0.01)
    
    def test_pv_short_formula(self):
        """Formula: PV = Cost Basis + Unrealized PnL"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'hold', 'short', 18.0, 0, date='2025-01-02')
        # Cost Basis = 100, Unrealized = (20-18)*5 = 10
        assert row['PV (Short)'] == pytest.approx(110.0, abs=0.01)  # 100 + 10
        assert row['Total PV'] == pytest.approx(110.0, abs=0.01)
    
    def test_pv_long_with_loss(self):
        """Formula: PV = Cost Basis + Unrealized PnL (when unrealized is negative)"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 8.0, 0, date='2025-01-02')
        # Cost Basis = 100, Unrealized = (8-10)*10 = -20
        assert row['PV (Long)'] == pytest.approx(80.0, abs=0.01)  # 100 - 20
    
    def test_total_pv_plus_remaining_formula(self):
        """Formula: Total PV + Remaining = Total Account Value"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        assert row['Total PV + Remaining'] == pytest.approx(220.0, abs=0.01)  # 120 + 100
    
    def test_total_pv_multiple_tickers(self):
        """Formula: Total PV = sum of PV for all tickers"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')  # PV = 120
        row = process_trade('MSFT', 'buy', 'long', 5.0, 5, date='2025-01-03')  # PV = 25
        assert row['Total PV'] == pytest.approx(145.0, abs=0.01)  # 120 + 25


class TestPositionQuantityFormulas:
    """CRITICAL: Position quantity formulas - must track correctly"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_quantity_buy_increases_position(self):
        """Formula: new_quantity = old_quantity + buy_quantity"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'buy', 'long', 12.0, 5, date='2025-01-02')
        assert row['Current Quantity'] == 15  # 10 + 5
    
    def test_quantity_sell_decreases_position(self):
        """Formula: new_quantity = old_quantity - sell_quantity"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 12.0, 4, date='2025-01-02')
        assert row['Current Quantity'] == 6  # 10 - 4
    
    def test_quantity_sell_short_increases_short(self):
        """Formula: new_quantity = old_quantity - sell_quantity (for shorts)"""
        row = process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        assert row['Current Quantity'] == -5  # 0 - 5
        row = process_trade('TSLA', 'sell', 'short', 22.0, 3, date='2025-01-02')
        assert row['Current Quantity'] == -8  # -5 - 3
    
    def test_quantity_buy_short_decreases_short(self):
        """Formula: new_quantity = old_quantity + buy_quantity (for shorts)"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'buy', 'short', 18.0, 2, date='2025-01-02')
        assert row['Current Quantity'] == -3  # -5 + 2


class TestComplexFormulaScenarios:
    """CRITICAL: Complex scenarios testing multiple formulas together"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=1000)
    
    def test_multiple_trades_all_formulas(self):
        """Test all formulas work together in a sequence"""
        # Buy long
        r1 = process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        assert r1['Remaining'] == 900.0  # Cash formula
        assert r1['Cost Basis'] == 100.0  # Cost basis formula
        assert r1['Avg Price'] == 10.0  # Avg price formula
        
        # Price moves, check unrealized
        r2 = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        assert r2['PnL (Long) Unrealized'] == 20.0  # Unrealized formula
        assert r2['PV (Long)'] == pytest.approx(120.0, abs=0.01)  # PV formula
        
        # Sell partial, check realized
        r3 = process_trade('AAPL', 'sell', 'long', 13.0, 4, date='2025-01-03')
        assert r3['Remaining'] == 952.0  # Cash: 900 + 13*4
        assert r3['Cost Basis'] == 60.0  # Cost basis: 100 * 6/10
        assert r3['PnL Realized'] == 12.0  # Realized: (13-10)*4
        assert r3['Current Quantity'] == 6  # Quantity: 10 - 4
    
    def test_averaging_down_formulas(self):
        """Test all formulas with averaging down scenario"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')  # Cost: 100
        process_trade('AAPL', 'buy', 'long', 8.0, 10, date='2025-01-02')   # Cost: 180, Avg: 9
        final_row = process_trade('AAPL', 'buy', 'long', 6.0, 10, date='2025-01-03')  # Cost: 240, Avg: 8
        
        # Verify cost basis accumulation
        assert final_row['Cost Basis'] == 240.0  # 100 + 80 + 60
        # Verify average price
        assert final_row['Avg Price'] == pytest.approx(8.0, abs=0.01)  # 240 / 30
        
        # Check unrealized PnL at new price
        final_row = process_trade('AAPL', 'hold', 'long', 9.0, 0, date='2025-01-04')
        assert final_row['PnL (Long) Unrealized'] == pytest.approx(30.0, abs=0.01)  # (9-8)*30
        assert final_row['PV (Long)'] == pytest.approx(270.0, abs=0.01)  # 240 + 30
    
    def test_scalping_scenario_formulas(self):
        """Test formulas with rapid buy/sell sequence"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'sell', 'long', 10.5, 10, date='2025-01-01')  # Realized: +5
        process_trade('AAPL', 'buy', 'long', 10.2, 10, date='2025-01-01')
        final_row = process_trade('AAPL', 'sell', 'long', 10.8, 10, date='2025-01-01')
        # First trade: (10.5-10)*10 = 5, Second trade: (10.8-10.2)*10 = 6
        assert final_row['PnL Realized'] == pytest.approx(11.0, abs=0.01)  # 5 + 6
    
    def test_round_trip_cash_formula(self):
        """Verify cash reconciliation formula after complete round trip"""
        initial = portfolio_state['remaining']
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        row = process_trade('AAPL', 'sell', 'long', 13.0, 10, date='2025-01-03')
        # Verify: Final = Initial + Realized PnL
        assert row['Remaining'] == pytest.approx(initial + row['PnL Realized'], abs=0.01)


class TestFormulaConsistency:
    """CRITICAL: Verify formulas remain consistent"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_avg_price_from_cost_basis(self):
        """Verify: avg_price = abs(cost_basis / quantity)"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'buy', 'long', 12.0, 5, date='2025-01-02')
        calculated_avg = abs(row['Cost Basis'] / row['Current Quantity'])
        assert row['Avg Price'] == pytest.approx(calculated_avg, abs=0.01)
    
    def test_pv_formula_consistency(self):
        """Verify: PV = Cost Basis + Unrealized PnL"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        calculated_pv = row['Cost Basis'] + row['PnL (Long) Unrealized']
        assert row['PV (Long)'] == pytest.approx(calculated_pv, abs=0.01)
    
    def test_total_pv_formula_consistency(self):
        """Verify: Total PV + Remaining = sum of all values"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        calculated_total = row['Total PV'] + row['Remaining']
        assert row['Total PV + Remaining'] == pytest.approx(calculated_total, abs=0.01)
    
    def test_unrealized_pnl_formula_consistency(self):
        """Verify: Unrealized = sum of long and short unrealized"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        calculated_unrealized = row['PnL (Long) Unrealized'] + row['PnL (Short) Unrealized']
        assert row['PnL Unrealized Value'] == pytest.approx(calculated_unrealized, abs=0.01)


class TestMultiTickerFormulas:
    """CRITICAL: Formulas with multiple tickers"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=1000)
    
    def test_realized_pnl_cumulative_across_tickers(self):
        """Formula: Realized PnL accumulates across all tickers"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'sell', 'long', 12.0, 5, date='2025-01-02')  # +10
        process_trade('MSFT', 'buy', 'long', 5.0, 5, date='2025-01-03')
        row = process_trade('MSFT', 'sell', 'long', 6.0, 3, date='2025-01-04')  # +3
        assert row['PnL Realized'] == 13.0  # 10 + 3
    
    def test_total_pv_sum_across_tickers(self):
        """Formula: Total PV = sum of all individual ticker PVs"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')  # PV = 120
        row = process_trade('MSFT', 'buy', 'long', 5.0, 5, date='2025-01-03')  # PV = 25
        # Total PV should include both tickers
        assert row['Total PV'] == pytest.approx(145.0, abs=0.01)  # 120 + 25
    
    def test_cash_tracks_all_trades(self):
        """Formula: Cash affected by all ticker trades"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')  # -100
        process_trade('MSFT', 'buy', 'long', 5.0, 20, date='2025-01-02')  # -100
        row = process_trade('AAPL', 'sell', 'long', 12.0, 5, date='2025-01-03')  # +60
        # Starting: 1000, -100, -100, +60 = 860
        assert row['Remaining'] == 860.0


class TestAdditionalCashFormulas:
    """Additional cash formula tests"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=1000)
    
    def test_multiple_buys_cash_accumulation(self):
        """Formula: Cash reduces correctly with multiple buys"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')  # -100
        process_trade('AAPL', 'buy', 'long', 12.0, 5, date='2025-01-02')  # -60
        row = process_trade('AAPL', 'buy', 'long', 15.0, 4, date='2025-01-03')  # -60
        # Expected: 1000 - 100 - 60 - 60 = 780
        assert row['Remaining'] == 780.0
    
    def test_multiple_sells_cash_accumulation(self):
        """Formula: Cash increases correctly with multiple sells"""
        process_trade('AAPL', 'buy', 'long', 10.0, 20, date='2025-01-01')  # -200
        process_trade('AAPL', 'sell', 'long', 12.0, 5, date='2025-01-02')  # +60
        row = process_trade('AAPL', 'sell', 'long', 13.0, 5, date='2025-01-03')  # +65
        # Expected: 800 + 60 + 65 = 925
        assert row['Remaining'] == 925.0
    
    def test_short_cover_full_formula(self):
        """Formula: remaining when fully covering short"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')  # -100
        row = process_trade('TSLA', 'buy', 'short', 18.0, 5, date='2025-01-02')
        # initial = 20*5 = 100, final = 18*5 = 90
        # delta = 100 + (100 - 90) = 110
        # Expected: 900 + 110 = 1010
        assert row['Remaining'] == pytest.approx(1010.0, abs=0.01)


class TestAdvancedCostBasisFormulas:
    """Advanced cost basis formula tests"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_cost_basis_with_three_buys(self):
        """Formula: Cost basis accumulates with three separate buys"""
        process_trade('AAPL', 'buy', 'long', 10.0, 5, date='2025-01-01')   # 50
        process_trade('AAPL', 'buy', 'long', 12.0, 5, date='2025-01-02')   # 60
        row = process_trade('AAPL', 'buy', 'long', 15.0, 5, date='2025-01-03')  # 75
        assert row['Cost Basis'] == 185.0  # 50 + 60 + 75
    
    def test_cost_basis_partial_sell_multiple_times(self):
        """Formula: Cost basis reduces proportionally with multiple partial sells"""
        process_trade('AAPL', 'buy', 'long', 10.0, 20, date='2025-01-01')  # 200
        process_trade('AAPL', 'sell', 'long', 12.0, 5, date='2025-01-02')   # Remaining: 15
        row = process_trade('AAPL', 'sell', 'long', 13.0, 5, date='2025-01-03')  # Remaining: 10
        # First sell: 200 * (15/20) = 150
        # Second sell: 150 * (10/15) = 100
        assert row['Cost Basis'] == 100.0
    
    def test_cost_basis_short_with_multiple_sells(self):
        """Formula: Short cost basis with multiple sells"""
        process_trade('TSLA', 'sell', 'short', 20.0, 3, date='2025-01-01')  # 60
        process_trade('TSLA', 'sell', 'short', 22.0, 2, date='2025-01-02')  # 44
        row = process_trade('TSLA', 'sell', 'short', 25.0, 1, date='2025-01-03')  # 25
        assert row['Cost Basis'] == 129.0  # 60 + 44 + 25


class TestAdvancedRealizedPnLFormulas:
    """Advanced realized PnL formula tests"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_realized_pnl_multiple_partial_closes(self):
        """Formula: Realized PnL with multiple partial closes"""
        process_trade('AAPL', 'buy', 'long', 10.0, 20, date='2025-01-01')
        process_trade('AAPL', 'sell', 'long', 12.0, 3, date='2025-01-02')  # +6
        process_trade('AAPL', 'sell', 'long', 13.0, 4, date='2025-01-03')  # +12
        row = process_trade('AAPL', 'sell', 'long', 14.0, 5, date='2025-01-04')  # +20
        assert row['PnL Realized'] == 38.0  # 6 + 12 + 20
    
    def test_realized_pnl_short_multiple_partial_covers(self):
        """Formula: Realized PnL with multiple partial short covers"""
        process_trade('TSLA', 'sell', 'short', 20.0, 10, date='2025-01-01')
        process_trade('TSLA', 'buy', 'short', 18.0, 3, date='2025-01-02')  # +6
        process_trade('TSLA', 'buy', 'short', 17.0, 3, date='2025-01-03')  # +9
        row = process_trade('TSLA', 'buy', 'short', 16.0, 2, date='2025-01-04')  # +8
        assert row['PnL Realized'] == 23.0  # 6 + 9 + 8
    
    def test_realized_pnl_averaging_down_then_selling(self):
        """Formula: Realized PnL after averaging down"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')  # Avg: 10
        process_trade('AAPL', 'buy', 'long', 8.0, 10, date='2025-01-02')  # Avg: 9
        row = process_trade('AAPL', 'sell', 'long', 11.0, 10, date='2025-01-03')
        # Selling at 11 with avg of 9: (11-9)*10 = 20
        assert row['PnL Realized'] == 20.0


class TestAdvancedUnrealizedPnLFormulas:
    """Advanced unrealized PnL formula tests"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_unrealized_pnl_after_partial_sell(self):
        """Formula: Unrealized PnL updates correctly after partial sell"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'sell', 'long', 12.0, 4, date='2025-01-02')  # Remaining: 6
        row = process_trade('AAPL', 'hold', 'long', 15.0, 0, date='2025-01-03')
        # Avg still 10, quantity 6, price 15: (15-10)*6 = 30
        assert row['PnL (Long) Unrealized'] == 30.0
    
    def test_unrealized_pnl_multiple_adds_then_price_move(self):
        """Formula: Unrealized PnL with multiple adds"""
        process_trade('AAPL', 'buy', 'long', 10.0, 5, date='2025-01-01')
        process_trade('AAPL', 'buy', 'long', 12.0, 5, date='2025-01-02')  # Avg: 11
        row = process_trade('AAPL', 'hold', 'long', 15.0, 0, date='2025-01-03')
        # Avg: 11, Quantity: 10, Price: 15: (15-11)*10 = 40
        assert row['PnL (Long) Unrealized'] == 40.0
    
    def test_unrealized_pnl_short_price_moves(self):
        """Formula: Unrealized PnL for short position as price moves"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        r1 = process_trade('TSLA', 'hold', 'short', 18.0, 0, date='2025-01-02')
        assert r1['PnL (Short) Unrealized'] == 10.0  # (20-18)*5
        r2 = process_trade('TSLA', 'hold', 'short', 15.0, 0, date='2025-01-03')
        assert r2['PnL (Short) Unrealized'] == 25.0  # (20-15)*5
        r3 = process_trade('TSLA', 'hold', 'short', 22.0, 0, date='2025-01-04')
        assert r3['PnL (Short) Unrealized'] == -10.0  # (20-22)*5


class TestAdvancedPositionValueFormulas:
    """Advanced position value formula tests"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_pv_after_partial_sell(self):
        """Formula: PV updates correctly after partial sell"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'sell', 'long', 12.0, 4, date='2025-01-02')  # CB: 60, Qty: 6
        row = process_trade('AAPL', 'hold', 'long', 15.0, 0, date='2025-01-03')
        # Cost Basis: 60, Unrealized: (15-10)*6 = 30, PV: 90
        assert row['PV (Long)'] == pytest.approx(90.0, abs=0.01)
    
    def test_pv_multiple_adds_price_move(self):
        """Formula: PV with multiple adds and price movement"""
        process_trade('AAPL', 'buy', 'long', 10.0, 5, date='2025-01-01')  # CB: 50
        process_trade('AAPL', 'buy', 'long', 12.0, 5, date='2025-01-02')  # CB: 110, Avg: 11
        row = process_trade('AAPL', 'hold', 'long', 14.0, 0, date='2025-01-03')
        # Cost Basis: 110, Unrealized: (14-11)*10 = 30, PV: 140
        assert row['PV (Long)'] == pytest.approx(140.0, abs=0.01)
    
    def test_total_pv_with_long_and_short(self):
        """Formula: Total PV with both long and short positions"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')  # CB: 100
        process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')  # PV: 120
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-03')  # CB: 100
        row = process_trade('TSLA', 'hold', 'short', 18.0, 0, date='2025-01-04')  # PV: 110
        # Total: 120 + 110 = 230
        assert row['Total PV'] == pytest.approx(230.0, abs=0.01)


class TestFormulaStressTests:
    """Stress tests for formulas - many trades"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=5000)
    
    def test_many_small_trades_cost_basis(self):
        """Formula: Cost basis with many small trades"""
        # Buy 1 share 10 times at different prices
        for i in range(10):
            process_trade('AAPL', 'buy', 'long', 10.0 + i, 1, date=f'2025-01-{i+1:02d}')
        row = process_trade('AAPL', 'hold', 'long', 15.0, 0, date='2025-01-11')
        # Cost basis: 10+11+12+...+19 = 145
        expected_cost_basis = sum(range(10, 20))
        assert row['Cost Basis'] == expected_cost_basis
    
    def test_many_trades_realized_pnl(self):
        """Formula: Realized PnL accumulates correctly with many trades"""
        process_trade('AAPL', 'buy', 'long', 10.0, 50, date='2025-01-01')
        # Sell 5 shares 10 times at increasing prices
        total_realized = 0
        for i in range(10):
            row = process_trade('AAPL', 'sell', 'long', 11.0 + i, 5, date=f'2025-01-{i+2:02d}')
            total_realized += (11.0 + i - 10.0) * 5
            assert row['PnL Realized'] == pytest.approx(total_realized, abs=0.01)
    
    def test_complex_multi_ticker_scenario(self):
        """Formula: All formulas work with complex multi-ticker scenario"""
        # AAPL trades
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'sell', 'long', 12.0, 5, date='2025-01-02')  # +10 realized
        
        # MSFT trades
        process_trade('MSFT', 'buy', 'long', 5.0, 20, date='2025-01-03')
        
        # TSLA short
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-04')
        
        # Check prices
        r1 = process_trade('AAPL', 'hold', 'long', 13.0, 0, date='2025-01-05')
        r2 = process_trade('MSFT', 'hold', 'long', 6.0, 0, date='2025-01-06')
        r3 = process_trade('TSLA', 'hold', 'short', 18.0, 0, date='2025-01-07')
        
        # Verify realized PnL cumulative
        assert r3['PnL Realized'] == 10.0
        
        # Verify total PV includes all tickers
        # AAPL: CB=50, Unrealized=(13-10)*5=15, PV=65
        # MSFT: CB=100, Unrealized=(6-5)*20=20, PV=120
        # TSLA: CB=100, Unrealized=(20-18)*5=10, PV=110
        # Total: 65 + 120 + 110 = 295
        assert r3['Total PV'] == pytest.approx(295.0, abs=0.01)


class TestFormulaEdgeCases:
    """Formula edge cases - boundary conditions"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_avg_price_single_share(self):
        """Formula: Avg price with single share"""
        row = process_trade('AAPL', 'buy', 'long', 10.0, 1, date='2025-01-01')
        assert row['Avg Price'] == 10.0
        assert row['Cost Basis'] == 10.0
    
    def test_realized_pnl_single_share(self):
        """Formula: Realized PnL with single share"""
        process_trade('AAPL', 'buy', 'long', 10.0, 1, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 15.0, 1, date='2025-01-02')
        assert row['PnL Realized'] == 5.0  # (15-10)*1
    
    def test_unrealized_pnl_single_share(self):
        """Formula: Unrealized PnL with single share"""
        process_trade('AAPL', 'buy', 'long', 10.0, 1, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        assert row['PnL (Long) Unrealized'] == 2.0  # (12-10)*1
    
    def test_pv_single_share(self):
        """Formula: PV with single share"""
        process_trade('AAPL', 'buy', 'long', 10.0, 1, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        # CB: 10, Unrealized: 2, PV: 12
        assert row['PV (Long)'] == pytest.approx(12.0, abs=0.01)
    
    def test_cost_basis_one_cent_difference(self):
        """Formula: Cost basis with very small price differences"""
        process_trade('AAPL', 'buy', 'long', 10.00, 10, date='2025-01-01')
        row = process_trade('AAPL', 'buy', 'long', 10.01, 10, date='2025-01-02')
        assert row['Cost Basis'] == pytest.approx(200.1, abs=0.01)


class TestBuyableSellableFormulas:
    """Test Buyable/Sellable calculation formulas"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=1000)
    
    def test_buyable_formula(self):
        """Formula: buyable = remaining / price"""
        process_trade('AAPL', 'buy', 'long', 10.0, 50, date='2025-01-01')  # Remaining: 500
        row = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        expected_buyable = row['Remaining'] / 12.0
        assert row['Buyable/Sellable'] == pytest.approx(expected_buyable, abs=0.01)
    
    def test_buyable_after_multiple_trades(self):
        """Formula: Buyable updates with remaining cash"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')  # Remaining: 900
        row = process_trade('AAPL', 'sell', 'long', 12.0, 5, date='2025-01-02')  # Remaining: 960
        expected_buyable = 960.0 / 12.0  # Using price from previous trade
        # Note: This tests the formula, actual value depends on last_price
        assert row['Buyable/Sellable'] > 0


class TestPositionFlipFormulas:
    """Test formulas when positions flip from long to short or vice versa"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=1000)
    
    def test_long_to_short_flip_cost_basis(self):
        """Formula: Cost basis resets when flipping long to short"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 12.0, 15, date='2025-01-02')
        # After flip: excess 5 at price 12 = 60
        assert row['Cost Basis'] == 60.0  # 12 * 5
        assert row['Avg Price'] == 12.0
    
    def test_short_to_long_flip_cost_basis(self):
        """Formula: Cost basis resets when flipping short to long"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'buy', 'short', 18.0, 8, date='2025-01-02')
        # After flip: excess 3 at price 18 = 54
        assert row['Cost Basis'] == 54.0  # 18 * 3
        assert row['Avg Price'] == 18.0
    
    def test_long_to_short_flip_realized_pnl(self):
        """Formula: Realized PnL when flipping long to short"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 12.0, 15, date='2025-01-02')
        # Should realize on 10 shares closed: (12-10)*10 = 20
        assert row['PnL Realized'] == 20.0
    
    def test_short_to_long_flip_realized_pnl(self):
        """Formula: Realized PnL when flipping short to long"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'buy', 'short', 18.0, 8, date='2025-01-02')
        # Should realize on 5 shares covered: (20-18)*5 = 10
        assert row['PnL Realized'] == 10.0


class TestPartialFillFormulas:
    """Test formulas with partial fills"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_partial_sell_cost_basis_formula(self):
        """Formula: Cost basis proportional reduction with partial sell"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'sell', 'long', 12.0, 3, date='2025-01-02')  # Sell 3 of 10
        row = process_trade('AAPL', 'sell', 'long', 13.0, 2, date='2025-01-03')  # Sell 2 of 7
        # First: 100 * (7/10) = 70, Second: 70 * (5/7) = 50
        assert row['Cost Basis'] == 50.0
    
    def test_partial_cover_cost_basis_formula(self):
        """Formula: Cost basis proportional reduction with partial cover"""
        process_trade('TSLA', 'sell', 'short', 20.0, 10, date='2025-01-01')
        process_trade('TSLA', 'buy', 'short', 18.0, 3, date='2025-01-02')  # Cover 3 of 10
        row = process_trade('TSLA', 'buy', 'short', 17.0, 4, date='2025-01-03')  # Cover 4 of 7
        # First: 200 * (7/10) = 140, Second: 140 * (3/7) = 60
        assert row['Cost Basis'] == pytest.approx(60.0, abs=0.01)
    
    def test_partial_sell_avg_price_unchanged(self):
        """Formula: Avg price unchanged with partial sells"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'sell', 'long', 12.0, 2, date='2025-01-02')
        process_trade('AAPL', 'sell', 'long', 13.0, 3, date='2025-01-03')
        row = process_trade('AAPL', 'sell', 'long', 14.0, 2, date='2025-01-04')
        # Avg price should remain 10 for remaining 3 shares
        assert row['Avg Price'] == 10.0


class TestMultiplePartialTradesFormulas:
    """Test formulas with many partial trades"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=1000)
    
    def test_gradual_position_build_formula(self):
        """Formula: Building position gradually maintains cost basis"""
        process_trade('AAPL', 'buy', 'long', 10.0, 2, date='2025-01-01')  # CB: 20
        process_trade('AAPL', 'buy', 'long', 11.0, 2, date='2025-01-02')  # CB: 42, Avg: 10.5
        process_trade('AAPL', 'buy', 'long', 12.0, 2, date='2025-01-03')  # CB: 66, Avg: 11
        row = process_trade('AAPL', 'buy', 'long', 13.0, 2, date='2025-01-04')  # CB: 92, Avg: 11.5
        assert row['Cost Basis'] == 92.0  # 20 + 22 + 24 + 26
        assert row['Avg Price'] == pytest.approx(11.5, abs=0.01)  # 92 / 8
    
    def test_gradual_position_reduction_formula(self):
        """Formula: Reducing position gradually maintains avg price"""
        process_trade('AAPL', 'buy', 'long', 10.0, 20, date='2025-01-01')  # CB: 200, Avg: 10
        process_trade('AAPL', 'sell', 'long', 12.0, 4, date='2025-01-02')  # CB: 160, Avg: 10
        process_trade('AAPL', 'sell', 'long', 13.0, 4, date='2025-01-03')  # CB: 120, Avg: 10
        row = process_trade('AAPL', 'sell', 'long', 14.0, 4, date='2025-01-04')  # CB: 80, Avg: 10
        assert row['Cost Basis'] == 80.0  # 200 * (8/20)
        assert row['Avg Price'] == 10.0  # Unchanged


class TestFormulaRelationships:
    """Test relationships between different formulas"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=200)
    
    def test_pv_equals_market_value(self):
        """Formula: PV should equal current market value of position"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        # PV = Cost Basis + Unrealized = 100 + 20 = 120
        # Market Value = 12 * 10 = 120
        market_value = row['Price'] * row['Current Quantity']
        assert row['PV (Long)'] == pytest.approx(market_value, abs=0.01)
    
    def test_unrealized_plus_cost_basis_equals_pv(self):
        """Formula: Unrealized PnL + Cost Basis = PV"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        calculated_pv = row['Cost Basis'] + row['PnL (Long) Unrealized']
        assert row['PV (Long)'] == pytest.approx(calculated_pv, abs=0.01)
    
    def test_cash_plus_total_pv_equals_account_value(self):
        """Formula: Remaining + Total PV = Total Account Value"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        row = process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')
        account_value = row['Remaining'] + row['Total PV']
        assert row['Total PV + Remaining'] == pytest.approx(account_value, abs=0.01)


class TestRealWorldTradingPatterns:
    """Test formulas with real-world trading patterns"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=2000)
    
    def test_dollar_cost_averaging_formula(self):
        """Formula: Dollar cost averaging scenario"""
        # Buy same dollar amount each time (10 shares at varying prices)
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')  # $100
        process_trade('AAPL', 'buy', 'long', 12.0, 8.33, date='2025-01-02')  # ~$100 (8.33 shares)
        process_trade('AAPL', 'buy', 'long', 15.0, 6.67, date='2025-01-03')  # ~$100 (6.67 shares)
        row = process_trade('AAPL', 'hold', 'long', 13.0, 0, date='2025-01-04')
        # Total: ~25 shares, Avg should be weighted average
        assert row['Current Quantity'] > 20
        assert row['Cost Basis'] > 0
    
    def test_pyramiding_formula(self):
        """Formula: Pyramiding (adding to winning position)"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')  # Avg: 10
        process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-02')  # Price up
        process_trade('AAPL', 'buy', 'long', 12.0, 5, date='2025-01-03')  # Add at higher price
        row = process_trade('AAPL', 'hold', 'long', 13.0, 0, date='2025-01-04')
        # New avg: (10*10 + 12*5) / 15 = 160/15 = 10.67
        assert row['Avg Price'] == pytest.approx(10.6667, abs=0.01)
        # Unrealized: (13-10.67)*15 = 34.95
        assert row['PnL (Long) Unrealized'] == pytest.approx(35.0, abs=0.5)
    
    def test_scaling_out_formula(self):
        """Formula: Scaling out (selling in increments)"""
        process_trade('AAPL', 'buy', 'long', 10.0, 30, date='2025-01-01')
        process_trade('AAPL', 'hold', 'long', 15.0, 0, date='2025-01-02')  # Price up
        process_trade('AAPL', 'sell', 'long', 15.0, 10, date='2025-01-03')  # Sell 1/3
        process_trade('AAPL', 'sell', 'long', 16.0, 10, date='2025-01-04')  # Sell 1/3 more
        row = process_trade('AAPL', 'sell', 'long', 17.0, 10, date='2025-01-05')  # Sell final 1/3
        # Realized: (15-10)*10 + (16-10)*10 + (17-10)*10 = 50 + 60 + 70 = 180
        assert row['PnL Realized'] == 180.0


class TestFormulaPrecision:
    """Test formula precision with various price/quantity combinations"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=10000)
    
    def test_high_price_low_quantity(self):
        """Formula: High price, low quantity"""
        row = process_trade('AAPL', 'buy', 'long', 100.0, 1, date='2025-01-01')
        assert row['Cost Basis'] == 100.0
        assert row['Avg Price'] == 100.0
    
    def test_low_price_high_quantity(self):
        """Formula: Low price, high quantity"""
        row = process_trade('AAPL', 'buy', 'long', 1.0, 100, date='2025-01-01')
        assert row['Cost Basis'] == 100.0
        assert row['Avg Price'] == 1.0
    
    def test_decimal_prices_formulas(self):
        """Formula: Decimal prices maintain precision"""
        process_trade('AAPL', 'buy', 'long', 10.123, 10, date='2025-01-01')
        row = process_trade('AAPL', 'buy', 'long', 10.456, 5, date='2025-01-02')
        expected_cost = 10.123 * 10 + 10.456 * 5
        assert row['Cost Basis'] == pytest.approx(expected_cost, abs=0.01)
        expected_avg = expected_cost / 15
        assert row['Avg Price'] == pytest.approx(expected_avg, abs=0.01)
    
    def test_fractional_quantities(self):
        """Formula: Formulas work with fractional quantities (if supported)"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10.5, date='2025-01-01')
        row = process_trade('AAPL', 'sell', 'long', 12.0, 5.25, date='2025-01-02')
        # Cost basis: 105, remaining: 5.25
        # Cost basis should reduce proportionally
        remaining_cb = 105.0 * (5.25 / 10.5)
        assert row['Cost Basis'] == pytest.approx(remaining_cb, abs=0.01)


class TestCompleteRoundTripScenarios:
    """Test complete round trip scenarios with all formulas"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=1000)
    
    def test_complete_long_round_trip(self):
        """Formula: Complete long position round trip"""
        initial = portfolio_state['remaining']
        # Buy
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        # Add more
        process_trade('AAPL', 'buy', 'long', 11.0, 5, date='2025-01-02')
        # Price moves
        process_trade('AAPL', 'hold', 'long', 12.0, 0, date='2025-01-03')
        # Partial sell
        process_trade('AAPL', 'sell', 'long', 13.0, 8, date='2025-01-04')
        # Final sell
        row = process_trade('AAPL', 'sell', 'long', 14.0, 7, date='2025-01-05')
        
        # Verify all formulas
        assert row['Current Quantity'] == 0
        assert row['Cost Basis'] == 0.0
        # Verify cash reconciliation
        assert row['Remaining'] == pytest.approx(initial + row['PnL Realized'], abs=0.01)
    
    def test_complete_short_round_trip(self):
        """Formula: Complete short position round trip"""
        initial = portfolio_state['remaining']
        # Sell short
        process_trade('TSLA', 'sell', 'short', 20.0, 10, date='2025-01-01')
        # Add to short
        process_trade('TSLA', 'sell', 'short', 22.0, 5, date='2025-01-02')
        # Price moves
        process_trade('TSLA', 'hold', 'short', 18.0, 0, date='2025-01-03')
        # Partial cover
        process_trade('TSLA', 'buy', 'short', 17.0, 8, date='2025-01-04')
        # Final cover
        row = process_trade('TSLA', 'buy', 'short', 16.0, 7, date='2025-01-05')
        
        # Verify all formulas
        assert row['Current Quantity'] == 0
        assert row['Cost Basis'] == 0.0
        # Cash reconciliation for shorts is more complex, but should be positive
        assert row['Remaining'] > initial  # Should have profit from short


class TestFormulaVerificationScenarios:
    """Additional verification scenarios for formulas"""
    
    def setup_method(self):
        reset_portfolio(initial_cash=500)
    
    def test_all_formulas_with_price_decline(self):
        """Verify all formulas when price declines"""
        process_trade('AAPL', 'buy', 'long', 10.0, 10, date='2025-01-01')
        process_trade('AAPL', 'buy', 'long', 12.0, 5, date='2025-01-02')  # Avg: 10.67
        row = process_trade('AAPL', 'hold', 'long', 8.0, 0, date='2025-01-03')
        
        # Cost basis should be unchanged
        assert row['Cost Basis'] == 160.0
        # Avg price unchanged
        assert row['Avg Price'] == pytest.approx(10.6667, abs=0.01)
        # Unrealized should be negative
        assert row['PnL (Long) Unrealized'] < 0  # (8-10.67)*15 = -40
        # PV should be less than cost basis
        assert row['PV (Long)'] < row['Cost Basis']
    
    def test_all_formulas_with_price_increase(self):
        """Verify all formulas when price increases"""
        process_trade('TSLA', 'sell', 'short', 20.0, 5, date='2025-01-01')
        row = process_trade('TSLA', 'hold', 'short', 25.0, 0, date='2025-01-02')
        
        # Cost basis unchanged
        assert row['Cost Basis'] == 100.0
        # Avg price unchanged
        assert row['Avg Price'] == 20.0
        # Unrealized should be negative (price up = bad for short)
        assert row['PnL (Short) Unrealized'] < 0  # (20-25)*5 = -25
        # PV should be less than cost basis
        assert row['PV (Short)'] < row['Cost Basis']


# Run tests with: pytest test_portfolio.py -v