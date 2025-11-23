import unittest
import numpy as np
import pandas as pd
from ssvi_surface import SSVIModel, forward_bs_price


class TestBidAskCrossings(unittest.TestCase):
    """Test bid/ask crossings method."""
    
    def setUp(self):
        self.model = SSVIModel()
        self.S0 = 100.0
        self.r = 0.025
        self.q = 0.0
        
        # Fit model with synthetic data
        T_array = np.array([0.25, 0.5])
        strikes_list = [
            np.array([90, 100, 110]),
            np.array([90, 100, 110])
        ]
        
        # Generate synthetic prices
        bids_list = []
        asks_list = []
        option_types_list = []
        
        for T, strikes in zip(T_array, strikes_list):
            F_T = self.S0 * np.exp((self.r - self.q) * T)
            k = np.log(strikes / F_T)
            iv = 0.2 + 0.1 * k**2
            iv = np.maximum(iv, 0.05)
            
            call_prices = forward_bs_price(self.S0, strikes, T, iv, self.r, self.q, True)
            
            spread = 0.02
            bids = call_prices - spread / 2
            asks = call_prices + spread / 2
            option_types = np.array(['call'] * len(strikes))
            
            bids_list.append(bids)
            asks_list.append(asks)
            option_types_list.append(option_types)
            
        self.model.fit(T_array, strikes_list, bids_list, asks_list, 
                       option_types_list, self.S0, self.r, self.q)
        
        # Create option DataFrame
        rows = []
        for i, (T, strikes) in enumerate(zip(T_array, strikes_list)):
            for j, strike in enumerate(strikes):
                rows.append({
                    'expiry': f'2024-{i+1:02d}-01',
                    'strike': strike,
                    'option_type': 'call',
                    'T': T,
                    'bid_usd': bids_list[i][j],
                    'ask_usd': asks_list[i][j]
                })
        self.option_df = pd.DataFrame(rows)
        
    def test_error_before_fitting(self):
        """Property: Should raise ValueError before fitting."""
        model = SSVIModel()
        with self.assertRaises(ValueError):
            model.check_bid_ask_crossings(self.option_df, self.S0, self.r, self.q)
            
    def test_crossing_detection(self):
        """Property: Verify crossing detection logic."""
        crossings_df = self.model.check_bid_ask_crossings(self.option_df, self.S0, self.r, self.q)
        
        # Check that crossing flags are set correctly
        for _, row in crossings_df.iterrows():
            if row['crosses_below_bid']:
                self.assertLess(row['model_price'], row['bid'])
                self.assertGreater(row['amount_below_bid'], 0)
            else:
                self.assertEqual(row['amount_below_bid'], 0)
                
            if row['crosses_above_ask']:
                self.assertGreater(row['model_price'], row['ask'])
                self.assertGreater(row['amount_above_ask'], 0)
            else:
                self.assertEqual(row['amount_above_ask'], 0)
                
    def test_amount_calculation(self):
        """Property: Verify amount calculations."""
        crossings_df = self.model.check_bid_ask_crossings(self.option_df, self.S0, self.r, self.q)
        
        for _, row in crossings_df.iterrows():
            if row['crosses_below_bid']:
                expected_amount = row['bid'] - row['model_price']
                self.assertAlmostEqual(row['amount_below_bid'], expected_amount, places=6)
            else:
                self.assertEqual(row['amount_below_bid'], 0.0)
                
            if row['crosses_above_ask']:
                expected_amount = row['model_price'] - row['ask']
                self.assertAlmostEqual(row['amount_above_ask'], expected_amount, places=6)
            else:
                self.assertEqual(row['amount_above_ask'], 0.0)
                
    def test_output_structure(self):
        """Property: Verify output DataFrame structure."""
        crossings_df = self.model.check_bid_ask_crossings(self.option_df, self.S0, self.r, self.q)
        
        expected_columns = [
            'expiry', 'option_type', 'strike', 'T', 'model_price',
            'bid', 'ask', 'crosses_below_bid', 'crosses_above_ask',
            'amount_below_bid', 'amount_above_ask'
        ]
        
        self.assertEqual(list(crossings_df.columns), expected_columns)
        self.assertGreater(len(crossings_df), 0)


if __name__ == '__main__':
    unittest.main()

