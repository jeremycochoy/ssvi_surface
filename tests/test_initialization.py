import unittest
import numpy as np
from ssvi_surface import SSVIModel


class TestInitialization(unittest.TestCase):
    def test_default_initialization(self):
        """Test that model initializes with default parameters."""
        model = SSVIModel()
        self.assertEqual(model.lr, 1e-3)
        self.assertEqual(model.outside_spread_penalty, 0.0)
        self.assertEqual(model.temporal_interp_method, 'linear')
        self.assertIsNone(model.rho)
        self.assertIsNone(model.eta)
        self.assertIsNone(model.gamma)
        self.assertIsNone(model.theta_t)
        self.assertIsNone(model.T_fitted)

    def test_custom_initialization(self):
        """Test that model initializes with custom parameters."""
        model = SSVIModel(
            lr=0.001,
            outside_spread_penalty=2.0,
            temporal_interp_method='cubic'
        )
        self.assertEqual(model.lr, 0.001)
        self.assertEqual(model.outside_spread_penalty, 2.0)
        self.assertEqual(model.temporal_interp_method, 'cubic')
        self.assertIsNone(model.rho)
        self.assertIsNone(model.eta)
        self.assertIsNone(model.gamma)


if __name__ == '__main__':
    unittest.main()

