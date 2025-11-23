import unittest
import numpy as np
from ssvi_surface import SSVIModel


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability with edge cases (valid inputs)."""
    
    def setUp(self):
        self.model = SSVIModel()
        self.model.rho = -0.3
        self.model.eta = 1.0
        self.model.gamma = 0.5
        self.model.T_fitted = np.array([0.1, 0.25, 0.5, 1.0])
        self.model.theta_t = np.array([0.01, 0.02, 0.03, 0.04])
        
    def test_very_large_k(self):
        """Test with very large |k| values (deep ITM/OTM)."""
        k_large = np.array([-5, -3, 3, 5])
        t = 0.5
        
        # Should not crash and produce finite results
        iv = self.model.predict(k_large, t)
        self.assertTrue(np.all(np.isfinite(iv)))
        self.assertTrue(np.all(iv > 0))
        
    def test_very_small_theta_t(self):
        """Test with very small theta_t values."""
        k = 0.1
        t = 0.5
        
        # Use very small theta_t
        self.model.theta_t = np.array([1e-6, 1e-5, 1e-4, 1e-3])
        iv = self.model.predict(k, t)
        
        self.assertTrue(np.isfinite(iv))
        self.assertTrue(iv > 0)
        
    def test_boundary_parameter_values(self):
        """Test with boundary parameter values."""
        k = 0.1
        t = 0.5
        
        # Test boundary rho values
        for rho in [-0.9, 0.0, 0.9]:
            self.model.rho = rho
            iv = self.model.predict(k, t)
            self.assertTrue(np.isfinite(iv))
            self.assertTrue(iv > 0)
            
        # Test boundary gamma values
        self.model.rho = -0.3
        for gamma in [0.01, 0.5, 0.99]:
            self.model.gamma = gamma
            iv = self.model.predict(k, t)
            self.assertTrue(np.isfinite(iv))
            self.assertTrue(iv > 0)
            
    def test_numerical_stability_ssvi(self):
        """Test SSVI formula numerical stability at extremes."""
        # Very large k
        k = 10.0
        w = self.model.ssvi(k, self.model.rho, self.model.eta, self.model.gamma, 0.04)
        self.assertTrue(np.isfinite(w))
        self.assertTrue(w > 0)
        
        # Very small theta_t
        w = self.model.ssvi(0.1, self.model.rho, self.model.eta, self.model.gamma, 1e-8)
        self.assertTrue(np.isfinite(w))
        self.assertTrue(w > 0)


if __name__ == '__main__':
    unittest.main()

