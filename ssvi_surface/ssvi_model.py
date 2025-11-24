from scipy.optimize import minimize
from scipy.stats import norm
from scipy.optimize import OptimizeResult, brentq
from scipy.interpolate import interp1d
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def forward_bs_price(S: Union[float, np.ndarray], K: Union[float, np.ndarray], T: Union[float, np.ndarray], iv: Union[float, np.ndarray], r: float, q: float, is_call: Union[bool, np.ndarray]) -> np.ndarray:
    """Forward Black-Scholes pricing using F_T = S * exp((r-q)*T)."""
    F_T = S * np.exp((r - q) * T)
    sqrt_T = np.sqrt(np.maximum(T, 1e-8))
    iv_safe = np.maximum(iv, 1e-8)
    d1 = (np.log(F_T / K) + 0.5 * iv_safe**2 * T) / (iv_safe * sqrt_T)
    d2 = d1 - iv_safe * sqrt_T
    call_price = np.exp(-r * T) * (F_T * norm.cdf(d1) - K * norm.cdf(d2))
    put_price = np.exp(-r * T) * (K * norm.cdf(-d2) - F_T * norm.cdf(-d1))
    return np.where(is_call, call_price, put_price)


def _invert_bs_implied_vol(S: float, K: float, T: float, price: float, r: float, q: float, is_call: bool, iv_min: float = 1e-6, iv_max: float = 5.0) -> float:
    """Invert Black-Scholes to get implied volatility from option price.
    
    Uses Brent's method to solve for σ such that BS(S, K, T, σ, r, q, is_call) = price.
    """
    def price_diff(iv):
        result = forward_bs_price(S, K, T, iv, r, q, is_call)
        # Ensure scalar result
        if isinstance(result, np.ndarray):
            return float(result.item()) - price
        return float(result) - price
    
    try:
        iv = brentq(price_diff, iv_min, iv_max, xtol=1e-6)
        return float(iv)
    except ValueError:
        # Fallback: return a reasonable default if inversion fails
        return 0.2


class SSVIModel:
    def __init__(self, lr: float = 1e-3, outside_spread_penalty: float = 0.0, maturity_weight_exponent: float = 1.0, temporal_interp_method: str = 'linear') -> None:
        self.rho: Optional[float] = None
        self.eta: Optional[float] = None
        self.gamma: Optional[float] = None
        self.theta_t: Optional[np.ndarray] = None
        self.T_fitted: Optional[np.ndarray] = None
        self.lr = lr
        self.outside_spread_penalty = outside_spread_penalty
        self.maturity_weight_exponent = maturity_weight_exponent
        self.temporal_interp_method = temporal_interp_method
    
    def theta_interp(self, T: Union[float, np.ndarray]) -> np.ndarray:
        """Interpolation of θ_t for arbitrary T."""
        if self.theta_t is None or self.T_fitted is None:
            raise ValueError("Model not fitted yet")
        T = np.asarray(T)
        
        # Handle case when there's only a single t value
        if len(self.T_fitted) == 1:
            # Return the single theta_t value for any input T
            return np.full_like(T, self.theta_t[0], dtype=self.theta_t.dtype)
        
        interp_func = interp1d(self.T_fitted, self.theta_t, kind=self.temporal_interp_method, 
                              bounds_error=False, fill_value='extrapolate')
        return interp_func(T)
    
    def predict(self, k: Union[float, np.ndarray], t: Union[float, np.ndarray]) -> np.ndarray:
        """Predict implied volatility for given log-moneyness k and time to expiry t."""
        if self.rho is None or self.eta is None or self.gamma is None:
            raise ValueError("Model not fitted yet")
        theta_t = self.theta_interp(t)
        w = self.ssvi(k, self.rho, self.eta, self.gamma, theta_t)
        t = np.asarray(t)
        return np.sqrt(np.maximum(w / t, 1e-8))
    
    def ssvi(self, k: Union[float, np.ndarray], rho: float, eta: float, gamma: float, theta_t: Union[float, np.ndarray]) -> np.ndarray:
        """Canonical SSVI: w(k,t) = (θ_t/2)[1 + ρ*φ(θ_t)*k + sqrt((φ(θ_t)*k + ρ)² + 1 - ρ²)]."""
        k = np.asarray(k)
        theta_t = np.asarray(theta_t)
        phi = eta * (theta_t ** (-gamma))
        phi_k = phi * k
        sqrt_term = np.sqrt((phi_k + rho)**2 + 1 - rho**2)
        return (theta_t / 2) * (1 + rho * phi_k + sqrt_term)
    
    def _compute_spreads(self, bids: np.ndarray, asks: np.ndarray) -> np.ndarray:
        """Compute spreads and fill NaN values with maximum spread."""
        spreads = asks - bids
        max_spread = np.nanmax(spreads)
        if not np.isnan(max_spread):
            spreads = np.where(np.isnan(spreads), max_spread, spreads)
        return spreads

    def _compute_bid_residuals(self, bids: np.ndarray, model_prices: np.ndarray, spreads: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        valid = ~np.isnan(bids) & ~np.isnan(spreads)
        if not np.any(valid):
            return np.array([]), np.array([])
        
        weights = 1.0 / spreads[valid]
        residuals = bids[valid] - model_prices[valid]
    
        return residuals, weights

    def _compute_ask_residuals(self, asks: np.ndarray, model_prices: np.ndarray, spreads: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        valid = ~np.isnan(asks) & ~np.isnan(spreads)
        if not np.any(valid):
            return np.array([]), np.array([])
        
        weights = 1.0 / spreads[valid]
        residuals = asks[valid] - model_prices[valid]
    
        return residuals, weights
    
    def objective(self, params: np.ndarray, T_array: np.ndarray, strikes_list: List[np.ndarray], bids_list: List[np.ndarray], asks_list: List[np.ndarray], option_types_list: List[np.ndarray], S0: float, r: Optional[float], q: Optional[float]) -> float:
        """Weighted least squares: params = [rho, eta, gamma, theta_t0, theta_t1, ..., r?, q?]."""
        rho, eta, gamma = params[0], params[1], params[2]
        n_theta = len(T_array)
        theta_t = params[3:3+n_theta]
        r_val = params[-2] if (r is None and q is None) else (params[-1] if r is None else r)
        q_val = params[-1] if q is None else q
        total_weighted_loss = 0.0
        sum_weights = 0.0
        
        for idx, (T, strikes, bids, asks, option_types) in enumerate(zip(T_array, strikes_list, bids_list, asks_list, option_types_list)):
            if not (np.any(~np.isnan(bids)) or np.any(~np.isnan(asks))):
                continue
            
            F_T = S0 * np.exp((r_val - q_val) * T)
            k = np.log(strikes / F_T)
            spreads = self._compute_spreads(bids, asks)

            w_model = self.ssvi(k, rho, eta, gamma, theta_t[idx])
            iv_model = np.sqrt(np.maximum(w_model / T, 1e-8))
            model_prices = forward_bs_price(S0, strikes, T, iv_model, r_val, q_val, option_types == 'call')

            bid_residuals, bid_weights = self._compute_bid_residuals(bids, model_prices, spreads)
            ask_residuals, ask_weights = self._compute_ask_residuals(asks, model_prices, spreads)
            
            bid_loss = np.sum(bid_residuals**2 * bid_weights) / np.sum(bid_weights)
            ask_loss = np.sum(ask_residuals**2 * ask_weights) / np.sum(ask_weights)
            bid_penalty = np.sum(np.maximum(0, bid_residuals) * bid_weights) / np.sum(bid_weights)
            ask_penalty = np.sum(np.maximum(0, ask_residuals) * ask_weights) / np.sum(ask_weights)
            
            loss = bid_loss + ask_loss + self.outside_spread_penalty * (bid_penalty + ask_penalty)
            weight = T ** self.maturity_weight_exponent
            total_weighted_loss += self.lr * loss * weight
            sum_weights += weight
        
        return total_weighted_loss / sum_weights if sum_weights > 0 else 0.0
    
    def _compute_theta_t_from_atm(self, T: float, strikes: np.ndarray, bids: np.ndarray, asks: np.ndarray, option_types: np.ndarray, S0: float, r: float, q: float) -> float:
        """Step 1 - Slice-wise ATM quantities.
        
        Mathematical model:
        For maturity t:
        1. Compute the forward F_t = S0 * exp((r-q)*t)
        2. Take the option with strike closest to F_t and invert BS to get ATM vol σ_ATM(t)
        3. Set θ_t = σ_ATM(t)^2 * t (with t in years)
        
        Returns:
            theta_t: The θ value for this maturity slice, or NaN if computation fails
        """
        F_t = S0 * np.exp((r - q) * T)
        
        # Find valid bid/ask pairs
        both_valid = ~(np.isnan(bids) | np.isnan(asks))
        if not np.any(both_valid):
            return np.nan
        
        # Find strike closest to forward (ATM)
        valid_strikes = strikes[both_valid]
        valid_bids = bids[both_valid]
        valid_asks = asks[both_valid]
        valid_option_types = option_types[both_valid]
        
        k_valid = np.log(valid_strikes / F_t)
        atm_idx = np.argmin(np.abs(k_valid))
        
        # Get mid price for ATM option
        mid_price = (valid_bids[atm_idx] + valid_asks[atm_idx]) / 2.0
        
        # Invert BS to get implied volatility
        is_call = (valid_option_types[atm_idx] == 'call')
        try:
            sigma_atm = _invert_bs_implied_vol(S0, valid_strikes[atm_idx], T, mid_price, r, q, is_call)
        except:
            return np.nan
        
        # θ_t = σ_ATM(t)^2 * t
        theta_t = sigma_atm ** 2 * T
        
        return max(theta_t, 1e-5)  # Ensure positive
    
    def _compute_atm_derivatives(self, T: float, strikes: np.ndarray, bids: np.ndarray, asks: np.ndarray, option_types: np.ndarray, S0: float, r: float, q: float, F_t: float) -> Tuple[float, float]:
        """Step 2 - Estimate ATM skew and curvature.
        
        Mathematical model:
        On the same slice:
        1. Work with log-moneyness k = log(K/F_t)
        2. For strikes around ATM (one OTM put, one OTM call), compute total variance
           w(k) = σ(k)^2 * t from the BS-implied vols
        3. Approximate first and second derivatives at ATM:
           - Choose two small symmetric points k_- < 0 < k_+
           - w'(0) ≈ (w(k_+) - w(k_-)) / (k_+ - k_-)
           - w''(0) ≈ (w(k_+) - 2w(0) + w(k_-)) / ((k_+ - k_-)/2)^2
        
        Returns:
            w_prime_0: First derivative w'(0) (skew), or NaN if computation fails
            w_double_prime_0: Second derivative w''(0) (curvature), or NaN if computation fails
        """
        # Find valid bid/ask pairs
        both_valid = ~(np.isnan(bids) | np.isnan(asks))
        if not np.any(both_valid):
            return np.nan, np.nan
        
        # Get valid data
        valid_strikes = strikes[both_valid]
        valid_bids = bids[both_valid]
        valid_asks = asks[both_valid]
        valid_option_types = option_types[both_valid]
        
        # Compute log-moneyness
        k_valid = np.log(valid_strikes / F_t)
        
        # Find ATM index (closest to k=0, true ATM)
        atm_idx = np.argmin(np.abs(k_valid))
        k_atm = k_valid[atm_idx]
        
        # Find symmetric points around k=0 (true ATM)
        # Look for OTM put (k < 0) and OTM call (k > 0)
        k_neg = k_valid[k_valid < 0]
        k_pos = k_valid[k_valid > 0]
        
        if len(k_neg) == 0 or len(k_pos) == 0:
            # Fallback: use points around the closest-to-ATM strike
            k_neg = k_valid[k_valid < k_atm]
            k_pos = k_valid[k_valid > k_atm]
            if len(k_neg) == 0 or len(k_pos) == 0:
                return np.nan, np.nan
        
        # Choose points closest to k=0 but on opposite sides
        k_minus = k_neg[np.argmax(k_neg)]  # Closest negative to 0 (largest negative)
        k_plus = k_pos[np.argmin(k_pos)]   # Closest positive to 0 (smallest positive)
        
        # Get indices in original valid arrays
        idx_minus = np.where(k_valid == k_minus)[0][0]
        idx_plus = np.where(k_valid == k_plus)[0][0]
        
        # Compute total variance w(k) = σ(k)^2 * T for each point
        def compute_w(k_idx):
            mid_price = (valid_bids[k_idx] + valid_asks[k_idx]) / 2.0
            is_call = (valid_option_types[k_idx] == 'call')
            try:
                sigma = _invert_bs_implied_vol(S0, valid_strikes[k_idx], T, mid_price, r, q, is_call)
                return sigma ** 2 * T
            except:
                return np.nan
        
        w_minus = compute_w(idx_minus)
        w_atm = compute_w(atm_idx)
        w_plus = compute_w(idx_plus)
        
        # Check for NaN values
        if np.isnan(w_minus) or np.isnan(w_atm) or np.isnan(w_plus):
            return np.nan, np.nan
        
        # Approximate derivatives
        # w'(0) ≈ (w(k_+) - w(k_-)) / (k_+ - k_-)
        w_prime_0 = (w_plus - w_minus) / (k_plus - k_minus)
        
        # w''(0) ≈ (w(k_+) - 2w(0) + w(k_-)) / ((k_+ - k_-)/2)^2
        delta_k = (k_plus - k_minus) / 2.0
        if abs(delta_k) < 1e-8:
            return np.nan, np.nan
        else:
            w_double_prime_0 = (w_plus - 2 * w_atm + w_minus) / (delta_k ** 2)
        
        return w_prime_0, max(w_double_prime_0, 1e-6)  # Ensure positive curvature
    
    def _compute_per_maturity_rho_phi(self, theta_t: float, w_prime_0: float, w_double_prime_0: float, epsilon: float = 1e-6) -> Tuple[float, float]:
        """Step 3 - Back-out per-maturity ρ_t and φ_t.
        
        Mathematical model:
        For SSVI:
        w(k,t) = (θ_t/2)[1 + ρ*φ(θ_t)*k + sqrt((φ(θ_t)*k + ρ)^2 + 1 - ρ^2)]
        
        Exact relationships at ATM:
        w'(0,t) = θ_t * φ(θ_t) * ρ
        w''(0,t) = (1/2) * θ_t * φ(θ_t)^2 * (1 - ρ^2)
        
        Solving:
        1. φ_t^2 = (2/θ_t) * (s_2 + s_1^2/(2*θ_t))
           where s_1 = w'(0), s_2 = w''(0)
        2. φ_t = sqrt(max(φ_t^2, ε))
        3. ρ_t = s_1 / (θ_t * φ_t)
           Clip ρ_t into (-0.999, 0.999)
        
        Returns:
            rho_t: Per-maturity correlation parameter, or NaN if inputs are invalid
            phi_t: Per-maturity φ parameter, or NaN if inputs are invalid
        """
        # Check for NaN inputs
        if np.isnan(theta_t) or np.isnan(w_prime_0) or np.isnan(w_double_prime_0) or theta_t <= 0:
            return np.nan, np.nan
        
        s_1 = w_prime_0
        s_2 = w_double_prime_0
        
        # φ_t^2 = (2/θ_t) * (s_2 + s_1^2/(2*θ_t))
        phi_t_squared = (2.0 / theta_t) * (s_2 + (s_1 ** 2) / (2.0 * theta_t))
        phi_t = np.sqrt(max(phi_t_squared, epsilon))
        
        # ρ_t = s_1 / (θ_t * φ_t)
        if abs(theta_t * phi_t) < 1e-10:
            rho_t = 0.0
        else:
            rho_t = s_1 / (theta_t * phi_t)

        return rho_t, phi_t
    
    def _compute_global_eta_gamma(self, theta_t_array: np.ndarray, phi_t_array: np.ndarray) -> Tuple[float, float]:
        """Step 3 - Compute global η and γ from canonical SSVI power law.
        
        Mathematical model:
        Impose canonical SSVI power law: φ(θ) = η * θ^(-γ)
        
        Fit linear regression:
        log(φ_t) ≈ log(η) - γ * log(θ_t)
        
        across maturities to get initial (η_0, γ_0).
        
        Returns:
            eta: Global η parameter, or NaN if computation fails
            gamma: Global γ parameter, or NaN if computation fails
        """
        # Filter out invalid values (NaN, zero, negative)
        valid_mask = (~np.isnan(theta_t_array)) & (~np.isnan(phi_t_array)) & (theta_t_array > 0) & (phi_t_array > 0)
        if not np.any(valid_mask):
            return np.nan, np.nan
        
        theta_valid = theta_t_array[valid_mask]
        phi_valid = phi_t_array[valid_mask]
        
        if len(theta_valid) < 2:
            return np.nan, np.nan  # Need at least 2 points for regression
        
        # Linear regression: log(φ_t) = log(η) - γ * log(θ_t)
        # y = log(φ_t), x = log(θ_t)
        # y = a + b*x, where a = log(η), b = -γ
        log_theta = np.log(theta_valid)
        log_phi = np.log(phi_valid)
        
        # Simple linear regression
        x_mean = np.mean(log_theta)
        y_mean = np.mean(log_phi)
        
        numerator = np.sum((log_theta - x_mean) * (log_phi - y_mean))
        denominator = np.sum((log_theta - x_mean) ** 2)
        
        if abs(denominator) < 1e-10:
            return np.nan, np.nan
        
        b = numerator / denominator  # b = -γ
        a = y_mean - b * x_mean      # a = log(η)
        
        gamma = -b
        eta = np.exp(a)

        return eta, gamma
    
    def _estimate_initial_params(self, T_array: np.ndarray, strikes_list: List[np.ndarray], bids_list: List[np.ndarray], asks_list: List[np.ndarray], option_types_list: List[np.ndarray], S0: float, r: Optional[float], q: Optional[float]) -> np.ndarray:
        """Estimate initial parameters using data-driven approach.
        
        Coordinates the three-step mathematical model:
        Step 1: Compute θ_t from ATM vols for each maturity
        Step 2: Estimate ATM skew and curvature (w'(0) and w''(0))
        Step 3: Back-out per-maturity ρ_t and φ_t, then global η and γ
        
        Returns:
            Initial parameter array: [rho, eta, gamma] + theta_t per maturity + r?, q?
        """
        r_init = r if r is not None else 0.045
        q_init = q if q is not None else 0.0
        
        # Step 1: Compute θ_t for each maturity
        theta_t_array = []
        rho_t_array = []
        phi_t_array = []
        
        for T, strikes, bids, asks, option_types in zip(T_array, strikes_list, bids_list, asks_list, option_types_list):
            # Step 1: Compute θ_t from ATM vol
            theta_t = self._compute_theta_t_from_atm(T, strikes, bids, asks, option_types, S0, r_init, q_init)
            theta_t_array.append(theta_t)
            
            # Step 2: Compute ATM derivatives
            F_t = S0 * np.exp((r_init - q_init) * T)
            w_prime_0, w_double_prime_0 = self._compute_atm_derivatives(
                T, strikes, bids, asks, option_types, S0, r_init, q_init, F_t
            )
            
            # Step 3: Compute per-maturity ρ_t and φ_t
            rho_t, phi_t = self._compute_per_maturity_rho_phi(theta_t, w_prime_0, w_double_prime_0)
            rho_t_array.append(rho_t)
            phi_t_array.append(phi_t)
        
        theta_t_array = np.array(theta_t_array)
        rho_t_array = np.array(rho_t_array)
        phi_t_array = np.array(phi_t_array)
        
        # Fill NaN values in theta_t_array: forward fill then backward fill
        theta_t_filled = theta_t_array.copy()
        # Forward fill
        for i in range(1, len(theta_t_filled)):
            if np.isnan(theta_t_filled[i]) and not np.isnan(theta_t_filled[i-1]):
                theta_t_filled[i] = theta_t_filled[i-1]
        # Backward fill
        for i in range(len(theta_t_filled) - 2, -1, -1):
            if np.isnan(theta_t_filled[i]) and not np.isnan(theta_t_filled[i+1]):
                theta_t_filled[i] = theta_t_filled[i+1]
        # If still NaN values remain, use default (0.1 * T for each maturity)
        for i in range(len(theta_t_filled)):
            if np.isnan(theta_t_filled[i]):
                theta_t_filled[i] = 0.1 * T_array[i]
        
        # Step 3: Compute global η and γ from regression (ignoring NaNs)
        eta_init, gamma_init = self._compute_global_eta_gamma(theta_t_array, phi_t_array)
        
        # For canonical SSVI (constant ρ), take median of per-maturity ρ_t (ignoring NaNs)
        rho_init = np.nanmedian(rho_t_array) if len(rho_t_array) > 0 else np.nan
        
        # Use defaults only at the very end if final parameters are NaN
        if np.isnan(rho_init):
            rho_init = 0.0
        if np.isnan(eta_init):
            eta_init = 1.0
        if np.isnan(gamma_init):
            gamma_init = 0.5
        
        # Build parameter array
        base_params = np.concatenate([[rho_init, eta_init, gamma_init], theta_t_filled])
        extra_params = []
        if r is None:
            extra_params.append(r_init)
        if q is None:
            extra_params.append(q_init)
        return np.concatenate([base_params, extra_params]) if extra_params else base_params
    
    def fit(self, T_array: Union[np.ndarray, List[float]], strikes_list: List[np.ndarray], bids_list: List[np.ndarray], asks_list: List[np.ndarray], option_types_list: List[np.ndarray], S0: float, r: Optional[float], q: Optional[float], initial_params: Optional[np.ndarray] = None) -> OptimizeResult:
        """Fit canonical SSVI: fixed [rho, eta, gamma] + theta_t array + optional r, q."""
        T_array = np.asarray(T_array)
        sort_idx = np.argsort(T_array)
        T_array = T_array[sort_idx]
        strikes_list = [strikes_list[i] for i in sort_idx]
        bids_list = [bids_list[i] for i in sort_idx]
        asks_list = [asks_list[i] for i in sort_idx]
        option_types_list = [option_types_list[i] for i in sort_idx]
        n_maturities = len(T_array)
        
        if initial_params is None:
            initial_params = self._estimate_initial_params(T_array, strikes_list, bids_list, asks_list, option_types_list, S0, r, q)
        else:
            initial_params = np.concatenate([initial_params[:3], initial_params[3:3+n_maturities][sort_idx]])
            extra_params = []
            if r is None:
                extra_params.append(0.025)
            if q is None:
                extra_params.append(0.0)
            if extra_params:
                initial_params = np.concatenate([initial_params, extra_params])
        
        def arb_constraint(params: np.ndarray) -> float:
            rho, eta = params[0], params[1]
            return 4 - eta * (1 + np.abs(rho))
        
        def monotonicity_constraint(params: np.ndarray) -> np.ndarray:
            theta_t = params[3:3+n_maturities]
            return np.diff(theta_t) if len(theta_t) > 1 else np.array([0.0])
        
        bounds = [(-0.9, 0.9), (1e-3, 3.0), (1e-4, 0.99)] + [(1e-5, 10.0)] * n_maturities
        if r is None:
            bounds.append((-0.1, 0.1))
        if q is None:
            bounds.append((-0.1, 0.1))
        
        result = minimize(
            self.objective,
            x0=initial_params,
            args=(T_array, strikes_list, bids_list, asks_list, option_types_list, S0, r, q),
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': arb_constraint}, {'type': 'ineq', 'fun': monotonicity_constraint}],
            options={'maxiter': 5000, 'ftol': 1e-6, 'eps': 1e-4}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        self.rho, self.eta, self.gamma = result.x[0], result.x[1], result.x[2]
        self.theta_t = result.x[3:3+n_maturities]
        self.r_fitted = result.x[-2] if (r is None and q is None) else (result.x[-1] if r is None else r)
        self.q_fitted = result.x[-1] if q is None else q
        self.T_fitted = T_array.copy()
        return result
    
    def check_bid_ask_crossings(self, option_df: pd.DataFrame, S0: float, r: float, q: float) -> pd.DataFrame:
        """Check if model prices cross outside bid/ask spread, grouped by maturity and contract type.
        
        Returns a DataFrame with columns:
        - expiry: expiry date string
        - option_type: 'call' or 'put'
        - strike: strike price
        - T: time to expiry (years)
        - model_price: model predicted price
        - bid: bid price
        - ask: ask price
        - crosses_below_bid: True if model_price < bid
        - crosses_above_ask: True if model_price > ask
        - amount_below_bid: bid - model_price if crosses below, else 0
        - amount_above_ask: model_price - ask if crosses above, else 0
        """
        if self.rho is None or self.eta is None or self.gamma is None or self.theta_t is None:
            raise ValueError("Model must be fitted before checking bid/ask crossings")
        
        rows = []
        expiries = sorted(option_df["expiry"].unique())
        
        for expiry in expiries:
            df_exp = option_df[option_df["expiry"] == expiry].sort_values(["strike", "option_type"])
            if len(df_exp) == 0:
                continue
            
            T = df_exp["T"].iloc[0]
            theta_t_val = self.theta_interp(T)
            
            for option_type in ['call', 'put']:
                df_type = df_exp[df_exp["option_type"] == option_type].sort_values("strike")
                if len(df_type) == 0:
                    continue
                
                strikes = df_type["strike"].values
                bids = df_type["bid_usd"].values
                asks = df_type["ask_usd"].values
                
                # Compute model prices
                F_T = S0 * np.exp((r - q) * T)
                k = np.log(strikes / F_T)
                w_model = self.ssvi(k, self.rho, self.eta, self.gamma, theta_t_val)
                iv_model = np.sqrt(np.maximum(w_model / T, 1e-8))
                is_call = (option_type == 'call')
                model_prices = forward_bs_price(S0, strikes, T, iv_model, r, q, is_call)
                
                # Check crossings
                for strike, model_price, bid, ask in zip(strikes, model_prices, bids, asks):
                    crosses_below_bid = False
                    crosses_above_ask = False
                    amount_below_bid = 0.0
                    amount_above_ask = 0.0
                    
                    if not np.isnan(bid) and model_price < bid:
                        crosses_below_bid = True
                        amount_below_bid = bid - model_price
                    
                    if not np.isnan(ask) and model_price > ask:
                        crosses_above_ask = True
                        amount_above_ask = model_price - ask
                    
                    rows.append({
                        "expiry": expiry,
                        "option_type": option_type,
                        "strike": strike,
                        "T": T,
                        "model_price": model_price,
                        "bid": bid,
                        "ask": ask,
                        "crosses_below_bid": crosses_below_bid,
                        "crosses_above_ask": crosses_above_ask,
                        "amount_below_bid": amount_below_bid,
                        "amount_above_ask": amount_above_ask
                    })
        
        return pd.DataFrame(rows)
