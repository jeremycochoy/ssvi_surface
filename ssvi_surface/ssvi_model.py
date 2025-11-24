from scipy.optimize import minimize
from scipy.stats import norm
from scipy.optimize import OptimizeResult
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
    
    def _fill_spreads(self, k: np.ndarray, bids: np.ndarray, asks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fill missing bids/asks with mid prices for spread calculation."""
        bids_filled = bids.copy()
        asks_filled = asks.copy()
        
        both_valid = ~(np.isnan(bids) | np.isnan(asks))
        if not np.any(both_valid):
            return bids_filled, asks_filled
        
        mids = np.where(both_valid, (bids + asks) / 2.0, np.nan)
        atm_idx = np.where(both_valid)[0][np.argmin(np.abs(k[both_valid]))]
        k_atm = k[atm_idx]
        
        for i in range(len(k)):
            if np.isnan(bids[i]) or np.isnan(asks[i]):
                k_min, k_max = min(k_atm, k[i]), max(k_atm, k[i])
                valid_mids = mids[(k >= k_min) & (k <= k_max) & both_valid]
                valid_mids = valid_mids[~np.isnan(valid_mids)]
                
                if len(valid_mids) > 0:
                    if np.isnan(bids[i]):
                        bids_filled[i] = np.min(valid_mids)
                    if np.isnan(asks[i]):
                        asks_filled[i] = np.max(valid_mids)
        
        return bids_filled, asks_filled
    
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
            bid_filled, ask_filled = self._fill_spreads(k, bids, asks)
            spreads = ask_filled - bid_filled
            
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
    
    def _estimate_initial_params(self, T_array: np.ndarray, strikes_list: List[np.ndarray], bids_list: List[np.ndarray], asks_list: List[np.ndarray], option_types_list: List[np.ndarray], S0: float, r: Optional[float], q: Optional[float]) -> np.ndarray:
        """Estimate initial parameters: [rho, eta, gamma] + theta_t per maturity + r?, q?."""
        r_init = r if r is not None else 0.025
        q_init = q if q is not None else 0.0
        theta_t_init = []
        for T, strikes, bids, asks in zip(T_array, strikes_list, bids_list, asks_list):
            F_T = S0 * np.exp((r_init - q_init) * T)
            k = np.log(strikes / F_T)
            both_valid = ~(np.isnan(bids) | np.isnan(asks))
            
            if np.any(both_valid):
                atm_idx = np.where(both_valid)[0][np.argmin(np.abs(k[both_valid]))]
                mid_price = (bids[atm_idx] + asks[atm_idx]) / 2.0
                theta_t_init.append(0.64 * T if mid_price >= 1e-6 else 0.1 * T)
            else:
                theta_t_init.append(0.1 * T)
        
        base_params = np.concatenate([[-0.3, 1.0, 0.5], theta_t_init])
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
