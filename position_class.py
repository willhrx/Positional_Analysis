import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass
from scipy.stats import norm

@dataclass
class MarketConditions:
    """Container for market conditions"""
    spot_price: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0

class Option:
    """Individual option contract"""
    def __init__(self, strike: float, expiry: datetime, option_type: str, 
                 position_size: int):
        """
        Args:
            strike: Strike price
            expiry: Expiration date
            option_type: 'call' or 'put'
            position_size: Number of contracts (negative for short positions)
        """
        self.strike = strike
        self.expiry = expiry
        self.option_type = option_type.lower()
        self.position_size = position_size  # Each contract = 100 shares
        
    def time_to_expiry(self, current_date: datetime = None) -> float:
        """Calculate time to expiry in years"""
        if current_date is None:
            current_date = datetime.now()
        days_to_expiry = (self.expiry - current_date).days
        return max(0, days_to_expiry / 365.25)
    
    def _calculate_d1_d2(self, S: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes formula"""
        d1 = (np.log(S / self.strike) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def calculate_price(self, market: MarketConditions, current_date: datetime = None) -> float:
        """Calculate option price using Black-Scholes"""
        T = self.time_to_expiry(current_date)
        if T == 0:
            # At expiration
            if self.option_type == 'call':
                return max(0, market.spot_price - self.strike)
            else:
                return max(0, self.strike - market.spot_price)
        
        S = market.spot_price
        K = self.strike
        r = market.risk_free_rate
        sigma = market.volatility
        
        d1, d2 = self._calculate_d1_d2(S, T, r, sigma)
        
        if self.option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    def calculate_greeks(self, market: MarketConditions, current_date: datetime = None) -> Dict[str, float]:
        """Calculate all Greeks for the option"""
        T = self.time_to_expiry(current_date)
        if T == 0:
            # At expiration, most Greeks are 0 or undefined
            return {
                'delta': 1.0 if self.option_type == 'call' and market.spot_price > self.strike else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        S = market.spot_price
        K = self.strike
        r = market.risk_free_rate
        sigma = market.volatility
        
        d1, d2 = self._calculate_d1_d2(S, T, r, sigma)
        
        # Delta: Rate of change of option price with respect to underlying price
        if self.option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma: Rate of change of delta with respect to underlying price
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta: Rate of change of option price with respect to time (per day)
        if self.option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.25
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.25
        
        # Vega: Rate of change of option price with respect to volatility (per 1% change)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho: Rate of change of option price with respect to interest rate (per 1% change)
        if self.option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

class Position:
    """Portfolio of options and underlying shares"""
    
    def __init__(self):
        self.underlying_shares = 0
        self.options: List[Option] = []
        self.initial_cost = 0.0  # Track cost basis
        
    def add_underlying(self, shares: int, price: float = None):
        """Add shares of underlying stock"""
        self.underlying_shares += shares
        if price:
            self.initial_cost += shares * price
        print(f"Added {shares} shares. Total shares: {self.underlying_shares}")
    
    def add_option(self, option: Option, premium: float = None):
        """Add option position"""
        self.options.append(option)
        if premium:
            # Negative because we pay premium to buy, receive premium to sell
            self.initial_cost += option.position_size * 100 * premium * (-1 if option.position_size > 0 else 1)
        
        position_type = "Long" if option.position_size > 0 else "Short"
        print(f"Added {position_type} {abs(option.position_size)} {option.option_type} @ {option.strike}")
    
    def remove_option(self, index: int):
        """Remove option by index"""
        if 0 <= index < len(self.options):
            removed = self.options.pop(index)
            print(f"Removed option: {removed.option_type} @ {removed.strike}")
    
    def calculate_total_greeks(self, market: MarketConditions, current_date: datetime = None) -> Dict[str, float]:
        """Calculate aggregate Greeks across all positions"""
        total_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        # Underlying delta is always 1 per share
        total_greeks['delta'] += self.underlying_shares
        
        # Add option Greeks
        for option in self.options:
            option_greeks = option.calculate_greeks(market, current_date)
            # Multiply by position size and contract multiplier (100 shares per contract)
            for greek, value in option_greeks.items():
                total_greeks[greek] += value * option.position_size * 100
        
        return total_greeks
    
    def calculate_position_value(self, market: MarketConditions, current_date: datetime = None) -> float:
        """Calculate current value of entire position"""
        # Underlying value
        value = self.underlying_shares * market.spot_price
        
        # Options value
        for option in self.options:
            option_price = option.calculate_price(market, current_date)
            # Each contract represents 100 shares
            value += option.position_size * 100 * option_price
        
        return value
    
    def calculate_pnl(self, market: MarketConditions, current_date: datetime = None) -> float:
        """Calculate profit/loss of position"""
        current_value = self.calculate_position_value(market, current_date)
        return current_value - self.initial_cost
    
    def calculate_pnl_range(self, spot_prices: np.ndarray, market: MarketConditions, 
                           current_date: datetime = None) -> np.ndarray:
        """Calculate P&L across a range of underlying prices"""
        pnls = []
        original_spot = market.spot_price
        
        for price in spot_prices:
            # Create temporary market conditions with new spot price
            temp_market = MarketConditions(
                spot_price=price,
                risk_free_rate=market.risk_free_rate,
                volatility=market.volatility,
                dividend_yield=market.dividend_yield
            )
            pnl = self.calculate_pnl(temp_market, current_date)
            pnls.append(pnl)
        
        return np.array(pnls)
    
    def get_breakeven_points(self, market: MarketConditions, current_date: datetime = None,
                           price_range: Tuple[float, float] = None) -> List[float]:
        """Find breakeven points for the position"""
        if price_range is None:
            # Default to +/- 50% of current price
            price_range = (market.spot_price * 0.5, market.spot_price * 1.5)
        
        # Create fine grid of prices
        prices = np.linspace(price_range[0], price_range[1], 1000)
        pnls = self.calculate_pnl_range(prices, market, current_date)
        
        # Find where P&L crosses zero
        breakevens = []
        for i in range(len(pnls) - 1):
            if pnls[i] * pnls[i + 1] < 0:  # Sign change
                # Linear interpolation for more precise breakeven
                breakeven = prices[i] - pnls[i] * (prices[i + 1] - prices[i]) / (pnls[i + 1] - pnls[i])
                breakevens.append(breakeven)
        
        return breakevens
    
    def get_max_profit_loss(self, market: MarketConditions, current_date: datetime = None,
                          price_range: Tuple[float, float] = None) -> Dict[str, float]:
        """Calculate maximum profit and loss within a price range"""
        if price_range is None:
            price_range = (0.01, market.spot_price * 3)
        
        prices = np.linspace(price_range[0], price_range[1], 500)
        pnls = self.calculate_pnl_range(prices, market, current_date)
        
        return {
            'max_profit': np.max(pnls),
            'max_profit_price': prices[np.argmax(pnls)],
            'max_loss': np.min(pnls),
            'max_loss_price': prices[np.argmin(pnls)]
        }
    
    def summarize_position(self) -> Dict:
        """Get summary of current position"""
        summary = {
            'underlying_shares': self.underlying_shares,
            'options': [],
            'total_options': len(self.options),
            'net_contracts': sum(opt.position_size for opt in self.options)
        }
        
        for i, option in enumerate(self.options):
            summary['options'].append({
                'index': i,
                'type': option.option_type,
                'strike': option.strike,
                'expiry': option.expiry.strftime('%Y-%m-%d'),
                'position': option.position_size,
                'position_type': 'Long' if option.position_size > 0 else 'Short'
            })
        
        return summary
    
    def calculate_probability_of_profit(self, market: MarketConditions, 
                                      current_date: datetime = None) -> float:
        """Estimate probability of profit at expiration using log-normal distribution"""
        # Find the nearest expiration
        if not self.options:
            return 0.5  # No options, depends on stock direction
        
        nearest_expiry = min(opt.expiry for opt in self.options)
        T = (nearest_expiry - (current_date or datetime.now())).days / 365.25
        
        if T <= 0:
            # Already expired
            current_pnl = self.calculate_pnl(market, current_date)
            return 1.0 if current_pnl > 0 else 0.0
        
        # Get breakeven points
        breakevens = self.get_breakeven_points(market, current_date)
        
        if not breakevens:
            # No breakeven points, always profitable or always loss
            test_pnl = self.calculate_pnl(market, current_date)
            return 1.0 if test_pnl > 0 else 0.0
        
        # Calculate probability using log-normal distribution
        S = market.spot_price
        r = market.risk_free_rate
        sigma = market.volatility
        
        # Expected value of ln(ST/S0)
        drift = (r - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T)
        
        # Calculate probability for each breakeven
        if len(breakevens) == 1:
            # Single breakeven
            breakeven = breakevens[0]
            z = (np.log(breakeven / S) - drift) / diffusion
            
            # Check which side is profitable
            test_price = breakeven * 1.01
            test_market = MarketConditions(test_price, r, sigma)
            test_pnl = self.calculate_pnl(test_market, current_date)
            
            if test_pnl > 0:
                # Profitable above breakeven
                return 1 - norm.cdf(z)
            else:
                # Profitable below breakeven
                return norm.cdf(z)
        else:
            # Multiple breakevens (e.g., iron condor)
            # This is simplified - assumes profitable between first and last breakeven
            z1 = (np.log(min(breakevens) / S) - drift) / diffusion
            z2 = (np.log(max(breakevens) / S) - drift) / diffusion
            return norm.cdf(z2) - norm.cdf(z1)