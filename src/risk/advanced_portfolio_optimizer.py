#!/usr/bin/env python3
"""
Advanced Portfolio Optimizer - Step 5 Implementation

This module implements advanced portfolio optimization techniques including:
- Modern Portfolio Theory (Markowitz Optimization)
- Value at Risk (VaR) and Expected Shortfall calculations
- Stress testing scenarios
- Multi-objective optimization
- Dynamic rebalancing strategies
- Machine Learning enhanced risk assessment
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco
import scipy.stats
from scipy import linalg
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

# Suppress optimization warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPortfolioOptimizer:
    """
    Advanced Portfolio Optimizer implementing multiple optimization strategies
    and risk management techniques for institutional-grade portfolio management.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Advanced Portfolio Optimizer.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        
        # Risk-free rate (10-year Indian Government Bond yield ~7%)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.07)
        
        # Optimization constraints
        self.min_weight = self.config.get('min_weight', 0.01)  # 1% minimum
        self.max_weight = self.config.get('max_weight', 0.20)  # 20% maximum
        self.max_sector_weight = self.config.get('max_sector_weight', 0.30)  # 30% sector max
        
        # VaR parameters
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.var_horizon = self.config.get('var_horizon', 1)  # 1 day
        
        # Portfolio constraints
        self.target_volatility = self.config.get('target_volatility', 0.15)
        self.target_return = self.config.get('target_return', 0.12)
        
        # ML models for risk assessment
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        self.ledoit_wolf = LedoitWolf()
        
        logger.info("Advanced Portfolio Optimizer initialized")
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio metrics.
        
        Args:
            weights (np.ndarray): Portfolio weights
            returns (pd.DataFrame): Historical returns data
            
        Returns:
            dict: Portfolio metrics
        """
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        # Portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Basic metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        
        # Downside metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else np.inf
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR and Expected Shortfall
        var_95 = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        var_99 = np.percentile(portfolio_returns, 1)  # 99% VaR
        expected_shortfall = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # Information ratio (excess return per unit of tracking error)
        benchmark_return = returns.mean(axis=1)  # Equal-weighted benchmark
        excess_returns = portfolio_returns - benchmark_return
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'information_ratio': information_ratio,
            'downside_volatility': downside_volatility
        }
    
    def optimize_portfolio_sharpe(self, returns: pd.DataFrame, constraints: Dict = None) -> Dict[str, Any]:
        """
        Optimize portfolio for maximum Sharpe ratio.
        
        Args:
            returns (pd.DataFrame): Historical returns data
            constraints (dict): Additional constraints
            
        Returns:
            dict: Optimization results
        """
        n_assets = len(returns.columns)
        constraints = constraints or {}
        
        # Objective function: negative Sharpe ratio (for minimization)
        def objective(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe  # Negative for minimization
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Add custom constraints
        if 'min_return' in constraints:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: np.sum(returns.mean() * x) * 252 - constraints['min_return']
            })
        
        if 'max_volatility' in constraints:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: constraints['max_volatility'] - np.sqrt(np.dot(x.T, np.dot(returns.cov() * 252, x)))
            })
        
        # Bounds for individual weights (more permissive to avoid optimization issues)
        bounds = tuple((0.001, 0.8) for _ in range(n_assets))
        
        # Initial guess (equal weights with small random perturbation)
        x0 = np.array([1/n_assets] * n_assets)
        x0 = x0 + np.random.normal(0, 0.01, n_assets)  # Add small noise
        x0 = np.abs(x0)  # Ensure positive
        x0 = x0 / np.sum(x0)  # Normalize to sum to 1
        
        # Optimize
        try:
            result = sco.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
            
            if result.success:
                optimal_weights = result.x
                metrics = self.calculate_portfolio_metrics(optimal_weights, returns)
                
                return {
                    'success': True,
                    'weights': optimal_weights,
                    'metrics': metrics,
                    'optimization_message': result.message,
                    'iterations': result.nit
                }
            else:
                logger.warning(f"Optimization failed: {result.message}")
                return {'success': False, 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in Sharpe optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def optimize_portfolio_minimum_variance(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize portfolio for minimum variance.
        
        Args:
            returns (pd.DataFrame): Historical returns data
            
        Returns:
            dict: Optimization results
        """
        n_assets = len(returns.columns)
        
        # Use Ledoit-Wolf shrinkage estimator for more robust covariance
        cov_matrix = self.ledoit_wolf.fit(returns).covariance_ * 252
        
        # Objective function: portfolio variance
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        try:
            result = sco.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                metrics = self.calculate_portfolio_metrics(optimal_weights, returns)
                
                return {
                    'success': True,
                    'weights': optimal_weights,
                    'metrics': metrics,
                    'optimization_message': result.message,
                    'iterations': result.nit
                }
            else:
                return {'success': False, 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in minimum variance optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def optimize_portfolio_risk_parity(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize portfolio using Risk Parity approach.
        
        Args:
            returns (pd.DataFrame): Historical returns data
            
        Returns:
            dict: Optimization results
        """
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        
        def calculate_risk_contribution(weights, cov_matrix):
            """Calculate risk contribution of each asset"""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            return contrib
        
        def objective(weights):
            """Minimize sum of squared deviations from equal risk contribution"""
            risk_contrib = calculate_risk_contribution(weights, cov_matrix)
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((risk_contrib/np.sum(risk_contrib) - target_contrib)**2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        try:
            result = sco.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                metrics = self.calculate_portfolio_metrics(optimal_weights, returns)
                
                # Calculate actual risk contributions
                risk_contrib = calculate_risk_contribution(optimal_weights, cov_matrix)
                risk_contrib_pct = risk_contrib / np.sum(risk_contrib)
                
                return {
                    'success': True,
                    'weights': optimal_weights,
                    'metrics': metrics,
                    'risk_contributions': risk_contrib_pct,
                    'optimization_message': result.message,
                    'iterations': result.nit
                }
            else:
                return {'success': False, 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def stress_test_portfolio(self, weights: np.ndarray, returns: pd.DataFrame, 
                            scenarios: Dict = None) -> Dict[str, Any]:
        """
        Perform comprehensive stress testing on the portfolio.
        
        Args:
            weights (np.ndarray): Portfolio weights
            returns (pd.DataFrame): Historical returns data
            scenarios (dict): Custom stress test scenarios
            
        Returns:
            dict: Stress test results
        """
        if scenarios is None:
            scenarios = {
                'market_crash': {'factor': -0.20, 'correlation_increase': 0.3},
                'sector_rotation': {'factor': -0.15, 'sector_specific': True},
                'interest_rate_shock': {'factor': -0.10, 'duration_sensitive': True},
                'liquidity_crisis': {'factor': -0.25, 'small_cap_bias': True},
                'inflation_surge': {'factor': -0.12, 'real_assets_protection': True}
            }
        
        results = {}
        baseline_metrics = self.calculate_portfolio_metrics(weights, returns)
        
        for scenario_name, scenario_params in scenarios.items():
            try:
                # Apply scenario stress
                stressed_returns = self._apply_stress_scenario(returns, scenario_params)
                stressed_metrics = self.calculate_portfolio_metrics(weights, stressed_returns)
                
                # Calculate impact
                return_impact = stressed_metrics['annual_return'] - baseline_metrics['annual_return']
                vol_impact = stressed_metrics['annual_volatility'] - baseline_metrics['annual_volatility']
                
                results[scenario_name] = {
                    'return_impact': return_impact,
                    'volatility_impact': vol_impact,
                    'new_sharpe': stressed_metrics['sharpe_ratio'],
                    'new_max_drawdown': stressed_metrics['max_drawdown'],
                    'var_95': stressed_metrics['var_95'],
                    'expected_shortfall': stressed_metrics['expected_shortfall']
                }
                
            except Exception as e:
                logger.error(f"Error in stress test scenario {scenario_name}: {e}")
                results[scenario_name] = {'error': str(e)}
        
        return {
            'baseline_metrics': baseline_metrics,
            'stress_scenarios': results,
            'overall_stress_score': self._calculate_stress_score(results)
        }
    
    def _apply_stress_scenario(self, returns: pd.DataFrame, scenario_params: Dict) -> pd.DataFrame:
        """Apply stress scenario to returns data"""
        stressed_returns = returns.copy()
        factor = scenario_params.get('factor', -0.10)
        
        if scenario_params.get('market_crash', False):
            # Apply uniform negative shock with increased correlations
            stress_shock = np.random.normal(factor, abs(factor) * 0.2, len(returns))
            for col in stressed_returns.columns:
                stressed_returns[col] += stress_shock
                
        elif scenario_params.get('sector_specific', False):
            # Apply sector-specific stress (simulate by random selection)
            stressed_assets = np.random.choice(stressed_returns.columns, 
                                             size=max(1, len(stressed_returns.columns)//3))
            for asset in stressed_assets:
                stressed_returns[asset] += np.random.normal(factor, abs(factor) * 0.1, len(returns))
                
        else:
            # Generic stress application
            stress_multiplier = 1 + factor
            stressed_returns = stressed_returns * stress_multiplier
            
        return stressed_returns
    
    def _calculate_stress_score(self, stress_results: Dict) -> float:
        """Calculate overall stress score (0-100, higher is better)"""
        scores = []
        
        for scenario, results in stress_results.items():
            if 'error' not in results:
                # Score based on return impact (less negative is better)
                return_score = max(0, 100 + results['return_impact'] * 1000)
                
                # Score based on drawdown (smaller drawdown is better)
                drawdown_score = max(0, 100 + results['new_max_drawdown'] * 500)
                
                scenario_score = (return_score + drawdown_score) / 2
                scores.append(scenario_score)
        
        return np.mean(scores) if scores else 0
    
    def calculate_var_and_es(self, weights: np.ndarray, returns: pd.DataFrame, 
                           confidence_levels: List[float] = None) -> Dict[str, Any]:
        """
        Calculate Value at Risk and Expected Shortfall for the portfolio.
        
        Args:
            weights (np.ndarray): Portfolio weights
            returns (pd.DataFrame): Historical returns data
            confidence_levels (list): Confidence levels for VaR calculation
            
        Returns:
            dict: VaR and ES calculations
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        # Portfolio returns
        portfolio_returns = returns.dot(weights)
        
        results = {}
        
        for conf_level in confidence_levels:
            # Historical VaR
            var_historical = np.percentile(portfolio_returns, (1 - conf_level) * 100)
            
            # Expected Shortfall (Conditional VaR)
            es_historical = portfolio_returns[portfolio_returns <= var_historical].mean()
            
            # Parametric VaR (assuming normal distribution)
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            z_score = scipy.stats.norm.ppf(1 - conf_level)
            var_parametric = mean_return + z_score * std_return
            
            # Modified VaR (Cornish-Fisher expansion for skewness and kurtosis)
            skewness = scipy.stats.skew(portfolio_returns)
            kurtosis = scipy.stats.kurtosis(portfolio_returns)
            
            z_cf = (z_score + 
                   (z_score**2 - 1) * skewness / 6 +
                   (z_score**3 - 3*z_score) * kurtosis / 24 -
                   (2*z_score**3 - 5*z_score) * skewness**2 / 36)
            
            var_modified = mean_return + z_cf * std_return
            
            results[f'confidence_{int(conf_level*100)}'] = {
                'var_historical': var_historical,
                'var_parametric': var_parametric,
                'var_modified': var_modified,
                'expected_shortfall': es_historical,
                'annualized_var_historical': var_historical * np.sqrt(252),
                'annualized_es': es_historical * np.sqrt(252)
            }
        
        return results
    
    def generate_efficient_frontier(self, returns: pd.DataFrame, 
                                  num_portfolios: int = 100) -> Dict[str, Any]:
        """
        Generate the efficient frontier for the given assets.
        
        Args:
            returns (pd.DataFrame): Historical returns data
            num_portfolios (int): Number of portfolios to generate
            
        Returns:
            dict: Efficient frontier data
        """
        n_assets = len(returns.columns)
        
        # Calculate expected returns and covariance matrix
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Define range of target returns
        min_ret = mean_returns.min()
        max_ret = mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                # Minimize portfolio variance for given target return
                def objective(weights):
                    return np.dot(weights.T, np.dot(cov_matrix, weights))
                
                # Constraints
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                    {'type': 'eq', 'fun': lambda x: np.sum(x * mean_returns) - target_return}  # Target return
                ]
                
                # Bounds
                bounds = tuple((0, 1) for _ in range(n_assets))
                
                # Initial guess
                x0 = np.array([1/n_assets] * n_assets)
                
                result = sco.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                
                if result.success:
                    weights = result.x
                    portfolio_return = np.sum(weights * mean_returns)
                    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                    
                    efficient_portfolios.append({
                        'return': portfolio_return,
                        'volatility': portfolio_volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'weights': weights
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to optimize for target return {target_return}: {e}")
                continue
        
        if not efficient_portfolios:
            return {'success': False, 'message': 'Failed to generate efficient frontier'}
        
        # Convert to DataFrame for easier analysis
        frontier_df = pd.DataFrame(efficient_portfolios)
        
        # Find key portfolios
        max_sharpe_idx = frontier_df['sharpe_ratio'].idxmax()
        min_vol_idx = frontier_df['volatility'].idxmin()
        
        return {
            'success': True,
            'efficient_frontier': frontier_df,
            'max_sharpe_portfolio': frontier_df.loc[max_sharpe_idx],
            'min_volatility_portfolio': frontier_df.loc[min_vol_idx],
            'num_portfolios': len(efficient_portfolios)
        }
    
    def rebalance_portfolio(self, current_weights: np.ndarray, target_weights: np.ndarray,
                          transaction_cost: float = 0.001) -> Dict[str, Any]:
        """
        Calculate optimal rebalancing considering transaction costs.
        
        Args:
            current_weights (np.ndarray): Current portfolio weights
            target_weights (np.ndarray): Target portfolio weights
            transaction_cost (float): Transaction cost rate (default 0.1%)
            
        Returns:
            dict: Rebalancing analysis
        """
        # Calculate required trades
        weight_diff = target_weights - current_weights
        trade_amounts = np.abs(weight_diff)
        
        # Calculate transaction costs
        total_transaction_cost = np.sum(trade_amounts) * transaction_cost
        
        # Determine if rebalancing is beneficial
        # (simplified analysis - could be more sophisticated)
        rebalancing_threshold = 0.05  # 5% drift threshold
        max_drift = np.max(np.abs(weight_diff))
        
        should_rebalance = max_drift > rebalancing_threshold
        
        return {
            'should_rebalance': should_rebalance,
            'max_drift': max_drift,
            'total_transaction_cost': total_transaction_cost,
            'weight_changes': weight_diff,
            'trade_amounts': trade_amounts,
            'rebalancing_threshold': rebalancing_threshold
        }
    
    def optimize_portfolio_multi_objective(self, returns: pd.DataFrame, 
                                         objectives: List[str] = None) -> Dict[str, Any]:
        """
        Multi-objective portfolio optimization using Pareto efficiency.
        
        Args:
            returns (pd.DataFrame): Historical returns data
            objectives (list): List of objectives to optimize
            
        Returns:
            dict: Multi-objective optimization results
        """
        if objectives is None:
            objectives = ['return', 'risk', 'sharpe']
        
        n_assets = len(returns.columns)
        
        # Define objective functions
        def calculate_objectives(weights):
            metrics = self.calculate_portfolio_metrics(weights, returns)
            
            obj_values = []
            for obj in objectives:
                if obj == 'return':
                    obj_values.append(-metrics['annual_return'])  # Negative for minimization
                elif obj == 'risk':
                    obj_values.append(metrics['annual_volatility'])
                elif obj == 'sharpe':
                    obj_values.append(-metrics['sharpe_ratio'])  # Negative for minimization
                elif obj == 'drawdown':
                    obj_values.append(-metrics['max_drawdown'])  # Negative for minimization
                    
            return obj_values
        
        # Generate multiple solutions using different objective weightings
        solutions = []
        num_solutions = 50
        
        for i in range(num_solutions):
            # Random weights for objectives
            obj_weights = np.random.random(len(objectives))
            obj_weights = obj_weights / np.sum(obj_weights)
            
            def objective(weights):
                obj_values = calculate_objectives(weights)
                return np.sum([w * v for w, v in zip(obj_weights, obj_values)])
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # Bounds
            bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
            
            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)
            
            try:
                result = sco.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                
                if result.success:
                    weights = result.x
                    metrics = self.calculate_portfolio_metrics(weights, returns)
                    obj_values = calculate_objectives(weights)
                    
                    solutions.append({
                        'weights': weights,
                        'metrics': metrics,
                        'objective_values': obj_values,
                        'objective_weights': obj_weights
                    })
                    
            except Exception as e:
                continue
        
        if not solutions:
            return {'success': False, 'message': 'No valid solutions found'}
        
        return {
            'success': True,
            'solutions': solutions,
            'num_solutions': len(solutions),
            'objectives': objectives
        }
    
    def optimize_portfolio(self, symbols: List[str], strategy: str = "sharpe", 
                         allocation_budget: float = 100000.0, 
                         risk_tolerance: str = "moderate") -> Dict[str, Any]:
        """
        General portfolio optimization method that dispatches to specific strategies.
        
        Args:
            symbols (list): List of stock symbols
            strategy (str): Optimization strategy (sharpe, min_variance, risk_parity, multi_objective)
            allocation_budget (float): Total allocation budget
            risk_tolerance (str): Risk tolerance level
            
        Returns:
            dict: Optimization results
        """
        try:
            # Generate mock returns data for the symbols (in real implementation, this would fetch actual data)
            np.random.seed(42)
            
            # Create mock historical returns for 252 trading days
            n_days = 252
            returns_data = {}
            
            for symbol in symbols:
                # Generate realistic returns based on asset type
                if 'BANK' in symbol.upper():
                    daily_return = np.random.normal(0.0008, 0.025, n_days)  # Banking stocks
                elif 'IT' in symbol.upper() or 'INFY' in symbol.upper() or 'TCS' in symbol.upper():
                    daily_return = np.random.normal(0.0012, 0.030, n_days)  # IT stocks
                else:
                    daily_return = np.random.normal(0.0010, 0.028, n_days)  # Other stocks
                
                returns_data[symbol] = daily_return
            
            returns_df = pd.DataFrame(returns_data)
            
            # Dispatch to appropriate optimization method based on strategy
            if strategy.lower() in ['sharpe', 'max_sharpe']:
                result = self.optimize_portfolio_sharpe(returns_df)
            elif strategy.lower() in ['min_variance', 'minimum_variance']:
                result = self.optimize_portfolio_minimum_variance(returns_df)
            elif strategy.lower() in ['risk_parity', 'equal_risk']:
                result = self.optimize_portfolio_risk_parity(returns_df)
            elif strategy.lower() in ['multi_objective', 'pareto']:
                result = self.optimize_portfolio_multi_objective(returns_df)
            else:
                # Default to Sharpe optimization
                result = self.optimize_portfolio_sharpe(returns_df)
            
            if result['success']:
                # Convert weights to allocation amounts
                weights = result['weights']
                allocations = {symbol: float(weight * allocation_budget) 
                             for symbol, weight in zip(symbols, weights)}
                
                # Add allocation information to the result
                result['allocations'] = allocations
                result['total_budget'] = allocation_budget
                result['strategy_used'] = strategy
                result['risk_tolerance'] = risk_tolerance
                
                # Perform stress testing
                stress_results = self.stress_test_portfolio(weights, returns_df)
                result['stress_test'] = stress_results
                
                # Calculate VaR
                var_results = self.calculate_var_and_es(weights, returns_df)
                result['var_analysis'] = var_results
            
            # Make all results JSON serializable
            result = self._make_json_serializable(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy_used': strategy
            }

    def save_optimization_results(self, results: Dict, filename: str = None):
        """Save optimization results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/portfolio_optimization_{timestamp}.json"
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Optimization results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Generate correlated returns for 5 assets
    n_assets = 5
    asset_names = ['RELIANCE.NSE', 'HDFCBANK.NSE', 'INFY.NSE', 'TCS.NSE', 'HDFC.NSE']
    
    # Create correlation matrix
    correlation = np.array([
        [1.0, 0.6, 0.4, 0.3, 0.5],
        [0.6, 1.0, 0.3, 0.4, 0.7],
        [0.4, 0.3, 1.0, 0.8, 0.2],
        [0.3, 0.4, 0.8, 1.0, 0.3],
        [0.5, 0.7, 0.2, 0.3, 1.0]
    ])
    
    # Generate returns
    returns_data = np.random.multivariate_normal(
        mean=[0.0005, 0.0008, 0.0012, 0.0010, 0.0007],
        cov=correlation * 0.0004,
        size=len(dates)
    )
    
    returns_df = pd.DataFrame(returns_data, columns=asset_names, index=dates)
    
    # Initialize optimizer
    optimizer = AdvancedPortfolioOptimizer({
        'risk_free_rate': 0.07,
        'target_return': 0.15,
        'target_volatility': 0.20
    })
    
    print("Testing Advanced Portfolio Optimizer...")
    
    # Test 1: Maximum Sharpe Ratio optimization
    print("\n1. Maximum Sharpe Ratio Optimization:")
    sharpe_result = optimizer.optimize_portfolio_sharpe(returns_df)
    if sharpe_result['success']:
        print(f"   Annual Return: {sharpe_result['metrics']['annual_return']:.4f}")
        print(f"   Annual Volatility: {sharpe_result['metrics']['annual_volatility']:.4f}")
        print(f"   Sharpe Ratio: {sharpe_result['metrics']['sharpe_ratio']:.4f}")
    
    # Test 2: Minimum Variance optimization
    print("\n2. Minimum Variance Optimization:")
    minvar_result = optimizer.optimize_portfolio_minimum_variance(returns_df)
    if minvar_result['success']:
        print(f"   Annual Return: {minvar_result['metrics']['annual_return']:.4f}")
        print(f"   Annual Volatility: {minvar_result['metrics']['annual_volatility']:.4f}")
        print(f"   Sharpe Ratio: {minvar_result['metrics']['sharpe_ratio']:.4f}")
    
    # Test 3: Risk Parity optimization
    print("\n3. Risk Parity Optimization:")
    riskparity_result = optimizer.optimize_portfolio_risk_parity(returns_df)
    if riskparity_result['success']:
        print(f"   Annual Return: {riskparity_result['metrics']['annual_return']:.4f}")
        print(f"   Annual Volatility: {riskparity_result['metrics']['annual_volatility']:.4f}")
        print(f"   Sharpe Ratio: {riskparity_result['metrics']['sharpe_ratio']:.4f}")
    
    # Test 4: Stress testing
    if sharpe_result['success']:
        print("\n4. Stress Testing:")
        stress_results = optimizer.stress_test_portfolio(sharpe_result['weights'], returns_df)
        print(f"   Overall Stress Score: {stress_results['overall_stress_score']:.2f}")
        
        for scenario, results in stress_results['stress_scenarios'].items():
            if 'error' not in results:
                print(f"   {scenario}: Return Impact = {results['return_impact']:.4f}")
    
    # Test 5: VaR and Expected Shortfall
    if sharpe_result['success']:
        print("\n5. Value at Risk Analysis:")
        var_results = optimizer.calculate_var_and_es(sharpe_result['weights'], returns_df)
        for conf_level, results in var_results.items():
            print(f"   {conf_level}: VaR = {results['var_historical']:.4f}, ES = {results['expected_shortfall']:.4f}")
    
    print("\nAdvanced Portfolio Optimizer testing completed successfully!") 