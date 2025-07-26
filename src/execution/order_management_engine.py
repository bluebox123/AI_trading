"""
Order Management Engine - Minimal Implementation

This module provides basic order management functionality for the trading system.
Minimal implementation to support Step 5 operations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

logger = logging.getLogger(__name__)

class OrderManagementEngine:
    """
    Basic Order Management Engine for trade execution and monitoring.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Order Management Engine.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.orders = {}
        self.order_history = []
        
        logger.info("Order Management Engine initialized")
    
    def place_order(self, symbol: str, order_type: str, side: str, quantity: int, price: float = None) -> Dict[str, Any]:
        """
        Place a new trading order.
        
        Args:
            symbol (str): Stock symbol
            order_type (str): Order type (MARKET, LIMIT, STOP_LOSS)
            side (str): Order side (BUY, SELL)
            quantity (int): Number of shares
            price (float, optional): Price for LIMIT/STOP_LOSS orders
            
        Returns:
            dict: Order result with order ID and status
        """
        try:
            # Generate unique order ID
            order_id = str(uuid.uuid4())
            
            # Get current market data for validation
            current_price = self._get_current_price(symbol)
            
            # Validate order
            validation_result = self._validate_order(symbol, order_type, side, quantity, price, current_price)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'order_id': None
                }
            
            # Calculate order value
            if order_type == "MARKET":
                estimated_price = current_price
            else:
                estimated_price = price
            
            order_value = estimated_price * quantity
            
            # Create order object
            order = Order(
                order_id=order_id,
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                status="PENDING",
                timestamp=datetime.now(),
                estimated_value=order_value
            )
            
            # Add to orders tracking
            self.orders[order_id] = order
            
            # Execute order based on type
            if order_type == "MARKET":
                execution_result = self._execute_market_order(order, current_price)
            elif order_type == "LIMIT":
                execution_result = self._execute_limit_order(order)
            elif order_type == "STOP_LOSS":
                execution_result = self._execute_stop_loss_order(order)
            else:
                return {
                    'success': False,
                    'error': f"Unsupported order type: {order_type}",
                    'order_id': order_id
                }
            
            # Log the order
            logger.info(f"Order placed: {order_id} - {side} {quantity} {symbol}")
            
            return {
                'success': True,
                'order_id': order_id,
                'status': order.status,
                'estimated_value': order_value,
                'execution_price': execution_result.get('execution_price'),
                'timestamp': order.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'order_id': None
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order"""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELLED'
            logger.info(f"Order cancelled: {order_id}")
            return self.orders[order_id]
        else:
            return {'error': 'Order not found'}
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of an order"""
        return self.orders.get(order_id, {'error': 'Order not found'})
    
    def get_all_orders(self) -> List[Dict[str, Any]]:
        """Get all orders"""
        return list(self.orders.values())
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """Get order history"""
        return self.order_history
    
    def create_order(self, symbol: str, order_type: str, side: str, quantity: int, price: float = None) -> Dict[str, Any]:
        """
        Create a new order (alias for place_order to match API expectations).
        
        Args:
            symbol (str): Stock symbol
            order_type (str): Order type (MARKET, LIMIT, STOP_LOSS)
            side (str): Order side (BUY, SELL)
            quantity (int): Number of shares
            price (float, optional): Price for LIMIT/STOP_LOSS orders
            
        Returns:
            dict: Order result
        """
        return self.place_order(symbol, order_type, side, quantity, price)
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol (mock implementation)"""
        import random
        # Mock price based on symbol hash for consistency
        random.seed(hash(symbol) % 1000)
        if 'RELIANCE' in symbol:
            return round(random.uniform(2800, 3200), 2)
        elif 'TCS' in symbol:
            return round(random.uniform(3400, 3600), 2)
        elif 'HDFC' in symbol:
            return round(random.uniform(1900, 2100), 2)
        else:
            return round(random.uniform(100, 500), 2)
    
    def _validate_order(self, symbol: str, order_type: str, side: str, quantity: int, 
                       price: float, current_price: float) -> Dict[str, Any]:
        """Validate order parameters"""
        if quantity <= 0:
            return {'valid': False, 'error': 'Quantity must be positive'}
        
        if order_type not in ['MARKET', 'LIMIT', 'STOP_LOSS']:
            return {'valid': False, 'error': f'Invalid order type: {order_type}'}
        
        if side not in ['BUY', 'SELL']:
            return {'valid': False, 'error': f'Invalid side: {side}'}
        
        if order_type in ['LIMIT', 'STOP_LOSS'] and price is None:
            return {'valid': False, 'error': f'{order_type} orders require a price'}
        
        return {'valid': True, 'error': None}
    
    def _execute_market_order(self, order: 'Order', current_price: float) -> Dict[str, Any]:
        """Execute a market order"""
        order.status = "FILLED"
        order.filled_quantity = order.quantity
        order.remaining_quantity = 0
        order.execution_price = current_price
        
        return {
            'status': 'FILLED',
            'execution_price': current_price,
            'filled_quantity': order.quantity
        }
    
    def _execute_limit_order(self, order: 'Order') -> Dict[str, Any]:
        """Execute a limit order (simplified - immediately filled for demo)"""
        order.status = "FILLED"
        order.filled_quantity = order.quantity
        order.remaining_quantity = 0
        order.execution_price = order.price
        
        return {
            'status': 'FILLED',
            'execution_price': order.price,
            'filled_quantity': order.quantity
        }
    
    def _execute_stop_loss_order(self, order: 'Order') -> Dict[str, Any]:
        """Execute a stop loss order (simplified)"""
        order.status = "PENDING"  # Stop loss orders wait for trigger
        
        return {
            'status': 'PENDING',
            'execution_price': None,
            'filled_quantity': 0
        }


class Order:
    """Simple order class for tracking order details"""
    def __init__(self, order_id: str, symbol: str, order_type: str, side: str, 
                 quantity: int, price: float, status: str, timestamp, estimated_value: float):
        self.order_id = order_id
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.status = status
        self.timestamp = timestamp
        self.estimated_value = estimated_value
        self.filled_quantity = 0
        self.remaining_quantity = quantity
        self.execution_price = None 