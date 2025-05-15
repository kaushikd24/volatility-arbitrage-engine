class PositionSizer:
    def __init__(self, capital: float, risk_per_trade: float = 0.01, max_position_pct: float = 0.05, 
                 absolute_max_contracts: int = 100):
        """
        :param capital: total capital available
        :param risk_per_trade: fraction of capital to risk per trade
        :param max_position_pct: maximum position size as percentage of capital
        :param absolute_max_contracts: absolute maximum number of contracts per trade regardless of price
        """
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.absolute_max_contracts = absolute_max_contracts

    def size_position(self, entry_price: float, confidence: float = None) -> int:
        """
        Determine how many contracts to trade
        """
        # Minimum price threshold to prevent excessive sizing on extremely low-priced options
        # For options priced below this threshold, we'll use this minimum for sizing calculations
        min_pricing_threshold = 0.1
        
        if entry_price <= 0:
            return 0
        
        # Use a minimum price for calculations to avoid excessive position sizes
        calculation_price = max(entry_price, min_pricing_threshold)
        
        # Calculate position size based on risk
        max_risk_amount = self.capital * self.risk_per_trade
        risk_based_quantity = max_risk_amount / calculation_price
        
        # Apply max position size cap
        max_position_size = self.capital * self.max_position_pct
        max_quantity = max_position_size / calculation_price
        
        # Use the smaller of the two quantities
        quantity = min(risk_based_quantity, max_quantity)
        
        # Adjust by confidence if provided
        if confidence is not None and 0 <= confidence <= 1:
            # Scale 0.5-1.0 based on confidence
            confidence_factor = 0.5 + confidence/2
            quantity = quantity * confidence_factor
        
        # Apply absolute maximum contract limit
        quantity = min(int(quantity), self.absolute_max_contracts)
        
        # Always return at least 1 contract, unless price is too high
        quantity = max(quantity, 1)
        
        # Final sanity check - ensure total cost doesn't exceed capital
        total_cost = quantity * entry_price
        if total_cost > self.capital:
            quantity = max(int(self.capital / entry_price), 0)
        
        return quantity
