class DrawdownLimiter:
    def __init__(self, starting_equity: float, max_drawdown_pct: float = 0.2):
        self.starting_equity = starting_equity
        self.max_drawdown_pct = max_drawdown_pct
        self.equity_curve = [starting_equity]

    def update_equity(self, new_value: float) -> bool:
        self.equity_curve.append(new_value)
        peak = max(self.equity_curve)
        drawdown = (peak - new_value) / peak

        if drawdown > self.max_drawdown_pct:
            print(f"⚠ Max drawdown exceeded: {drawdown:.2%}")
            return False  # Trading should stop
        return True  # Continue trading 