import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

def calculate_cagr(initial_capital, final_capital, start_date, end_date):
    """
    Calculate the Compound Annual Growth Rate (CAGR).
    
    Parameters:
    -----------
    initial_capital : float
        Initial investment amount
    final_capital : float
        Final investment value
    start_date : datetime
        Start date of the investment period
    end_date : datetime
        End date of the investment period
        
    Returns:
    --------
    float
        CAGR as decimal (e.g., 0.12 for 12%)
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Calculate years
    days = (end_date - start_date).days
    years = days / 365.25
    
    # Calculate CAGR
    if years > 0 and initial_capital > 0 and final_capital > 0:
        cagr = (final_capital / initial_capital) ** (1 / years) - 1
        return cagr
    else:
        return 0.0

def main():
    """Run CAGR tests"""
    # Test with different scenarios
    test_cases = [
        {
            'name': '1 year growth',
            'initial': 100000,
            'final': 110000,
            'start': '2022-01-01',
            'end': '2023-01-01',
            'expected': 0.10  # 10% annual growth
        },
        {
            'name': '3 year growth',
            'initial': 100000,
            'final': 133100,
            'start': '2020-01-01',
            'end': '2023-01-01',
            'expected': 0.10  # Compounded at 10% for 3 years
        },
        {
            'name': 'SPY-like 2023 performance',
            'initial': 100000,
            'final': 124500,  # Approximately 24.5% growth
            'start': '2023-01-01',
            'end': '2023-12-31',
            'expected': 0.245  # 24.5% annual growth
        }
    ]
    
    print("=== CAGR Calculation Tests ===")
    for tc in test_cases:
        cagr = calculate_cagr(tc['initial'], tc['final'], tc['start'], tc['end'])
        print(f"{tc['name']}:")
        print(f"  Initial: ${tc['initial']:,.2f}, Final: ${tc['final']:,.2f}")
        print(f"  Period: {tc['start']} to {tc['end']}")
        print(f"  CAGR: {cagr:.2%} (Expected: {tc['expected']:.2%})")
        print(f"  Accuracy: {abs(cagr - tc['expected']) < 0.001}")
        print()
    
    # Generate a visualization of CAGR for different time periods
    initial = 100000
    annual_growth_rates = [0.05, 0.10, 0.15, 0.20, 0.25]  # 5% to 25%
    years = list(range(1, 11))  # 1 to 10 years
    
    plt.figure(figsize=(10, 6))
    
    for rate in annual_growth_rates:
        final_values = [initial * (1 + rate) ** year for year in years]
        plt.plot(years, final_values, marker='o', label=f"{rate:.0%} Annual Growth")
    
    plt.title('Growth of $100,000 at Different Annual Rates')
    plt.xlabel('Years')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cagr_visualization.png')
    print("CAGR visualization saved to cagr_visualization.png")

if __name__ == "__main__":
    main() 