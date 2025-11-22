"""
Comparative Analysis: September vs October 2025
Compare model performance across different months
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_comparison_report():
    """Create a detailed comparison report"""
    
    print("="*80)
    print("COMPARATIVE ANALYSIS: SEPTEMBER vs OCTOBER 2025")
    print("Model trained on: August 2025 hourly data")
    print("="*80)
    
    # Data from our runs
    results = {
        'Month': ['September', 'October'],
        'Candles': [720, 744],
        'Patterns_Detected': [401, 407],
        'Total_Trades': [262, 232],
        'Initial_Capital': [10000, 10000],
        'Final_Capital': [18203.68, 1971.01],
        'Return_Pct': [82.04, -80.29],
        'Win_Rate_Pct': [42.75, 25.00],
        'Profit_Factor': [1.08, 0.56],
        'Max_Drawdown_Pct': [78.81, 82.87],
        'Sharpe_Ratio': [0.61, -0.76],
        'Avg_Win': [943.52, 177.56],
        'Avg_Loss': [649.80, 105.33],
        'Winning_Trades': [112, 58],
        'Losing_Trades': [150, 174]
    }
    
    df = pd.DataFrame(results)
    
    # Print summary table
    print("\n📊 PERFORMANCE SUMMARY")
    print("-" * 80)
    print(f"{'Metric':<30} {'September':<25} {'October':<25}")
    print("-" * 80)
    print(f"{'Total Candles Analyzed':<30} {720:<25} {744:<25}")
    print(f"{'Patterns Detected':<30} {401:<25} {407:<25}")
    print(f"{'Total Trades':<30} {262:<25} {232:<25}")
    print()
    print(f"{'Initial Capital':<30} {'$10,000':<25} {'$10,000':<25}")
    print(f"{'Final Capital':<30} {'$18,203.68':<25} {'$1,971.01':<25}")
    print(f"{'Total P&L':<30} {'+$8,203.68':<25} {'-$8,028.99':<25}")
    print(f"{'Return %':<30} {'+82.04%':<25} {'-80.29%':<25}")
    print()
    print(f"{'Win Rate':<30} {'42.75%':<25} {'25.00%':<25}")
    print(f"{'Profit Factor':<30} {'1.08':<25} {'0.56':<25}")
    print(f"{'Sharpe Ratio':<30} {'0.61':<25} {'-0.76':<25}")
    print(f"{'Max Drawdown':<30} {'78.81%':<25} {'82.87%':<25}")
    print()
    print(f"{'Average Win':<30} {'$943.52':<25} {'$177.56':<25}")
    print(f"{'Average Loss':<30} {'$649.80':<25} {'$105.33':<25}")
    print("-" * 80)
    
    # Pattern performance comparison
    print("\n\n📈 PATTERN-SPECIFIC PERFORMANCE")
    print("-" * 80)
    
    # September patterns
    sept_patterns = {
        'ascending_triangle': {'win_rate': 48.9, 'trades': 133, 'pnl': 19635.76},
        'cup_and_handle': {'win_rate': 51.5, 'trades': 33, 'pnl': -1019.70},
        'descending_triangle': {'win_rate': 31.2, 'trades': 96, 'pnl': -10412.38}
    }
    
    # October patterns
    oct_patterns = {
        'ascending_triangle': {'win_rate': 19.1, 'trades': 115, 'pnl': -7003.73},
        'cup_and_handle': {'win_rate': 0.0, 'trades': 14, 'pnl': -2694.69},
        'descending_triangle': {'win_rate': 35.0, 'trades': 103, 'pnl': 1669.43}
    }
    
    print("\nSEPTEMBER Pattern Performance:")
    print(f"{'Pattern':<25} {'Win Rate':<15} {'Trades':<10} {'P&L':<15}")
    print("-" * 65)
    for pattern, stats in sept_patterns.items():
        pnl_str = f"${stats['pnl']:,.2f}" if stats['pnl'] >= 0 else f"-${abs(stats['pnl']):,.2f}"
        print(f"{pattern:<25} {stats['win_rate']:.1f}%{'':<10} {stats['trades']:<10} {pnl_str:<15}")
    
    print("\n\nOCTOBER Pattern Performance:")
    print(f"{'Pattern':<25} {'Win Rate':<15} {'Trades':<10} {'P&L':<15}")
    print("-" * 65)
    for pattern, stats in oct_patterns.items():
        pnl_str = f"${stats['pnl']:,.2f}" if stats['pnl'] >= 0 else f"-${abs(stats['pnl']):,.2f}"
        print(f"{pattern:<25} {stats['win_rate']:.1f}%{'':<10} {stats['trades']:<10} {pnl_str:<15}")
    
    # Key insights
    print("\n\n🔍 KEY INSIGHTS")
    print("-" * 80)
    print("\n✅ SEPTEMBER 2025 (Profitable):")
    print("   • Strong performance with +82% return")
    print("   • Ascending Triangle was the star performer (+$19,636)")
    print("   • Win rate of 42.75% with positive profit factor (1.08)")
    print("   • Model successfully identified profitable patterns")
    
    print("\n❌ OCTOBER 2025 (Loss):")
    print("   • Significant loss of -80% return")
    print("   • Only Descending Triangle was profitable (+$1,669)")
    print("   • Win rate dropped to 25% with poor profit factor (0.56)")
    print("   • Ascending Triangle and Cup & Handle both failed")
    print("   • Market conditions likely changed significantly")
    
    print("\n⚠️  OBSERVATIONS:")
    print("   • Same patterns performed differently in different months")
    print("   • October market behaved differently than August training data")
    print("   • Pattern recognition was consistent (407 vs 401 patterns)")
    print("   • Execution and market conditions were the differentiators")
    
    print("\n💡 RECOMMENDATIONS:")
    print("   • Consider adaptive position sizing based on recent performance")
    print("   • Implement market regime detection (trending vs ranging)")
    print("   • Add filters for market volatility and trend strength")
    print("   • Consider retraining model monthly or using rolling window")
    print("   • Implement stop-loss tightening in adverse conditions")
    
    print("\n📊 STATISTICAL COMPARISON:")
    print("-" * 80)
    return_diff = 82.04 - (-80.29)
    wr_diff = 42.75 - 25.00
    pf_diff = 1.08 - 0.56
    
    print(f"   • Return Difference: {return_diff:.2f} percentage points")
    print(f"   • Win Rate Difference: {wr_diff:.2f} percentage points")
    print(f"   • Profit Factor Difference: {pf_diff:.2f}")
    print(f"   • Trade Count Difference: {262 - 232} trades")
    
    print("\n" + "="*80)
    print("CONCLUSION: Model performance is highly dependent on market conditions.")
    print("The August-trained model worked well in September but failed in October.")
    print("Consider implementing adaptive strategies and risk management improvements.")
    print("="*80)
    
    # Create visualization
    create_comparison_charts()


def create_comparison_charts():
    """Create comparison visualizations"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('September vs October 2025 - Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. Return Comparison
    ax1 = axes[0, 0]
    months = ['September', 'October']
    returns = [82.04, -80.29]
    colors = ['green', 'red']
    ax1.bar(months, returns, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Total Return Comparison')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Win Rate Comparison
    ax2 = axes[0, 1]
    win_rates = [42.75, 25.00]
    ax2.bar(months, win_rates, color=['skyblue', 'orange'], alpha=0.7)
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Profit Factor Comparison
    ax3 = axes[0, 2]
    profit_factors = [1.08, 0.56]
    ax3.bar(months, profit_factors, color=['lightgreen', 'lightcoral'], alpha=0.7)
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Break-even')
    ax3.set_ylabel('Profit Factor')
    ax3.set_title('Profit Factor Comparison')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Trade Distribution
    ax4 = axes[1, 0]
    categories = ['Winning\nTrades', 'Losing\nTrades']
    sept_trades = [112, 150]
    oct_trades = [58, 174]
    x = np.arange(len(categories))
    width = 0.35
    ax4.bar(x - width/2, sept_trades, width, label='September', alpha=0.7)
    ax4.bar(x + width/2, oct_trades, width, label='October', alpha=0.7)
    ax4.set_ylabel('Number of Trades')
    ax4.set_title('Trade Distribution')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Pattern Performance (September)
    ax5 = axes[1, 1]
    sept_pattern_names = ['Asc\nTriangle', 'Cup &\nHandle', 'Desc\nTriangle']
    sept_pattern_pnl = [19635.76, -1019.70, -10412.38]
    colors_sept = ['green' if x > 0 else 'red' for x in sept_pattern_pnl]
    ax5.bar(sept_pattern_names, sept_pattern_pnl, color=colors_sept, alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax5.set_ylabel('P&L ($)')
    ax5.set_title('September - Pattern P&L')
    ax5.grid(axis='y', alpha=0.3)
    ax5.tick_params(axis='x', rotation=0)
    
    # 6. Pattern Performance (October)
    ax6 = axes[1, 2]
    oct_pattern_names = ['Asc\nTriangle', 'Cup &\nHandle', 'Desc\nTriangle']
    oct_pattern_pnl = [-7003.73, -2694.69, 1669.43]
    colors_oct = ['green' if x > 0 else 'red' for x in oct_pattern_pnl]
    ax6.bar(oct_pattern_names, oct_pattern_pnl, color=colors_oct, alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax6.set_ylabel('P&L ($)')
    ax6.set_title('October - Pattern P&L')
    ax6.grid(axis='y', alpha=0.3)
    ax6.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    save_path = 'september_vs_october_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n\n📊 Comparison chart saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    import numpy as np
    create_comparison_report()
