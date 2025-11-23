import pandas as pd
import sys

# Read Excel file
df_dict = pd.read_excel('stat/backtest_report_20251123_023022.xlsx', sheet_name=None)

print('=== AVAILABLE SHEETS ===')
print(list(df_dict.keys()))
print('\n')

for sheet_name, data in df_dict.items():
    print(f'\n{"="*60}')
    print(f'SHEET: {sheet_name}')
    print(f'{"="*60}')
    print(data.to_string())
    print('\n')
