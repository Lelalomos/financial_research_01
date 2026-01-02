"""
Backtester for CRNN Financial Prediction Model.

This module provides:
- Trading strategy simulation
- Portfolio return calculation
- Risk metrics calculation
- Report generation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path

from config.model_config import ModelConfig
from src.utils.logger import EvaluationLogger
from .metrics import (
    evaluate_model,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    print_metrics
)


class Backtester:
    """
    Backtester for trading strategies.

    Simulates trading based on model predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: str = 'cuda'
    ):
        """
        Initialize backtester.

        Args:
            model: PyTorch model
            config: ModelConfig instance
            device: Device to use
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = EvaluationLogger(log_dir="logs")

    def run_backtest(
        self,
        test_loader: DataLoader,
        prediction_threshold: float = 0.0,
        initial_capital: float = 100000.0,
        commission: float = 0.001
    ) -> Dict[str, any]:
        """
        Run backtest simulation.

        Args:
            test_loader: Test data loader
            prediction_threshold: Minimum prediction magnitude to take position
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)

        Returns:
            Dictionary with backtest results
        """
        self.logger.info("Running backtest...")

        self.model.eval()

        all_predictions = []
        all_targets = []
        all_dates = []
        all_tickers = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                stock_id = batch['stock_id'].to(self.device)
                group_id = batch['group_id'].to(self.device)
                day = batch['day'].to(self.device)
                month = batch['month'].to(self.device)
                target = batch['target'].to(self.device)

                output = self.model(features, stock_id, group_id, day, month)

                all_predictions.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        # Calculate returns based on strategy
        strategy_returns = calculate_returns(predictions, targets, threshold=prediction_threshold)

        # Calculate portfolio value over time
        cumulative_returns = np.cumprod(1 + strategy_returns / 100)
        portfolio_values = initial_capital * cumulative_returns

        # Calculate metrics
        total_return = (portfolio_values[-1] / initial_capital - 1) * 100
        sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
        sortino_ratio = calculate_sortino_ratio(strategy_returns)
        max_drawdown = calculate_max_drawdown(strategy_returns) * 100  # Convert to percent

        # Win rate
        winning_trades = strategy_returns > 0
        win_rate = np.mean(winning_trades) * 100 if len(strategy_returns) > 0 else 0

        # Average win/loss
        avg_win = np.mean(strategy_returns[winning_trades]) if np.any(winning_trades) else 0
        avg_loss = np.mean(strategy_returns[~winning_trades]) if np.any(~winning_trades) else 0

        # Profit factor
        total_wins = np.sum(strategy_returns[winning_trades]) if np.any(winning_trades) else 0
        total_losses = abs(np.sum(strategy_returns[~winning_trades])) if np.any(~winning_trades) else 1
        profit_factor = total_wins / total_losses if total_losses != 0 else 0

        results = {
            # Portfolio metrics
            'initial_capital': initial_capital,
            'final_capital': portfolio_values[-1],
            'total_return_pct': total_return,
            'total_return_value': portfolio_values[-1] - initial_capital,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,

            # Trade statistics
            'num_trades': len(strategy_returns),
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,

            # Prediction metrics
            'predictions': predictions,
            'targets': targets,
            'returns': strategy_returns,
            'portfolio_values': portfolio_values,
        }

        # Print summary
        self._print_backtest_summary(results)

        return results

    def _print_backtest_summary(self, results: Dict[str, any]):
        """Print backtest summary."""
        print("\n" + "=" * 70)
        print("BACKTEST SUMMARY")
        print("=" * 70)

        print("\nPORTFOLIO METRICS:")
        print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"  Final Capital:   ${results['final_capital']:,.2f}")
        print(f"  Total Return:    {results['total_return_pct']:+.2f}% (${results['total_return_value']:,.2f})")
        print(f"  Sharpe Ratio:    {results['sharpe_ratio']:.4f}")
        print(f"  Sortino Ratio:   {results['sortino_ratio']:.4f}")
        print(f"  Max Drawdown:    {results['max_drawdown_pct']:.2f}%")

        print("\nTRADE STATISTICS:")
        print(f"  Number of Trades: {results['num_trades']}")
        print(f"  Win Rate:         {results['win_rate_pct']:.2f}%")
        print(f"  Avg Win:          {results['avg_win_pct']:.4f}%")
        print(f"  Avg Loss:         {results['avg_loss_pct']:.4f}%")
        print(f"  Profit Factor:    {results['profit_factor']:.4f}")

        print("\n" + "=" * 70)

    def generate_report(
        self,
        results: Dict[str, any],
        output_path: str,
        format: str = 'excel'
    ):
        """
        Generate backtest report.

        Args:
            results: Backtest results dictionary
            output_path: Output file path
            format: Output format ('excel', 'csv', 'json')
        """
        self.logger.info(f"Generating backtest report: {output_path}")

        if format == 'excel':
            self._generate_excel_report(results, output_path)
        elif format == 'csv':
            self._generate_csv_report(results, output_path)
        elif format == 'json':
            self._generate_json_report(results, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_excel_report(self, results: Dict[str, any], output_path: str):
        """Generate Excel report."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Initial Capital',
                    'Final Capital',
                    'Total Return (%)',
                    'Total Return ($)',
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Max Drawdown (%)',
                    'Number of Trades',
                    'Win Rate (%)',
                    'Avg Win (%)',
                    'Avg Loss (%)',
                    'Profit Factor',
                ],
                'Value': [
                    results['initial_capital'],
                    results['final_capital'],
                    results['total_return_pct'],
                    results['total_return_value'],
                    results['sharpe_ratio'],
                    results['sortino_ratio'],
                    results['max_drawdown_pct'],
                    results['num_trades'],
                    results['win_rate_pct'],
                    results['avg_win_pct'],
                    results['avg_loss_pct'],
                    results['profit_factor'],
                ]
            }

            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Trades sheet
            trades_df = pd.DataFrame({
                'Prediction': results['predictions'],
                'Target': results['targets'],
                'Return (%)': results['returns'],
                'Portfolio Value': results['portfolio_values'],
            })

            trades_df.to_excel(writer, sheet_name='Trades', index=False)

        self.logger.info(f"Excel report saved to {output_path}")

    def _generate_csv_report(self, results: Dict[str, any], output_path: str):
        """Generate CSV report."""
        # Save summary
        summary_path = Path(output_path).parent / 'backtest_summary.csv'
        summary_data = {
            'Metric': [
                'Initial Capital',
                'Final Capital',
                'Total Return (%)',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Max Drawdown (%)',
                'Number of Trades',
                'Win Rate (%)',
                'Profit Factor',
            ],
            'Value': [
                results['initial_capital'],
                results['final_capital'],
                results['total_return_pct'],
                results['sharpe_ratio'],
                results['sortino_ratio'],
                results['max_drawdown_pct'],
                results['num_trades'],
                results['win_rate_pct'],
                results['profit_factor'],
            ]
        }

        pd.DataFrame(summary_data).to_csv(summary_path, index=False)

        # Save trades
        trades_df = pd.DataFrame({
            'Prediction': results['predictions'],
            'Target': results['targets'],
            'Return (%)': results['returns'],
        })

        trades_df.to_csv(output_path, index=False)

        self.logger.info(f"CSV reports saved")

    def _generate_json_report(self, results: Dict[str, any], output_path: str):
        """Generate JSON report."""
        import json

        # Filter out array data for cleaner JSON
        report = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"JSON report saved to {output_path}")
