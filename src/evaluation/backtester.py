"""
Backtester for Multi-Model Financial Forecasting.

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

from src.config import load_config
from src.utils.logger import EvaluationLogger
from .metrics import (
    evaluate_model,
    calculate_returns,
    calculate_turnover,
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
        config,
        device: str = 'cuda'
    ):
        """
        Initialize backtester.

        Args:
            model: PyTorch model
            config instance
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
        commission: float = 0.001,
        stock_id_to_ticker: Optional[Dict[int, str]] = None,
        group_id_to_sector: Optional[Dict[int, str]] = None
    ) -> Dict[str, any]:
        """
        Run backtest simulation.

        Args:
            test_loader: Test data loader
            prediction_threshold: Minimum prediction magnitude to take position
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)
            stock_id_to_ticker: Optional mapping from stock_id to ticker name
            group_id_to_sector: Optional mapping from group_id to sector name

        Returns:
            Dictionary with backtest results
        """
        self.logger.info("Running backtest...")

        self.model.eval()

        all_predictions = []
        all_targets = []
        all_dates = []
        all_tickers = []
        all_stock_ids = []
        all_group_ids = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                stock_id = batch['stock_id'].to(self.device)
                group_id = batch['group_id'].to(self.device)
                day = batch['day'].to(self.device)
                month = batch['month'].to(self.device)
                target = batch['target'].to(self.device)

                # Get dividend_flag if available
                if 'dividend_flag' in batch:
                    dividend_flag = batch['dividend_flag'].to(self.device)
                else:
                    dividend_flag = torch.ones(features.shape[0], features.shape[1],
                                                  dtype=torch.long, device=self.device)

                output = self.model(features, stock_id, group_id, day, month, dividend_flag)

                all_predictions.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
                all_stock_ids.extend(stock_id[:, 0].cpu().numpy().flatten())
                all_group_ids.extend(group_id[:, 0].cpu().numpy().flatten())

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        stock_ids = np.array(all_stock_ids)
        group_ids = np.array(all_group_ids)

        # Calculate gross returns based on strategy and apply trading friction.
        gross_returns = calculate_returns(predictions, targets, threshold=prediction_threshold)
        turnover = calculate_turnover(predictions, threshold=prediction_threshold)
        transaction_costs = turnover * commission * 100.0
        strategy_returns = gross_returns - transaction_costs

        # Calculate portfolio value over time
        cumulative_returns = np.cumprod(1 + strategy_returns / 100)
        portfolio_values = initial_capital * cumulative_returns

        # Calculate metrics
        total_return = (portfolio_values[-1] / initial_capital - 1) * 100
        sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
        sortino_ratio = calculate_sortino_ratio(strategy_returns)
        max_drawdown = calculate_max_drawdown(strategy_returns) * 100  # Convert to percent
        average_turnover = float(np.mean(turnover)) if len(turnover) > 0 else 0.0
        total_turnover = float(np.sum(turnover)) if len(turnover) > 0 else 0.0
        num_position_changes = int(np.count_nonzero(turnover))
        total_transaction_cost_pct = float(np.sum(transaction_costs)) if len(transaction_costs) > 0 else 0.0
        total_transaction_cost_value = initial_capital * (total_transaction_cost_pct / 100.0)

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

        # Create ticker and sector labels
        if stock_id_to_ticker:
            tickers = np.array([stock_id_to_ticker.get(sid, f"stock_{sid}") for sid in stock_ids])
        else:
            tickers = np.array([f"stock_{sid}" for sid in stock_ids])

        if group_id_to_sector:
            sectors = np.array([group_id_to_sector.get(gid, f"sector_{gid}") for gid in group_ids])
        else:
            sectors = np.array([f"sector_{gid}" for gid in group_ids])

        # Calculate direction scores by sector
        direction_scores = (np.sign(predictions) == np.sign(targets)).astype(int)
        sector_stats = {}
        for sector in np.unique(sectors):
            sector_mask = sectors == sector
            sector_count = sector_mask.sum()
            sector_correct = direction_scores[sector_mask].sum()
            sector_accuracy = sector_correct / sector_count if sector_count > 0 else 0

            sector_stats[sector] = {
                'total': int(sector_count),
                'correct': int(sector_correct),
                'accuracy': float(sector_accuracy),
            }

        results = {
            # Portfolio metrics
            'initial_capital': initial_capital,
            'final_capital': portfolio_values[-1],
            'total_return_pct': total_return,
            'total_return_value': portfolio_values[-1] - initial_capital,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'risk_adjusted_return': sharpe_ratio,

            # Trade statistics
            'num_trades': len(strategy_returns),
            'num_position_changes': num_position_changes,
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'average_turnover': average_turnover,
            'total_turnover': total_turnover,
            'commission_rate': commission,
            'total_transaction_cost_pct': total_transaction_cost_pct,
            'total_transaction_cost_value': total_transaction_cost_value,

            # Prediction metrics
            'predictions': predictions,
            'targets': targets,
            'returns': strategy_returns,
            'gross_returns': gross_returns,
            'transaction_costs': transaction_costs,
            'turnover': turnover,
            'portfolio_values': portfolio_values,

            # Sector analysis
            'tickers': tickers,
            'sectors': sectors,
            'direction_scores': direction_scores,
            'sector_stats': sector_stats,
            'stock_ids': stock_ids,
            'group_ids': group_ids,
        }

        # Print summary
        self._print_backtest_summary(results)

        # Print sector stats if available
        if sector_stats:
            self._print_sector_stats(sector_stats)

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
        print(f"  Risk-Adj Return: {results['risk_adjusted_return']:.4f}")

        print("\nTRADE STATISTICS:")
        print(f"  Number of Trades:  {results['num_trades']}")
        print(f"  Position Changes:  {results['num_position_changes']}")
        print(f"  Win Rate:          {results['win_rate_pct']:.2f}%")
        print(f"  Avg Win:           {results['avg_win_pct']:.4f}%")
        print(f"  Avg Loss:          {results['avg_loss_pct']:.4f}%")
        print(f"  Profit Factor:     {results['profit_factor']:.4f}")
        print(f"  Avg Turnover:      {results['average_turnover']:.4f}")
        print(f"  Total Turnover:    {results['total_turnover']:.2f}")
        print(f"  Total Tx Cost:     {results['total_transaction_cost_pct']:.4f}%")

        print("\n" + "=" * 70)

    def _print_sector_stats(self, sector_stats: Dict[str, Dict[str, any]]):
        """Print sector direction accuracy statistics."""
        print("\n" + "=" * 70)
        print("DIRECTION ACCURACY BY SECTOR")
        print("=" * 70)
        print(f"{'Sector':<30} | {'Correct':<10} | {'Total':<10} | {'Accuracy':<12}")
        print("-" * 70)

        for sector, stats in sorted(sector_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            correct = stats['correct']
            total = stats['total']
            accuracy = stats['accuracy']
            print(f"{sector:<30} | {correct:<10} | {total:<10} | {accuracy:<12.2%}")

        print("-" * 70)
        print("=" * 70)

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
        """Generate Excel report with detailed trades and sector analysis."""
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
                    'Risk-Adjusted Return',
                    'Number of Trades',
                    'Position Changes',
                    'Win Rate (%)',
                    'Avg Win (%)',
                    'Avg Loss (%)',
                    'Profit Factor',
                    'Average Turnover',
                    'Total Turnover',
                    'Commission Rate',
                    'Transaction Cost (%)',
                    'Transaction Cost ($)',
                ],
                'Value': [
                    results['initial_capital'],
                    results['final_capital'],
                    results['total_return_pct'],
                    results['total_return_value'],
                    results['sharpe_ratio'],
                    results['sortino_ratio'],
                    results['max_drawdown_pct'],
                    results['risk_adjusted_return'],
                    results['num_trades'],
                    results['num_position_changes'],
                    results['win_rate_pct'],
                    results['avg_win_pct'],
                    results['avg_loss_pct'],
                    results['profit_factor'],
                    results['average_turnover'],
                    results['total_turnover'],
                    results['commission_rate'],
                    results['total_transaction_cost_pct'],
                    results['total_transaction_cost_value'],
                ]
            }

            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Detailed trades sheet with ticker, sector, direction score
            trades_data = {
                'Ticker': results.get('tickers', [f"stock_{i}" for i in range(len(results['predictions']))]),
                'Sector': results.get('sectors', [f"sector_{i}" for i in range(len(results['predictions']))]),
                'Real Target': results['targets'],
                'Predict Target': results['predictions'],
                'Distance': results['predictions'] - results['targets'],
                'Direction Score': results.get('direction_scores', [0] * len(results['predictions'])),
                'Turnover': results.get('turnover', np.zeros(len(results['predictions']))),
                'Transaction Cost (%)': results.get('transaction_costs', np.zeros(len(results['predictions']))),
                'Gross Return (%)': results.get('gross_returns', results['returns']),
                'Return (%)': results['returns'],
                'Portfolio Value': results['portfolio_values'],
            }

            trades_df = pd.DataFrame(trades_data)

            # Add per-ticker std columns
            trades_df['Std Real'] = trades_df.groupby('Ticker')['Real Target'].transform(lambda x: x.std())
            trades_df['Std Predict'] = trades_df.groupby('Ticker')['Predict Target'].transform(lambda x: x.std())

            trades_df.to_excel(writer, sheet_name='Trades', index=False)

            # Sector statistics sheet
            if 'sector_stats' in results and results['sector_stats']:
                sector_df = pd.DataFrame({
                    sector: {
                        'Total': stats['total'],
                        'Correct': stats['correct'],
                        'Accuracy': stats['accuracy'],
                    }
                    for sector, stats in results['sector_stats'].items()
                }).T

                sector_df.to_excel(writer, sheet_name='Sector Stats')

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
                'Risk-Adjusted Return',
                'Number of Trades',
                'Position Changes',
                'Win Rate (%)',
                'Profit Factor',
                'Average Turnover',
                'Total Turnover',
                'Commission Rate',
                'Transaction Cost (%)',
                'Transaction Cost ($)',
            ],
            'Value': [
                results['initial_capital'],
                results['final_capital'],
                results['total_return_pct'],
                results['sharpe_ratio'],
                results['sortino_ratio'],
                results['max_drawdown_pct'],
                results['risk_adjusted_return'],
                results['num_trades'],
                results['num_position_changes'],
                results['win_rate_pct'],
                results['profit_factor'],
                results['average_turnover'],
                results['total_turnover'],
                results['commission_rate'],
                results['total_transaction_cost_pct'],
                results['total_transaction_cost_value'],
            ]
        }

        pd.DataFrame(summary_data).to_csv(summary_path, index=False)

        # Save trades
        trades_df = pd.DataFrame({
            'Prediction': results['predictions'],
            'Target': results['targets'],
            'Turnover': results.get('turnover', np.zeros(len(results['predictions']))),
            'Transaction Cost (%)': results.get('transaction_costs', np.zeros(len(results['predictions']))),
            'Gross Return (%)': results.get('gross_returns', results['returns']),
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
