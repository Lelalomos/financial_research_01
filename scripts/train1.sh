docker exec crnn_predictor python scripts/preprocess_data.py --skip-download
docker exec crnn_predictor python scripts/train.py --model-type crnn_attention --backend lightning --epochs 30 --batch-size 32 --lr 0.0001
docker exec crnn_predictor python scripts/backtest.py --model best --model-type crnn_attention --data-dir data/processed --split test --output outputs/backtest_report_fixed.xlsx
