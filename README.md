# TLP Project: Product Market Share Analysis

A machine learning tool that calculates and predicts product type market shares per Origin-Destination (O&D) route. For any given origin–destination route and time period, this tool computes how each cargo "product" (General, Perishables, Pharma, DGR, etc.) shares the total market volume.

## Key Features

This tool provides two main capabilities:
1. **Prediction**: Forecasts the market share per product for the next month
2. **Trend Identification**: Identifies the product trend that is likely to lead the market on each route

## Business Value

- **Route-level demand visibility**: See which products dominate each O&D and how concentrated or diversified the mix is.
- **Capacity and pricing levers**: Align capacity and pricing by product and corridor using next-month share forecasts and trend signals.
- **Growth targeting**: Spot rising products/routes early to prioritize sales pushes or service enhancements.
- **Portfolio risk control**: Flag declining or over-concentrated routes to rebalance network and reduce exposure.

## Questions This Tool Answers

- Which products are strong in specific markets?
- Where should we deploy new service offerings?
- Which markets are experiencing rising demand for specific product types and how quickly are they growing?
- Which routes are the most "concentrated" in a single product versus those with diversified mixes?


## How It Works

The pipeline performs the following steps:

1. **Data Ingestion**: Ingests monthly O→D shipment data (from GCS)
2. **Data Processing**: Aggregates by product and route to calculate each product's weight share, filters routes by market size threshold
3. **Feature Engineering**: Builds revenue, seasonality, lag, ratio, route, competition, and cross-route pattern features
4. **Feature Scaling**: Applies scaling (simple MinMax 0-100 or domain-specific scaling) to prepare features for modeling
5. **Model Training**: Trains and compares three models using temporal validation:
   - Training: Jan 2024 - Nov 2024
   - Validation: Dec 2024
   - Test: Jan 2025
   - Models: LightGBM, Random Forest, Linear Regression
6. **Model Selection**: Selects best model based on validation RMSE (lowest error)
7. **Prediction**: Generates next-month share forecasts for each product-route combination using the best model
8. **Output**: Provides predictions with all features for further analysis in Power BI or other tools

By surfacing which products dominate which corridors and where demand is shifting, it supports pricing, capacity planning, performance monitoring, and risk management in a single, end-to-end pipeline.



## Prerequisites

- Python 3.11+
- Poetry (install from https://python-poetry.org/docs/#installation)
- Access to Google Cloud Storage (GCS) if using cloud data sources
- Required credentials for GCS stored in `key/key_cloudstorage.json`

## Installation

### Option 1: Using Poetry (Recommended)

```bash
# Install Poetry if you haven't already
# Windows: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Install dependencies and create virtual environment
poetry install

# Activate the virtual environment
poetry shell
```

### Option 2: Using pip (Alternative)

```bash
# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Docker

```bash
# Build the Docker image
docker build -t tlp-project .

# Run the container
docker run -v $(pwd)/data:/app/data tlp-project
```

## Configuration

The project uses `settings.yml` for configuration. Key settings include:

- **Data Source**: Configure GCS bucket and paths, or use local file paths
  - `gcs_bucket`: GCS bucket name
  - `gcs_input_path`: Path to input CSV in GCS
  - `project_input`: Local input directory path
  - `WACD_Local`: Local input filename
- **Model Parameters**: Adjust hyperparameters for LightGBM, Random Forest, and Linear Regression
- **Prediction Date**: Set the target month for predictions (e.g., "2025-02")
- **Scaling Method**: Choose between "simple" or "domain" specific scaling
- **Trend Threshold**: Set threshold for trend identification (default: 0.02 for 2%)
- **Market Size**: Minimum market size threshold for route filtering (default: 25000)

See `settings.yml` for all available configuration options with detailed comments.

## Usage

### Basic Usage (Command Line)

1. **Configure your settings** in `settings.yml`:
   ```yaml
   # For local data
   project_input: "data/input/"
   WACD_Local: 'your_data_file.csv'
   
   # Or for GCS
   gcs_bucket: "your_bucket_name"
   gcs_input_path: "path/to/your/data.csv"
   ```

2. **Set up GCS credentials** (if using cloud storage):
   - Place your GCS service account key file at `key/key_cloudstorage.json`
   - Ensure the service account has read permissions for the input bucket

3. **Run the pipeline**:
   ```bash
   # If using Poetry
   poetry run python main.py
   
   # Or if using pip/venv
   python main.py
   ```

4. **Check the output**:
   - Predictions are saved to `data/output/prediction_YYYYMM.csv` (e.g., `prediction_202502.csv`)
   - Logs provide information about model performance, selected best model, and metrics (RMSE, MAE, R², MAPE)

### API Usage (Flask Server)

Start the Flask API server:

```bash
# If using Poetry
poetry run python server.py

# Or if using pip/venv
python server.py
```

The API will be available at `http://localhost:8080/`. Accessing the root endpoint (`/`) will run the full pipeline and return a success message.

## Project Structure

```
TLPProject/
├── main.py              # Main pipeline execution script
├── server.py            # Flask API server
├── settings.yml         # Configuration file
├── pyproject.toml       # Poetry configuration and dependencies
├── requirements.txt     # Python dependencies (pip fallback)
├── Dockerfile           # Docker configuration
├── README.md            # This file
├── data/
│   ├── input/          # Input data files
│   └── output/         # Generated predictions
├── key/
│   └── key_cloudstorage.json  # GCS credentials (not in version control)
├── src/
│   ├── reader.py       # Data reading utilities (GCS/local)
│   ├── processor.py    # Data processing and aggregation
│   ├── feature.py      # Feature engineering
│   ├── scaler.py       # Feature scaling (SimpleScaler, DomainSpecificScaler)
│   ├── modeler.py      # Model training and comparison (MultiModelComparer)
│   ├── writer.py       # Output writing utilities
│   └── logger.py       # Logging configuration
└── tests/
    └── test_main.py    # Unit tests
```

## Models

The tool automatically compares three models and selects the best performer using temporal validation:

1. **LightGBM**: Gradient boosting framework optimized for speed and performance
   - Uses validation set for early stopping
   - Configurable via `lightgbm_params` in settings.yml
2. **Random Forest**: Ensemble method providing robust predictions
   - Configurable via `rf_params` in settings.yml
3. **Linear Regression**: Baseline model for comparison
   - Configurable via `lr_params` in settings.yml

Model hyperparameters can be configured in `settings.yml`. The best model is selected based on the lowest validation RMSE. All models are evaluated on train, validation, and test sets with metrics including RMSE, MAE, R², and MAPE.

## Output Format

The output CSV contains one row per product-route combination for the target prediction month, including:

- **Identifiers**: `origin_city`, `destination_city`, `product`, `date`
- **Raw Metrics**: `benchmark_actual_weight`, `benchmark_chargeable_weight`, `benchmark_revenue`, `total_weight`
- **Calculated Shares**: `weight_share` (historical), `share_revenue`
- **Features**: All engineered features used in the model (lags, moving averages, ratios, route characteristics, etc.)
- **Predictions**: `pred_share` - predicted market share percentage for that product-route
- **Trend Indicators**: `product_trend` (stable/growth/decline/new/disappeared), `weight_share_ma3`

The leading product per route can be identified by sorting by `pred_share` descending for each origin-destination pair.

Example output file: `data/output/prediction_202502.csv`

## Testing

Run the test suite:

```bash
# If using Poetry
poetry run pytest tests/

# Or if using pip/venv
pytest tests/
```

## Data Requirements

Input data should be a tab-separated CSV file with the following columns:
- `product`: Product type (General, Perishables, Pharma, DGR, etc.)
- `origin_city`: Origin city code (3-letter IATA code)
- `destination_city`: Destination city code (3-letter IATA code)
- `date`: Date in format readable by pandas (YYYY-MM-DD or similar)
- `benchmark_actual_weight`: Actual weight metric
- `benchmark_chargeable_weight`: Chargeable weight metric
- `benchmark_revenue`: Revenue metric

## Notes

- For GCS usage, ensure `key/key_cloudstorage.json` is not committed to version control (add to `.gitignore`)
- Logging is configurable via `settings.yml` and provides detailed pipeline execution information

---

**Version**: v2025.02
