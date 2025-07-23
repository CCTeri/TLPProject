import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Union
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

class Modeler:
    """
    Encapsulates feature engineering and LightGBM modeling
    for one-month-ahead share forecasts per product × O→D.
    """
    def __init__(self, settings: dict, logger, verbose: bool = False):

        self.settings = settings
        self.logger = logger
        self.verbose = verbose

        # pull params from settings or fall back
        self.lgb_params = settings.get('model_params', {
            'objective': 'regression',
            'n_estimators': 500,
            'learning_rate': 0.05
        })

        # If my validation RMSE doesn’t get any better after 20 new trees in a row,
        # stop running more trees (out of 500) - there’s no point overfitting or wasting time
        # topping point 10 = stop sooner, can underfit / 50 = more room for improvements but cost longer training
        self.early_stopping_rounds = settings.get('early_stopping_rounds', 20)

        # up to fold n_splits, which trains on the first n_splits chunks and tests on the last chunk.
        # 2 = faster but noiser metric / 5 = more validation, better error estimate, more time
        self.n_splits = settings.get('n_splits', 3)

        # LGBM: gradient-boosted decision trees
        # smaller sample sizes, interpretability via feature importance.
        # It’s fast to train and often outperforms neural nets on structured data.
        self.model: lgb.LGBMRegressor = None
        self.logger.info(f"[Modeler] params={self.lgb_params}, "
                         f"early_stop={self.early_stopping_rounds}, splits={self.n_splits}")


    def train(self, df_route):
        """
        Build features from df_route, then fit LightGBM on share_actual.
        Test set is Jan 2025, and check the accuracy.
        """
        TARGET = 'weight_share'
        FEATURES = [col for col in df_route.columns
                    if col not in {TARGET, 'date', 'product', 'origin_city', 'destination_city', 'product_trend'}]
        LAST = df_route['date'].max()

        # Split the data based on time
        train_df = df_route[df_route['date'] < LAST].copy()
        val_df = df_route[df_route['date'] == LAST].copy()

        # Cast strings to categorical
        # cat_cols = ['product', 'origin_city', 'destination_city']
        # train_df.loc[:, cat_cols] = train_df[cat_cols].astype('category')
        # val_df.loc[:, cat_cols] = val_df[cat_cols].astype('category')

        X_train, y_train = train_df[FEATURES], train_df[TARGET]
        X_val, y_val = val_df[FEATURES], val_df[TARGET]

        self.logger.info(f"[Modeler] Training on {X_train.shape[0]} rows, "
                         f"validating on {X_val.shape[0]} rows")

        # Model and fit
        self.model = lgb.LGBMRegressor(**self.lgb_params)
        self.model.fit(X_train, y_train)

        # Log the validation RMSE
        preds = self.model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        self.logger.info(f"[Modeler] Validation RMSE: {rmse:.4f}")

    def predict_future(self, df_route: pd.DataFrame, date: Union[str,pd.Timestamp]) -> pd.DataFrame:
        """
        Build a future feature frame for the given date,
        predict share_actual, and return top‐product per O→D.
        """

        # 1) Copy and find the last historic period
        last_period = df_route['date'].max()

        # 2) Take only the last-period rows (one row per product×O×D with all lag/time features)
        last_feat = df_route[df_route['date'] == last_period].copy()

        # 3) Build future frame by reusing those last-period feature rows
        df_fut = last_feat.copy()

        # 4) Overwrite the date to the target future month
        fut_month = pd.to_datetime(date)
        df_fut['date'] = fut_month
        df_fut['month'] = fut_month.month
        df_fut['quarter'] = fut_month.quarter
        # “t” is a simple month index from the earliest year in your history
        min_year = df_route['date'].dt.year.min()
        df_fut['t'] = (fut_month.year - min_year) * 12 + fut_month.month

        # 6) Define the exact same feature list used in training
        TARGET = 'weight_share'
        FEATURES = [col for col in df_route.columns
                    if col not in {TARGET, 'date', 'product', 'origin_city', 'destination_city', 'product_trend'}]

        # 7) Run the model
        df_fut['pred_share'] = self.model.predict(df_fut[FEATURES])

        # 8) Pick the winning product per route by highest pred_share
        top = (
            df_fut
            .sort_values(['origin_city', 'destination_city', 'pred_share'], ascending=[True,True,False])
            .groupby(['origin_city', 'destination_city'], as_index=False)
            .first()     # first() after sorting gives the max-share row
        )

        return top
