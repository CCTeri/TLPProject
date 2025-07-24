import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler


class SimpleScaler:
    """
    Simple uniform scaler that scales all numeric features to 0-100 range.
    Best for: Quick prototyping, when features have similar distributions.
    """

    def __init__(self, logger=None):
        self.scaler = MinMaxScaler(feature_range=(0, 100))
        self.fitted = False
        self.feature_names = []
        self.logger = logger

    def fit(self, df):
        """
        Fit the scaler on training data.

        Args:
            df: Training DataFrame
        """
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            self.scaler.fit(df[numeric_cols])
            self.fitted = True
            self.feature_names = numeric_cols

            if self.logger:
                self.logger.info(f"[SimpleScaler] Fitted on {len(numeric_cols)} numeric features")

        return self

    def transform(self, df):
        """
        Transform data using fitted scaler.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with scaled features (0-100 range)
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transformation")

        df_scaled = df.copy()

        # Scale available features
        available_features = [col for col in self.feature_names if col in df.columns]

        if available_features:
            df_scaled[available_features] = self.scaler.transform(df[available_features])

            if self.logger:
                self.logger.info(f"[SimpleScaler] Transformed {len(available_features)} features")

        return df_scaled

    def fit_transform(self, df):
        """
        Fit and transform in one step.

        Args:
            df: Training DataFrame

        Returns:
            DataFrame with scaled features
        """
        return self.fit(df).transform(df)


class DomainSpecificScaler:
    """
    Domain-aware scaler that applies different scaling methods based on feature types.
    Best for: Production models, when preserving business meaning matters.
    """

    def __init__(self, logger=None):
        self.fitted = False
        self.logger = logger

        # Individual scalers for each domain
        self.revenue_scaler = MinMaxScaler(feature_range=(0, 100))
        self.ratio_scaler = RobustScaler()
        self.ratio_final_scaler = MinMaxScaler(feature_range=(0, 100))
        self.count_scaler = MinMaxScaler(feature_range=(0, 100))
        self.weight_scaler = MinMaxScaler(feature_range=(0, 100))

        # Track processed features
        self.processed_features = {
            'revenue': [],
            'share': [],
            'ratio': [],
            'count': [],
            'weight': [],
            'binary': []
        }

    def _get_feature_categories(self, df):
        """
        Categorize features based on domain knowledge.

        Args:
            df: DataFrame to categorize

        Returns:
            Dict of feature categories
        """
        categories = {
            'revenue': [
                'benchmark_revenue', 'total_revenue', 'ma3_revenue',
                'lag1_revenue', 'weighted_revenue'
            ],
            'share': [
                'weight_share', 'share_revenue', 'lag1_share', 'ma3_share_wt',
                'product_avg_share', 'similar_routes_performance', 'market_concentration',
                'weight_share_ma3', 'ma3_share_rev', 'top3_concentration'
            ],
            'ratio': [
                'revenue_per_kg', 'revenue_per_chargeable_kg', 'chargeability_ratio',
                'route_growth_trend', 'share_vs_peers'
            ],
            'count': [
                'product_rank', 'route_diversity', 'active_count_ma3',
                'route_maturity', 'month', 'quarter', 't', 'route_size_quartile'
            ],
            'weight': [
                'benchmark_actual_weight', 'benchmark_chargeable_weight',
                'total_weight', 'route_total_volume', 'lag1_actual_wt', 'lag1_chargeable_wt'
            ],
            'binary': [
                'is_market_leader'
            ]
        }

        # Filter to only include features present in the data
        filtered_categories = {}
        for category, features in categories.items():
            available_features = [f for f in features if f in df.columns]
            if available_features:
                filtered_categories[category] = available_features

        return filtered_categories

    def fit(self, df):
        """
        Fit scalers on training data.

        Args:
            df: Training DataFrame
        """
        categories = self._get_feature_categories(df)

        if self.logger:
            self.logger.info("[DomainScaler] Fitting domain-specific scalers...")

        # Fit scalers for each category

        # Revenue features: Log transform + MinMax scaling
        if 'revenue' in categories:
            revenue_cols = categories['revenue']
            revenue_log = np.log1p(df[revenue_cols].clip(lower=0))
            self.revenue_scaler.fit(revenue_log)
            self.processed_features['revenue'] = revenue_cols

            if self.logger:
                self.logger.info(f"  Revenue: {len(revenue_cols)} features (log + scale)")

        # Share features: Direct percentage conversion (no fitting needed)
        if 'share' in categories:
            self.processed_features['share'] = categories['share']

            if self.logger:
                self.logger.info(f"  Share: {len(categories['share'])} features (to percentage)")

        # Ratio features: Robust scaling + MinMax
        if 'ratio' in categories:
            ratio_cols = categories['ratio']
            self.ratio_scaler.fit(df[ratio_cols])
            ratio_scaled = self.ratio_scaler.transform(df[ratio_cols])
            self.ratio_final_scaler.fit(ratio_scaled)
            self.processed_features['ratio'] = ratio_cols

            if self.logger:
                self.logger.info(f"  Ratio: {len(ratio_cols)} features (robust + scale)")

        # Count features: Simple MinMax scaling
        if 'count' in categories:
            count_cols = categories['count']
            self.count_scaler.fit(df[count_cols])
            self.processed_features['count'] = count_cols

            if self.logger:
                self.logger.info(f"  Count: {len(count_cols)} features (simple scale)")

        # Weight features: Simple MinMax scaling
        if 'weight' in categories:
            weight_cols = categories['weight']
            self.weight_scaler.fit(df[weight_cols])
            self.processed_features['weight'] = weight_cols

            if self.logger:
                self.logger.info(f"  Weight: {len(weight_cols)} features (simple scale)")

        # Binary features: Direct conversion (no fitting needed)
        if 'binary' in categories:
            self.processed_features['binary'] = categories['binary']

            if self.logger:
                self.logger.info(f"  Binary: {len(categories['binary'])} features (to 0/100)")

        self.fitted = True

        if self.logger:
            total_features = sum(len(features) for features in self.processed_features.values())
            self.logger.info(f"[DomainScaler] Fitted scalers for {total_features} features")

        return self

    def transform(self, df):
        """
        Transform data using fitted scalers.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with scaled features
        """
        if not self.fitted:
            raise ValueError("Scalers must be fitted before transformation")

        df_scaled = df.copy()

        # Apply transformations for each category

        # Revenue features: Log + scale
        if self.processed_features['revenue']:
            revenue_cols = [col for col in self.processed_features['revenue'] if col in df.columns]
            if revenue_cols:
                revenue_log = np.log1p(df[revenue_cols].clip(lower=0))
                df_scaled[revenue_cols] = self.revenue_scaler.transform(revenue_log)

        # Share features: Convert to percentage
        if self.processed_features['share']:
            share_cols = [col for col in self.processed_features['share'] if col in df.columns]
            if share_cols:
                df_scaled[share_cols] = df[share_cols] * 100

        # Ratio features: Robust + scale
        if self.processed_features['ratio']:
            ratio_cols = [col for col in self.processed_features['ratio'] if col in df.columns]
            if ratio_cols:
                ratio_scaled = self.ratio_scaler.transform(df[ratio_cols])
                df_scaled[ratio_cols] = self.ratio_final_scaler.transform(ratio_scaled)

        # Count features: Simple scale
        if self.processed_features['count']:
            count_cols = [col for col in self.processed_features['count'] if col in df.columns]
            if count_cols:
                df_scaled[count_cols] = self.count_scaler.transform(df[count_cols])

        # Weight features: Simple scale
        if self.processed_features['weight']:
            weight_cols = [col for col in self.processed_features['weight'] if col in df.columns]
            if weight_cols:
                df_scaled[weight_cols] = self.weight_scaler.transform(df[weight_cols])

        # Binary features: Convert to 0/100
        if self.processed_features['binary']:
            binary_cols = [col for col in self.processed_features['binary'] if col in df.columns]
            if binary_cols:
                df_scaled[binary_cols] = df[binary_cols] * 100

        if self.logger:
            processed_count = sum(len([col for col in features if col in df.columns])
                                  for features in self.processed_features.values())
            self.logger.info(f"[DomainScaler] Transformed {processed_count} features")

        return df_scaled

    def fit_transform(self, df):
        """
        Fit and transform in one step.

        Args:
            df: Training DataFrame

        Returns:
            DataFrame with scaled features
        """
        return self.fit(df).transform(df)