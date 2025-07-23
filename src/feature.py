from typing import Tuple
import pandas as pd
import numpy as np


def _build_feature_market_competition(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create market competition and concentration features.

    These features help the model understand competitive dynamics:
    - How concentrated is the market? (few dominant products vs many competing)
    - Is this product a market leader or follower?
    - How stable is the competitive landscape?

    Returns features:
    - market_concentration: Herfindahl index (0=very competitive, 1=monopoly)
    - product_rank: This product's rank by weight share on this route
    - top3_concentration: Combined share of top 3 products
    - is_market_leader: Boolean if this product has highest share
    """
    self.logger.info('\t[>] Building features: market competition metrics')

    df_comp = df.copy()

    # Calculate market concentration using Herfindahl-Hirschman Index
    # Sum of squared market shares - higher = more concentrated market
    df_comp['market_concentration'] = (
        df_comp.groupby(['origin_city', 'destination_city', 'date'])['weight_share']
        .transform(lambda x: (x ** 2).sum())
    )

    # Rank products by weight share within each route-month
    df_comp['product_rank'] = (
        df_comp.groupby(['origin_city', 'destination_city', 'date'])['weight_share']
        .rank(method='dense', ascending=False)
    )

    # Calculate combined share of top 3 products (market concentration indicator)
    top3_shares = (
        df_comp.groupby(['origin_city', 'destination_city', 'date'])['weight_share']
        .transform(lambda x: x.nlargest(3).sum())
    )
    df_comp['top3_concentration'] = top3_shares

    # Flag if this product is the market leader
    df_comp['is_market_leader'] = (df_comp['product_rank'] == 1).astype(int)

    return df_comp


def _build_feature_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create volatility and stability features.

    Volatility features help predict which products have stable vs unpredictable demand:
    - Historical variance in market share
    - Coefficient of variation
    - Streak of growth/decline periods

    Returns features:
    - share_volatility: 6-month rolling standard deviation of weight share
    - share_cv: Coefficient of variation (volatility relative to mean)
    - consecutive_growth: Number of consecutive months of share growth
    - consecutive_decline: Number of consecutive months of share decline
    """
    self.logger.info('\t[>] Building features: volatility and stability metrics')

    df_vol = df.copy()

    # Sort for proper time series operations
    df_vol = df_vol.sort_values(['product', 'origin_city', 'destination_city', 'date'])

    group_cols = ['product', 'origin_city', 'destination_city']

    # 6-month rolling volatility of market share
    df_vol['share_volatility'] = (
        df_vol.groupby(group_cols)['weight_share']
        .transform(lambda x: x.rolling(6, min_periods=2).std())
    )

    # Coefficient of variation (volatility relative to mean level)
    rolling_mean = df_vol.groupby(group_cols)['weight_share'].transform(
        lambda x: x.rolling(6, min_periods=2).mean()
    )
    df_vol['share_cv'] = df_vol['share_volatility'] / (
                rolling_mean + 0.001)  # Add small constant to avoid division by zero

    # Calculate consecutive growth/decline streaks
    # First, determine if share increased or decreased vs previous month
    df_vol['share_change'] = df_vol.groupby(group_cols)['weight_share'].diff()
    df_vol['is_growing'] = (df_vol['share_change'] > 0).astype(int)
    df_vol['is_declining'] = (df_vol['share_change'] < 0).astype(int)

    # Count consecutive periods
    df_vol['consecutive_growth'] = df_vol.groupby(group_cols)['is_growing'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    df_vol['consecutive_decline'] = df_vol.groupby(group_cols)['is_declining'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )

    # Clean up temporary columns
    df_vol = df_vol.drop(['share_change', 'is_growing', 'is_declining'], axis=1)
    df_vol = df_vol.fillna(0)

    return df_vol


def _build_feature_route_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create route-level characteristics that affect product demand.

    Different routes have different characteristics that favor certain products:
    - Route maturity (how long has it been active?)
    - Route size (total volume)
    - Route diversity (how many products compete?)
    - Route growth trend

    Returns features:
    - route_total_volume: Total weight across all products on this route
    - route_diversity: Number of active products on this route
    - route_maturity: Number of months this route has been active
    - route_growth_trend: 6-month trend in total route volume
    """
    self.logger.info('\t[>] Building features: route characteristics')

    df_route = df.copy()

    # Total route volume per month
    df_route['route_total_volume'] = (
        df_route.groupby(['origin_city', 'destination_city', 'date'])['benchmark_actual_weight']
        .transform('sum')
    )

    # Route diversity - number of products with >1% market share
    df_route['route_diversity'] = (
        df_route.groupby(['origin_city', 'destination_city', 'date'])['weight_share']
        .transform(lambda x: (x > 0.01).sum())
    )

    # Route maturity - how many months has this route been active?
    df_route['route_maturity'] = (
        df_route.groupby(['origin_city', 'destination_city'])['date']
        .transform(lambda x: (x.max() - x.min()).days / 30)  # Convert to months
    )

    # Route growth trend - 6-month rolling growth rate of total volume
    df_route = df_route.sort_values(['origin_city', 'destination_city', 'date'])
    df_route['route_growth_trend'] = (
        df_route.groupby(['origin_city', 'destination_city'])['route_total_volume']
        .transform(lambda x: x.pct_change(periods=6))
    )

    df_route = df_route.fillna(0)

    return df_route


def _build_feature_cross_route_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on patterns across multiple routes.

    These features help identify products that are strong in similar markets:
    - How does this product perform on similar routes?
    - Are there regional patterns?
    - Cross-route momentum indicators

    Returns features:
    - product_avg_share: This product's average share across all routes
    - origin_product_strength: Product's average share from this origin
    - destination_product_strength: Product's average share to this destination
    - similar_routes_performance: Average performance on routes with similar characteristics
    """
    self.logger.info('\t[>] Building features: cross-route patterns')

    df_cross = df.copy()

    # Product's overall average market share across all routes
    df_cross['product_avg_share'] = (
        df_cross.groupby(['product', 'date'])['weight_share']
        .transform('mean')
    )

    # Product strength by origin city (some products stronger from certain origins)
    df_cross['origin_product_strength'] = (
        df_cross.groupby(['product', 'origin_city', 'date'])['weight_share']
        .transform('mean')
    )

    # Product strength by destination city
    df_cross['destination_product_strength'] = (
        df_cross.groupby(['product', 'destination_city', 'date'])['weight_share']
        .transform('mean')
    )

    # Performance on similar-sized routes (routes with similar total volume)
    # First, create route size categories
    df_cross['route_size_quartile'] = (
        df_cross.groupby('date')['route_total_volume']
        .transform(lambda x: pd.qcut(x, q=4, labels=[1, 2, 3, 4], duplicates='drop'))
    )

    # Average performance on routes of similar size
    df_cross['similar_routes_performance'] = (
        df_cross.groupby(['product', 'route_size_quartile', 'date'])['weight_share']
        .transform('mean')
    )

    # Fill any remaining NaN values
    df_cross = df_cross.fillna(0)

    return df_cross


def _build_feature_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create momentum and acceleration features.

    These features capture whether a product is gaining or losing momentum:
    - Rate of change in market share
    - Acceleration (change in the rate of change)
    - Momentum relative to competitors

    Returns features:
    - share_momentum_1m: 1-month change in weight share
    - share_momentum_3m: 3-month change in weight share
    - share_acceleration: Change in momentum (2nd derivative)
    - relative_momentum: This product's momentum vs route average
    """
    self.logger.info('\t[>] Building features: momentum indicators')

    df_momentum = df.copy()
    df_momentum = df_momentum.sort_values(['product', 'origin_city', 'destination_city', 'date'])

    group_cols = ['product', 'origin_city', 'destination_city']

    # 1-month and 3-month momentum
    df_momentum['share_momentum_1m'] = (
        df_momentum.groupby(group_cols)['weight_share'].diff(periods=1)
    )
    df_momentum['share_momentum_3m'] = (
        df_momentum.groupby(group_cols)['weight_share'].diff(periods=3)
    )

    # Acceleration (change in momentum)
    df_momentum['share_acceleration'] = (
        df_momentum.groupby(group_cols)['share_momentum_1m'].diff(periods=1)
    )

    # Route-level average momentum for comparison
    route_avg_momentum = (
        df_momentum.groupby(['origin_city', 'destination_city', 'date'])['share_momentum_1m']
        .transform('mean')
    )

    # Relative momentum vs other products on same route
    df_momentum['relative_momentum'] = df_momentum['share_momentum_1m'] - route_avg_momentum

    df_momentum = df_momentum.fillna(0)

    return df_momentum