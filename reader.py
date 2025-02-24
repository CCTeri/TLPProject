import pandas as pd

class Reader(object):
    """
    Class to read in the input data

    Args:
        settings (dict): Project settings
        logger (logging.Logger): Logger object
    """

    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger

        pass

    def read_data(self, from_sql = True) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Read the input data

        Args:
            from_sql (bool): True if flows are to be read from SQL, False if they are to be read locally

        Returns:
            df (pandas.DataFrame): Dataframe that contains the flows and truck flows
            df_geo (pandas.DataFrame): Dataframe that contains the geographic information
            df_top30 (pandas.DataFrame): DataFrame with the top cities O&D combinations
            df_wacd_carriers (pandas.DataFrame): DataFrame with the WACD carriers
            df_wacd (pandas.DataFrame): Market data from WACD for every carrier on city-city level per period (month)
        """
        self.logger.info(f'[>] Reading data')

        df_wacd = self._read_market_data(df_top30)

        return df_wacd


def _read_market_data(self):
    """
    Read the WACD market data files on city - city level with three carrier groups (AF, KL and OTH).

    Args:
        df_top30 (pandas.DataFrame): DataFrame with the top cities O&D combinations

    Returns:
        pandas.DataFrame: Market data from WACD for every carrier on city-city level per period (month)
    """
    self.logger.info(f'[>] Reading WACD market data from SQL')

    engine = self._init_engine()
    query = f"""
    SELECT
        OriginCity AS DepartureCity,
        DestinationCity AS ArrivalCity,
        Period AS YearMonth,
        Carrier,
        ChargeableWeight
    FROM
        RM_History.dbo.tblAGGWACD
    WHERE
        LEN(OriginCity) = 3 AND
        LEN(DestinationCity) = 3 AND
        ChargeableWeight > 0 AND
        Period >= '{self.start_date}' AND
        Period <= '{self.end_date}'
    """

    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)
    df['YearMonth'] = pd.to_datetime(df['YearMonth']).dt.strftime('%Y%m').astype(int)
    # Chargeable weight is in kilos
    df['ChargeableWeight'] = df['ChargeableWeight'].astype(int)

    # Only keep O&Ds which are within the top cities we defined
    df['OD'] = df['DepartureCity'] + '-' + df['ArrivalCity']

    return df
