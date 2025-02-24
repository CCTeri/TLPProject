import yaml, os
from dotenv import load_dotenv
from src.logger import init_logger
from src.reader import Reader

# Load environment variables
load_dotenv()
ftp_info = {
    'host': os.environ['FTP_HOST'],
    'user': os.environ['FTP_USER'],
    'password': os.environ['FTP_PASSWORD']
}

# Load settings from settings.yml file and environment variables
with open('settings.yml', 'r') as f:
    settings = yaml.load(f, Loader = yaml.FullLoader)
    settings['sql_connection'] = os.environ['secret_SQL']

# Initialize logger for a selected period
logger = init_logger(settings)
logger.info(f'[>] Running TLP Project: Niche Market Research for Cargo')

# Read data
df_wacd = Reader(settings, logger).read_data()

if df_wacd is not None:
    print("Market data loaded successfully!")
    print(df_wacd.head())

logger.info('[V] Finished')


