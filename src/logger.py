import logging, os, time

def init_logger(settings):
    log_level = settings['log_level']
    name = settings['project_name']

    # Set up loggers (Verbose for illustration purposes)
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    level = logging.getLevelName(log_level.upper())
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Delete old handlers
    logger.handlers = []

    # Setup logger to console
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger