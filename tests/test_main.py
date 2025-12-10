import logging
from main import init_logger, settings


# Test 1: Verify Logger Initialization
def test_logger_initialization():
    """
    Test if the logger initializes correctly.
    """
    logger = init_logger(settings)
    assert isinstance(logger, logging.Logger), "Logger is not initialized correctly"

