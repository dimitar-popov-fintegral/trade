import logging

################################################################################
def get_logging_handler(level=logging.DEBUG):
    logging_handler = logging.StreamHandler(sys.stdout)
    logging_handler.setLevel(level)
    logging_handler.setFormatter()
