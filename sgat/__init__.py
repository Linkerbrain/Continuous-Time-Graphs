import logging

# STEP 1
# Make logger
logger = logging.getLogger(__name__)


# STEP 2
# specifies the lowest severity for logging
logger.setLevel(logging.INFO)

# STEP 3
# set a destination for your logs or a "handler"
# here, we choose to print on console (a consoler handler)
console_handler = logging.StreamHandler()

# STEP 4
# set the logging format for your handler
log_format = '%(asctime)s | %(levelname)s: %(message)s'
console_handler.setFormatter(logging.Formatter(log_format))

# finally, we add the handler to the logger
logger.addHandler(console_handler)