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

class Task(object):
    def __init__(self, name):
        self.name = name
        self.started = False
        self.isdone = False

    def start(self):
        logger.info(self.name + '...')
        self.started = True
        return self

    def done(self):
        if not self.started:
            raise Exception(f"Task '{self.name}' done before it started.")
        if self.isdone:
            raise Exception(f"Task '{self.name}' already done!")
        logger.info(self.name + '...' + 'Done')
        self.isdone = True
        return self


def task(name):
    def decorator(function):
        def wrapper(*args, **kwargs):
            t = Task(name).start()
            result = function(*args, **kwargs)
            t.done()
            return result
        return wrapper
    return decorator
