from prometheus_client import Summary


def time(name=None, documentation=None):
    summary = Summary(name, documentation)

    def time_decorator(f):

        def wrapped(*args, **kwargs):
            with summary.time():
                return f(*args, **kwargs)

        return wrapped

    return time_decorator
