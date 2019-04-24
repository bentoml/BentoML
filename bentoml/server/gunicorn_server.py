import multiprocessing

import gunicorn.app.base
from gunicorn.six import iteritems


def get_gunicorn_worker_count():
    """
    Generate an recommend gunicorn worker process count

    Gunicorn's documentation recommand 2 times of cpu cores + 1.
    For ml model serving, it might consumer more computing resources, therefore
    we recommend half of the number of cpu cores + 1
    """

    return (multiprocessing.cpu_count() // 2) + 1


class GunicornApplication(gunicorn.app.base.BaseApplication):  # pylint: disable=abstract-method
    """
    A custom Gunicorn application.

    Usage::

        >>> import GunicornApplication
        >>>
        >>> gunicorn_app = GunicornApplication(app, 5000, 2)
        >>> gunicorn_app.run()

    :param app: a Flask app, flask.Flask.app
    :param port: the port you want to run gunicorn server on
    :param workers: number of worker processes
    """

    def __init__(self, app, port, workers):
        self.options = {'workers': workers, 'bind': '%s:%s' % ('127.0.0.1', port)}
        self.application = app
        super(GunicornApplication, self).__init__()

    def load_config(self):
        config = dict([(key, value) for key, value in iteritems(self.options)
                       if key in self.cfg.settings and value is not None])
        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application
