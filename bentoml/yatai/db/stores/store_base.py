class Store(object):
    def add(self, session, *args, **kwargs):
        raise NotImplemented()

    def update(self, session, *args, **kwargs):
        raise NotImplemented()

    def list(self, session, *args, **kwargs):
        raise NotImplemented()

    def delete(self, session, *args, **kwargs):
        raise NotImplemented()