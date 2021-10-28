import click


@click.group()
def models():
    pass


@models.command()
def get():
    pass


@models.command(name="list")
def list_models():
    pass


@models.command()
def delete():
    pass


@models.command()
def push():
    pass


@models.command()
def pull():
    pass


@models.command()
def export():
    pass


@models.command(name="import")
def import_model():
    pass
