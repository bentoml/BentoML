"""add labels for repository

Revision ID: 719dd2aacc9a
Revises: a6b00ae45279
Create Date: 2020-08-05 22:41:30.611193

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '719dd2aacc9a'
down_revision = 'a6b00ae45279'
branch_labels = None
depends_on = None


def upgrade():
    pass
    # deployments = sa.Table(
    #     'deployments',
    #     sa.MetaData(),
    #     sa.Column('id', sa.Integer),
    #     sa.Column('labels', sa.JSON),
    # )
    # labels = sa.Table(
    #     'labels',
    #     sa.MetaData(),
    #     sa.Column('resource_type', sa.String),
    #     sa.Column('resource_id', sa.String),
    #     sa.Column('key', sa.String),
    #     sa.Column('value', sa.String),
    # )
    # connection = op.get_bind()
    # results = connection.execute(sa.select([deployments.c.id, deployments.c.labels]))
    # for row in results:
    #     connection.execute(labels.create())
    #
    # op.drop_column('deployments', 'labels')


def downgrade():
    pass
    # op.add_column(
    #     'deployments', sa.Column('labels', sa.JSON, nullable=True, default={})
    # )
    # deployments = sa.Table(
    #     'deployments',
    #     sa.MetaData(),
    #     sa.Column('id', sa.Integer),
    #     sa.Column('labels', sa.JSON),
    # )
    # connection = op.get_bind()
    # results = connection.execute(
    #     sa.select([deployments.c.id, deployments.c.labels])
    # ).fetchall()
    # for id in results:
    #     connection.execute(
    #         deployments.update().where(deployments.c.id == id[0]).values(labels={})
    #     )
