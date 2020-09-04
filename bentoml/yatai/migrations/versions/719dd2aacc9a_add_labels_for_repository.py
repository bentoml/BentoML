"""add labels for repository

Revision ID: 719dd2aacc9a
Revises: a6b00ae45279
Create Date: 2020-08-05 22:41:30.611193

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm

from bentoml.yatai.label_store import Label
from bentoml.yatai.deployment.store import Deployment

# revision identifiers, used by Alembic.
revision = '719dd2aacc9a'
down_revision = 'a6b00ae45279'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)
    Label.__table__.create(bind)

    deployments = sa.Table(
        'deployments',
        sa.MetaData(),
        sa.Column('id', sa.Integer),
        sa.Column('labels', sa.JSON),
    )
    result = bind.execute(sa.select([deployments.c.id, deployments.c.labels]))
    labels_need_to_add = []
    for row in result:
        for key in row.labels:
            labels_need_to_add.append(
                Label(
                    resource_type='deployment',
                    resource_id=row.id,
                    key=key,
                    value=row.labels[key],
                )
            )
    session.add_all(labels_need_to_add)

    session.commit()
    with op.batch_alter_table('deployments') as batch_op:
        batch_op.drop_column('labels')


def downgrade():
    op.add_column('deployments', sa.Column('labels', sa.JSON, default={}))
    op.drop_table('labels')
