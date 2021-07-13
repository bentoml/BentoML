"""add labels table

Revision ID: 719dd2aacc9a
Revises: a6b00ae45279
Create Date: 2020-08-05 22:41:30.611193

"""
from alembic import op
import sqlalchemy as sa

from bentoml.yatai.db.stores.label import Label

# revision identifiers, used by Alembic.
revision = '719dd2aacc9a'
down_revision = 'a6b00ae45279'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    Label.__table__.create(bind)
    with op.batch_alter_table('deployments') as batch_op:
        batch_op.drop_column('labels')


def downgrade():
    op.add_column('deployments', sa.Column('labels', sa.JSON, default={}))
    op.drop_table('labels')
