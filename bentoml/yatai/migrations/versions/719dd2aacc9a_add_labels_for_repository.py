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
    op.add_column('bentos', sa.Column('labels', sa.JSON))


def downgrade():
    op.drop_column('bentos', 'labels')
    pass
