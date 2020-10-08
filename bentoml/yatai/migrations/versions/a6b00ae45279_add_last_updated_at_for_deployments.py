"""add last_updated_at for deployments

Revision ID: a6b00ae45279
Revises: 095fb029da39
Create Date: 2019-10-15 15:59:25.742280

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a6b00ae45279'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('deployments', sa.Column('last_updated_at', sa.DateTime))


def downgrade():
    op.drop_column('deployments', 'last_updated_at')
