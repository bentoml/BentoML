"""add last_update_at timestamp for deployment

Revision ID: 5e1fc4452ff2
Revises: 
Create Date: 2019-10-14 14:53:42.915298

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5e1fc4452ff2'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('deployments', sa.Column('last_updated_at', sa.DateTime))


def downgrade():
    op.drop_column('deployments', 'last_updated_at')
