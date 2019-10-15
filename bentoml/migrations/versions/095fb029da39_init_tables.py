"""init tables

Revision ID: 095fb029da39
Revises: 
Create Date: 2019-10-15 14:27:22.948583

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

from bentoml.db import Base
from bentoml.deployment.store import Deployment
from bentoml.repository.metadata_store import Bento

# revision identifiers, used by Alembic.

revision = '095fb029da39'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    tables = inspector.get_table_names()
    if 'deployments' not in tables and 'bentos' not in tables:
        Base.metadata.create_all(bind=bind)
    else:
        op.add_column('deployments', sa.Column('last_updated_at', sa.DateTime))


def downgrade():
    pass
