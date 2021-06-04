"""create lock table

Revision ID: af18f0b83286
Revises: 719dd2aacc9a
Create Date: 2021-05-18 18:33:10.622364

"""
from alembic import op


# revision identifiers, used by Alembic.
from bentoml.yatai.db.stores.lock import Lock

revision = 'af18f0b83286'
down_revision = '719dd2aacc9a'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    Lock.__table__.create(bind)


def downgrade():
    op.drop_table('locks')
