"""update lock table column

Revision ID: 370a2bf2e594
Revises: af18f0b83286
Create Date: 2021-06-15 13:41:18.015802

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '370a2bf2e594'
down_revision = 'af18f0b83286'
branch_labels = None
depends_on = None


def upgrade():
    # SQLite does not support making modifications to existing columns.
    # Alembic "batch mode" is designed to overcome this limitation.
    # https://alembic.sqlalchemy.org/en/latest/batch.html#batch-mode-with-autogenerate
    with op.batch_alter_table('locks') as batch_op:
        batch_op.alter_column(
            column_name='resource_id', type_=sa.String, existing_type=sa.INTEGER
        )


def downgrade():
    with op.batch_alter_table('locks') as batch_op:
        batch_op.alter_column(
            column_name='resource_id', type_=sa.INTEGER, existing_type=sa.String
        )
