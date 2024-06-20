"""Add req_id and best_params columns

Revision ID: fb77561988bd
Revises: 1f210c97db6f
Create Date: 2020-04-17 13:43:27.129388

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "fb77561988bd"
down_revision = "1f210c97db6f"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("jobs", sa.Column("req_id", sa.String(length=100), nullable=True))
    op.add_column("results", sa.Column("best_params", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("results", "best_params")
    op.drop_column("jobs", "req_id")
