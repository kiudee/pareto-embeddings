"""initial db state

Revision ID: 5e588093a806
Revises: 
Create Date: 2020-03-30 11:33:08.637648

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "5e588093a806"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("jobs", sa.Column("host_name", sa.String(length=100), nullable=True))
    op.drop_constraint("results_job_id_fkey", "results", type_="foreignkey")
    op.create_foreign_key(
        None,
        "results",
        "jobs",
        ["job_id"],
        ["id"],
        source_schema="pareto",
        referent_schema="pareto",
    )


def downgrade():
    op.drop_constraint(None, "results", schema="pareto", type_="foreignkey")
    op.create_foreign_key("results_job_id_fkey", "results", "jobs", ["job_id"], ["id"])
    op.drop_column("jobs", "host_name")
