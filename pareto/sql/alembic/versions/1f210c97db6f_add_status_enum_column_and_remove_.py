"""Add status Enum column and remove allocated flag

Revision ID: 1f210c97db6f
Revises: 5e588093a806
Create Date: 2020-04-02 10:58:39.695327

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "1f210c97db6f"
down_revision = "5e588093a806"
branch_labels = None
depends_on = None


def upgrade():
    jobstatus = postgresql.ENUM(
        "QUEUED", "ALLOCATED", "FINISHED", "ERRORED", name="jobstatus"
    )
    jobstatus.create(op.get_bind())

    op.add_column(
        "jobs",
        sa.Column(
            "status",
            sa.Enum("QUEUED", "ALLOCATED", "FINISHED", "ERRORED", name="jobstatus"),
            nullable=False,
            server_default="QUEUED",
        ),
    )
    op.drop_column("jobs", "allocated")
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
    op.add_column(
        "jobs",
        sa.Column("allocated", sa.BOOLEAN(), autoincrement=False, nullable=False),
    )
    op.drop_column("jobs", "status")
    jobstatus = postgresql.ENUM(
        "QUEUED", "ALLOCATED", "FINISHED", "ERRORED", name="jobstatus"
    )
    jobstatus.drop(op.get_bind())
