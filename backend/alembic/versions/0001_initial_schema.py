"""Initial SovereignAI API schema.

Revision ID: 0001_initial_schema
Revises:
"""
from alembic import op
from backend.app.database.base import Base
from backend.app import models  # noqa: F401
revision="0001_initial_schema"; down_revision=None; branch_labels=None; depends_on=None
def upgrade():
    # Kept in the migration (not application startup) so production schema
    # changes are versioned and Alembic can stamp/upgrade existing databases.
    for table in Base.metadata.sorted_tables: table.create(op.get_bind(),checkfirst=True)
def downgrade():
    for table in reversed(Base.metadata.sorted_tables): table.drop(op.get_bind(),checkfirst=True)
