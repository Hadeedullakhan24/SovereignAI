"""initial backend schema

Revision ID: 0001_initial
"""
revision="0001_initial"
down_revision=None
branch_labels=None
depends_on=None
from alembic import op
from backend.app.database.base import Base
from backend.app.database import models
def upgrade(): Base.metadata.create_all(op.get_bind())
def downgrade(): Base.metadata.drop_all(op.get_bind())
