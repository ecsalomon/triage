"""Add subsets

Revision ID: 5bf4cdc4d4dc
Revises: 38f37d013686
Create Date: 2018-12-04 10:26:49.878110

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '5bf4cdc4d4dc'
down_revision = '38f37d013686'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('subsets',
        sa.Column('subset_hash', sa.String(), nullable=False),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('subset_hash'),
        schema='model_metadata'
    )
    op.create_table('subset_evaluations',
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('evaluation_start_time', sa.DateTime(), nullable=False),
        sa.Column('evaluation_end_time', sa.DateTime(), nullable=False),
        sa.Column('as_of_date_frequency', sa.Interval(), nullable=False),
        sa.Column('subset_hash', sa.String(), nullable=False),
        sa.Column('metric', sa.String(), nullable=False),
        sa.Column('parameter', sa.String(), nullable=False),
        sa.Column('value', sa.Numeric(), nullable=True),
        sa.Column('num_labeled_examples', sa.Integer(), nullable=True),
        sa.Column('num_labeled_above_threshold', sa.Integer(), nullable=True),
        sa.Column('num_positive_labels', sa.Integer(), nullable=True),
        sa.Column('sort_seed', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['model_metadata.models.model_id'], ),
        sa.ForeignKeyConstraint(['subset_hash'], ['model_metadata.subsets.subset_hash'], ),
        sa.PrimaryKeyConstraint(
            'model_id',
            'evaluation_start_time',
            'evaluation_end_time',
            'as_of_date_frequency',
            'subset_hash',
            'metric',
            'parameter'
        ),
        schema='test_results'
    )
    op.create_table('subset_evaluations',
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('evaluation_start_time', sa.DateTime(), nullable=False),
        sa.Column('evaluation_end_time', sa.DateTime(), nullable=False),
        sa.Column('as_of_date_frequency', sa.Interval(), nullable=False),
        sa.Column('subset_hash', sa.String(), nullable=False),
        sa.Column('metric', sa.String(), nullable=False),
        sa.Column('parameter', sa.String(), nullable=False),
        sa.Column('value', sa.Numeric(), nullable=True),
        sa.Column('num_labeled_examples', sa.Integer(), nullable=True),
        sa.Column('num_labeled_above_threshold', sa.Integer(), nullable=True),
        sa.Column('num_positive_labels', sa.Integer(), nullable=True),
        sa.Column('sort_seed', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['model_metadata.models.model_id'], ),
        sa.ForeignKeyConstraint(['subset_hash'], ['model_metadata.subsets.subset_hash'], ),
        sa.PrimaryKeyConstraint(
            'model_id',
            'evaluation_start_time',
            'evaluation_end_time',
            'as_of_date_frequency',
            'subset_hash',
            'metric',
            'parameter'
        ),
        schema='train_results'
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('subset_evaluations', schema='train_results')
    op.drop_table('subset_evaluations', schema='test_results')
    op.drop_table('subsets', schema='model_metadata')
    # ### end Alembic commands ###
