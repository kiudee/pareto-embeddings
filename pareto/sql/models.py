import time
from enum import Enum

from sqlalchemy import ARRAY, JSON, BigInteger, Column
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import ForeignKey, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from pareto.sql.base_sql_model import Base

__all__ = ["Job", "Result", "JobStatus"]
SCHEMA_NAME = "pareto"


class JobStatus(Enum):
    QUEUED = 1
    ALLOCATED = 2
    FINISHED = 3
    ERRORED = 4


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = {"schema": SCHEMA_NAME}
    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    queue_time = Column(BigInteger, default=int(time.time()))
    start_time = Column(BigInteger, nullable=True, default=None)
    end_time = Column(BigInteger, nullable=True, default=None)
    dataset = Column(String(256), nullable=False)
    learner = Column(String(256), nullable=False)
    fold_id = Column(Integer, nullable=False)
    outer_cv_folds = Column(Integer, nullable=False, default=5)
    outer_cv_test = Column(Numeric, nullable=False, default=0.1)
    inner_cv_folds = Column(Integer, nullable=False, default=1)
    inner_cv_val = Column(Numeric, nullable=False, default=0.2)
    dataset_opts = Column(JSON, default=None)
    learner_opts = Column(JSON, default=None)
    hyperopt_opts = Column(JSON, default=None)
    search_space = Column(JSON, default=None)
    metrics = Column(ARRAY(String(64)), nullable=False)
    random_seed = Column(Integer, nullable=False, default=123)
    status = Column(SQLEnum(JobStatus), default=JobStatus.QUEUED, nullable=False)
    host_name = Column(String(100), nullable=True, default=None)
    req_id = Column(String(100), nullable=True, default=None)

    result = relationship("Result", uselist=False, back_populates="job")

    def __repr__(self):
        return f"<Job (id={self.id}, name={self.name}, dataset={self.dataset}, learner={self.learner})>"

    def to_dict(self):
        return {
            "dataset": self.dataset,
            "learner": self.learner,
            "fold_id": self.fold_id,
            "outer_cv_folds": self.outer_cv_folds,
            "outer_cv_test": self.outer_cv_test,
            "inner_cv_folds": self.inner_cv_folds,
            "inner_cv_val": self.inner_cv_val,
            "dataset_opts": self.dataset_opts,
            "learner_opts": self.learner_opts,
            "hyperopt_opts": self.hyperopt_opts,
            "search_space": self.search_space,
            "metrics": self.metrics,
            "random_seed": self.random_seed,
        }


class Result(Base):
    __tablename__ = "results"
    __table_args__ = {"schema": SCHEMA_NAME}
    id = Column(Integer, primary_key=True)
    metrics = Column(JSON, nullable=False)
    model_uuid = Column(UUID(as_uuid=True), unique=True, nullable=False)
    best_params = Column(JSON, nullable=True, default=None)
    job_id = Column(Integer, ForeignKey(f"{SCHEMA_NAME}.jobs.id"))

    job = relationship("Job", back_populates="result")

    def __repr__(self):
        return f"<Result (id={self.id}, uuid={self.model_uuid}, job_id={self.job_id})>"
