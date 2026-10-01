from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column




class Base(DeclarativeBase):
    pass


class Candidate(Base):

    __tablename__ = "candidate_pool"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    experiment_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    run_id: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False,
    )

    model_family: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    val_pr_auc: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    artifact_path: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    state: Mapped[str] = mapped_column(
        String,
        default="retained",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    inference_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    explainability: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
