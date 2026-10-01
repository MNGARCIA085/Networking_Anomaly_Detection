from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from .models import Base, Candidate


class CandidateRegistry:

    def __init__(self, db_url):

        self.engine = create_engine(db_url)

        Base.metadata.create_all(self.engine)

    def add(
        self,
        experiment_id,
        run_id,
        model_family,
        val_pr_auc,
        artifact_path=None,
    ):

        candidate = Candidate(
            experiment_id=experiment_id,
            run_id=run_id,
            model_family=model_family,
            val_pr_auc=val_pr_auc,
            artifact_path=artifact_path,
        )

        with Session(self.engine) as session:
            session.add(candidate)
            session.commit()

    def get_candidates(self, experiment_id):

        stmt = (
            select(Candidate)
            .where(
                Candidate.experiment_id == experiment_id,
                Candidate.state == "retained",
            )
            .order_by(Candidate.val_pr_auc.desc())
        )

        with Session(self.engine) as session:
            return session.scalars(stmt).all()



    def get_all(self, experiment_id):

        stmt = (
            select(Candidate)
            .where(
                Candidate.experiment_id == experiment_id,
            )
            .order_by(Candidate.val_pr_auc.desc())
        )

        with Session(self.engine) as session:
            return session.scalars(stmt).all()



    def get_worst(self, experiment_id):

        stmt = (
            select(Candidate)
            .where(
                Candidate.experiment_id == experiment_id,
            )
            .order_by(Candidate.val_pr_auc.asc())
            .limit(1)
        )

        with Session(self.engine) as session:
            return session.scalars(stmt).first()

    def count(self, experiment_id):

        stmt = (
            select(Candidate)
            .where(
                Candidate.experiment_id == experiment_id,
            )
        )

        with Session(self.engine) as session:
            return len(session.scalars(stmt).all())

    def evict(self, run_id):

        stmt = (
            update(Candidate)
            .where(Candidate.run_id == run_id)
            .values(state="evicted")
        )

        with Session(self.engine) as session:
            session.execute(stmt)
            session.commit()



    def update_selection_metrics(
        self,
        run_id,
        inference_ms=None,
        explainability=None,
    ):
        with Session(self.engine) as session:

            candidate = session.scalar(
                select(Candidate).where(
                    Candidate.run_id == run_id
                )
            )

            if candidate is None:
                raise ValueError(
                    f"Candidate not found: {run_id}"
                )

            candidate.inference_ms = inference_ms
            candidate.explainability = explainability

            session.commit()


    #---------------here just for now (LATER MOVE, is presentation not persisence)----------#
    def print_candidates(self, experiment_id, include_evicted=False):

        if include_evicted:
            candidates = self.get_all(experiment_id)
        else:
            candidates = self.get_retained(experiment_id)

        if not candidates:
            print("No candidates found.")
            return

        print()
        print(f"Experiment: {experiment_id}")
        print()

        print(
            f"{'Rank':<6}"
            f"{'Model':<15}"
            f"{'PR-AUC':<10}"
            f"{'State':<12}"
            f"{'Run ID'}"
        )

        print("-" * 70)

        for rank, candidate in enumerate(candidates, start=1):

            print(
                f"{rank:<6}"
                f"{candidate.model_family:<15}"
                f"{candidate.val_pr_auc:<10.4f}"
                f"{candidate.state:<12}"
                f"{candidate.run_id}"
            )

        print()



    def candidate_records(
        self,
        experiment_id,
        include_evicted=False,
    ):
        if include_evicted:
            candidates = self.get_all(experiment_id)
        else:
            candidates = self.get_candidates(experiment_id)

        return [
            {
                "run_id": candidate.run_id,
                "model_family": candidate.model_family,
                "val_pr_auc": candidate.val_pr_auc,
                "inference_ms": candidate.inference_ms,
                "explainability": candidate.explainability,
                "state": candidate.state,
                "artifact_path": candidate.artifact_path,
                "created_at": candidate.created_at,
            }
            for candidate in candidates
        ]