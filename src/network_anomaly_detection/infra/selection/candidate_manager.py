



class CandidateManager:

    def __init__(
        self,
        registry,
        logger,
        candidate_pool_size=5,
        min_pr_auc=0.70,
        max_candidates_per_model=2,
    ):
        self.registry = registry
        self.logger = logger

        self.candidate_pool_size = candidate_pool_size
        self.min_pr_auc = min_pr_auc
        self.max_candidates_per_model = max_candidates_per_model

    # ---------------------------------------------------------
    # Candidate selection
    # ---------------------------------------------------------

    def should_retain(
        self,
        experiment_id,
        model_family,
        val_pr_auc,
    ):
        if val_pr_auc < self.min_pr_auc:
            return False

        candidates = self.registry.get_candidates(
            experiment_id
        )

        family_candidates = [
            c for c in candidates
            if c.model_family == model_family
        ]

        # Model-family limit
        if len(family_candidates) >= self.max_candidates_per_model:

            weakest_family = min(
                family_candidates,
                key=lambda c: c.val_pr_auc,
            )

            if val_pr_auc <= weakest_family.val_pr_auc:
                return False

        # Pool has room
        if len(candidates) < self.candidate_pool_size:
            return True

        # Pool full: must beat weakest candidate
        weakest = min(
            candidates,
            key=lambda c: c.val_pr_auc,
        )

        return val_pr_auc > weakest.val_pr_auc

    # ---------------------------------------------------------
    # Registration
    # ---------------------------------------------------------

    def register_candidate(
        self,
        experiment_id,
        run_id,
        model_family,
        val_pr_auc,
        artifact_path="model",
    ):
        self.registry.add(
            experiment_id=experiment_id,
            run_id=run_id,
            model_family=model_family,
            val_pr_auc=val_pr_auc,
            artifact_path=artifact_path,
        )

        self._enforce_limits(experiment_id)

    # ---------------------------------------------------------
    # Pool management
    # ---------------------------------------------------------

    def _enforce_limits(self, experiment_id):

        candidates = self.registry.get_candidates(
            experiment_id
        )

        while len(candidates) > self.candidate_pool_size:

            weakest = min(
                candidates,
                key=lambda c: c.val_pr_auc,
            )

            self._evict_candidate(weakest)

            candidates = self.registry.get_candidates(
                experiment_id
            )

    # ---------------------------------------------------------
    # Eviction
    # ---------------------------------------------------------

    def _evict_candidate(self, candidate):

        self.logger.delete_artifact(
            run_id=candidate.run_id,
            artifact_path=candidate.artifact_path,
        )

        self.registry.evict(
            candidate.run_id
        )