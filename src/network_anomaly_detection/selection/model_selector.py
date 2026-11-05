class ModelSelector:

    def __init__(self, registry):
        self.registry = registry

    def get_candidates(self, experiment_id):
        return self.registry.get_candidates( # bfore get_retained; now get_cnds=get_ret
            experiment_id
        )


    def select(
        self,
        experiment_id,
        pr_auc_tolerance=0.005,
        max_inference_ms=None,
        min_explainability=None,
    ):
        candidates = self.get_candidates(
            experiment_id
        )

        if not candidates:
            raise ValueError(
                "No retained candidates available."
            )

        # --------------------------------------------------
        # 1. Optional hard constraints
        # --------------------------------------------------

        if max_inference_ms is not None:
            candidates = [
                c for c in candidates
                if c.inference_ms is not None
                and c.inference_ms <= max_inference_ms
            ]

        if min_explainability is not None:
            candidates = [
                c for c in candidates
                if c.explainability is not None
                and c.explainability >= min_explainability
            ]

        if not candidates:
            raise ValueError(
                "No candidates satisfy the constraints."
            )

        # --------------------------------------------------
        # 2. Find best detection performance
        # --------------------------------------------------

        best_pr_auc = max(
            c.val_pr_auc
            for c in candidates
        )

        # --------------------------------------------------
        # 3. Keep models close enough to the best
        # --------------------------------------------------

        candidates = [
            c for c in candidates
            if c.val_pr_auc
            >= best_pr_auc - pr_auc_tolerance
        ]

        # --------------------------------------------------
        # 4. Among comparable models, prefer faster model
        # --------------------------------------------------

        selected = min(
            candidates,
            key=lambda c: c.inference_ms
        )

        return selected

