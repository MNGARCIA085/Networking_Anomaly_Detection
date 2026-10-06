from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# evaluator only evaluates
# dont need x_train or x_val.....


class Evaluator:

    def evaluate(
        self,
        scores, # always evaluated
        y_true=None,
        predictions=None, # optional!
    ):
        result = {
            "mean_score": scores.mean(),
        }

        if y_true is None:
            return result

        # Threshold-independent metrics
        result["auc"] = roc_auc_score(
            y_true,
            scores,
        )

        result["pr_auc"] = average_precision_score(
            y_true,
            scores,
        )

        # Threshold-dependent metrics
        if predictions is not None:

            result["precision"] = precision_score(
                y_true,
                predictions,
                zero_division=0,
            )

            result["recall"] = recall_score(
                y_true,
                predictions,
                zero_division=0,
            )

            result["f1"] = f1_score(
                y_true,
                predictions,
                zero_division=0,
            )

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1],
            ).ravel()

            result["tn"] = tn
            result["fp"] = fp
            result["fn"] = fn
            result["tp"] = tp

        return result



