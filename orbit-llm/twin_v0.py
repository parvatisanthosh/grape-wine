"""Performance Twin v0 (Layer 1): predict TTFT, TPOT and peak memory
before running a request, and measure how good those predictions are.

Two predictors are compared:

- analytical: per-configuration linear laws taken from the findings
  (TTFT grows linearly with prompt length, a prefix-cache hit has a
  near-constant TTFT, TPOT grows slowly with context, memory grows
  linearly with tokens), fitted by least squares;
- gbm: gradient-boosted trees on the same pre-run features.

Evaluation holds out whole configurations (GroupKFold on config_key),
so every prediction is for a configuration+workload the model has never
seen. Errors are compared with the noise floor: how far a run lands from
the median of the *other* runs of the same configuration.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from build_dataset import (
    CONFIG_FEATURES,
    MACHINE_FEATURES,
    OUTPUT_CSV as DATASET_CSV,
    TARGETS,
    WORKLOAD_FEATURES,
)


EVALUATION_CSV = DATASET_CSV.with_name("twin_v0_evaluation.csv")
PREDICTIONS_CSV = DATASET_CSV.with_name("twin_v0_predictions.csv")

FOLDS = 5
SEED = 0

# Bytes per token per KV element type, relative to u8.
KV_BYTES = {"u8": 1, "f16": 2, "bf16": 2, "f32": 4, "u4": 0.5}


# ---------------------------------------------------------------------------
# Analytical predictor


class GroupedLinearLaw:
    """y = slope * x + intercept, fitted separately per group.

    Groups are tried from most to least specific; a configuration whose
    exact group was never seen falls back to a coarser one.

    A group observed at a single x cannot give a slope. single_point says
    what to assume then: "constant" (y does not depend on x) or
    "proportional" (y = k * x, e.g. prefill time vs prompt length).
    """

    def __init__(self, x_function, group_levels, single_point="constant"):
        self.x_function = x_function
        self.group_levels = group_levels
        self.single_point = single_point
        self.fits = []

    def fit(self, frame, target):
        x = self.x_function(frame)
        self.fits = []

        for level in self.group_levels:
            level_fits = {}

            for key, index in frame.groupby(level).groups.items():
                xs = x.loc[index].to_numpy(dtype=float)
                ys = frame.loc[index, target].to_numpy(dtype=float)

                if len(np.unique(xs)) >= 2:
                    slope, intercept = np.polyfit(xs, ys, 1)
                elif self.single_point == "proportional":
                    slope, intercept = float(np.median(ys / xs)), 0.0
                else:
                    slope, intercept = 0.0, float(np.median(ys))

                level_fits[key if isinstance(key, tuple) else (key,)] = (
                    slope,
                    intercept,
                )

            self.fits.append(level_fits)

        self.global_fit = (0.0, float(frame[target].median()))
        return self

    def predict(self, frame):
        x = self.x_function(frame)
        predictions = []

        for row_index, row in frame.iterrows():
            slope, intercept = self.global_fit

            for level, level_fits in zip(self.group_levels, self.fits):
                key = tuple(row[column] for column in level)

                if key in level_fits:
                    slope, intercept = level_fits[key]
                    break

            predictions.append(slope * x.loc[row_index] + intercept)

        return np.maximum(np.array(predictions), 1e-3)


class AnalyticalTwin:
    PLACEMENT = ["precision", "device", "backend", "cores", "threads"]

    def __init__(self):
        self.ttft = GroupedLinearLaw(
            lambda frame: frame["prompt_tokens"],
            [self.PLACEMENT, ["precision", "device"], ["device"]],
            single_point="proportional",
        )
        # A cache hit only computes the uncached tail of the prompt, so its
        # TTFT is near-constant; it gets its own law.
        self.ttft_hit = GroupedLinearLaw(
            lambda frame: frame["prompt_tokens"],
            [["precision", "device"], ["device"]],
        )
        self.tpot = GroupedLinearLaw(
            lambda frame: frame["prompt_tokens"] + frame["output_tokens"] / 2,
            [self.PLACEMENT, ["precision", "device"], ["device"]],
        )
        self.memory = GroupedLinearLaw(
            lambda frame: frame["prompt_tokens"] + frame["output_tokens"],
            [
                ["precision", "device", "backend", "kv_precision"],
                ["precision", "device", "backend"],
                ["precision", "device"],
            ],
        )

    def fit(self, frame):
        misses = frame[frame["cache_hit"] == 0]
        hits = frame[frame["cache_hit"] == 1]

        self.ttft.fit(misses, "ttft_ms")
        self.has_hits = len(hits) > 0
        if self.has_hits:
            self.ttft_hit.fit(hits, "ttft_ms")

        self.tpot.fit(frame, "tpot_ms_per_token")
        self.memory.fit(frame, "sampled_peak_rss_mib")
        return self

    def predict(self, frame):
        ttft = self.ttft.predict(frame)
        if self.has_hits:
            hit_mask = frame["cache_hit"].to_numpy() == 1
            if hit_mask.any():
                ttft[hit_mask] = self.ttft_hit.predict(frame[hit_mask])

        return {
            "ttft_ms": ttft,
            "tpot_ms_per_token": self.tpot.predict(frame),
            "sampled_peak_rss_mib": self.memory.predict(frame),
        }


# ---------------------------------------------------------------------------
# Gradient-boosted predictor


class BoostedTwin:
    def __init__(self, features):
        self.features = features
        self.models = {}

    def _matrix(self, frame):
        matrix = frame[self.features].copy()

        for column in matrix.columns:
            if matrix[column].dtype == object or matrix[column].dtype == bool:
                matrix[column] = matrix[column].astype(str).astype("category")

        return matrix

    def fit(self, frame):
        matrix = self._matrix(frame)
        self.categories = {
            column: matrix[column].cat.categories
            for column in matrix.columns
            if str(matrix[column].dtype) == "category"
        }

        for target in TARGETS:
            model = HistGradientBoostingRegressor(
                categorical_features="from_dtype",
                max_iter=300,
                learning_rate=0.05,
                min_samples_leaf=5,
                random_state=SEED,
            )
            # Errors are relative, so learn in log space.
            model.fit(matrix, np.log(frame[target]))
            self.models[target] = model

        return self

    def predict(self, frame):
        matrix = self._matrix(frame)

        for column, categories in self.categories.items():
            matrix[column] = pd.Categorical(
                matrix[column].astype(str),
                categories=categories,
            )

        return {
            target: np.exp(model.predict(matrix))
            for target, model in self.models.items()
        }


# ---------------------------------------------------------------------------
# Evaluation


def noise_floor(frame, target):
    """Prediction from the median of the other runs of the same config."""
    predictions = np.empty(len(frame))

    for _, index in frame.groupby("config_key").groups.items():
        values = frame.loc[index, target].to_numpy(dtype=float)
        positions = frame.index.get_indexer(index)

        for offset, position in enumerate(positions):
            others = np.delete(values, offset)
            predictions[position] = (
                np.median(others) if len(others) else np.nan
            )

    return predictions


def error_summary(actual, predicted):
    mask = ~np.isnan(predicted)
    relative = np.abs(predicted[mask] - actual[mask]) / actual[mask]

    return {
        "mape_percent": 100 * relative.mean(),
        "median_ape_percent": 100 * np.median(relative),
        "within_20_percent": 100 * (relative <= 0.20).mean(),
    }


def cross_validated_predictions(frame, make_model):
    predictions = {target: np.empty(len(frame)) for target in TARGETS}
    splitter = GroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    for train_index, test_index in splitter.split(
        frame,
        groups=frame["config_key"],
    ):
        model = make_model().fit(frame.iloc[train_index])
        fold_predictions = model.predict(frame.iloc[test_index])

        for target in TARGETS:
            predictions[target][test_index] = fold_predictions[target]

    return predictions


def main():
    frame = pd.read_csv(DATASET_CSV)

    all_features = CONFIG_FEATURES + WORKLOAD_FEATURES + MACHINE_FEATURES
    no_machine = CONFIG_FEATURES + WORKLOAD_FEATURES

    predictors = {
        "analytical": AnalyticalTwin,
        "gbm": lambda: BoostedTwin(all_features),
        "gbm_no_machine_state": lambda: BoostedTwin(no_machine),
    }

    predictions = {
        name: cross_validated_predictions(frame, make_model)
        for name, make_model in predictors.items()
    }

    rows = []

    for target in TARGETS:
        actual = frame[target].to_numpy(dtype=float)

        rows.append(
            {
                "target": target,
                "predictor": "noise_floor",
                **error_summary(actual, noise_floor(frame, target)),
            }
        )

        for name in predictors:
            rows.append(
                {
                    "target": target,
                    "predictor": name,
                    **error_summary(actual, predictions[name][target]),
                }
            )

    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(EVALUATION_CSV, index=False)

    output = frame[["experiment", "config_key"] + TARGETS].copy()
    for name in predictors:
        for target in TARGETS:
            output[f"{name}_{target}"] = predictions[name][target]
    output.to_csv(PREDICTIONS_CSV, index=False)

    print(
        f"\nPERFORMANCE TWIN v0 — {len(frame)} runs, "
        f"{frame['config_key'].nunique()} configurations, "
        f"{FOLDS}-fold CV holding out whole configurations\n"
    )
    print(
        evaluation.to_string(
            index=False,
            float_format=lambda value: f"{value:.1f}",
        )
    )
    print(f"\nEvaluation saved to: {EVALUATION_CSV.resolve()}")
    print(f"Predictions saved to: {PREDICTIONS_CSV.resolve()}")


if __name__ == "__main__":
    main()
