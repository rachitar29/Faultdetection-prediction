
import re
from difflib import SequenceMatcher

import pandas as pd


# Canonical names expected by the AI4I model.
# Keep these aliases as the default mapping configuration.
SENSOR_ALIASES = {
    "Air temperature [K]": [
        "air temperature [k]",
        "air temperature",
        "air temp",
        "air_temperature",
        "temperature",
        "ambient temperature",
    ],
    "Process temperature [K]": [
        "process temperature [k]",
        "process temperature",
        "process temp",
        "process_temperature",
    ],
    "Rotational speed [rpm]": [
        "rotational speed [rpm]",
        "rotational speed",
        "rotation speed",
        "rpm",
        "motor speed",
        "spindle speed",
    ],
    "Torque [Nm]": [
        "torque [nm]",
        "torque",
        "motor torque",
    ],
    "Tool wear [min]": [
        "tool wear [min]",
        "tool wear",
        "tool usage",
        "tool life usage",
    ],
    "Type": [
        "type",
        "machine type",
        "product type",
        "equipment type",
    ],
    "Machine failure": [
        "machine failure",
        "failure",
        "failure status",
        "fault",
        "fault status",
        "is failure",
    ],
}


DEFAULT_REQUIRED_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]


def normalize_column_name(name):
    """Normalize column names for safe alias matching."""
    name = str(name).strip().lower()
    name = name.replace("_", " ")
    name = re.sub(r"[\[\](){}]", " ", name)
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def build_alias_lookup(aliases=None):
    """Build a normalized alias-to-canonical lookup."""
    aliases = SENSOR_ALIASES if aliases is None else aliases

    if not isinstance(aliases, dict):
        raise TypeError("Aliases must be supplied as a dictionary.")

    lookup = {}

    for canonical, alias_list in aliases.items():
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(
                "Every canonical column name must be a non-empty string."
            )

        if not isinstance(alias_list, (list, tuple, set)):
            raise TypeError(
                f"Aliases for '{canonical}' must be a list, tuple or set."
            )

        for alias in [canonical, *alias_list]:
            normalized = normalize_column_name(alias)

            if not normalized:
                raise ValueError(
                    f"An empty alias was found for '{canonical}'."
                )

            existing = lookup.get(normalized)

            if existing is not None and existing != canonical:
                raise ValueError(
                    f"Alias conflict detected: '{normalized}' maps to "
                    f"both '{existing}' and '{canonical}'."
                )

            lookup[normalized] = canonical

    return lookup


ALIAS_LOOKUP = build_alias_lookup()


def map_sensor_columns(
    dataframe,
    required_features=None,
    aliases=None,
):
    """
    Map uploaded sensor columns to canonical column names.

    Unknown columns are preserved.

    Fuzzy matching is not used for automatic renaming.

    Args:
        dataframe: Input pandas DataFrame.
        required_features: Canonical columns required by the caller.
            Defaults to the AI4I model's sensor features.
        aliases: Optional custom alias dictionary. When omitted,
            the default AI4I aliases are used.

    Returns:
        Dictionary containing:
            dataframe
            mapping
            unmatched_columns
            missing_features
            mapped_count
            total_columns
            is_valid

    Raises:
        TypeError: If the input is not a DataFrame.
        ValueError: If the input is empty or duplicate mappings exist.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Uploaded dataset is empty.")

    if dataframe.columns.duplicated().any():
        duplicate_columns = (
            dataframe.columns[
                dataframe.columns.duplicated(keep=False)
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate input column names detected: {duplicate_columns}"
        )

    # Prevent collisions in the string-keyed mapping report.
    display_names = [str(column) for column in dataframe.columns]

    if len(display_names) != len(set(display_names)):
        raise ValueError(
            "Different input column labels have the same string "
            "representation. Rename them before mapping."
        )

    alias_lookup = (
        ALIAS_LOOKUP
        if aliases is None
        else build_alias_lookup(aliases)
    )

    mapping = {}
    unmatched_columns = []
    used_canonical_names = {}
    rename_mapping = {}

    for original_column in dataframe.columns:
        original_name = str(original_column)
        normalized = normalize_column_name(original_name)

        canonical = alias_lookup.get(normalized)

        if canonical is None:
            mapping[original_name] = None
            unmatched_columns.append(original_name)
            continue

        if canonical in used_canonical_names:
            previous = used_canonical_names[canonical]

            raise ValueError(
                f"Duplicate mapping detected: '{previous}' and "
                f"'{original_name}' both map to '{canonical}'."
            )

        used_canonical_names[canonical] = original_name
        mapping[original_name] = canonical
        rename_mapping[original_column] = canonical

    mapped_dataframe = dataframe.rename(
        columns=rename_mapping
    )

    if required_features is None:
        required_features = DEFAULT_REQUIRED_FEATURES.copy()
    else:
        required_features = list(required_features)

    if not all(
        isinstance(feature, str) and feature.strip()
        for feature in required_features
    ):
        raise ValueError(
            "required_features must contain non-empty strings."
        )

    if len(required_features) != len(set(required_features)):
        raise ValueError(
            "required_features contains duplicate column names."
        )

    missing_features = [
        feature
        for feature in required_features
        if feature not in mapped_dataframe.columns
    ]

    return {
        "dataframe": mapped_dataframe,
        "mapping": mapping,
        "unmatched_columns": unmatched_columns,
        "missing_features": missing_features,
        "mapped_count": len(rename_mapping),
        "total_columns": len(dataframe.columns),
        "is_valid": len(missing_features) == 0,
    }


def suggest_column_matches(column_name, limit=3):
    """
    Suggest possible canonical column matches.

    Suggestions are informational only. This function never renames
    columns or changes a DataFrame.
    """

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    normalized = normalize_column_name(column_name)

    if not normalized:
        return []

    scores = []

    for alias, canonical in ALIAS_LOOKUP.items():
        score = SequenceMatcher(
            None,
            normalized,
            alias,
        ).ratio()

        scores.append({
            "canonical_name": canonical,
            "matched_alias": alias,
            "similarity": round(score, 3),
        })

    scores.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return scores[:limit]
