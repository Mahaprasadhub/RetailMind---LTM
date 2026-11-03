"""Reusable, inference-safe domain features for Telco churn prediction."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


ENGINEERED_NUMERIC_FEATURES = [
    "is_month_to_month",
    "has_security_service",
    "has_tech_support",
    "has_multiple_services",
    "average_charge_per_tenure",
    "monthly_to_average_charge_ratio",
]
ENGINEERED_CATEGORICAL_FEATURES = ["tenure_group"]
EXPECTED_RAW_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class ChurnFeatureEngineer(TransformerMixin, BaseEstimator):
    """Add domain features while keeping the original columns available."""

    def __init__(self, add_features=True):
        self.add_features = add_features

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ChurnFeatureEngineer expects a pandas DataFrame.")
        self.feature_names_in_ = X.columns.to_numpy()
        if self.add_features:
            required = {
                "Contract",
                "OnlineSecurity",
                "TechSupport",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineBackup",
                "DeviceProtection",
                "StreamingTV",
                "StreamingMovies",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
            }
            missing = required.difference(X.columns)
            if missing:
                raise ValueError(
                    "Cannot create churn features; missing columns: "
                    + ", ".join(sorted(missing))
                )
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ChurnFeatureEngineer expects a pandas DataFrame.")
        missing = set(self.feature_names_in_).difference(X.columns)
        if missing:
            raise ValueError(
                "Input is missing fitted feature columns: "
                + ", ".join(sorted(missing))
            )

        transformed = X.loc[:, self.feature_names_in_].copy()
        if not self.add_features:
            return transformed

        tenure = pd.to_numeric(transformed["tenure"], errors="coerce")
        monthly_charges = pd.to_numeric(
            transformed["MonthlyCharges"], errors="coerce"
        )
        total_charges = pd.to_numeric(
            transformed["TotalCharges"], errors="coerce"
        )

        transformed["tenure_group"] = pd.cut(
            tenure,
            bins=[-1, 6, 12, 36, float("inf")],
            labels=[
                "New customer",
                "Short-term customer",
                "Medium-term customer",
                "Long-term customer",
            ],
        ).astype("object")
        transformed["is_month_to_month"] = (
            transformed["Contract"].eq("Month-to-month").astype("int64")
        )
        transformed["has_security_service"] = (
            transformed["OnlineSecurity"].eq("Yes").astype("int64")
        )
        transformed["has_tech_support"] = (
            transformed["TechSupport"].eq("Yes").astype("int64")
        )

        service_flags = pd.DataFrame(
            {
                "phone": transformed["PhoneService"].eq("Yes"),
                "multiple_lines": transformed["MultipleLines"].eq("Yes"),
                "internet": transformed["InternetService"].notna()
                & transformed["InternetService"].ne("No"),
                "online_security": transformed["OnlineSecurity"].eq("Yes"),
                "online_backup": transformed["OnlineBackup"].eq("Yes"),
                "device_protection": transformed["DeviceProtection"].eq("Yes"),
                "tech_support": transformed["TechSupport"].eq("Yes"),
                "streaming_tv": transformed["StreamingTV"].eq("Yes"),
                "streaming_movies": transformed["StreamingMovies"].eq("Yes"),
            },
            index=transformed.index,
        )
        transformed["has_multiple_services"] = service_flags.sum(axis=1).astype(
            "int64"
        )

        average_charge = total_charges.div(tenure.where(tenure > 0))
        transformed["average_charge_per_tenure"] = average_charge
        transformed["monthly_to_average_charge_ratio"] = monthly_charges.div(
            average_charge.where(average_charge > 0)
        )
        return transformed

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        raw_features = list(self.feature_names_in_)
        if self.add_features:
            raw_features.extend(ENGINEERED_CATEGORICAL_FEATURES)
            raw_features.extend(ENGINEERED_NUMERIC_FEATURES)
        return pd.Index(raw_features, dtype="object").to_numpy()
