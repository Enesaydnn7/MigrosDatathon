import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTENC


def clean_isolation_forest(train, anomalies):
    # Anomaly detection among response = 0 using sklearn.IsolationForest

    clf = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=anomalies,
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )

    _train = train.copy()
    _train.drop(
        columns=[
            "level1_relevant_category_volume_per_day",
            "level2_relevant_category_volume_per_day",
            "level3_relevant_category_volume_per_day",
            "level4_relevant_category_volume_per_day",
            "level1_relevant_category_quantity_per_day",
            "level2_relevant_category_quantity_per_day",
            "level3_relevant_category_quantity_per_day",
            "level4_relevant_category_quantity_per_day",
            "discount_per_day",
            "total_money_spent_per_day",
        ],
        inplace=True,
    )
    _train["gender"].fillna("K", inplace=True)
    _train.fillna(0, inplace=True)
    _train.age.iloc[_train.query("age<=0").index] = _train.age.median()

    x = _train[_train["response"] == 0]

    # to_model_columns = ["level1_relevant_category_volume", "level2_relevant_category_volume", "level3_relevant_category_volume", "level4_relevant_category_volume", "level1_relevant_category_quantity", "level2_relevant_category_quantity", "level3_relevant_category_quantity", "level4_relevant_category_quantity"]
    to_model_columns = [
        "level1_relevant_category_volume",
        "level1_relevant_category_quantity",
        "total_money_spent",
        "total_discount",
        "sanal_percent",
        "days_shopped",
        "months_since_last_shopping",
        "shop_count",
    ]

    clf.fit(x[to_model_columns])
    pred = clf.predict(x[to_model_columns])

    x["anomaly"] = pred

    # Find the number of anomalies and normal points here points classified -1 are anomalous
    x_anomalies = x[x["anomaly"] == -1]

    train_anomalies_dropped = pd.merge(
        _train,
        x_anomalies[["individualnumber", "anomaly"]],
        how="left",
        on="individualnumber",
    )
    train_anomalies_dropped.fillna(1, inplace=True)
    train_anomalies_dropped = train_anomalies_dropped[
        train_anomalies_dropped["anomaly"] != -1
    ]
    train_anomalies_dropped.drop(columns="anomaly", inplace=True)

    return train_anomalies_dropped


def clean_isolation_forest_lower(train, anomalies):
    quantile_selection = "total_discount"
    quantile_limit_selection = 0.9

    limit = train[quantile_selection].quantile(quantile_limit_selection)
    train_lower = train[train[quantile_selection] < limit].copy()

    # Anomaly detection among response = 0 using sklearn.IsolationForest

    _train_lower = train_lower.copy()
    x = _train_lower[_train_lower["response"] == 0]

    # to_model_columns = ["level1_relevant_category_volume", "level2_relevant_category_volume", "level3_relevant_category_volume", "level4_relevant_category_volume", "level1_relevant_category_quantity", "level2_relevant_category_quantity", "level3_relevant_category_quantity", "level4_relevant_category_quantity"]
    to_model_columns = [
        "level1_relevant_category_quantity",
        "level2_relevant_category_quantity",
        "level3_relevant_category_quantity",
        "level4_relevant_category_quantity",
        "total_money_spent",
        "sanal_percent",
        "days_shopped",
        "months_since_last_shopping",
        "shop_count",
    ]

    clf = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=anomalies,
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    clf.fit(x[to_model_columns])
    pred = clf.predict(x[to_model_columns])

    x["anomaly"] = pred
    outliers = x.loc[x["anomaly"] == -1]

    outlier_index = list(outliers.index)

    # Find the number of anomalies and normal points here points classified -1 are anomalous
    print(x["anomaly"].value_counts())
    x_anomalies = x[x["anomaly"] == -1]

    train_lower_anomalies_dropped = pd.merge(
        _train_lower,
        x_anomalies[["individualnumber", "anomaly"]],
        how="left",
        on="individualnumber",
    )
    train_lower_anomalies_dropped.fillna(1, inplace=True)
    train_lower_anomalies_dropped = train_lower_anomalies_dropped[
        train_lower_anomalies_dropped["anomaly"] != -1
    ]
    train_lower_anomalies_dropped.drop(columns="anomaly", inplace=True)
    train_lower_anomalies_dropped

    train_higher = train[train[quantile_selection] >= limit]
    train_new = train_higher.append(train_lower_anomalies_dropped, ignore_index=True)

    return train_new


def oversample_train(train):
    y = train["response"].copy()
    X = train.drop(columns="response").copy()

    X = X.drop(columns=["individualnumber", "gender", "odul/hakkedis"])

    oversample = SMOTENC(sampling_strategy=0.5, categorical_features=[0])
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(X, y)
    # summarize class distribution
    oversampled_data = X_over.copy()
    oversampled_data["response"] = y_over
    return oversampled_data
