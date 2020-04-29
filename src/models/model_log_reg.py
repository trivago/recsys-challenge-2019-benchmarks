from sklearn.linear_model import LogisticRegression

import data_handling.helper_functions as f
import data_handling.feature_building as fb
import data_handling.data_frame_functions as dff


class ModelLogReg():
    """
    Model class for the logistic regression model.

    Methods
        fit(df): Fit the model on training data
        predict(df): Calculate recommendations for test data        
    """

    def fit(self, df):
        """Train the logistic regression model."""

        df_impressions = fb.build_features(df)

        # Target column, item that was clicked
        f.print_time("target column")
        df_impressions.loc[:, "is_clicked"] = (
            df_impressions["referenced_item"] == df_impressions["impressed_item"]
        ).astype(int)

        features = [
            "position",
            "prices",
            "interaction_count",
            "is_last_interacted",
        ]

        X = df_impressions[features]
        y = df_impressions.is_clicked

        # Training the actual model
        f.print_time("training logistic regression model")
        self.logreg = LogisticRegression(solver="lbfgs", max_iter=100, tol=1e-11, C=1e10).fit(X, y)


    def predict(self, df):
        """Calculate click probability based on trained logistic regression model."""

        df_impressions = fb.build_features(df)

        # Target row, withheld item ID that needs to be predicted
        df_impressions = df_impressions[df_impressions.referenced_item.isna()]

        features = [
            "position",
            "prices",
            "interaction_count",
            "is_last_interacted"
        ]

        # Predict clickout probabilities for each impressed item
        f.print_time("predict clickout item")
        df_impressions.loc[:, "click_probability"] = (
            self
            .logreg
            .predict_proba(df_impressions[features])[:, 1]
        )

        # Summarize recommendations
        f.print_time("summarize recommendations")
        df_rec = dff.summarize_recs(df_impressions, "click_probability")

        return df_rec
