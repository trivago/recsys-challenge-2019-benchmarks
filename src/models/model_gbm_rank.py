import lightgbm as lgb

import data_handling.helper_functions as f
import data_handling.feature_building as fb
import data_handling.data_frame_functions as dff


class ModelGbmRank():
    """
    Model class for the lightGBM model.

    Methods
        fit(df): Fit the model on training data
        predict(df): Calculate recommendations for test data        
    """

    def fit(self, df):
        """Train the lightGBM model."""

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

        # Bring to format suitable for lightGBM
        f.print_time("lightGBM format")
        X = df_impressions[features]
        y = df_impressions.is_clicked

        q = (
            df_impressions
            .groupby(["user_id", "session_id", "timestamp", "step"])
            .size()
            .reset_index(name="query_length")
            .query_length
        )

        # Training the actual model
        f.print_time("training lightGBM model")
        self.gbm = lgb.LGBMRanker()
        self.gbm.fit(X, y, group=q, verbose=True)


    def predict(self, df):
        """Calculate item ranking based on trained lightGBM model."""

        df_impressions = fb.build_features(df)

        # Target row, withheld item ID that needs to be predicted
        df_impressions = df_impressions[df_impressions.referenced_item.isna()]

        features = [
            "position",
            "prices",
            "interaction_count",
            "is_last_interacted"
        ]

        df_impressions.loc[:, "click_propensity"] = self.gbm.predict(df_impressions[features])

        # Summarize recommendations
        f.print_time("summarize recommendations")
        df_rec = dff.summarize_recs(df_impressions, "click_propensity")
         
        return df_rec
