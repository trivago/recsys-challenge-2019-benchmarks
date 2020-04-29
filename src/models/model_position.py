import data_handling.helper_functions as f
import data_handling.data_frame_functions as dff


class ModelPosition():
    """
    Model class for the model based on the original position in displayed list.

    Methods
        fit(df): Not needed. Only added for consistency with other model classes
        predict(df): Calculate recommendations for test data        
    """
    def fit(self, _):
        pass


    def predict(self, df):
        """Return items in impressions list in original order."""

        # Target row, withheld item ID that needs to be predicted
        f.print_time("target rows")
        df_target = dff.get_target_rows(df.copy())

        # Summarize recommendations
        f.print_time("summarize recommendations")
        df_target["item_recommendations"] = (
            df_target
            .apply(lambda x: x.impressions.replace("|", " "), axis=1)
        )

        cols_rec = ["user_id", "session_id", "timestamp", "step", "item_recommendations"]
        df_rec = df_target.loc[:, cols_rec]

        return df_rec
