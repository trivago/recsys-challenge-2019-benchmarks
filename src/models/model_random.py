import random

import data_handling.helper_functions as f
import data_handling.data_frame_functions as dff


class ModelRandom():
    """
    Model class for the random ordering model.

    Methods
        fit(df): Not needed. Only added for consistency with other model classes
        predict(df): Calculate recommendations for test data        
    """
    def fit(self, _):
        pass


    def predict(self, df):
        """Randomly sort the impressions list."""

        # Target row, withheld item ID that needs to be predicted
        f.print_time("target rows")
        df_target = dff.get_target_rows(df.copy())

        # Summarize recommendations
        f.print_time("summarize recommendations")
        random.seed(10121)
        df_target.loc[:, "item_recs_list"] = (
            df_target
            .loc[:, "impressions"].str.split("|")
            .map(lambda x: sorted(x, key=lambda k: random.random()))
        )

        df_target.loc[:, "item_recommendations"] = (
           df_target["item_recs_list"]
           .map(lambda arr: ' '.join(arr))
        )

        cols_rec = ["user_id", "session_id", "timestamp", "step", "item_recommendations"]
        df_rec = df_target.loc[:, cols_rec]

        return df_rec
