import data_handling.helper_functions as f
import data_handling.data_frame_functions as dff


class ModelPopUsers():
    """
    Model class for the popularity model based on distinct users.

    Methods
        fit(df): Fit the model on training data
        predict(df): Calculate recommendations for test data        
    """

    def fit(self, df):
        """Count the number of distinct users that click on an item."""

        # Select columns that are of interest for this method
        f.print_time("start")
        cols = ['user_id', 'session_id', 'timestamp', 'step',
                'action_type', 'reference']
        df_cols = df.loc[:, cols] 

        # We only need to count clickouts per item
        f.print_time("clicks per item")
        df_item_clicks = (
            df_cols
            .loc[df_cols["action_type"] == "clickout item", :]
            .groupby("reference")
            .user_id
            .nunique()
            .reset_index(name="n_users")
            .rename(columns={"reference": "item"})
        )

        self.df_pop = df_item_clicks


    def predict(self, df):
        """Sort the impression list by number of distinct users in the training phase."""

        # Select columns that are of interest for this method
        f.print_time("start")
        cols = ['user_id', 'session_id', 'timestamp', 'step',
                'action_type', 'reference', "impressions"]
        df_cols = df.loc[:, cols] 

        # Target row, withheld item ID that needs to be predicted
        f.print_time("target rows")
        df_target = dff.get_target_rows(df_cols)

        # Explode to impression level
        f.print_time("explode impression array")
        df_impressions = (
            dff.explode(df_target, "impressions")
            .rename(columns={"impressions": "impressed_item"})
        )
        df_impressions = (
            df_impressions
            .merge(
                self.df_pop,
                left_on="impressed_item",
                right_on="item",
                how="left"
            )
        )

        # Summarize recommendations
        f.print_time("summarize recommendations")
        df_rec = dff.summarize_recs(df_impressions, "n_users")

        return df_rec
