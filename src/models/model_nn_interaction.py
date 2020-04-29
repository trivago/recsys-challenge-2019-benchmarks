import data_handling.helper_functions as f
import data_handling.nearest_neighbor_crunching as nn
import data_handling.data_frame_functions as dff


class ModelNNInteraction():
    """
    Model class for the session co-occurrence nearest neighbor model.

    Methods
        fit(df): Fit the model on training data
        predict(df): Calculate recommendations for test data        
    """

    def fit(self, df):
        """Calculate item similarity based on session co-occurrence."""

        # Select columns that are of interest for this method
        f.print_time("start")
        cols = ['user_id', 'session_id', 'timestamp', 'step',
                'action_type', 'reference']
        df_cols = df.loc[:, cols] 

        # We are only interested in action types, for wich the reference is an item ID
        f.print_time("filter interactions")
        item_interactions = [
            'clickout item', 'interaction item deals', 'interaction item image',
            'interaction item info', 'interaction item rating', 'search for item'
        ]
        df_actions = (
            df_cols
            .loc[df_cols.action_type.isin(item_interactions), :]
            .rename(columns={'reference': 'item'})
            .drop(columns='action_type')
        )

        df_item_sim = nn.calc_item_sims(df_actions, "item", "session_id")

        self.df_item_sim = df_item_sim


    def predict(self, df):
        """Sort impression list by similarity."""

        df_rec = nn.predict_nn(df, self.df_item_sim)

        return df_rec
