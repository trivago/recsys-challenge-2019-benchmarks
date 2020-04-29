import data_handling.helper_functions as f
import data_handling.nearest_neighbor_crunching as nn
import data_handling.data_frame_functions as dff


class ModelNNItem():
    """
    Model class for the item metadata nearest neighbor model.

    Methods
        fit(df): Fit the model on training data
        predict(df): Calculate recommendations for test data        
    """

    def fit(self, df):
        """Calculate item similarity based on item metadata."""

        # Explode property arrays
        f.print_time("explode properties")
        df_properties = dff.explode(df, "properties")

        df_item_sim = nn.calc_item_sims(df_properties, "item_id", "properties")

        self.df_item_sim = df_item_sim


    def predict(self, df):
        """Sort impression list by similarity."""

        df_rec = nn.predict_nn(df, self.df_item_sim)

        return df_rec
