import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from game_sales.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing video game sales DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df
        self.config = config
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the DataFrame stored in self.df.

        This method handles missing values, converts data types, and prepares features.
        """
        # Convert numeric features to numeric dtype
        for col in self.config.num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Fill missing numeric values with column mean
        for col in self.config.num_features:
            mean_val = self.df[col].mean()
            self.df[col].fillna(mean_val, inplace=True)

        # Convert categorical features and fill missing with 'Unknown'
        for cat_col in self.config.cat_features:
            self.df[cat_col] = self.df[cat_col].fillna("Unknown").astype("category")

        # Ensure target is numeric and drop rows without target
        target = self.config.target
        self.df[target] = pd.to_numeric(self.df[target], errors="coerce")
        self.df.dropna(subset=[target], inplace=True)

        # Select only relevant columns for modeling
        features = self.config.cat_features + self.config.num_features
        self.df = self.df[features + [target]]

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame into train and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks Delta tables."""
        # Add UTC timestamp
        train_df = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_df = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Write tables
        train_df.write.mode("overwrite").saveAsTable(f"{self.config.catalog_name}.{self.config.schema_name}.train")
        test_df.write.mode("overwrite").saveAsTable(f"{self.config.catalog_name}.{self.config.schema_name}.test")

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test tables."""
        catalog = self.config.catalog_name
        schema = self.config.schema_name
        self.spark.sql(f"ALTER TABLE {catalog}.{schema}.train SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
        self.spark.sql(f"ALTER TABLE {catalog}.{schema}.test SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
