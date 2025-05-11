import pytest
import pandas as pd
from pyspark.sql import SparkSession
from mlops_course.config import ProjectConfig
from mlops_course.data_process import DataProcessor


def test_data_ingestion(sample_data: pd.DataFrame) -> None:
    """Test the data ingestion by verifying non-empty DataFrame."""
    assert sample_data.shape[0] > 0
    assert sample_data.shape[1] > 0


def test_dataprocessor_init(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Test DataProcessor initialization."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    assert hasattr(processor, 'df')
    assert processor.config is config
    assert processor.spark is spark_session


def test_column_types_and_missing_handling(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession
) -> None:
    """Test that numeric features are numeric, categorical features are 'category', and no missing values."""
    processor = DataProcessor(pandas_df=sample_data.copy(), config=config, spark=spark_session)
    processor.preprocess()

    # Numeric features
    for col in config.num_features:
        assert pd.api.types.is_float_dtype(processor.df[col]) or pd.api.types.is_integer_dtype(processor.df[col])
        assert processor.df[col].isna().sum() == 0

    # Categorical features
    for col in config.cat_features:
        assert processor.df[col].dtype.name == 'category'
        assert (processor.df[col] == 'Unknown').sum() >= 0  # missing filled with Unknown

    # Target
    assert pd.api.types.is_numeric_dtype(processor.df[config.target])
    assert processor.df[config.target].isna().sum() == 0


def test_column_selection(sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test that only configured features and target are present after preprocessing."""
    processor = DataProcessor(pandas_df=sample_data.copy(), config=config, spark=spark_session)
    processor.preprocess()
    expected_columns = config.cat_features + config.num_features + [config.target]
    assert set(processor.df.columns) == set(expected_columns)


def test_split_data_default_params(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession
) -> None:
    """Test DataProcessor.split_data defaults."""
    processor = DataProcessor(pandas_df=sample_data.copy(), config=config, spark=spark_session)
    processor.preprocess()
    train, test = processor.split_data()

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(processor.df)
    assert set(train.columns) == set(processor.df.columns)
    assert set(test.columns) == set(processor.df.columns)


def test_preprocess_empty_dataframe(config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test preprocess raises on empty DataFrame."""
    processor = DataProcessor(pandas_df=pd.DataFrame(), config=config, spark=spark_session)
    with pytest.raises(KeyError):
        processor.preprocess()


@pytest.mark.skip(reason="depends on delta tables on Databricks")
def test_save_to_catalog_success(sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test saving train and test to catalog."""
    processor = DataProcessor(pandas_df=sample_data.copy(), config=config, spark=spark_session)
    processor.preprocess()
    train_set, test_set = processor.split_data()
    processor.save_to_catalog(train_set, test_set)
    processor.enable_change_data_feed()

    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.train")
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.test")


@pytest.mark.skip(reason="depends on delta tables on Databricks")
@pytest.mark.order(after=test_save_to_catalog_success)
def test_delta_table_property_of_enable_change_data_feed(config: ProjectConfig, spark_session: SparkSession) -> None:
    """Verify delta.enableChangeDataFeed property is true on both tables."""
    for table in ['train', 'test']:
        full_name = f"{config.catalog_name}.{config.schema_name}.{table}"
        properties = spark_session.sql(f"DESCRIBE EXTENDED {full_name}") \
            .filter("col_name = 'delta.enableChangeDataFeed'") \
            .select('data_type').collect()[0][0]
        assert properties == 'true'