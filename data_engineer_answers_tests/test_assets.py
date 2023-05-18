from dagster import materialize
from data_engineer_answers.assets import (
    load_data,
    transform_data,
    preprocessed_data,
)


def test_asset_graph():
    assets = [load_data, transform_data, preprocessed_data]
    result = materialize(assets=assets)
    assert result.success
    df = result.output_for_node("preprocessed_data")
    assert len(df) == 28151758
    assert df.columns.tolist() == ["Symbol", "Date", "Open", "High", "Low",
                                   "Close", "Volume", "Adj Close", "vol_moving_avg", "adj_close_rolling_med"]
