"""Integration tests for map namespace expressions.

These tests require Ray and test end-to-end map namespace expression evaluation.
"""

import pandas as pd
import pyarrow as pa
import pytest
from packaging import version

import ray
from ray.data._internal.util import rows_same
from ray.data.expressions import col
from ray.data.tests.conftest import *  # noqa
from ray.tests.conftest import *  # noqa

pytestmark = pytest.mark.skipif(
    version.parse(pa.__version__) < version.parse("19.0.0"),
    reason="Namespace expressions tests require PyArrow >= 19.0",
)


def _create_dataset(items_data, dataset_format, arrow_table=None):
    if dataset_format == "arrow":
        if arrow_table is not None:
            ds = ray.data.from_arrow(arrow_table)
        else:
            table = pa.Table.from_pylist(items_data)
            ds = ray.data.from_arrow(table)
    elif dataset_format == "pandas":
        if arrow_table is not None:
            df = arrow_table.to_pandas()
        else:
            df = pd.DataFrame(items_data)
        ds = ray.data.from_blocks([df])
    return ds


DATASET_FORMATS = ["pandas", "arrow"]


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestMapNamespace:
    """Tests for map namespace operations."""

    def test_map_keys(self, ray_start_regular_shared, dataset_format):
        """Test map.keys() extracts keys."""
        arrow_table = pa.table(
            {
                "metadata": pa.array(
                    [
                        {"color": "red", "size": "large"},
                        {"color": "blue", "size": "small"},
                    ],
                    type=pa.map_(pa.string(), pa.string()),
                )
            }
        )
        ds = _create_dataset(None, dataset_format, arrow_table)

        result = ds.with_column("keys", col("metadata").map.keys()).to_pandas()
        expected = pd.DataFrame(
            {
                "metadata": [
                    {"color": "red", "size": "large"},
                    {"color": "blue", "size": "small"},
                ],
                "keys": [["color", "size"], ["color", "size"]],
            }
        )
        assert rows_same(result, expected)

    def test_map_values(self, ray_start_regular_shared, dataset_format):
        """Test map.values() extracts values."""
        arrow_table = pa.table(
            {
                "metadata": pa.array(
                    [
                        {"color": "red", "size": "large"},
                        {"color": "blue", "size": "small"},
                    ],
                    type=pa.map_(pa.string(), pa.string()),
                )
            }
        )
        ds = _create_dataset(None, dataset_format, arrow_table)

        result = ds.with_column("values", col("metadata").map.values()).to_pandas()
        expected = pd.DataFrame(
            {
                "metadata": [
                    {"color": "red", "size": "large"},
                    {"color": "blue", "size": "small"},
                ],
                "values": [["red", "large"], ["blue", "small"]],
            }
        )
        assert rows_same(result, expected)

    def test_map_keys_string_int(self, ray_start_regular_shared, dataset_format):
        """Test map.keys() with string keys and int values."""
        arrow_table = pa.table(
            {
                "counts": pa.array(
                    [
                        {"a": 1, "b": 2},
                        {"c": 3, "d": 4},
                    ],
                    type=pa.map_(pa.string(), pa.int32()),
                )
            }
        )
        ds = _create_dataset(None, dataset_format, arrow_table)

        result = ds.with_column("keys", col("counts").map.keys()).to_pandas()
        expected = pd.DataFrame(
            {
                "counts": [{"a": 1, "b": 2}, {"c": 3, "d": 4}],
                "keys": [["a", "b"], ["c", "d"]],
            }
        )
        assert rows_same(result, expected)

    def test_map_values_string_int(self, ray_start_regular_shared, dataset_format):
        """Test map.values() with string keys and int values."""
        arrow_table = pa.table(
            {
                "counts": pa.array(
                    [
                        {"a": 1, "b": 2},
                        {"c": 3, "d": 4},
                    ],
                    type=pa.map_(pa.string(), pa.int32()),
                )
            }
        )
        ds = _create_dataset(None, dataset_format, arrow_table)

        result = ds.with_column("values", col("counts").map.values()).to_pandas()
        expected = pd.DataFrame(
            {
                "counts": [{"a": 1, "b": 2}, {"c": 3, "d": 4}],
                "values": [[1, 2], [3, 4]],
            }
        )
        assert rows_same(result, expected)

    def test_map_keys_empty_map(self, ray_start_regular_shared, dataset_format):
        """Test map.keys() with empty maps."""
        arrow_table = pa.table(
            {
                "metadata": pa.array(
                    [{}, {"color": "red"}],
                    type=pa.map_(pa.string(), pa.string()),
                )
            }
        )
        ds = _create_dataset(None, dataset_format, arrow_table)

        result = ds.with_column("keys", col("metadata").map.keys()).to_pandas()
        expected = pd.DataFrame(
            {
                "metadata": [{}, {"color": "red"}],
                "keys": [[], ["color"]],
            }
        )
        assert rows_same(result, expected)

    def test_map_values_empty_map(self, ray_start_regular_shared, dataset_format):
        """Test map.values() with empty maps."""
        arrow_table = pa.table(
            {
                "metadata": pa.array(
                    [{}, {"color": "red"}],
                    type=pa.map_(pa.string(), pa.string()),
                )
            }
        )
        ds = _create_dataset(None, dataset_format, arrow_table)

        result = ds.with_column("values", col("metadata").map.values()).to_pandas()
        expected = pd.DataFrame(
            {
                "metadata": [{}, {"color": "red"}],
                "values": [[], ["red"]],
            }
        )
        assert rows_same(result, expected)

    def test_map_keys_null_maps(self, ray_start_regular_shared, dataset_format):
        """Test map.keys() with null map entries."""
        arrow_table = pa.table(
            {
                "metadata": pa.array(
                    [
                        {"color": "red", "size": "large"},
                        None,
                        {"color": "blue"},
                    ],
                    type=pa.map_(pa.string(), pa.string()),
                )
            }
        )
        ds = _create_dataset(None, dataset_format, arrow_table)

        result = ds.with_column("keys", col("metadata").map.keys()).to_pandas()
        expected = pd.DataFrame(
            {
                "metadata": [
                    {"color": "red", "size": "large"},
                    None,
                    {"color": "blue"},
                ],
                "keys": [["color", "size"], None, ["color"]],
            }
        )
        assert rows_same(result, expected)

    def test_map_values_null_maps(self, ray_start_regular_shared, dataset_format):
        """Test map.values() with null map entries."""
        arrow_table = pa.table(
            {
                "metadata": pa.array(
                    [
                        {"color": "red", "size": "large"},
                        None,
                        {"color": "blue"},
                    ],
                    type=pa.map_(pa.string(), pa.string()),
                )
            }
        )
        ds = _create_dataset(None, dataset_format, arrow_table)

        result = ds.with_column("values", col("metadata").map.values()).to_pandas()
        expected = pd.DataFrame(
            {
                "metadata": [
                    {"color": "red", "size": "large"},
                    None,
                    {"color": "blue"},
                ],
                "values": [["red", "large"], None, ["blue"]],
            }
        )
        assert rows_same(result, expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
