"""Map namespace for expression operations on map-typed columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow
import pyarrow.compute as pc

from ray.data.datatype import DataType
from ray.data.expressions import pyarrow_udf

if TYPE_CHECKING:
    from ray.data.expressions import Expr, UDFExpr


@dataclass
class _MapNamespace:
    """Namespace for map operations on expression columns.

    This namespace provides methods for operating on map-typed columns using
    PyArrow compute functions.

    Example:
        >>> from ray.data.expressions import col
        >>> # Extract keys from a map column
        >>> expr = col("metadata").map.keys()
        >>> # Extract values from a map column
        >>> expr = col("metadata").map.values()
    """

    _expr: Expr

    def keys(self) -> "UDFExpr":
        """Extract keys from each map.

        Returns:
            UDFExpr that extracts a list of keys from each map value.

        Example:
            >>> from ray.data.expressions import col
            >>> import ray
            >>> import pyarrow as pa
            >>> ds = ray.data.from_arrow(pa.table({
            ...     "metadata": pa.array([
            ...         {"color": "red", "size": "large"},
            ...         {"color": "blue", "size": "small"}
            ...     ], type=pa.map_(pa.string(), pa.string()))
            ... }))
            >>> ds = ds.with_column("keys", col("metadata").map.keys())
            >>> # Result: keys = [["color", "size"], ["color", "size"]]
        """
        # Infer return type: list<key_type> from map<key_type, value_type>
        return_dtype = DataType(object)  # fallback
        if self._expr.data_type.is_arrow_type():
            arrow_type = self._expr.data_type.to_arrow_dtype()
            if pyarrow.types.is_map(arrow_type):
                key_type = arrow_type.key_type
                # Return as list type
                return_dtype = DataType.from_arrow(pyarrow.list_(key_type))

        @pyarrow_udf(return_dtype=return_dtype)
        def _map_keys(arr: pyarrow.Array) -> pyarrow.Array:
            return pc.map_keys(arr)

        return _map_keys(self._expr)

    def values(self) -> "UDFExpr":
        """Extract values from each map.

        Returns:
            UDFExpr that extracts a list of values from each map value.

        Example:
            >>> from ray.data.expressions import col
            >>> import ray
            >>> import pyarrow as pa
            >>> ds = ray.data.from_arrow(pa.table({
            ...     "metadata": pa.array([
            ...         {"color": "red", "size": "large"},
            ...         {"color": "blue", "size": "small"}
            ...     ], type=pa.map_(pa.string(), pa.string()))
            ... }))
            >>> ds = ds.with_column("values", col("metadata").map.values())
            >>> # Result: values = [["red", "large"], ["blue", "small"]]
        """
        # Infer return type: list<value_type> from map<key_type, value_type>
        return_dtype = DataType(object)  # fallback
        if self._expr.data_type.is_arrow_type():
            arrow_type = self._expr.data_type.to_arrow_dtype()
            if pyarrow.types.is_map(arrow_type):
                value_type = arrow_type.item_type
                # Return as list type
                return_dtype = DataType.from_arrow(pyarrow.list_(value_type))

        @pyarrow_udf(return_dtype=return_dtype)
        def _map_values(arr: pyarrow.Array) -> pyarrow.Array:
            return pc.map_values(arr)

        return _map_values(self._expr)
