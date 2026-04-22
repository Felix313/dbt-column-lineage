"""Tests for get_column_lineage() public API.

No warehouse connection required — catalog resolver and registry are mocked.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dbt_column_lineage.api import ColumnLineageResult, get_column_lineage, _resolve_progenitor
from dbt_column_lineage.models.schema import Column, ColumnLineage, Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(name: str, cols: dict[str, list[ColumnLineage]]) -> Model:
    columns = {
        col_name: Column(name=col_name, model_name=name, lineage=lineage_list)
        for col_name, lineage_list in cols.items()
    }
    return Model(
        name=name,
        schema="main",
        database="dev",
        columns=columns,
        resource_type="model",
    )


def _make_manifest(tmp_path: Path) -> Path:
    manifest = {
        "metadata": {"adapter_type": "snowflake"},
        "nodes": {},
        "sources": {},
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def _make_catalog(tmp_path: Path) -> Path:
    catalog = {"nodes": {}, "sources": {}}
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(catalog))
    return p


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_both_catalog_and_live_db_raises(tmp_path):
    m = _make_manifest(tmp_path)
    c = _make_catalog(tmp_path)
    with pytest.raises(ValueError, match="not both"):
        get_column_lineage(str(m), catalog_path=str(c), live_db=True)


def test_neither_catalog_nor_live_db_raises(tmp_path):
    m = _make_manifest(tmp_path)
    with pytest.raises(ValueError, match="required"):
        get_column_lineage(str(m))


# ---------------------------------------------------------------------------
# Catalog resolver path
# ---------------------------------------------------------------------------

def test_catalog_path_returns_results(tmp_path):
    m = _make_manifest(tmp_path)
    c = _make_catalog(tmp_path)

    direct_lin = ColumnLineage(
        source_columns={"stg_orders.id"},
        transformation_type="direct",
    )
    rename_lin = ColumnLineage(
        source_columns={"stg_orders.user_id"},
        transformation_type="renamed",
    )

    models = {
        "orders": _make_model("orders", {
            "order_id": [direct_lin],
            "customer_id": [rename_lin],
        }),
    }

    with patch("dbt_column_lineage.artifacts.registry.ModelRegistry") as MockRegistry:
        instance = MockRegistry.return_value
        instance.get_models.return_value = models
        instance.load.return_value = None

        results = get_column_lineage(str(m), catalog_path=str(c))

    assert len(results) == 2
    by_col = {r.column: r for r in results}

    direct = by_col["order_id"]
    assert direct.is_rename is False
    assert direct.source_column is None
    assert direct.progenitor_model == "stg_orders"
    assert direct.progenitor_column == "id"

    renamed = by_col["customer_id"]
    assert renamed.is_rename is True
    assert renamed.source_column == "user_id"
    assert renamed.progenitor_model == "stg_orders"
    assert renamed.progenitor_column == "user_id"


# ---------------------------------------------------------------------------
# Live DB resolver path (adapter mocked — no warehouse)
# ---------------------------------------------------------------------------

def test_live_db_path_calls_live_db_reader(tmp_path):
    m = _make_manifest(tmp_path)

    mock_reader = MagicMock()
    mock_reader.get_models_nodes.return_value = {
        "payments": _make_model("payments", {
            "payment_id": [
                ColumnLineage(
                    source_columns={"raw_payments.id"},
                    transformation_type="renamed",
                )
            ]
        })
    }

    with (
        patch("dbt_column_lineage.api._build_catalog_reader", return_value=mock_reader),
        patch("dbt_column_lineage.artifacts.registry.ModelRegistry") as MockRegistry,
    ):
        instance = MockRegistry.return_value
        instance.get_models.return_value = {
            "payments": _make_model("payments", {
                "payment_id": [
                    ColumnLineage(
                        source_columns={"raw_payments.id"},
                        transformation_type="renamed",
                    )
                ]
            })
        }
        instance.load.return_value = None

        results = get_column_lineage(
            str(m),
            live_db=True,
            project_dir=str(tmp_path),
            profiles_dir=str(tmp_path),
        )

    assert len(results) == 1
    r = results[0]
    assert r.model == "payments"
    assert r.column == "payment_id"
    assert r.is_rename is True
    assert r.source_column == "id"


# ---------------------------------------------------------------------------
# Model filter
# ---------------------------------------------------------------------------

def test_model_filter_restricts_output(tmp_path):
    m = _make_manifest(tmp_path)
    c = _make_catalog(tmp_path)

    lin = ColumnLineage(source_columns={"src.col"}, transformation_type="direct")
    all_models = {
        "model_a": _make_model("model_a", {"col1": [lin]}),
        "model_b": _make_model("model_b", {"col2": [lin]}),
    }

    with patch("dbt_column_lineage.artifacts.registry.ModelRegistry") as MockRegistry:
        instance = MockRegistry.return_value
        instance.get_models.return_value = all_models
        instance.load.return_value = None

        results = get_column_lineage(str(m), catalog_path=str(c), models=["model_a"])

    assert all(r.model == "model_a" for r in results)


# ---------------------------------------------------------------------------
# _resolve_progenitor helper
# ---------------------------------------------------------------------------

def test_resolve_progenitor_qualified():
    lin = ColumnLineage(source_columns={"my_model.my_col"}, transformation_type="direct")
    model, col = _resolve_progenitor(lin)
    assert model == "my_model"
    assert col == "my_col"


def test_resolve_progenitor_unqualified():
    lin = ColumnLineage(source_columns={"bare_col"}, transformation_type="direct")
    model, col = _resolve_progenitor(lin)
    assert model is None
    assert col == "bare_col"


def test_resolve_progenitor_empty():
    lin = ColumnLineage(source_columns=set(), transformation_type="direct")
    model, col = _resolve_progenitor(lin)
    assert model is None
    assert col is None
