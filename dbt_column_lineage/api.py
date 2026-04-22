"""Clean public API for column lineage resolution.

This is the stable contract consumed by dbt-osmosis and other integrators.
Do not break the signatures of ``ColumnLineageResult`` or ``get_column_lineage``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ColumnLineageResult:
    """Fully-resolved lineage for a single model column."""

    model: str
    """Short model name (e.g. ``stg_orders``)."""

    column: str
    """Column name (lowercase)."""

    progenitor_model: Optional[str]
    """Direct upstream model that provides this column's value, or None for source columns."""

    progenitor_column: Optional[str]
    """Column name in *progenitor_model*, or None."""

    is_rename: bool
    """True when this column is a pure alias of a single upstream column with no transformation."""

    source_column: Optional[str]
    """Original column name before the rename, or None when *is_rename* is False."""


def get_column_lineage(
    manifest_path: str,
    catalog_path: Optional[str] = None,
    live_db: bool = False,
    project_dir: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    target: Optional[str] = None,
    models: Optional[List[str]] = None,
    dialect: Optional[str] = None,
) -> List[ColumnLineageResult]:
    """Resolve column-level lineage for dbt models.

    Exactly one of *catalog_path* or *live_db=True* must be supplied to
    provide column schema information.  Both cannot be set simultaneously.

    Args:
        manifest_path: Path to ``manifest.json``.
        catalog_path: Path to ``catalog.json`` (mutually exclusive with *live_db*).
        live_db: When True, query the live database for column schemas via the
            dbt adapter (requires *profiles_dir* and a dbt project at *project_dir*).
        project_dir: dbt project root directory.  Required when *live_db=True*.
        profiles_dir: Directory containing ``profiles.yml``.  Defaults to the
            current directory when *live_db=True*.
        target: dbt target profile name.  Uses profile default when omitted.
        models: Optional list of model **names** (not node IDs) to restrict
            results to.  When None, all models in the manifest are included.
        dialect: SQL dialect override (e.g. ``"snowflake"``).  When None, the
            adapter type from the manifest is used automatically.

    Returns:
        List of :class:`ColumnLineageResult` — one entry per (model, column) pair
        with resolved lineage.  Columns whose lineage cannot be parsed are omitted
        silently (the parser logs a warning for each).

    Raises:
        ValueError: When both *catalog_path* and *live_db* are supplied, or
            neither is supplied.
        RuntimeError: When the dbt adapter cannot be bootstrapped (live_db mode).
        FileNotFoundError: When *manifest_path* or *catalog_path* do not exist.
    """
    if catalog_path and live_db:
        raise ValueError("Provide either catalog_path or live_db=True, not both.")
    if not catalog_path and not live_db:
        raise ValueError("Either catalog_path or live_db=True is required.")

    catalog_reader = _build_catalog_reader(
        manifest_path=manifest_path,
        catalog_path=catalog_path,
        live_db=live_db,
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target=target,
    )

    from dbt_column_lineage.artifacts.registry import ModelRegistry

    registry = ModelRegistry(
        catalog_path=None,       # type: ignore[arg-type]  # overridden below
        manifest_path=manifest_path,
        adapter_override=dialect,
        _catalog_reader_override=catalog_reader,
    )
    registry.load()

    model_filter = {m.lower() for m in models} if models else None
    results: List[ColumnLineageResult] = []

    for model_name, model_obj in sorted(registry.get_models().items()):
        if model_filter and model_name not in model_filter:
            continue
        if model_obj.resource_type not in ("model",):
            continue

        for col_name, col_obj in sorted(model_obj.columns.items()):
            if not col_obj.lineage:
                results.append(
                    ColumnLineageResult(
                        model=model_name,
                        column=col_name,
                        progenitor_model=None,
                        progenitor_column=None,
                        is_rename=False,
                        source_column=None,
                    )
                )
                continue

            # Take the first lineage entry (highest precedence after parsing)
            lin = col_obj.lineage[0]
            progenitor_model, progenitor_column = _resolve_progenitor(lin)

            results.append(
                ColumnLineageResult(
                    model=model_name,
                    column=col_name,
                    progenitor_model=progenitor_model,
                    progenitor_column=progenitor_column,
                    is_rename=lin.is_rename,
                    source_column=lin.source_column,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_catalog_reader(
    manifest_path: str,
    catalog_path: Optional[str],
    live_db: bool,
    project_dir: Optional[str],
    profiles_dir: Optional[str],
    target: Optional[str],
):
    if live_db:
        from dbt_column_lineage.artifacts.live_db import LiveDbCatalogReader

        return LiveDbCatalogReader(
            manifest_path=manifest_path,
            project_dir=project_dir or ".",
            profiles_dir=profiles_dir or ".",
            target=target,
        )

    from dbt_column_lineage.artifacts.catalog import CatalogReader

    return CatalogReader(catalog_path=catalog_path)  # type: ignore[arg-type]


def _resolve_progenitor(lin) -> tuple[Optional[str], Optional[str]]:
    """Extract (model, column) from the first source_column entry of a ColumnLineage."""
    if not lin.source_columns:
        return None, None

    src = next(iter(sorted(lin.source_columns)))
    if "." not in src:
        return None, src.lower()

    parts = src.rsplit(".", 1)
    return parts[0].lower(), parts[1].lower()
