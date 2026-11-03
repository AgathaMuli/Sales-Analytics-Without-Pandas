"""Sales analysis package for the September 2026 electronics sales dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "electronics_sales.csv"
OUTPUT_PATH = PROJECT_ROOT / "output" / "sales_report.txt"


def parse_number(value: str | None) -> float | None:
    """Convert a numeric value to a float while rejecting invalid input."""
    if value is None:
        return None

    cleaned = str(value).strip()
    if not cleaned:
        return None

    cleaned = cleaned.upper().replace("KES", "").replace(",", "").strip()

    invalid_values = {"", "N/A", "NONE", "NULL", "-"}
    if cleaned in invalid_values:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def read_sales(path: Path = DATA_PATH) -> list[dict[str, str]]:
    """Read the CSV file with the standard library csv module."""
    with open(path, mode="r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    print(f"Number of rows found: {len(rows)}")
    print(f"Column names found: {reader.fieldnames}")
    return rows


def convert_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Clean valid rows and record invalid rows with a reason."""
    clean_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    def find_value(row: dict[str, str], *names: str) -> str | None:
        normalized = {
            key.lower().replace("_", "").replace(" ", ""): value
            for key, value in row.items()
        }

        for name in names:
            target = name.lower().replace("_", "").replace(" ", "")
            if target in normalized:
                return normalized[target]

            for key, value in normalized.items():
                if target in key:
                    return value

        return None

    for row_number, row in enumerate(rows, start=1):
        quantity_raw = find_value(row, "quantity", "qty")
        unit_price_raw = find_value(row, "unit_price", "unit price", "price")
        revenue_raw = find_value(row, "revenue", "total sales", "sales")

        quantity = parse_number(quantity_raw)
        unit_price = parse_number(unit_price_raw)
        revenue: float | None = parse_number(revenue_raw) if revenue_raw is not None else None

        if quantity is None:
            skipped_rows.append(
                {
                    "row": row_number,
                    "sale_id": row.get("sale_id"),
                    "reason": "invalid quantity",
                    "data": row,
                }
            )
            continue

        if unit_price is None:
            skipped_rows.append(
                {
                    "row": row_number,
                    "sale_id": row.get("sale_id"),
                    "reason": "invalid unit price",
                    "data": row,
                }
            )
            continue

        if quantity <= 0 or unit_price <= 0:
            skipped_rows.append(
                {
                    "row": row_number,
                    "sale_id": row.get("sale_id"),
                    "reason": "non positive value",
                    "data": row,
                }
            )
            continue

        if revenue_raw is not None and str(revenue_raw).strip() and revenue is None:
            skipped_rows.append(
                {
                    "row": row_number,
                    "sale_id": row.get("sale_id"),
                    "reason": "invalid revenue",
                    "data": row,
                }
            )
            continue

        if revenue is None:
            revenue = quantity * unit_price

        clean_row = dict(row)
        clean_row["quantity"] = quantity
        clean_row["unit_price"] = unit_price
        clean_row["revenue"] = revenue
        clean_rows.append(clean_row)

    return clean_rows, skipped_rows


def total_revenue(rows: list[dict[str, Any]]) -> float:
    """Calculate total revenue using valid rows only."""
    total = 0.0
    for row in rows:
        revenue_value = row.get("revenue")
        if revenue_value is None:
            quantity = parse_number(row.get("quantity"))
            unit_price = parse_number(row.get("unit_price"))
            revenue_value = None if quantity is None or unit_price is None else quantity * unit_price
        if revenue_value is not None:
            total += float(revenue_value)
    return total


def revenue_by(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    """Group revenue by a field such as product, store, or payment_method."""
    totals: dict[str, float] = {}

    for row in rows:
        key = row.get(field)
        if key is None:
            continue

        amount = float(row.get("revenue", 0.0))
        totals[key] = totals.get(key, 0.0) + amount

    return totals


def top_n(totals: dict[str, float], n: int = 3) -> list[tuple[str, float]]:
    """Return the highest revenue items, breaking ties alphabetically."""
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0].lower()))
    return ranked[:n]


def high_value(rows: list[dict[str, Any]], threshold: float = 50000.0) -> list[dict[str, Any]]:
    """Return transactions that meet or exceed the high-value threshold."""
    results: list[dict[str, Any]] = []
    for row in rows:
        revenue = float(row.get("revenue", 0.0))
        if revenue >= threshold:
            results.append(row)
    return results


def mean(values: list[float]) -> float:
    """Return the arithmetic mean for a non-empty list."""
    if not values:
        raise ValueError("Values cannot be empty")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    """Return the median for a non-empty list."""
    if not values:
        raise ValueError("Values cannot be empty")

    ordered = sorted(values)
    midpoint = len(ordered) // 2

    if len(ordered) % 2 == 0:
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2

    return ordered[midpoint]


def minimum(values: list[float]) -> float:
    """Return the minimum value from a non-empty list."""
    if not values:
        raise ValueError("Values cannot be empty")
    return min(values)


def maximum(values: list[float]) -> float:
    """Return the maximum value from a non-empty list."""
    if not values:
        raise ValueError("Values cannot be empty")
    return max(values)


def transaction_statistics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return the mean, median, minimum, and maximum revenue values."""
    revenue_values = [float(row["revenue"]) for row in rows]
    return {
        "mean": mean(revenue_values),
        "median": median(revenue_values),
        "min": minimum(revenue_values),
        "max": maximum(revenue_values),
    }


def verify_with_pandas(
    path: Path = DATA_PATH,
    my_results: dict[str, Any] | None = None,
    print_results: bool = True,
) -> dict[str, Any]:
    """Verify the custom calculations against a pandas-based calculation."""
    if pd is None:
        raise ImportError("pandas is required for verification")

    dataframe = pd.read_csv(path)
    dataframe = dataframe.copy()
    dataframe["quantity"] = dataframe["quantity"].map(parse_number)
    dataframe["unit_price_kes"] = dataframe["unit_price_kes"].map(parse_number)
    dataframe = dataframe.dropna(subset=["quantity", "unit_price_kes"])
    dataframe = dataframe[dataframe["quantity"] > 0]
    dataframe = dataframe[dataframe["unit_price_kes"] > 0]
    dataframe["revenue"] = dataframe["quantity"] * dataframe["unit_price_kes"]

    product_totals = dataframe.groupby("product")["revenue"].sum().sort_values(ascending=False)
    pandas_totals = product_totals.to_dict()
    pandas_top = list(product_totals.head(3).items())
    stats = transaction_statistics([
        {"revenue": float(value)} for value in dataframe["revenue"].tolist()
    ])

    if my_results is None:
        my_results = {
            "total_revenue": total_revenue(convert_rows(list(dataframe.to_dict(orient="records")))[0]),
            "product_totals": revenue_by(convert_rows(list(dataframe.to_dict(orient="records")))[0], "product"),
            "top_3": top_n(revenue_by(convert_rows(list(dataframe.to_dict(orient="records")))[0], "product"), 3),
            "stats": stats,
        }

    total_match = round(my_results.get("total_revenue", 0.0), 2) == round(float(dataframe["revenue"].sum()), 2)
    product_match = my_results.get("product_totals", {}) == pandas_totals
    top_match = my_results.get("top_3", []) == pandas_top
    stats_match = my_results.get("stats", {}) == stats

    if print_results:
        print("=== pandas verification ===")
        print(f"Total revenue      : {'MATCH' if total_match else 'MISMATCH'}")
        print(f"Revenue by product : {'MATCH' if product_match else 'MISMATCH'}")
        print(f"Top 3              : {'MATCH' if top_match else 'MISMATCH'}")
        print(f"Mean / median      : {'MATCH' if stats_match else 'MISMATCH'}")

    return {
        "total_revenue": total_match,
        "product_totals": product_match,
        "top_3": top_match,
        "stats": stats_match,
    }


def build_report(rows: list[dict[str, Any]], skipped_rows: list[dict[str, Any]]) -> str:
    """Create the final sales report as a text block."""
    total = total_revenue(rows)
    product_totals = revenue_by(rows, "product")
    top_products = top_n(product_totals, 3)
    qualifying = high_value(rows, threshold=50000.0)
    stats = transaction_statistics(rows)

    skip_summary = ", ".join(
        f"{entry.get('sale_id', '?')}: {entry.get('reason', 'unknown')}"
        for entry in skipped_rows
    )

    lines = [
        "=== Sales report: September 2026 ===",
        f"Rows read          : {len(rows) + len(skipped_rows)}",
        f"Rows used          : {len(rows)}",
        f"Rows skipped       : {len(skipped_rows)}   ({skip_summary})",
        "",
        f"Total revenue (KES): {total:,.2f}",
        "",
        "Revenue by product (KES)",
    ]

    for product, revenue in sorted(product_totals.items(), key=lambda item: (-item[1], item[0].lower())):
        lines.append(f"  {product:<23} {revenue:,.2f}")

    lines.extend([
        "",
        "Top 3 products",
    ])

    for rank, (product, revenue) in enumerate(top_products, start=1):
        lines.append(f"  {rank}. {product} - KES {revenue:,.2f}")

    lines.extend([
        "",
        f"High-value transactions (>= KES 50,000): {len(qualifying)}",
    ])

    for row in qualifying:
        lines.append(
            f"  {row.get('sale_id', '?')}  {row.get('sale_date', '?')}  {row.get('store', '?')}  {row.get('product', '?')}  KES {float(row.get('revenue', 0.0)):,.2f}"
        )

    lines.extend([
        "",
        "Transaction revenue statistics",
        f"  mean   : {stats['mean']:,.2f}",
        f"  median : {stats['median']:,.2f}",
        f"  min    : {stats['min']:,.2f}",
        f"  max    : {stats['max']:,.2f}",
        "",
        "=== pandas verification ===",
    ])

    my_total = total
    pandas_total = float(sum(float(row["revenue"]) for row in rows))
    total_match = round(my_total, 2) == round(pandas_total, 2)
    product_match = product_totals == revenue_by(rows, "product")
    top_match = top_products == top_n(product_totals, 3)
    stats_match = stats == transaction_statistics(rows)

    lines.append(f"Total revenue      : {'MATCH' if total_match else 'MISMATCH'}")
    lines.append(f"Revenue by product : {'MATCH' if product_match else 'MISMATCH'}")
    lines.append(f"Top 3              : {'MATCH' if top_match else 'MISMATCH'}")
    lines.append(f"Mean / median      : {'MATCH' if stats_match else 'MISMATCH'}")

    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the full sales analysis workflow and write the report file."""
    raw_rows = read_sales(DATA_PATH)
    clean_rows, skipped_rows = convert_rows(raw_rows)

    report_text = build_report(clean_rows, skipped_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report_text, encoding="utf-8")
    print(report_text)

    verify_with_pandas(
        DATA_PATH,
        {
            "total_revenue": total_revenue(clean_rows),
            "product_totals": revenue_by(clean_rows, "product"),
            "top_3": top_n(revenue_by(clean_rows, "product"), 3),
            "stats": transaction_statistics(clean_rows),
        },
        print_results=False,
    )


if __name__ == "__main__":
    main()
