"""
Portfolio Validator App — v3.0
==============================
Flask web app that lets you upload a trades CSV and inspect the full
portfolio output produced by the v3.0 portfolio engine (portfolio.py).

Usage:
    python portfolio_validator_app.py
    → open http://127.0.0.1:5000 in your browser

CSV requirements (column names are case-insensitive):
    Required : ticker / symbol / tvId / tv_Id
               side   (buy | sell | hold)
               price  / entryPrice
               quantity
    Optional : date / cts / mts / entryDate
               asset_type / assetClass
               market_cap / market cap
               industry
               sector
"""

import io
from typing import Any, Optional

import pandas as pd
from flask import Flask, render_template_string, request, send_file

# ---------- Import v3.0 engine ----------
import portfolio as _eng   # portfolio.py must be in the same directory

# ---------- Flask app ----------
app = Flask(__name__)

# Module-level cache for the last computed result (used by /download)
last_result_df: Optional[pd.DataFrame] = None


# ---------- HTML template (identical look-and-feel to v2.0 validator) ----------
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Portfolio Validator v3.0</title>
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
      rel="stylesheet"
    />
    <link
      href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css"
      rel="stylesheet"
    />
    <style>
      body {
        min-height: 100vh;
        background: radial-gradient(circle at top, #1f2937 0, #020617 55%, #000 100%);
        color: #e5e7eb;
      }
      .app-shell {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem 3rem;
      }
      .card-glass {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.7);
      }
      .card-glass-header {
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
      }
      .hero-title {
        letter-spacing: 0.05em;
      }
      .subtitle {
        color: #9ca3af;
      }
      .badge-v3 {
        font-size: 0.65rem;
        vertical-align: middle;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #fff;
        padding: 0.2em 0.55em;
        border-radius: 0.4rem;
        letter-spacing: 0.05em;
      }
      .table-wrapper {
        position: relative;
        max-height: 70vh;
        overflow-x: auto;
        overflow-y: auto;
        border-radius: 0.75rem;
        background: #020617;
        padding: 0 !important;
      }
      .table-wrapper table {
        margin: 0 !important;
      }
      table.dataTable thead th {
        position: sticky !important;
        top: 0;
        background: #020617;
        color: #e5e7eb;
        z-index: 5;
      }
      table.dataTable tbody tr:hover {
        background-color: #0b1120;
      }
      table.dataTable thead th,
      table.dataTable tbody td {
        padding: 0.3rem 0.5rem;
        font-size: 0.8rem;
        white-space: nowrap;
      }
      .card-glass-header .dataTables_filter label {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.25rem;
        margin: 0;
        color: #e5e7eb;
        font-size: 0.8rem;
        text-align: left !important;
      }
      .card-glass-header .dataTables_filter input[type="search"] {
        width: 220px;
      }
      .card-glass .text-muted {
        color: #cbd5f5 !important;
      }
      .form-text {
        color: #9ca3af;
      }
    </style>
  </head>
  <body>
    <div class="app-shell">
      <header class="mb-4 text-center">
        <h1 class="hero-title display-6 fw-semibold text-light mb-2">
          Portfolio Validator
          <span class="badge-v3">v3.0</span>
        </h1>
        <p class="subtitle mb-0">
          Upload a trades CSV and explore full portfolio metrics, PnL, and risk in one view.
        </p>
      </header>

      <section class="mb-4">
        <div class="card-glass">
          <div class="card-body p-4">
            <form method="post" enctype="multipart/form-data" class="row gx-3 gy-1 align-items-end">
              <div class="col-md-5 col-lg-4">
                <label for="csv_file" class="form-label text-light">Trades CSV</label>
                <input
                  class="form-control"
                  type="file"
                  id="csv_file"
                  name="csv_file"
                  accept=".csv"
                  required
                />
              </div>
              <div class="col-md-3 col-lg-2">
                <label for="initial_cash" class="form-label text-light">Initial Balance</label>
                <input
                  type="text"
                  class="form-control"
                  id="initial_cash"
                  name="initial_cash"
                  value="{{ initial_cash }}"
                />
              </div>

              <div class="col-12">
                <div class="border rounded-3 border-secondary-subtle p-3 mt-1">
                  <p class="mb-2 small text-light fw-semibold">
                    Optional equity metadata from CSV
                  </p>
                  <p class="mb-2 small text-muted">
                    If your CSV has these columns, you can feed them directly into the engine.
                    When unchecked, the app will auto-infer using built-in mappings.
                  </p>
                  <div class="row g-2">
                    <div class="col-6 col-md-3">
                      <div class="form-check form-switch">
                        <input
                          class="form-check-input"
                          type="checkbox"
                          id="include_asset_type"
                          name="include_asset_type"
                        />
                        <label class="form-check-label small text-light" for="include_asset_type">
                          Use <code>Asset Type</code> Column
                        </label>
                      </div>
                    </div>
                    <div class="col-6 col-md-3">
                      <div class="form-check form-switch">
                        <input
                          class="form-check-input"
                          type="checkbox"
                          id="include_market_cap"
                          name="include_market_cap"
                        />
                        <label class="form-check-label small text-light" for="include_market_cap">
                          Use <code>Market Cap</code> Column
                        </label>
                      </div>
                    </div>
                    <div class="col-6 col-md-3">
                      <div class="form-check form-switch">
                        <input
                          class="form-check-input"
                          type="checkbox"
                          id="include_industry"
                          name="include_industry"
                        />
                        <label class="form-check-label small text-light" for="include_industry">
                          Use <code>Industry</code> Column
                        </label>
                      </div>
                    </div>
                    <div class="col-6 col-md-3">
                      <div class="form-check form-switch">
                        <input
                          class="form-check-input"
                          type="checkbox"
                          id="include_sector"
                          name="include_sector"
                        />
                        <label class="form-check-label small text-light" for="include_sector">
                          Use <code>Sector</code> Column
                        </label>
                      </div>
                    </div>
                  </div>
                  <p class="mb-0 mt-2 small text-muted">
                    Column names are case-insensitive. Missing columns are treated as empty.
                  </p>
                </div>
              </div>

              <div class="col-md-3 col-lg-2 mt-2">
                <label class="form-label text-light d-block mb-0 visually-hidden">Run Portfolio</label>
                <button class="btn btn-primary w-100" type="submit">
                  Run Portfolio
                </button>
              </div>
              <div class="col-12">
                <div class="form-text mt-1">
                  Required (case-insensitive): ticker/symbol/tvId/tv_Id, side, price, quantity.
                  Optional: date/cts/mts/entryDate, asset_type/assetClass, market_cap, industry, sector.
                </div>
              </div>
            </form>
          </div>
        </div>
      </section>

      {% if error %}
        <section class="mb-3">
          <div class="alert alert-danger shadow-sm mb-0" role="alert">
          {{ error }}
        </div>
        </section>
      {% endif %}

      {% if df_html %}
        <section class="mt-3">
          <div class="card-glass">
            <div
              class="card-glass-header d-flex justify-content-between align-items-center px-4 py-3"
            >
              <div>
                <h2 class="h5 mb-0 text-light">Portfolio Results</h2>
                <small class="subtitle">Sortable, searchable trade and metric table</small>
              </div>
              <!-- DataTables search box will be moved into this header via JS -->
            </div>
            <div class="d-flex justify-content-end px-4 pt-2 pb-2">
              <a href="{{ url_for('download_csv') }}" class="btn btn-outline-light btn-sm">
                Download CSV
              </a>
            </div>
            <div class="table-wrapper p-3">
              {{ df_html | safe }}
            </div>
          </div>
        </section>
      {% endif %}
    </div>

    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    <script>
      document.addEventListener('DOMContentLoaded', function () {
        // Live-format initial balance with commas while typing
        const cashInput = document.getElementById('initial_cash');
        if (cashInput) {
          const formatWithCommas = (val) => {
            if (val === '' || val === null || val === undefined) return '';
            const str = String(val).replace(/,/g, '');
            if (!str) return '';
            const parts = str.split('.');
            const intPart = parts[0].replace(/[^0-9-]/g, '');
            const decPart = parts[1] ? parts[1].replace(/[^0-9]/g, '') : '';
            if (!intPart) return decPart ? '0.' + decPart : '';
            const intNum = Number(intPart);
            if (Number.isNaN(intNum)) return val;
            const formattedInt = intNum.toLocaleString('en-US');
            return decPart ? formattedInt + '.' + decPart : formattedInt;
          };

          cashInput.addEventListener('input', function () {
            const formattedFull = formatWithCommas(this.value);
            this.value = formattedFull;
            this.selectionStart = this.selectionEnd = this.value.length;
          });
        }

        // Freeze first N columns horizontally while allowing sideways scroll
        function freezeColumns(tableEl, count) {
          if (!tableEl || !tableEl.tHead) return;
          const headerCells = tableEl.tHead.rows[0].cells;
          const max = Math.min(count, headerCells.length);

          const leftOffsets = [];
          let left = 0;
          for (let i = 0; i < max; i++) {
            leftOffsets[i] = left;
            left += headerCells[i].offsetWidth;
          }

          for (let i = 0; i < max; i++) {
            const selector = `#${tableEl.id} thead th:nth-child(${i + 1}), #${tableEl.id} tbody td:nth-child(${i + 1}), #${tableEl.id} tbody th:nth-child(${i + 1})`;
            tableEl.querySelectorAll(selector).forEach((cell) => {
              cell.style.position = 'sticky';
              cell.style.left = leftOffsets[i] + 'px';
              const computedBg = window.getComputedStyle(cell).backgroundColor;
              cell.style.backgroundColor = computedBg;
              if (cell.tagName === 'TH' || cell.tagName === 'TD') {
                cell.style.boxShadow = '2px 0 6px rgba(15, 23, 42, 0.85)';
              }
              cell.style.zIndex = cell.tagName === 'TH' ? 6 : 4;
            });
          }
        }

        const table = document.getElementById('results-table');
        if (table) {
          const dt = new DataTable(table, {
            paging: false,
            info: false,
            ordering: true,
            searching: true
          });

          // Move the DataTables search bar into the card header
          const wrapper = table.closest('.dataTables_wrapper');
          if (wrapper) {
            const filter = wrapper.querySelector('.dataTables_filter');
            const lengthCtrl = wrapper.querySelector('.dataTables_length');
            if (lengthCtrl) lengthCtrl.style.display = 'none';
            if (filter) {
              const header = document.querySelector('.card-glass-header');
              if (header) {
                header.appendChild(filter);
                const label = filter.querySelector('label');
                const input = filter.querySelector('input');
                if (label) label.classList.add('mb-0', 'text-light');
                if (input) input.classList.add('form-control', 'form-control-sm', 'bg-dark', 'text-light');
              }
            }
          }

          // Freeze first 7 visible columns: Index, Date, Ticker, Side, Quantity Buy, Price, Current Quantity
          freezeColumns(table, 7);
        }
      });
    </script>
  </body>
</html>
"""


# ---------- Helpers ----------

def _normalize_trade_date(value: Any) -> Optional[str]:
    """
    Normalize any incoming date value to '%m/%d/%Y' string, or None.

    Delegates to the engine's _parse_date_value parser so the validator
    uses the exact same date-parsing logic as portfolio.py itself.
    Handles: datetime objects, pandas Timestamps, and string formats
    including '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', and variants with time.
    """
    if value is None:
        return None
    dt = _eng._parse_date_value(value)
    if dt is None:
        return None
    return dt.strftime("%m/%d/%Y %H:%M:%S")


def _get_col(cols: dict, candidates: list, required: bool = True, label: Optional[str] = None):
    """Return the first matching column name from candidates (case-insensitive lookup)."""
    label = label or ",".join(candidates)
    for cand in candidates:
        key = str(cand).lower()
        if key in cols:
            return cols[key]
    if required:
        raise ValueError(
            f"CSV must contain column(s) {candidates} for '{label}' (case-insensitive). "
            f"Found: {list(cols.values())}"
        )
    return None


# ---------- Core driver ----------

def _run_portfolio_on_dataframe(
    trades: pd.DataFrame,
    initial_cash: float,
    include_asset_type: bool = False,
    include_market_cap: bool = False,
    include_industry: bool = False,
    include_sector: bool = False,
) -> pd.DataFrame:
    """
    Feed a trades DataFrame into the v3.0 portfolio engine and return the result.

    Column detection is case-insensitive and supports multiple aliases:
      - ticker  : ticker / symbol / tvId / tv_Id
      - price   : price / entryPrice
      - date    : date / cts / mts / entryDate  (optional)
    """
    # Build case-insensitive column lookup
    cols = {str(c).lower(): c for c in trades.columns}

    # Core required columns
    ticker_col   = _get_col(cols, ["ticker", "symbol", "tvid", "tv_id"], label="ticker")
    side_col     = _get_col(cols, ["side"],                               label="side")
    price_col    = _get_col(cols, ["price", "entryprice"],                label="price")
    quantity_col = _get_col(cols, ["quantity","quantity buy"],                           label="quantity")

    # Optional columns
    date_col = _get_col(cols, ["date", "cts", "mts", "entrydate"], required=False, label="date")

    asset_type_col = market_cap_col = industry_col = sector_col = None
    if include_asset_type:
        asset_type_col = _get_col(cols, ["asset_type", "assetclass"], required=False, label="asset_type")
    if include_market_cap:
        market_cap_col = _get_col(cols, ["market_cap", "market cap"], required=False, label="market_cap")
    if include_industry:
        industry_col = _get_col(cols, ["industry"], required=False, label="industry")
    if include_sector:
        sector_col = _get_col(cols, ["sector"], required=False, label="sector")

    # Reset engine state
    _eng.reset_portfolio(initial_cash)

    for _, row in trades.iterrows():
        ticker   = str(row[ticker_col]).upper()
        side_raw = str(row[side_col]).strip().lower()

        # Only accept valid sides; skip anything else
        if side_raw not in {"buy", "sell", "hold"}:
            continue

        price = float(row[price_col])
        qty   = float(row[quantity_col])

        trade_date = _normalize_trade_date(row[date_col]) if date_col is not None else None

        # Optional metadata values (only if user toggled them on)
        asset_type_val = market_cap_val = industry_val = sector_val = None

        if include_asset_type and asset_type_col is not None:
            raw = row.get(asset_type_col)
            if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                s = str(raw).strip()
                asset_type_val = s or None

        if include_market_cap and market_cap_col is not None:
            raw = row.get(market_cap_col)
            if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                s = str(raw).strip()
                market_cap_val = s or None

        if include_industry and industry_col is not None:
            raw = row.get(industry_col)
            if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                s = str(raw).strip()
                industry_val = s or None

        if include_sector and sector_col is not None:
            raw = row.get(sector_col)
            if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                s = str(raw).strip()
                sector_val = s or None

        _eng.add_trade(
            ticker=ticker,
            asset_type=asset_type_val,
            side=side_raw,
            price=price,
            quantity_buy=qty,
            date=trade_date,
            market_cap=market_cap_val,
            industry=industry_val,
            sector=sector_val,
        )

    return _eng.get_portfolio_df()


# ---------- Routes ----------

@app.route("/", methods=["GET", "POST"])
def index():
    df_html = None
    error: Optional[str] = None
    default_initial_cash = 200.0
    global last_result_df

    if request.method == "POST":
        file = request.files.get("csv_file")
        initial_cash_raw = request.form.get("initial_cash", "").strip()

        include_asset_type = bool(request.form.get("include_asset_type"))
        include_market_cap = bool(request.form.get("include_market_cap"))
        include_industry   = bool(request.form.get("include_industry"))
        include_sector     = bool(request.form.get("include_sector"))

        try:
            cleaned      = initial_cash_raw.replace(",", "")
            initial_cash = float(cleaned) if cleaned else default_initial_cash
        except ValueError:
            initial_cash = default_initial_cash

        if not file or file.filename == "":
            error = "Please select a CSV file."
        else:
            try:
                content   = file.read()
                trades_df = pd.read_csv(io.BytesIO(content))
                result_df = _run_portfolio_on_dataframe(
                    trades_df,
                    initial_cash,
                    include_asset_type=include_asset_type,
                    include_market_cap=include_market_cap,
                    include_industry=include_industry,
                    include_sector=include_sector,
                )
                last_result_df = result_df
                df_html = result_df.to_html(
                    classes="table table-striped table-sm",
                    border=0,
                    table_id="results-table",
                )
            except Exception as exc:
                error = f"Error processing file: {exc}"

    return render_template_string(
        HTML_TEMPLATE,
        df_html=df_html,
        error=error,
        initial_cash=default_initial_cash,
    )


@app.route("/download", methods=["GET"])
def download_csv():
    """Download the last computed portfolio results as a CSV file."""
    global last_result_df
    if last_result_df is None or last_result_df.empty:
        return "No portfolio results to download. Please run the portfolio first.", 400

    csv_buffer = io.StringIO()
    last_result_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="portfolio_results_v3.csv",
    )


# ---------- Entry point ----------

from werkzeug.serving import run_simple

if __name__ == "__main__":
    run_simple(
        "127.0.0.1",
        5000,
        app,
        use_reloader=True,
        use_debugger=True,
        extra_files=[],
        exclude_patterns=["*/site-packages/*"],
    )
