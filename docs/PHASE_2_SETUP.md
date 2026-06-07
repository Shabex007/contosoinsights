# Phase 2: Python Data Extraction Setup

This phase connects Python to PostgreSQL, extracts the `public.customer_ml_features` view created in Phase 1, validates it, and saves ML-ready files.

## 1. Add these files to your project

```text
.env.example
requirements.txt
src/config.py
src/database.py
src/extract_data.py
src/quick_check.py
data/raw/
data/processed/
```

## 2. Create virtual environment

```bash
python -m venv .venv
```

Activate it:

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Windows CMD:

```bash
.venv\Scripts\activate.bat
```

Mac/Linux:

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Create `.env`

Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

On Mac/Linux:

```bash
cp .env.example .env
```

Then update:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=contoso
DB_USER=postgres
DB_PASSWORD=your_real_password
DB_SCHEMA=public
```

## 5. Confirm Phase 1 view exists

Run in PostgreSQL:

```sql
SELECT *
FROM public.customer_ml_features
LIMIT 10;
```

## 6. Run extraction

From project root:

```bash
python src/extract_data.py
```

Expected outputs:

```text
data/processed/customer_ml_features.csv
data/processed/customer_ml_features.parquet
data/raw/sales_transaction_sample.csv
data/raw/sales_transaction_sample.parquet
```

## 7. Run quick check

```bash
python src/quick_check.py
```

This confirms the extracted data shape, missing values, churn label distribution, and sample rows.

## Common errors

### `password authentication failed`

Your `.env` database password is wrong.

### `relation public.customer_ml_features does not exist`

You have not run Phase 1 SQL yet, or your schema is not `public`.

### `ModuleNotFoundError: No module named dotenv`

Run:

```bash
pip install -r requirements.txt
```

### `customer_ml_features returned 0 rows`

Check whether your Contoso sales table has data and whether the Phase 1 SQL view was created correctly.
