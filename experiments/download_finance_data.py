#!/usr/bin/env python
"""
Download financial data using the financedatabase package.

This script demonstrates how to access and download various types of financial
data including stock information, ETF data, indices, and more.
"""

import financedatabase as fd
import pandas as pd

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.pead import paths


def main():
    # Create output directory (repo-root-anchored)
    output_dir = paths.FINANCE_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Financial Database - Data Download")
    print("=" * 80)

    # 1. Download Equities data
    print("\n1. Fetching Equities data...")
    try:
        equities = fd.Equities()
        equities_data = equities.select()
        print(f"   Found {len(equities_data)} equities")
        equities_df = pd.DataFrame(equities_data).T
        equities_df.to_csv(output_dir / "equities.csv")
        print(f"   Saved to: financial_data/equities.csv")
    except Exception as e:
        print(f"   Error: {e}")

    # 2. Download ETF data
    print("\n2. Fetching ETF data...")
    try:
        etfs = fd.ETFs()
        etfs_data = etfs.select()
        print(f"   Found {len(etfs_data)} ETFs")
        etfs_df = pd.DataFrame(etfs_data).T
        etfs_df.to_csv(output_dir / "etfs.csv")
        print(f"   Saved to: financial_data/etfs.csv")
    except Exception as e:
        print(f"   Error: {e}")

    # 3. Download indices data
    print("\n3. Fetching indices data...")
    try:
        indices = fd.Indices()
        indices_data = indices.select()
        print(f"   Found {len(indices_data)} indices")
        indices_df = pd.DataFrame(indices_data).T
        indices_df.to_csv(output_dir / "indices.csv")
        print(f"   Saved to: financial_data/indices.csv")
    except Exception as e:
        print(f"   Error: {e}")

    # 4. Download crypto data
    print("\n4. Fetching cryptocurrencies data...")
    try:
        crypto = fd.Cryptos()
        crypto_data = crypto.select()
        print(f"   Found {len(crypto_data)} cryptocurrencies")
        crypto_df = pd.DataFrame(crypto_data).T
        crypto_df.to_csv(output_dir / "cryptocurrencies.csv")
        print(f"   Saved to: financial_data/cryptocurrencies.csv")
    except Exception as e:
        print(f"   Error: {e}")

    # 5. Download forex pairs data
    print("\n5. Fetching currencies (forex) data...")
    try:
        currencies = fd.Currencies()
        currencies_data = currencies.select()
        print(f"   Found {len(currencies_data)} currencies")
        currencies_df = pd.DataFrame(currencies_data).T
        currencies_df.to_csv(output_dir / "currencies.csv")
        print(f"   Saved to: financial_data/currencies.csv")
    except Exception as e:
        print(f"   Error: {e}")

    # 6. Download Funds data
    print("\n6. Fetching Funds data...")
    try:
        funds = fd.Funds()
        funds_data = funds.select()
        print(f"   Found {len(funds_data)} funds")
        funds_df = pd.DataFrame(funds_data).T
        funds_df.to_csv(output_dir / "funds.csv")
        print(f"   Saved to: financial_data/funds.csv")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 80)
    print("Download complete!")
    print(f"All data saved to: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
