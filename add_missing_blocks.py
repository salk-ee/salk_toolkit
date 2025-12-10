"""Script to add missing mured and plussid_factors blocks to pref.parquet metadata.

Usage:
  conda run -n salk python add_missing_blocks.py /path/to/pref.parquet

This script uses salk_toolkit.io functions to properly read/write metadata.
"""

import sys
import shutil
import pandas as pd
from salk_toolkit.io import read_parquet_with_metadata, write_parquet_with_metadata


def add_missing_blocks(parquet_path: str, output_path: str | None = None, backup: bool = True):
    """Add missing mured and plussid_factors blocks to parquet metadata."""
    
    if output_path is None:
        output_path = parquet_path  # Overwrite in place
    
    # Create backup if overwriting in place
    if backup and output_path == parquet_path:
        backup_path = parquet_path + '.backup'
        print(f"Creating backup at {backup_path}...")
        shutil.copy2(parquet_path, backup_path)
    
    # Read existing data and metadata using salk_toolkit functions
    print("Reading existing parquet file...")
    df, full_meta = read_parquet_with_metadata(parquet_path, lazy=False)
    
    if full_meta is None:
        raise ValueError("No metadata found in parquet file")
    
    data_meta = full_meta.data

    # Get the columns from the actual data
    mured_cols = sorted([c for c in df.columns if c.startswith('mured_') and not c.startswith('mured_factors_')])
    mured_factor_cols = sorted([c for c in df.columns if c.startswith('mured_factors_')])
    plussid_factor_cols = sorted([c for c in df.columns if c.startswith('plussid_factors_')])

    print(f"Found {len(mured_cols)} mured columns")
    print(f"Found {len(mured_factor_cols)} mured_factors columns")
    print(f"Found {len(plussid_factor_cols)} plussid_factors columns")

    # Get existing metadata as dict
    existing_meta = full_meta.model_dump(mode='json')
    data_dict = existing_meta['data']

    # Check if structure exists
    if 'structure' not in data_dict or data_dict['structure'] is None:
        data_dict['structure'] = []

    # Create the new blocks
    # mured block (using col_prefix='mured_')
    mured_block = {
        'name': 'mured',
        'scale': {
            'continuous': True,
            'col_prefix': 'mured_',
            'label': 'Mured (concerns)'
        },
        'columns': [[c[6:], c[6:]] for c in mured_cols]  # [name, source] - name is without prefix
    }

    # mured_factors block (using col_prefix='mured_factors_')
    mured_factors_block = {
        'name': 'mured_factors',
        'scale': {
            'continuous': True,
            'col_prefix': 'mured_factors_',
            'label': 'Mured factors'
        },
        'columns': [[c[14:], c[14:]] for c in mured_factor_cols]  # Remove 'mured_factors_' prefix
    }

    # plussid_factors block (using col_prefix='plussid_factors_')
    plussid_factors_block = {
        'name': 'plussid_factors',
        'scale': {
            'continuous': True,
            'col_prefix': 'plussid_factors_',
            'label': 'Plussid factors'
        },
        'columns': [[c[16:], c[16:]] for c in plussid_factor_cols]  # Remove 'plussid_factors_' prefix
    }

    # Add new blocks to structure (as list for JSON serialization)
    structure_list = list(data_dict['structure'].values()) if isinstance(data_dict['structure'], dict) else data_dict['structure']

    # Check if blocks already exist
    existing_names = {b['name'] for b in structure_list}
    blocks_to_add = []

    if 'mured' not in existing_names and mured_cols:
        blocks_to_add.append(mured_block)
        print("Adding 'mured' block")
    elif 'mured' in existing_names:
        print("'mured' block already exists, skipping")

    if 'mured_factors' not in existing_names and mured_factor_cols:
        blocks_to_add.append(mured_factors_block)
        print("Adding 'mured_factors' block")
    elif 'mured_factors' in existing_names:
        print("'mured_factors' block already exists, skipping")

    if 'plussid_factors' not in existing_names and plussid_factor_cols:
        blocks_to_add.append(plussid_factors_block)
        print("Adding 'plussid_factors' block")
    elif 'plussid_factors' in existing_names:
        print("'plussid_factors' block already exists, skipping")

    if not blocks_to_add:
        print("No blocks to add, exiting")
        return

    # Add new blocks
    structure_list.extend(blocks_to_add)
    data_dict['structure'] = structure_list

    # Update the full metadata
    existing_meta['data'] = data_dict

    # Write updated parquet using salk_toolkit function (uses correct 'salk-toolkit-meta' key)
    print(f"\nWriting updated parquet to {output_path}...")
    write_parquet_with_metadata(df, existing_meta, output_path)

    print(f"Done! Updated parquet saved to: {output_path}")
    if backup and output_path == parquet_path:
        print(f"Backup available at: {parquet_path}.backup")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_missing_blocks.py <parquet_path> [output_path]")
        sys.exit(1)
    
    parquet_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    add_missing_blocks(parquet_path, output_path)
