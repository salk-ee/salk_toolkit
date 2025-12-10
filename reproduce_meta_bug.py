from pydantic import BaseModel
from typing import Dict, Optional

class DataMeta(BaseModel):
    structure: Dict[str, str] = {}

# Simulate the logic in sampling.py
data_meta = DataMeta(structure={'initial': 'block'})
full_meta = {'data': data_meta, 'model': {}}

print(f"Initial full_meta['data'].structure: {full_meta['data'].structure}")

# Simulate adding a new block via model_copy (as done in sampling.py)
new_structure = dict(data_meta.structure)
new_structure['mured'] = 'new_block'
data_meta = data_meta.model_copy(update={'structure': new_structure})

print(f"Updated data_meta.structure: {data_meta.structure}")
print(f"full_meta['data'].structure (should be STALE): {full_meta['data'].structure}")

if 'mured' not in full_meta['data'].structure:
    print("\nBUG REPRODUCED: full_meta['data'] does not contain the new block!")
else:
    print("\nBug not reproduced.")
