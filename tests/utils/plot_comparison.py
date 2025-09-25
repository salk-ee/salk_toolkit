"""Utilities for comparing plot JSON outputs with normalization of non-deterministic elements."""

import json
import re


def compare_json_with_tolerance(json1, json2, float_tolerance=1e-5):
    """Compare two JSON objects with floating point tolerance."""
    def compare_recursive(obj1, obj2):
        if type(obj1) != type(obj2):
            return False
        
        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            return all(compare_recursive(obj1[k], obj2[k]) for k in obj1.keys())
        
        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                return False
            return all(compare_recursive(a, b) for a, b in zip(obj1, obj2))
        
        elif isinstance(obj1, float) and isinstance(obj2, float):
            return abs(obj1 - obj2) <= float_tolerance
        
        else:
            return obj1 == obj2
    
    return compare_recursive(json1, json2)


def normalize_chart_json(chart_json):
    """Normalize chart JSON by removing non-deterministic fields and sorting datasets."""
    if isinstance(chart_json, str):
        chart_json = json.loads(chart_json)
    
    def normalize_recursive(obj):
        if isinstance(obj, dict):
            # Remove non-deterministic fields
            normalized = {k: v for k, v in obj.items() 
                         if k not in ['$schema', 'config', 'usermeta']}
            
            # Process each field
            for key, value in list(normalized.items()):
                if key == 'name' and isinstance(value, str) and re.match(r'data-[a-f0-9]+', value):
                    # Normalize random data names
                    normalized[key] = 'data-normalized'
                
                elif key == 'datasets' and isinstance(value, dict):
                    # Normalize dataset keys and sort data
                    new_datasets = {}
                    for dataset_key, dataset_value in value.items():
                        if re.match(r'data-[a-f0-9]+', dataset_key):
                            normalized_data = normalize_recursive(dataset_value)
                            # Sort dataset records for deterministic comparison
                            if isinstance(normalized_data, list) and normalized_data:
                                normalized_data = sort_dataset_records(normalized_data)
                            new_datasets['data-normalized'] = normalized_data
                        else:
                            new_datasets[dataset_key] = normalize_recursive(dataset_value)
                    normalized[key] = new_datasets
                
                else:
                    normalized[key] = normalize_recursive(value)
            
            return normalized
        
        elif isinstance(obj, list):
            return [normalize_recursive(item) for item in obj]
        
        elif isinstance(obj, str) and re.match(r'data-[a-f0-9]+', obj):
            # Normalize random data references in strings
            return 'data-normalized'
        
        else:
            return obj
    
    return normalize_recursive(chart_json)


def sort_dataset_records(records):
    """Sort dataset records using categorical fields first for deterministic ordering."""
    if not records or not isinstance(records[0], dict):
        return records

    unnamed = [k for k in records[0].keys() if k.startswith('level_')]
    if unnamed:
        raise ValueError(f"Unnamed index ({', '.join(unnamed)}) found in dataset records. This can lead to unstable tests and should be fixed.")

    # Numerical fields that should always be last
    # Important for density plots like facet_dist or density
    lesser_fields = ['density', 'probability']
    
    def sort_key(record):
        key_parts = []
        skeys = sorted(record.keys())
        
        # Add categorical fields (str)
        for field in skeys:
            if isinstance(record[field],str):
                key_parts.append(record[field])

        # Add all other fields
        for field in skeys:
            if field not in lesser_fields and not isinstance(record[field],str):
                key_parts.append(str(record[field]))

        # Add lesser fields last
        for field in lesser_fields:
            if field in record:
                key_parts.append(str(record[field]))

        return tuple(key_parts)
    
    return sorted(records, key=sort_key)


def pretty_print_json_differences(json1, json2, float_tolerance=1e-5, max_differences=10):
    """Pretty print differences between two JSON objects for better error messages."""
    def find_differences(obj1, obj2, path=''):
        diffs = []
        
        if type(obj1) != type(obj2):
            diffs.append({
                'path': path,
                'type': 'type_mismatch',
                'old': str(type(obj1).__name__),
                'new': str(type(obj2).__name__)
            })
            return diffs
        
        if isinstance(obj1, dict):
            all_keys = set(obj1.keys()) | set(obj2.keys())
            for key in all_keys:
                new_path = f'{path}.{key}' if path else key
                if key not in obj1:
                    diffs.append({
                        'path': new_path,
                        'type': 'missing_in_old',
                        'new': str(obj2[key])[:100] + ('...' if len(str(obj2[key])) > 100 else '')
                    })
                elif key not in obj2:
                    diffs.append({
                        'path': new_path,
                        'type': 'missing_in_new',
                        'old': str(obj1[key])[:100] + ('...' if len(str(obj1[key])) > 100 else '')
                    })
                else:
                    diffs.extend(find_differences(obj1[key], obj2[key], new_path))
        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                diffs.append({
                    'path': path,
                    'type': 'list_length',
                    'old': len(obj1),
                    'new': len(obj2),
                    'diff': len(obj2) - len(obj1),
                    'old_list': obj1,
                    'new_list': obj2
                })
                return diffs
            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                diffs.extend(find_differences(item1, item2, f'{path}[{i}]'))
        else:
            if isinstance(obj1, float) and isinstance(obj2, float):
                if abs(obj1 - obj2) > float_tolerance:
                    diffs.append({
                        'path': path,
                        'type': 'float_difference',
                        'old': obj1,
                        'new': obj2,
                        'diff': abs(obj1 - obj2),
                        'tolerance': float_tolerance
                    })
            elif obj1 != obj2:
                diffs.append({
                    'path': path,
                    'type': 'value_mismatch',
                    'old': obj1,
                    'new': obj2
                })
        
        return diffs
    
    differences = find_differences(json1, json2)
    
    if not differences:
        return "✅ No differences found"
    
    # Group differences by type for better organization
    by_type = {}
    for diff in differences:
        diff_type = diff['type']
        if diff_type not in by_type:
            by_type[diff_type] = []
        by_type[diff_type].append(diff)
    
    result = [f"❌ Found {len(differences)} differences:"]
    result.append("")
    
    # Show summary by type
    for diff_type, diffs in by_type.items():
        count = len(diffs)
        type_name = diff_type.replace('_', ' ').title()
        result.append(f"📊 {type_name}: {count} difference{'s' if count != 1 else ''}")
    
    result.append("")
    
    # Show detailed differences (limited)
    shown = 0
    for diff_type, type_diffs in by_type.items():
        if shown >= max_differences:
            break
            
        result.append(f"🔍 {diff_type.replace('_', ' ').title()}:")
        
        for diff in type_diffs[:min(5, max_differences - shown)]:
            if diff['type'] == 'list_length':
                result.append(f"  📏 Location: {diff['path']}")
                result.append(f"     Length: {diff['old']} → {diff['new']} records ({diff['diff']:+d})")
                
                # Show all entries for both lists
                old_list = diff['old_list']
                new_list = diff['new_list']
                
                result.append(f"     📋 Old list ({len(old_list)} items):")
                for i, item in enumerate(old_list):
                    item_str = str(item)[:80] + ('...' if len(str(item)) > 80 else '')
                    result.append(f"       [{i}] {item_str}")
                
                result.append(f"     📋 New list ({len(new_list)} items):")
                for i, item in enumerate(new_list):
                    item_str = str(item)[:80] + ('...' if len(str(item)) > 80 else '')
                    result.append(f"       [{i}] {item_str}")
                
            elif diff['type'] == 'float_difference':
                result.append(f"  🔢 Location: {diff['path']}")
                result.append(f"     Values: {diff['old']:.6f} → {diff['new']:.6f} (Δ{diff['diff']:.2e}, tol: {diff['tolerance']:.2e})")
                
            elif diff['type'] == 'value_mismatch':
                old_str = str(diff['old'])[:100] + ('...' if len(str(diff['old'])) > 100 else '')
                new_str = str(diff['new'])[:100] + ('...' if len(str(diff['new'])) > 100 else '')
                result.append(f"  📝 Location: {diff['path']}")
                result.append(f"     Old: '{old_str}'")
                result.append(f"     New: '{new_str}'")
                
            elif diff['type'] == 'type_mismatch':
                result.append(f"  🔄 Location: {diff['path']}")
                result.append(f"     Types: {diff['old']} → {diff['new']}")
                
            elif diff['type'] == 'missing_in_old':
                result.append(f"  ➕ Location: {diff['path']}")
                result.append(f"     Added: '{diff['new']}'")
                
            elif diff['type'] == 'missing_in_new':
                result.append(f"  ➖ Location: {diff['path']}")
                result.append(f"     Removed: '{diff['old']}'")
            
            shown += 1
            if shown >= max_differences:
                break
        
        if len(type_diffs) > 5:
            remaining = len(type_diffs) - 5
            result.append(f"  ... and {remaining} more {diff_type.replace('_', ' ')} differences")
        
        result.append("")
    
    if len(differences) > max_differences:
        result.append(f"... and {len(differences) - shown} more differences not shown")
        result.append("")
    
    result.append("💡 Use --recompute to update the reference file")
    
    return "\n".join(result)
