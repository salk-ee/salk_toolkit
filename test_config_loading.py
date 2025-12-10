"""Diagnostic script to check if altair_custom_config.json is being loaded"""
import os
import sys
import json

print("="*60)
print("DIAGNOSTIC: Checking altair_custom_config.json loading")
print("="*60)

# Check current working directory
print(f"\n1. Current working directory: {os.getcwd()}")

# Check if file exists in cwd
cwd_path = os.path.join(os.getcwd(), "altair_custom_config.json")
print(f"\n2. Checking for file in cwd: {cwd_path}")
print(f"   Exists: {os.path.exists(cwd_path)}")

# Check if file exists relative to package
package_path = os.path.join(os.path.dirname(__file__), "altair_custom_config.json")
print(f"\n3. Checking for file relative to script: {package_path}")
print(f"   Exists: {os.path.exists(package_path)}")

# Now import utils and check what was loaded
print("\n4. Importing salk_toolkit.utils...")
try:
    from salk_toolkit import utils
    print("   ✅ Import successful")
    
    print(f"\n5. Checking utils.altair_custom_config:")
    if utils.altair_custom_config:
        print(f"   ✅ Config loaded! Keys: {list(utils.altair_custom_config.keys())}")
        if 'legend' in utils.altair_custom_config:
            print(f"   Legend config: {utils.altair_custom_config['legend']}")
        if 'background' in utils.altair_custom_config:
            print(f"   Background: {utils.altair_custom_config['background']}")
    else:
        print(f"   ❌ Config is empty: {utils.altair_custom_config}")
    
    print(f"\n6. Checking utils.altair_custom_chart:")
    if utils.altair_custom_chart:
        print(f"   ⚠️  Custom chart is defined (this takes precedence over config!)")
        print(f"   Chart keys: {list(utils.altair_custom_chart.keys())}")
    else:
        print(f"   ✅ No custom chart (config should be used)")
        
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)
if os.path.exists(cwd_path):
    with open(cwd_path, 'r') as f:
        config = json.load(f)
    print(f"✅ Config file found with {len(config)} top-level keys")
    print(f"   File location: {cwd_path}")
    print(f"\n   If config is not being applied, the module may be cached.")
    print(f"   Try restarting your Python/Streamlit process.")
else:
    print(f"❌ No config file found in current directory")
    print(f"   Expected location: {cwd_path}")
    print(f"   Create the file there or in the package root")

print("="*60)
