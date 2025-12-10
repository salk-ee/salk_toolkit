"""
Simple test to verify config loading works by directly importing utils module
"""
import sys
import os

# Add the salk_toolkit directory to path
sys.path.insert(0, '/Users/erik/salk/salk_toolkit')

# Change to the directory where altair_custom_config.json is
os.chdir('/Users/erik/salk/salk_toolkit')

print(f"Current directory: {os.getcwd()}")
print(f"Config file exists: {os.path.exists('altair_custom_config.json')}")

# Now directly import just the utils module (not the whole package)
import importlib.util
spec = importlib.util.spec_from_file_location("utils", "/Users/erik/salk/salk_toolkit/salk_toolkit/utils.py")
utils = importlib.util.module_from_spec(spec)

print("\nLoading utils module...")
try:
    spec.loader.exec_module(utils)
    print("✅ Module loaded successfully")
    
    print(f"\naltair_custom_config type: {type(utils.altair_custom_config)}")
    print(f"altair_custom_config empty: {len(utils.altair_custom_config) == 0}")
    
    if utils.altair_custom_config:
        print(f"✅ Config loaded with {len(utils.altair_custom_config)} keys")
        print(f"   Keys: {list(utils.altair_custom_config.keys())[:5]}...")
        if 'background' in utils.altair_custom_config:
            print(f"   Background color: {utils.altair_custom_config['background']}")
    else:
        print("❌ Config is empty!")
        
    if utils.altair_custom_chart:
        print(f"\n⚠️  altair_custom_chart is defined (takes precedence!)")
    else:
        print(f"\n✅ altair_custom_chart is None (config will be used)")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
