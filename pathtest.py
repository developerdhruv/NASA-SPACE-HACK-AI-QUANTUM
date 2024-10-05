import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Attempt the import
try:
    from pinn.main import SeismicPINN, prepare_training_data
    print("Import successful!")
except ModuleNotFoundError as e:
    print(f"Import failed: {e}")
