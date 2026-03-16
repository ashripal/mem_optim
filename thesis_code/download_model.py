#filepath: /Users/ashripal/mem_optim/download_model.py
import os
from huggingface_hub import snapshot_download

# --- Configuration ---
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
# Expand the '~' to the user's full home directory path
local_dir = os.path.expanduser("~/models/qwen2.5-1.5b-instruct")
# --- End Configuration ---

print(f"Downloading model '{model_id}' to '{local_dir}'...")

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Use snapshot_download to get the model files
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False, # This is safer and copies files directly
    # You can add resume_download=True if the download gets interrupted
)

print("\nDownload complete.")
print(f"Model files are saved in: {local_dir}")
# print("\nYou can now run your tests with:")
# print(f'export MEMARCH_TEST_MODEL_PATH="{local_dir}"')
# print("pytest thesis_code/tests/test_generator_integration.py")