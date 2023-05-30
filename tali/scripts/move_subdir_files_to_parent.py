import os
import shutil

# Define your top-level directory
top_dir = "."

# Loop through all subdirectories in the top-level directory
for subdir in os.listdir(top_dir):
    if os.path.isdir(
        os.path.join(top_dir, subdir)
    ):  # Make sure it's a directory
        for filename in os.listdir(os.path.join(top_dir, subdir)):
            # Move each file to the top-level directory
            shutil.move(os.path.join(top_dir, subdir, filename), top_dir)

print("File moving completed!")
