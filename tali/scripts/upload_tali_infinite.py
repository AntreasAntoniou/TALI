import subprocess
import time
import signal

def run_command():
    command = [
        "/opt/conda/envs/main/bin/python",
        "tali/scripts/upload_dataset_from_disk_to_hf.py",
        "--dataset_name",
        "Antreas/TALI-big-2.0",
        "--train_data_percentage",
        "1.0",
        "--num_workers",
        "1",
    ]

    while True:
        process = subprocess.Popen(command)
        start_time = time.time()

        # Monitor the process for 15 minutes or until it exits
        while True:
            time_elapsed = time.time() - start_time

            # If 15 minutes have passed, terminate the process
            if time_elapsed >= 15 * 60:
                process.send_signal(signal.SIGTERM)
                break

            # Check if the process has terminated (due to error or completion)
            return_code = process.poll()
            if return_code is not None:
                print(f"Process exited with return code {return_code}. Restarting...")
                break

            # Sleep for a short duration to avoid busy-waiting
            time.sleep(1)

        # Wait for the process to terminate before restarting
        process.wait()

if __name__ == "__main__":
    run_command()
