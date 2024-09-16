import os
import subprocess

def run_script(script_path):
    try:
        print(f"Running {script_path}...")
        subprocess.run(['python', script_path], check=True)
        print(f"{script_path} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}: {e}")

def main():
    run_script('app.py')

    run_script('Training.py')

  
    run_script('app2024.py')

    run_script('Training_2024_data.py')

if __name__ == "__main__":
    main()
