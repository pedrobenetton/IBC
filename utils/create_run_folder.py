import os

def create_run_folder(project_name):
    os.makedirs(project_name, exist_ok=True)
    i = 0
    while True:
        run_folder = os.path.join(project_name, f"run{i}")
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
            return run_folder
        i += 1
