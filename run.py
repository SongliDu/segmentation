import subprocess

epoch_num = 290

while epoch_num < 400:
    command  = ["python", "train.py", f"--epoch={epoch_num}"]
    subprocess.run(command)