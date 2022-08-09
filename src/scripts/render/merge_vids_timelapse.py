import os
from pathlib import Path

parent_dir = Path("/home/pavel/vids/new")

list = parent_dir / "list.txt"

list_str = ""

dirs = [dir for dir in parent_dir.iterdir() if dir.is_dir()]

dirs = sorted(dirs)
for dir in dirs:
    for file in dir.iterdir():
        if "000000" in file.name and file.suffix == ".mp4":
            list_str += f"file '{str(file)}'\n"
list.write_text(list_str)

os.system(f"ffmpeg -y -f concat -safe 0 -i {list} -c copy output.mp4")
os.system('ffmpeg -y -i output.mp4 -filter:v "setpts=3*PTS" output-slow.mp4')
