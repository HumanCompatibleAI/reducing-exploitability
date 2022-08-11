import os
from pathlib import Path

dir = Path("/home/pavel/vids/new")

list = dir / "list.txt"
list_str = ""

for file in dir.iterdir():
    if file.suffix == ".mp4":
        list_str += f"file '{str(file)}'\n"
list.write_text(list_str)


os.system(f"ffmpeg -y -f concat -safe 0 -i {list} -c copy output.mp4")
os.system('ffmpeg -y -i output.mp4 -filter:v "setpts=4*PTS" output-slow.gif')
os.system("ffmpeg -y -i output.mp4 output.gif")
