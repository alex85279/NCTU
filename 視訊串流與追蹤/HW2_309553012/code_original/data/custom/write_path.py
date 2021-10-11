from os import walk
from os.path import join
mypath = "D:\desktop\PyTorch-YOLOv3-master\data\custom\images"
for root, dirs, files in walk(mypath):
  for f in files:
    fullpath = join(root, f)
    print(fullpath)