import os
import shutil

# 指定源文件夹和目标文件夹
source_dir = "E:/cocostuff/test2017"
dest_dir = "E:/cocostuff/train2017"

# 获取源文件夹中所有文件的列表
files = os.listdir(source_dir)

# 遍历文件列表
for file in files:
    # 构造源文件路径
    source_file = os.path.join(source_dir, file)

    # 检查文件是否为图像文件
    if os.path.isfile(source_file) and file.endswith((".jpg", ".png", ".gif", ".bmp")):
        # 构造目标文件路径
        dest_file = os.path.join(dest_dir, file)

        # 复制文件
        shutil.copy2(source_file, dest_file)
        print(f"Copied {file} to {dest_dir}")

print("Done!")
