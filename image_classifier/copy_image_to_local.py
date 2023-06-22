import argparse
import shutil
from tqdm import tqdm


if __name__ == '__main__':
    """
    获取空白照片识别的正负例数据：
    输入：人的照片json路径，动物照片json路径，没有人和动物的json路径
    输出：
        正例的图片路径的文件:have_target_images.csv
        负例的图片路径的文件:no_target_images.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str,
                        required=True, help='需要拷贝的文件路径')
    parser.add_argument('-t', '--target', type=str,
                        required=True, help='拷贝到那个文件夹中')
    parser.add_argument('-i', '--invalid', type=str,
                        required=True, help='拷贝失败的路径')
    args = parser.parse_args()

    failed_file = open(args.invalid, 'w')
    if args.target[-1] != "/":
        args.target += "/"

    num_files = sum([1 for i in open(args.file, "r")])
    print(f"需要拷贝的文件数量：{num_files}.")

    with tqdm(total=num_files) as pbar:
        for line in open(args.file):
            path = line.strip()
            fileName = path.split("/")[-1]
            try:
                shutil.copyfile(path, args.target + fileName)
            except Exception:
                failed_file.write(line)
            pbar.update(1)
    failed_file.close()
