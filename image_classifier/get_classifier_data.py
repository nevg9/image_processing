import json
import hashlib
import argparse
import shutil
from PIL import Image
from pathlib import Path
from tqdm import tqdm


copy_file_failed_fd = open("copy_to_local_image_failed.txt", "w")


def get_file_path(content):
    # 计算图片id的md5
    pic_id = content["图片id"]
    md5hash = hashlib.md5(pic_id.encode('utf-8'))
    md5 = md5hash.hexdigest()

    root_dir = "/hpc_input_fs"

    first = md5[0]
    second = md5[1:3]
    third = md5[3:6]
    path = "/".join([root_dir, first, second, third, pic_id])
    my_file = Path(path)
    if my_file.is_file():
        content["有效路径"] = path
        return path
    else:
        new_path = "/".join([root_dir, first, second, third, md5])
        my_file = Path(new_path)
        if my_file.is_file():
            content["有效路径"] = new_path
            return new_path
        else:
            content["无效路径"] = path + "\t" + new_path
            return None


def copy_image_to_local(content, root_dir):
    try:
        Image.open(content["有效路径"])
        content["本地路径"] = root_dir + "/" + content["有效路径"].split("/")[-1]
        shutil.copyfile(content["有效路径"], content["本地路径"])
        return True
    except Exception:
        json.dump(content, copy_file_failed_fd, ensure_ascii=False)
        copy_file_failed_fd.write("\n")
        return False


def write_json_file(file_name, data):
    with open(file_name, "w") as json_file:
        json.dump(data, json_file, ensure_ascii=False)


def json_contents_to_path(actions, local_files, dir_name):
    invalid_json = []
    root_dir = "/home/data/image_data/"
    json_file = root_dir + dir_name + ".json"
    json_file_fd = open(json_file, 'w')
    path_not_exist = 0
    image_corrupted = 0
    for content in tqdm(actions):
        path = get_file_path(content)
        if path:
            if copy_image_to_local(content, root_dir=root_dir + "/" + dir_name):
                json.dump(content, json_file_fd, ensure_ascii=False)
                json_file_fd.write("\n")
                local_files.append(content["本地路径"])
            else:
                invalid_json.append(content)
                image_corrupted += 1
        else:
            invalid_json.append(content)
            path_not_exist += 1

    json_file_fd.close()

    print(f"{dir_name},路径非法总数:{path_not_exist},照片损坏总数:{image_corrupted}.")

    with open(root_dir + dir_name + "_failed.json", "w") as json_file:
        json.dump(invalid_json, json_file, ensure_ascii=False)


def get_image_classifier_data(human_action, animal_action, no_action):
    pos_file_list = []
    neg_file_list = []
    json_contents_to_path(human_action, pos_file_list, "human_images")
    json_contents_to_path(animal_action, pos_file_list, "animal_images")
    json_contents_to_path(no_action, neg_file_list, "no_target_images")
    with open('have_target_images.txt', 'w', newline='') as f:
        f.writelines(line + '\n' for line in pos_file_list)

    with open('no_target_images.txt', 'w', newline='') as f:
        f.writelines(line + '\n' for line in neg_file_list)

    print("正例照片数量:{},负例照片数量:{}.".format(len(pos_file_list), len(neg_file_list)))


if __name__ == '__main__':
    """
    获取空白照片识别的正负例数据：
    输入：人的照片json路径，动物照片json路径，没有人和动物的json路径
    输出：
        正例的图片路径的文件:have_target_images.csv
        负例的图片路径的文件:no_target_images.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--human', type=str,
                        required=True, help='有人的照片的json路径')
    parser.add_argument('-a', '--animal', type=str,
                        required=True, help='有动物的照片的json路径')
    parser.add_argument('-n', '--no', type=str,
                        required=True, help='没有人和动物的json路径')
    args = parser.parse_args()

    with open(args.human, 'r') as f:
        human_action = json.load(f)

    with open(args.animal, 'r') as f:
        animal_action = json.load(f)

    with open(args.no, 'r') as f:
        no_action = json.load(f)

    print("人物照片数量:{},动物照片数量:{},空白照片数量:{}".format(
        len(human_action), len(animal_action), len(no_action)))

    get_image_classifier_data(human_action, animal_action, no_action)

    copy_file_failed_fd.close()
