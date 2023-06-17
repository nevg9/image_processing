import json
import hashlib
import argparse


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

    return path


def get_image_classifier_data(human_action, animal_action, no_action):
    pos_file_list = []
    neg_file_list = []

    for content in human_action:
        pos_file_list.append(get_file_path(content))

    for content in animal_action:
        pos_file_list.append(get_file_path(content))

    for content in no_action:
        neg_file_list.append(get_file_path(content))

    with open('have_target_images.txt', 'w', newline='') as f:
        f.writelines(line + '\n' for line in pos_file_list)

    with open('no_target_images.txt', 'w', newline='') as f:
        f.writelines(line + '\n' for line in neg_file_list)

    print("正例照片数量:{},负例照片数量:{}".format(len(pos_file_list), len(neg_file_list)))


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
