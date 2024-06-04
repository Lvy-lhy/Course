import json
import os

name2id = {'mask': 0, 'nomask': 1}  # 标签名称


def convert(img_size, box):
    dw = 1.0 / img_size[0]
    dh = 1.0 / img_size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x *= dw
    w *= dw
    y *= dh
    h *= dh
    return x, y, w, h


def decode_json(json_folder_path, json_name):
    txt_name = os.path.join('/experiment/12109990929_ex5/1210999092_数据集划分/Annotations',
                            json_name.replace('.json', '.txt'))
    os.makedirs(os.path.dirname(txt_name), exist_ok=True)  # 确保目录存在

    try:
        with open(txt_name, 'w') as txt_file, open(os.path.join(json_folder_path, json_name), 'r', encoding='gb2312',
                                                   errors='ignore') as json_file:
            data = json.load(json_file)

            img_w = data['imageWidth']
            img_h = data['imageHeight']

            for shape in data['shapes']:
                label_name = shape['label']
                if shape['shape_type'] == 'rectangle':
                    x1, y1 = map(int, shape['points'][0])
                    x2, y2 = map(int, shape['points'][1])
                    bbox = convert((img_w, img_h), (x1, y1, x2, y2))
                    txt_file.write(f"{name2id[label_name]} {' '.join(map(str, bbox))}\n")

        # 成功转换后删除原始 JSON 文件
        os.remove(os.path.join(json_folder_path, json_name))

    except KeyError as e:
        print(f"键错误：{e} 在文件 {json_name} 中")
    except json.JSONDecodeError as e:
        print(f"JSON 解码错误：{e} 在文件 {json_name} 中")
    except Exception as e:
        print(f"意外错误：{e} 在文件 {json_name} 中")


if __name__ == "__main__":
    json_folder_path = '/experiment/12109990929_ex5/1210999092_数据集划分/Annotations'
    json_names = [file for file in os.listdir(json_folder_path) if file.endswith('.json')]

    for json_name in json_names:
        decode_json(json_folder_path, json_name)
