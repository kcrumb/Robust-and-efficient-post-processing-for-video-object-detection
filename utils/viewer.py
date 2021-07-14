def read_yolo_preds():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    from pathlib import Path

    threshold = 0.05

    folder = Path('D:/Development/REPP/data/1')
    images = [x for x in folder.iterdir() if x.name.endswith('.png')]
    images.sort(key=lambda x: int(x.name[:x.name.index(x.suffix)]))

    bb_files = [x for x in folder.iterdir() if x.name.endswith('.txt')]
    bb_files.sort(key=lambda x: int(x.name[:x.name.index(x.suffix)]))

    for idx, img in enumerate(images):
        im = Image.open(img)

        annotation_file = bb_files[idx]
        with annotation_file.open() as f:
            lines = [l.split(' ') for l in f.read().splitlines()]

        boxes = []
        for l in lines:

            if threshold is None or float(l[5]) < threshold:
                continue

            box_width = float(l[3]) * im.width
            box_height = float(l[4]) * im.height
            x = (float(l[1]) * im.width) - (box_width / 2)
            y = (float(l[2]) * im.height) - (box_height / 2)
            boxes.append(patches.Rectangle((x, y), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none'))

        fig, ax = plt.subplots()
        ax.imshow(im)
        ax.set_title(img.name + ' - ' + annotation_file.name)
        for b in boxes:
            ax.add_patch(b)
        plt.show()


def read_json_post():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    from pathlib import Path
    import json


    folder = Path('D:/Development/REPP/data/1')
    images = [x for x in folder.iterdir() if x.name.endswith('.png')]
    images.sort(key=lambda x: int(x.name[:x.name.index(x.suffix)]))

    bb_json = json.load(open('D:/Development/REPP/repp_1.pkl_coco.json', mode='r'))
    boxes_dict = {}
    for b in bb_json:
        img_id = b['image_id']
        if img_id in boxes_dict:
            boxes_dict[img_id].append(b)
        else:
            boxes_dict[img_id] = [b]

    for idx, img in enumerate(images):
        im = Image.open(img)

        boxes = []
        if idx in boxes_dict:
            for box in boxes_dict[idx]:
                boxes.append(patches.Rectangle((box['bbox'][0], box['bbox'][1]), box['bbox'][2], box['bbox'][3], linewidth=1, edgecolor='r', facecolor='none'))

        fig, ax = plt.subplots()
        ax.imshow(im)
        ax.set_title(img.name)
        for b in boxes:
            ax.add_patch(b)
        plt.show()


if __name__ == '__main__':
    # read_yolo_preds()
    read_json_post()
