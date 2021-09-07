import json

# num = 1
# for num in range(1, 19):
for num in ['ci_3(1)']:
    # path = 'D:/Development/REPP/data_repp/repp_{}/'.format(num)
    path = 'D:/Development/REPP/validation_repp/uniklinikum-endo_{}/'.format(num)
    # with open('D:/Development/REPP/data_repp/repp_{}.pkl_coco.json'.format(num), mode='r') as f:
    with open('D:/Development/REPP/validation_videos_split/repp_uniklinikum-endo_{}.pkl_coco.json'.format(num), mode='r') as f:
        repp_org = json.load(f)
    print('Converting REPP JSON to individual files:', path)
    preds = {}
    for p in repp_org:
        value = preds[p['image_id']] if p['image_id'] in preds else []
        value.append(p)
        preds[p['image_id']] = value
    for k in preds.keys():
        with open(path + '{}.txt'.format(k), mode='w') as f:
            for p in preds[k]:
                f.write('{class_name} {conf} {left} {top} {right} {bottom}\n'.format(
                    class_name='polyp', conf=p['score'],
                    left=p['bbox'][0], top=p['bbox'][1],
                    right=p['bbox'][0] + p['bbox'][2],
                    bottom=p['bbox'][1] + p['bbox'][3]))
    print('DONE')
