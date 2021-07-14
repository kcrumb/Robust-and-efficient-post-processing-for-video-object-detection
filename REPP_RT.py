#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:46:22 2020

@author: asabater
"""

import copy
import json
import pickle

import numpy as np
from scipy import signal, ndimage

from repp_utils import get_iou, get_pair_features

INF = 9e15


# =============================================================================
# Robust and Efficient Post-Processing for Video Object Detection (REPP)
# =============================================================================


class REPP():

    def __init__(self, min_tubelet_score, add_unmatched, min_pred_score,
                 distance_func, clf_thr, clf_mode, appearance_matching,
                 recoordinate, recoordinate_std, window_size,
                 store_coco=False, store_imdb=False,
                 annotations_filename='',
                 **kwargs):

        self.min_tubelet_score = min_tubelet_score  # threshold to filter out low-scoring tubelets
        self.min_pred_score = min_pred_score  # threshold to filter out low-scoring base predictions
        self.add_unmatched = add_unmatched  # True to add unlinked detections to the final set of detections. Leads to a lower mAP

        self.distance_func = distance_func  # LogReg to use the learning-based linking model. 'def' to use the baseline from SBM
        self.clf_thr = clf_thr  # threshold to filter out detection linkings
        self.clf_mode = clf_mode  # Relation between the logreg score and the semmantic similarity. 'dot' recommended
        self.appearance_matching = appearance_matching  # True to use appearance similarity features

        self.recoordinate = recoordinate  # True to perform a recordinating step
        self.recoordinate_std = recoordinate_std  # Strength of the recoordinating step
        self.store_coco = store_coco  # True to store predictions with the COCO format
        self.store_imdb = store_imdb  # True to store predictions with the IMDB format. Needed for evaluation

        self.pairs = []
        self.unmatched_pairs = []
        self.window_size = window_size

        if self.distance_func == 'def':
            self.match_func = self.distance_def
        elif self.distance_func == 'logreg':
            if self.appearance_matching:
                print('Loading clf matching model:', './REPP_models/matching_model_logreg_appearance.pckl')
                self.clf_match, self.matching_feats = pickle.load(
                    open('./REPP_models/matching_model_logreg_appearance.pckl', 'rb'))
            else:
                print('Loading clf matching model:', './REPP_models/matching_model_logreg.pckl')
                self.clf_match, self.matching_feats = pickle.load(
                    open('./REPP_models/matching_model_logreg.pckl', 'rb'))
            self.match_func = self.distance_logreg
        else:
            raise ValueError('distance_func not recognized:', self.distance_func)

        if self.store_imdb:
            imageset_filename = motion_utils.annotations_to_imageset(annotations_filename)
            with open(imageset_filename, 'r') as f: image_set = f.read().splitlines()
            self.image_set = {l.split()[0]: int(l.split()[1]) for l in image_set}

    def distance_def(self, p1, p2):
        iou = get_iou(p1['bbox'][:], p2['bbox'][:])
        score = np.dot(p1['scores'], p2['scores'])
        div = iou * score
        if div == 0: return INF
        return 1 / div

    # Computes de linking score between a pair of detections
    def distance_logreg(self, p1, p2):
        pair_features = get_pair_features(p1, p2, self.matching_feats)  # , image_size[0], image_size[1]
        score = self.clf_match.predict_proba(np.array([[pair_features[col] for col in self.matching_feats]]))[:, 1]
        if score < self.clf_thr: return INF

        if self.clf_mode == 'max':
            score = p1['scores'].max() * p2['scores'].max() * score
        elif self.clf_mode == 'dot':
            score = np.dot(p1['scores'], p2['scores']) * score
        elif self.clf_mode == 'dot_plus':
            score = np.dot(p1['scores'], p2['scores']) + score
        elif self.clf_mode == 'def':
            return self.distance_def(p1, p2)
        elif self.clf_mode == 'raw':
            pass
        else:
            raise ValueError('error post_clf')
        return 1 - score

    # Return a list of pairs of frames linked across frames
    def get_video_pairs(self, preds_frame):
        num_frames = len(preds_frame)
        frames = list(preds_frame.keys())
        frames = sorted(frames, key=int)

        pairs, unmatched_pairs = [], []
        for i in range(num_frames - 1):

            pairs_i = []
            frame_1, frame_2 = frames[i], frames[i + 1]
            preds_frame_1, preds_frame_2 = preds_frame[frame_1], preds_frame[frame_2]
            num_preds_1, num_preds_2 = len(preds_frame_1), len(preds_frame_2)

            # Any frame has no preds -> save empty pairs
            if num_preds_1 != 0 and num_preds_2 != 0:
                # Get distance matrix
                distances = np.zeros((num_preds_1, num_preds_2))
                for i, p1 in enumerate(preds_frame_1):
                    for j, p2 in enumerate(preds_frame_2):
                        distances[i, j] = self.match_func(p1, p2)

                # Get frame pairs
                pairs_i = self.solve_distances_def(distances, maximization_problem=False)

            unmatched_pairs_i = [i for i in range(num_preds_1) if i not in [p[0] for p in pairs_i]]
            pairs.append(pairs_i)
            unmatched_pairs.append(unmatched_pairs_i)

        return pairs, unmatched_pairs

    # Return a list of pairs of frames linked across frames
    def get_video_pairs_rt(self, preds_frame, first_time: bool = False):

        if first_time:
            self.pairs, self.unmatched_pairs = self.get_video_pairs(preds_frame)
        else:
            self.pairs.pop(0)
            self.unmatched_pairs.pop(0)

            frames = list(preds_frame.keys())
            frames = sorted(frames, key=int)
            pairs_i = []
            frame_1, frame_2 = frames[-2], frames[-1]
            preds_frame_1, preds_frame_2 = preds_frame[frame_1], preds_frame[frame_2]
            num_preds_1, num_preds_2 = len(preds_frame_1), len(preds_frame_2)
            if num_preds_1 != 0 and num_preds_2 != 0:
                # Get distance matrix
                distances = np.zeros((num_preds_1, num_preds_2))
                for i, p1 in enumerate(preds_frame_1):
                    for j, p2 in enumerate(preds_frame_2):
                        distances[i, j] = self.match_func(p1, p2)
                pairs_i = self.solve_distances_def(distances, maximization_problem=False)

            unmatched_pairs_i = [i for i in range(num_preds_1) if i not in [p[0] for p in pairs_i]]

            self.pairs.append(pairs_i)
            self.unmatched_pairs.append(unmatched_pairs_i)

        return copy.deepcopy(self.pairs), copy.deepcopy(self.unmatched_pairs)

    # Solve distance matrix and return a list of pair of linked detections from two consecutive frames
    def solve_distances_def(self, distances, maximization_problem):
        pairs = []
        if maximization_problem:
            while distances.min() != -1:
                inds = np.where(distances == distances.max())
                a, b = inds if len(inds[0]) == 1 else (inds[0][0], inds[1][0])
                a, b = int(a), int(b)
                pairs.append((a, b))
                distances[a, :] = -1
                distances[:, b] = -1
        else:
            while distances.min() != INF:
                inds = np.where(distances == distances.min())
                a, b = inds if len(inds[0]) == 1 else (inds[0][0], inds[1][0])
                a, b = int(a), int(b)
                pairs.append((a, b))
                distances[a, :] = INF
                distances[:, b] = INF

        return pairs

    # Create tubelets from list of linked pairs
    def get_tubelets(self, preds_frame, pairs):

        num_frames = len(preds_frame)
        frames = list(preds_frame.keys())
        tubelets, tubelets_count = [], 0

        first_frame = 0

        while first_frame != num_frames - 1:
            ind = None
            for current_frame in range(first_frame, num_frames - 1):

                # Continue tubelet
                if ind is not None:
                    pair = [p for p in pairs[current_frame] if p[0] == ind]
                    # Tubelet ended
                    if len(pair) == 0:
                        tubelets[tubelets_count].append((current_frame, preds_frame[frames[current_frame]][ind]))
                        tubelets_count += 1
                        ind = None
                        break

                        # Continue tubelet
                    else:
                        pair = pair[0]
                        del pairs[current_frame][pairs[current_frame].index(pair)]
                        tubelets[tubelets_count].append((current_frame, preds_frame[frames[current_frame]][ind]))
                        ind = pair[1]

                # Looking for a new tubelet
                else:
                    # No more candidates in current frame -> keep searching
                    if len(pairs[current_frame]) == 0:
                        first_frame = current_frame + 1
                        continue
                    # Beginning a new tubelet in current frame
                    else:
                        pair = pairs[current_frame][0]
                        del pairs[current_frame][0]
                        tubelets.append([(current_frame,
                                          preds_frame[frames[current_frame]][pair[0]])])
                        ind = pair[1]

            # Tubelet has finished in the last frame
            if ind != None:
                tubelets[tubelets_count].append((current_frame + 1, preds_frame[frames[current_frame + 1]][ind]))  # 4
                tubelets_count += 1
                ind = None

        return tubelets

    # Performs the re-scoring refinment
    def rescore_tubelets(self, tubelets):
        for t_num in range(len(tubelets)):
            t_scores = [p['scores'] for _, p in tubelets[t_num]]
            new_scores = np.mean(t_scores, axis=0)
            for i in range(len(tubelets[t_num])): tubelets[t_num][i][1]['scores'] = new_scores

            for i in range(len(tubelets[t_num])):
                if 'emb' in tubelets[t_num][i][1]: del tubelets[t_num][i][1]['emb']

        return tubelets

    # Performs de re-coordinating refinment
    def recoordinate_tubelets_full(self, tubelets, ms=-1):

        if ms == -1: ms = 40
        for t_num in range(len(tubelets)):
            t_coords = np.array([p['bbox'] for _, p in tubelets[t_num]])
            w = signal.gaussian(len(t_coords) * 2 - 1, std=self.recoordinate_std * 100 / ms)
            w /= sum(w)

            for num_coord in range(4):
                t_coords[:, num_coord] = ndimage.convolve(t_coords[:, num_coord], w, mode='reflect')

            for num_bbox in range(len(tubelets[t_num])):
                tubelets[t_num][num_bbox][1]['bbox'] = t_coords[num_bbox, :].tolist()

        return tubelets

    # Extracts predictions from tubelets
    def tubelets_to_predictions(self, tubelets_video, preds_format):

        preds, track_id_num = [], 0
        for tub in tubelets_video:
            for _, pred in tub:
                for cat_id, s in enumerate(pred['scores']):
                    if s < self.min_pred_score: continue
                    if preds_format == 'coco':
                        preds.append({
                            'image_id': pred['image_id'],
                            'bbox': list(map(float, pred['bbox'])),
                            'score': float(s),
                            'category_id': cat_id,
                            'track_id': track_id_num,
                        })
                    elif preds_format == 'imdb':
                        preds.append('{} {} {} {} {} {} {}'.format(
                            self.image_set['/'.join(pred['image_id'].split('/')[-2:])],
                            cat_id + 1,
                            float(s),
                            pred['bbox'][0], pred['bbox'][1],
                            pred['bbox'][0] + pred['bbox'][2], pred['bbox'][1] + pred['bbox'][3]
                        ))
                    else:
                        raise ValueError('Predictions format not recognized')
            track_id_num += 1
        return preds

    def __call__(self, video_predictions, ft: bool = False):
        # Filter out low-score predictions
        for frame in video_predictions.keys():
            video_predictions[frame] = [p for p in video_predictions[frame] if
                                        max(p['scores']) >= self.min_tubelet_score]

        video_predictions = dict(sorted(video_predictions.items()))

        # import time

        # start_time = time.time()
        # pairs, unmatched_pairs = self.get_video_pairs(video_predictions)
        pairs, unmatched_pairs = self.get_video_pairs_rt(video_predictions, first_time=ft)
        # print("========== get_video_pairs:\t\t%.6f sec ==========" % (time.time() - start_time))

        # start_time = time.time()
        tubelets = self.get_tubelets(video_predictions, pairs)
        # print("========== get_tubelets:\t\t%.6f sec ==========" % (time.time() - start_time))

        # start_time = time.time()
        tubelets = self.rescore_tubelets(tubelets)
        # print("========== rescore_tubelets:\t%.6f sec ==========\n" % (time.time() - start_time))

        if self.recoordinate: tubelets = self.recoordinate_tubelets_full(tubelets)

        if self.add_unmatched:
            print('Adding unmatched')
            tubelets += self.add_unmatched_pairs_as_single_tubelet(unmatched_pairs, video_predictions)

        if self.store_coco:
            predictions_coco = self.tubelets_to_predictions(tubelets, 'coco')
        else:
            predictions_coco = []
        if self.store_imdb:
            predictions_imdb = self.tubelets_to_predictions(tubelets, 'imdb')
        else:
            predictions_imdb = []

        return predictions_coco, predictions_imdb


def get_vid_preds(path: str):
    from PIL import Image
    from pathlib import Path

    folder = Path(path)
    images = [x for x in folder.iterdir() if x.name.endswith('.png')]
    images.sort(key=lambda x: int(x.name[:x.name.index(x.suffix)]))
    bb_files = [x for x in folder.iterdir() if x.name.endswith('.txt')]
    bb_files.sort(key=lambda x: int(x.name[:x.name.index(x.suffix)]))

    preds = {}
    for idx, img in enumerate(images):
        im = Image.open(img)
        annotation_file = bb_files[idx]
        # get frame number
        num = annotation_file.name
        num = num[:num.index(annotation_file.suffix)]
        # read boxes
        with annotation_file.open() as f:
            lines = [l.split(' ') for l in f.read().splitlines()]
        # convert boxes to COCO format
        boxes = []
        for l in lines:
            box_width = float(l[3]) * im.width
            box_height = float(l[4]) * im.height
            x = (float(l[1]) * im.width) - (box_width / 2)
            y = (float(l[2]) * im.height) - (box_height / 2)
            # create box entry
            entry = {'image_id': num, 'bbox': [x, y, box_width, box_height],
                     'bbox_center': [float(l[1]), float(l[2])], 'scores': [float(l[5])]}
            # add to list of frame boxes
            boxes.append(entry)
        # add to list of frames predictions
        preds[int(num)] = boxes
    return preds


if __name__ == '__main__':

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Apply REPP to a saved predictions file')
    parser.add_argument('--repp_cfg', help='repp cfg filename', type=str)
    parser.add_argument('--predictions_file', help='predictions filename', type=str)
    # parser.add_argument('--from_python_2', help='predictions filename', action='store_true')
    parser.add_argument('--evaluate', help='evaluate motion mAP', action='store_true')
    parser.add_argument('--annotations_filename', help='ILSVRC annotations. Needed for ILSVRC evaluation',
                        required=False, type=str)
    parser.add_argument('--path_dataset', help='path of the Imagenet VID dataset. Needed for ILSVRC evaluation',
                        required=False, type=str)
    parser.add_argument('--store_coco', help='store processed predictions in coco format', action='store_true')
    parser.add_argument('--store_imdb', help='store processed predictions in imdb format', action='store_true')
    parser.add_argument('--store', help='folder path to store processed predications into individual files', type=Path)
    parser.add_argument('--window_size', help='size of buffer', type=int)
    parser.add_argument('--vid_path', help='path to folder with video frames', type=str)
    args = parser.parse_args()

    assert not (args.evaluate and args.annotations_filename is None), \
        'Annotations filename is required for ILSVRC evaluation'
    assert not (args.evaluate and args.path_dataset is None), \
        'Dataset path is required for ILSVRC evaluation'

    print(' * Loading REPP cfg')
    repp_params = json.load(open(args.repp_cfg, 'r'))
    print(repp_params)
    predictions_file_out = args.predictions_file.replace('.pckl', '_repp')

    repp = REPP(**repp_params, window_size=args.window_size, annotations_filename=args.annotations_filename,
                store_coco=args.store_coco, store_imdb=args.store_imdb or args.evaluate)

    from tqdm import tqdm
    import sys

    total_preds_coco, total_preds_imdb = [], []
    print(' * Applying repp')
    if args.evaluate:
        with open(args.annotations_filename, 'r') as f: annotations = sorted(f.read().splitlines())
        pbar = tqdm(total=len(annotations), file=sys.stdout)

    # simulate real-time video feed
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # trigger: pred=detection(frame)

    vid_preds = get_vid_preds(args.vid_path)
    folder = Path(args.vid_path)
    images = [x for x in folder.iterdir() if x.name.endswith('.png')]
    images.sort(key=lambda x: int(x.name[:x.name.index(x.suffix)]))

    window_preds = {}
    window_numbers = []

    fig, ax = plt.subplots()
    plt.show(block=False)

    first_time = True
    total_preds = {}

    for img in images:
        # delete oldest predication if window is full
        if len(window_preds) == repp.window_size:
            num = window_numbers.pop(0)
            del window_preds[num]
        # add predication to window buffer
        frame_number = int(img.name[:img.name.index(img.suffix)])
        window_numbers.append(frame_number)
        window_preds[frame_number] = vid_preds[frame_number]

        # show frame
        ax.imshow(Image.open(img))
        ax.set_title(img.name)

        # display detected boxes (window_preds will be modified in REPP)
        for pred_box in window_preds[frame_number]:
            # only show boxes that exceeds min_tubelet_score
            if max(pred_box['scores']) >= repp_params['min_tubelet_score']:
                post_box = patches.Rectangle((pred_box['bbox'][0], pred_box['bbox'][1]),
                                             pred_box['bbox'][2], pred_box['bbox'][3],
                                             linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(post_box)

        latest_pred = None
        if len(window_preds) == repp.window_size:
            predictions_coco, predictions_imdb = repp(window_preds, first_time)
            first_time = False
            total_preds_coco += predictions_coco
            total_preds_imdb += predictions_imdb

            # get predictions of frame
            latest_pred = [e for e in predictions_coco if e['image_id'] == str(frame_number)]
            # latest_pred = predictions_coco[-1] if len(predictions_coco) > 0 else None

        # TODO: Liefert predictions_coco immer nur eine Box pro frame?
        # display boxes of post-processing
        if latest_pred is not None and len(latest_pred) > 0:
            for lp in latest_pred:
                post_box = patches.Rectangle((lp['bbox'][0], lp['bbox'][1]), lp['bbox'][2], lp['bbox'][3],
                                             linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(post_box)

            if args.store:
                lines = []
                for lp in latest_pred:
                    lines.append('{class_name} {conf} {left} {top} {right} {bottom}'.format(
                    class_name='polyp', conf=lp['score'],left=lp['bbox'][0], top=lp['bbox'][1],
                    right=lp['bbox'][0] + lp['bbox'][2], bottom=lp['bbox'][1] + lp['bbox'][3]))
                    total_preds[frame_number] = lines

        # pause an reload next frame
        plt.pause(0.1)
        ax.clear()
        fig.canvas.draw()

    if args.store:
        print(' * Dumping predictions as individual files:', args.store)
        args.store.mkdir()
        for img_id in total_preds.keys():
            with open(args.store.joinpath('{}.txt'.format(img_id)), mode='w') as f:
                f.write('\n'.join(total_preds[img_id]))

    # if args.store_imdb:
    #     print(' * Dumping predictions with the IMDB format:', predictions_file_out + '_imdb.txt')
    #     with open(predictions_file_out + '_imdb.txt', 'w') as f:
    #         for p in total_preds_imdb: f.write(p + '\n')
    #
    # if args.store_coco:
    #     print(' * Dumping predictions with the COCO format:', predictions_file_out + '_coco.json')
    #     json.dump(total_preds_coco, open(predictions_file_out + '_coco.json', 'w'))

    # if args.evaluate:
    #
    #     print(' * Evaluating REPP predictions')
    #
    #     import sys
    #
    #     sys.path.append('ObjectDetection_mAP_by_motion')
    #     from ObjectDetection_mAP_by_motion import motion_utils
    #     from ObjectDetection_mAP_by_motion.imagenet_vid_eval_motion import get_motion_mAP
    #     import os
    #
    #     stats_file_motion = predictions_file_out.replace('preds', 'stats').replace('.txt', '.json')
    #     motion_iou_file_orig = './ObjectDetection_mAP_by_motion/imagenet_vid_groundtruth_motion_iou.mat'
    #     imageset_filename_orig = os.path.join(args.path_dataset, 'ImageSets/VID/val.txt')
    #
    #     if os.path.isfile(stats_file_motion): os.remove(stats_file_motion)
    #     stats = get_motion_mAP(args.annotations_filename, args.path_dataset,
    #                            predictions_file_out + '_imdb.txt', stats_file_motion,
    #                            motion_iou_file_orig, imageset_filename_orig)
    #
    #     print(stats)
    #     print(' * Stats stored:', stats_file_motion)
