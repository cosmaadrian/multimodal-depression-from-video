import os
import sys
import cv2
import json
from argparse import ArgumentParser
from threading import Thread
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import collate, scatter
from tqdm import tqdm
import math
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose
from mmcv.cnn.utils.flops_counter import add_flops_counting_methods, flops_to_string, params_to_string
from mmdet.core import build_assigner
from mmcv import ConfigDict
import time
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint',help='Checkpoint file')
    parser.add_argument(
        '--root', default="/home/dgimeno/phd/old_perceiving-depression/databases/D-vlog/old_data/videos/", help='Path to image file')
    parser.add_argument(
        '--dest-root', default="./home/dgimeno/phd/old_perceiving-depression/databases/D-vlog/old_data/blinking_patterns/", help='Path to image file')
    parser.add_argument(
        '--dest-no-idxs-root', default="./home/dgimeno/phd/old_perceiving-depression/databases/D-vlog/old_data/no_blink_idxs/", help='Path to image file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        "--left-index", type=int, default=0, help="Position index from where to start to process videos")
    parser.add_argument(
        "--right-index", type=int, default=861, help="Position index where to finish to process videos")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def load_datas(data, test_pipeline, datas):
    datas.append(test_pipeline(data))

def compute_iou(assigner, previous, cur):
    iou = assigner.assign(previous, cur)
    return iou

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()

def blink_patterns_detection(videoID):
    # getting video frames
    video_root_dir = os.path.join(args.root, videoID)
    imgs = sorted(os.listdir(video_root_dir))

    video_det_bboxes = []
    video_det_blinks = []

    datas, threads = [], []
    video_length = len(imgs)

    # computing number of clips according to the 'clip_len' variable
    if video_length <= clip_len:
        clip_num = 1
    else:
        clip_num = math.ceil((video_length-clip_len)/stride) + 1

    # processing clips
    for clip_index in range(0, clip_num):
        # if it is not the last clip
        if clip_index!=clip_num-1:
            cur_clip = imgs[clip_index*stride:clip_index*stride + clip_len]
            clip_overlap = clip_len - stride
        # If it is the last clip, take the last clip_num frame backwards
        else:   
            cur_clip = imgs[-clip_len:]
            if (video_length-clip_len)%stride:
                clip_overlap = clip_len - (video_length-clip_len)%stride
            else:
                clip_overlap = clip_len - stride

        # loading clip frames
        threads = []
        datas = []
        for img in cur_clip:
            data = dict(img_info=dict(filename=img), img_prefix=video_root_dir)
            threads.append(Thread(target=load_datas, args=(data, test_pipeline, datas)))
            threads[-1].start()
        for thread in threads:
            thread.join()

        datas = sorted(datas, key=lambda x:x['img_metas'].data['filename'])
        datas = collate(datas, samples_per_gpu=len(cur_clip)) # form the input batch
        datas['img_metas'] = datas['img_metas'].data
        datas['img'] = datas['img'].data
        datas = scatter(datas, [args.device])[0]

        # processing clip
        with torch.no_grad():
            model.start_flops_count()
            (det_bboxes, det_labels), det_blinks = model(
                return_loss=False,
                rescale=True,
                format=False,
                **datas)
            model.stop_flops_count()
            
        # performing inter-clip matching
        # if it is not the first clip
        if clip_index!=0:

            previous_det_bboxes_for_match = video_det_bboxes[:,-clip_overlap:,:]

            det_bboxes = torch.stack(det_bboxes)
            det_blinks = torch.stack(det_blinks)
            det_bboxes = det_bboxes.permute(1, 0, 2)
            det_blinks = det_blinks.permute(1, 0, 2)

            # filter prediction results by a confidence threshold
            det_blinks = det_blinks[torch.where(det_bboxes[:, 0, -1] > person_threshold)]
            det_bboxes = det_bboxes[torch.where(det_bboxes[:, 0, -1] > person_threshold)]

            previous_person_num = previous_det_bboxes_for_match.size(0)

            # Next, perform pre-padding foe the upcoming clip, length=clip_len-clip_overlap bbox:[0,0,0,0], blink:[0]
            next_padding_bboxes = torch.zeros([previous_person_num,clip_len-clip_overlap,5]).to(video_det_bboxes.device) 
            video_det_bboxes = torch.cat((video_det_bboxes, next_padding_bboxes),1)
            
            next_padding_blinks = torch.zeros([previous_person_num,clip_len-clip_overlap,1]).to(video_det_blinks.device)
            video_det_blinks = torch.cat((video_det_blinks, next_padding_blinks),1)

            # perform matching
            previous_det_bboxes_for_iou = previous_det_bboxes_for_match.permute(1,0,2)
            det_boxes_for_iou = det_bboxes.permute(1,0,2)[:clip_overlap,:,:]
            mat = assigner.assign(previous_det_bboxes_for_iou, det_boxes_for_iou, datas['img_metas'][0][0])

            det_assigned = torch.zeros(det_bboxes.shape[0])
            for i in range(0, min(mat.shape)):

                tar = np.unravel_index(mat.argmax(), mat.shape)
                if mat[tar[0], tar[1]] < iou_threshold:  # below the threshold value, indicating the appearance of new id, added to the registry

                    new_person_bboxes = det_bboxes[tar[1], -(clip_len):, :].unsqueeze(0) 
                    new_person_blinks = det_blinks[tar[1], -(clip_len):, :].unsqueeze(0)
                    new_person_pre_bboxes = torch.zeros([1,video_det_bboxes.size(1)-(clip_len),5]).to(video_det_bboxes.device)
                    new_person_pre_blinks = torch.zeros([1,video_det_blinks.size(1)-(clip_len),1]).to(video_det_blinks.device)

                    new_person_bboxes = torch.cat((new_person_pre_bboxes, new_person_bboxes), 1)
                    new_person_blinks = torch.cat((new_person_pre_blinks, new_person_blinks), 1)
                    video_det_bboxes = torch.cat((video_det_bboxes, new_person_bboxes), 0)
                    video_det_blinks = torch.cat((video_det_blinks, new_person_blinks), 0)
                    mat[tar[0],:] = -10000
                    mat[:,tar[1]] = -10000
                    det_assigned[tar[1]] = 1   # Mark the new prediction result for index = tar[1] has been processed
                else: # the current match is satisfying the threshold
                    mat[tar[0], :] = -10000
                    mat[:, tar[1]] = -10000
                    video_det_bboxes[tar[0], -(clip_len-clip_overlap):, :] = det_bboxes[tar[1], -(clip_len-clip_overlap):, :]
                    video_det_blinks[tar[0], -(clip_len-clip_overlap):, :] = det_blinks[tar[1], -(clip_len-clip_overlap):, :]
                    # Average the result on overlapping parts
                    video_det_bboxes[tar[0], -clip_len:-(clip_len-clip_overlap), :] = (video_det_bboxes[tar[0], -clip_len:-(clip_len-clip_overlap), :] + det_bboxes[tar[1], -clip_len:-(clip_len-clip_overlap), :])/2
                    video_det_blinks[tar[0], -clip_len:-(clip_len-clip_overlap), :] = (video_det_blinks[tar[0], -clip_len:-(clip_len-clip_overlap), :] + det_blinks[tar[1], -clip_len:-(clip_len-clip_overlap), :])/2
                    det_assigned[tar[1]] = 1    # Mark the new prediction result for index = tar[1] has been processed
            for index in range(0, det_assigned.shape[0]):
                if det_assigned[index] == 0: # This new prediction result has not been processed yet and is a new id

                    new_person_bboxes = det_bboxes[index, -(clip_len):, :].unsqueeze(0)  
                    new_person_blinks = det_blinks[index, -(clip_len):, :].unsqueeze(0)
                    new_person_pre_bboxes = torch.zeros([1,video_det_bboxes.size(1)-(clip_len),5]).to(video_det_bboxes.device)
                    new_person_pre_blinks = torch.zeros([1,video_det_blinks.size(1)-(clip_len),1]).to(video_det_blinks.device)

                    new_person_bboxes = torch.cat((new_person_pre_bboxes, new_person_bboxes), 1)
                    new_person_blinks = torch.cat((new_person_pre_blinks, new_person_blinks), 1)
                    video_det_bboxes = torch.cat((video_det_bboxes, new_person_bboxes), 0)
                    video_det_blinks = torch.cat((video_det_blinks, new_person_blinks), 0)
                    
                    det_assigned[index] = 1    # Mark the new prediction result for index = tar[1] has been processed

                
        # for the first video_cilp
        else:
            det_bboxes = torch.stack(det_bboxes)
            det_blinks = torch.stack(det_blinks)
            det_bboxes = det_bboxes.permute(1,0,2)
            det_blinks = det_blinks.permute(1,0,2)

            video_det_blinks = det_blinks[torch.where(det_bboxes[:,0,-1]>person_threshold)]
            video_det_bboxes = det_bboxes[torch.where(det_bboxes[:,0,-1]>person_threshold)]


    # video bounding boxes and blinks
    det_bboxes =  video_det_bboxes.permute(1,0,2)
    det_blinks = video_det_blinks.permute(1,0,2)

    # computing blinking patterns and no blinking indeces
    blinking_patterns = (det_blinks > eyeblink_threshold).squeeze().long().detach().cpu().numpy()
    no_blink_idxs = np.where(blinking_patterns==0)[0]

    # saving numpy arrays
    dest_path = os.path.join(args.dest_root, videoID + ".npz")
    np.savez_compressed(dest_path, data=blinking_patterns)

    dest_no_idxs_path = os.path.join(args.dest_no_idxs_root, videoID + ".npz")
    np.savez_compressed(dest_no_idxs_path, data=no_blink_idxs)

if __name__ == "__main__":
    args = parse_args()

    eyeblink_threshold = 0.3

    os.makedirs(args.dest_root, exist_ok=True)
    os.makedirs(args.dest_no_idxs_root, exist_ok=True)

    # building blinking detector model
    model = init_detector(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=args.cfg_options)

    model = add_flops_counting_methods(model)
    cfg = model.cfg
    test_pipeline = Compose(cfg.data.test.pipeline)

    # configuration details
    results = []
    clip_len = 36   # define the video clip length for a single forward propagation
    stride = 18     # define the stride
    matcher_config = ConfigDict(dict(type='FaceLinkerCalculator',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='InferenceBBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='iou',
                                  weight=1.0)))
    assigner = build_assigner(matcher_config)
    iou_threshold = 0.2
    person_threshold = 0.5

    # processing videos
    videos = sorted(os.listdir(args.root))[args.left_index:args.right_index]
    # videos = [line.strip() for line in open("./video_completed.txt", "r").readlines()][args.left_index:args.right_index]

    for videoID in tqdm(videos):
        blink_patterns_detection(videoID)
