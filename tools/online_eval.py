# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from vis_utils import *
import pandas as pd
import time
import copy


# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"
result = pd.read_csv('/share/j_sun/xy468/sam2/eval/unlimited_user_prompt/VOST/results.csv')
result = result[1:]
result = result[result['prompt_num'] > 0]
result['obj'] = result['obj'].apply(lambda x: int(x))
# group result by sequence
result = result.groupby('sequence').agg({'obj': list}).reset_index() 
# change it to dict
video_obj_dict = dict(zip(result['sequence'], result['obj']))

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[(object_ids > 0)].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    sufix = '.bmp' if 'endovis' in input_mask_dir else '.png'
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}{sufix}")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}{sufix}"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            # output_mask[output_mask > 0] = object_id
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


def get_input_per_obj(
    base_video_dir,
    input_mask_dir,
    video_name,
    use_all_masks=False,
    per_obj_png_file=False,
    track_object_appearing_later_in_video=False,
):
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png', '.bmp']
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].replace('frame', '').split('_')[-1]))
    if not track_object_appearing_later_in_video:
        frame_names = frame_names[:1]

    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    sufix = '.bmp' if 'endovis' in base_video_dir else '.png'
    for idx, name in enumerate(frame_names):
        if per_obj_png_file or os.path.exists(
            os.path.join(input_mask_dir, video_name, f"{name}{sufix}")
        ):
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[idx],
                per_obj_png_file=per_obj_png_file,
                allow_missing=True,
            )
            for object_id, object_mask in per_obj_input_mask.items():
                # skip empty masks
                if not np.any(object_mask):
                    continue
                # if `use_all_masks=False`, we only use the first mask for each object
                if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                    continue
                print(f"adding mask from frame {idx} as input for {object_id=}")
                inputs_per_object[object_id][idx] = object_mask
    return input_palette, inputs_per_object

def vos_online_one_pass_user_correction(
    predictor,
    inference_state,
    inputs_per_object,
    obj_gt_mask_per_object,
    video_segments,
    object_id,
    score_thresh,
    correct_threshold,
    correct_type,
    correction_num_limit,
    max_num_click_per_frame=3,
):
    input_frame_inds = sorted(inputs_per_object[object_id])
    cur_correction_num = 0
    output_gt_iou_per_object = []
    output_prompt_per_object = {}
    finish_process_obj = False
    predictor.reset_state(inference_state)
    for input_frame_idx in input_frame_inds:
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=input_frame_idx,
            obj_id=object_id,
            mask=inputs_per_object[object_id][input_frame_idx],
        )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    # run propagation throughout the video and collect the results in a dict
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=min(input_frame_inds),
        reverse=False,
    ):
        video_segments[out_frame_idx][object_id] = (out_mask_logits[0] > 0).cpu().numpy()
        mask = (out_mask_logits[0] > score_thresh).cpu().numpy()
        mask = mask.reshape(height, width)
        if obj_gt_mask_per_object[object_id][out_frame_idx] is not None:
            gt_mask = obj_gt_mask_per_object[object_id][out_frame_idx]
            gt_iou = get_iou(mask, gt_mask)
            
            input_gt_mask = ((gt_mask > 0) & (gt_mask != 255)).astype(np.uint8)
            if gt_iou < correct_threshold and (correction_num_limit == None or cur_correction_num < correction_num_limit):
                print(f"frame: {out_frame_idx}, object: {object_id}, gt_iou: {gt_iou}, cur_correction_num: {cur_correction_num}, p: {correction_num_limit}")
                if correct_type == 'mask':
                    _, _, correct_res_mask = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=out_frame_idx,
                        obj_id=object_id,
                        mask=input_gt_mask,
                        run_mem_encoder=True
                    )
                    gt_iou = 1
                else:
                    correct_result = predictor.correct_by_iou(
                        inference_state=inference_state,
                        frame_idx=out_frame_idx,
                        obj_id=object_id,
                        cur_iou=gt_iou,
                        gt_mask=gt_mask,
                        pred_mask_tensor=out_mask_logits>0,
                        correct_threshold=correct_threshold,
                        max_num_clicks=max_num_click_per_frame,
                    )
                    gt_iou = correct_result['correct_iou']
                    correct_res_mask = correct_result['correct_res_mask']
                video_segments[out_frame_idx][object_id] = (correct_res_mask[0] > 0).cpu().numpy()
                output_gt_iou_per_object.append(gt_iou)
                # predictor.propagate_in_video_preflight(inference_state)
                output_prompt_per_object[out_frame_idx] = {
                    'prev_iou': float(gt_iou), 
                    'prev_mask': bmask_to_rle(mask.astype(np.bool)), 
                    'correct_type': correct_result['correct_type'], 
                    'correct_num_clicks': correct_result['correct_num_clicks'],
                }
                cur_correction_num += 1
            output_gt_iou_per_object.append(gt_iou)
    if min(output_gt_iou_per_object) >= correct_threshold:
        finish_process_obj = True


    return output_gt_iou_per_object, output_prompt_per_object, finish_process_obj

def vos_online_one_pass_lora_correction(
    predictor,
    inference_state,
    inputs_per_object,
    obj_gt_mask_per_object,
    video_segments,
    object_id,
    score_thresh,
    correct_threshold,
    correct_type,
    correction_num_limit,
    max_num_click_per_frame=3,
    max_lora_num=3,
):
    trained_lora = False
    input_frame_inds = sorted(inputs_per_object[object_id])
    output_gt_iou_per_object = []
    output_prompt_per_object = {}
    finish_process_obj = False
    cur_correction_num = 0
    predictor.reset_state(inference_state)
    predictor.reset_lora()
    for input_frame_idx in input_frame_inds:
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=input_frame_idx,
            obj_id=object_id,
            mask=inputs_per_object[object_id][input_frame_idx],
        )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    # run propagation throughout the video and collect the results in a dict
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=min(input_frame_inds),
        reverse=False,
    ):
        video_segments[out_frame_idx][object_id] = (out_mask_logits[0] > 0).cpu().numpy()
        mask = (out_mask_logits[0] > score_thresh).cpu().numpy().reshape(height, width)
        if obj_gt_mask_per_object[object_id][out_frame_idx] is not None:
            gt_mask = obj_gt_mask_per_object[object_id][out_frame_idx]
            gt_iou = get_iou(mask, gt_mask)
            input_gt_mask = ((gt_mask > 0) & (gt_mask != 255)).astype(np.uint8)
            if gt_iou < correct_threshold: 
            # and (correction_num_limit == None or cur_correction_num < correction_num_limit):
                print(f"frame: {out_frame_idx}, object: {object_id}, gt_iou: {gt_iou}, cur_correction_num: {cur_correction_num}, p: {correction_num_limit}")
                if not trained_lora:
                    if correct_type == 'mask':
                        _, _, correct_res_mask = predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=out_frame_idx,
                            obj_id=object_id,
                            mask=input_gt_mask,
                            run_mem_encoder=True
                        )
                    else:
                        correct_result = predictor.correct_by_iou(
                            inference_state=inference_state,
                            frame_idx=out_frame_idx,
                            obj_id=object_id,
                            cur_iou=gt_iou,
                            gt_mask=gt_mask,
                            pred_mask_tensor=out_mask_logits>0,
                            correct_threshold=correct_threshold
                        )
                        user_correct_type = correct_result['correct_type']
                        user_correct_num_clicks = correct_result['correct_num_clicks']
                        gt_iou = correct_result['correct_iou']
                        correct_res_mask = correct_result['correct_res_mask']
                    video_segments[out_frame_idx][object_id] = (correct_res_mask[0] > 0).cpu().numpy()
                    torch.cuda.synchronize()
                    start_time = time.time()
                    correction_type = 'user_correction'
                    mask_decoder_lora = copy.deepcopy(predictor.sam_mask_decoder)
                    predictor.convert_to_lora(mask_decoder_lora.transformer)
                    predictor.freeze_non_lora(mask_decoder_lora)
                    if predictor.train_lora(mask_decoder_lora, gt_mask, training_epoch=100):
                        trained_lora = True
                    torch.cuda.synchronize()
                    end_time = time.time()
                    train_lora_time = end_time - start_time
                    train_lora_type = 'init'
                    lora_inference_time = None
                    cur_correction_num += 1
                else:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    successful_predict, best_lora_iou, best_lora_idx, best_lora_predicted_mask_score = predictor.lora_predict(
                        inference_state, 
                        out_frame_idx, 
                        gt_mask, 
                        correct_threshold
                    )
                    torch.cuda.synchronize()
                    end_time = time.time()
                    lora_inference_time = end_time - start_time
                    
                    if not successful_predict and (correction_num_limit == None or cur_correction_num < correction_num_limit):
                        if correct_type == 'mask':
                            _, _, correct_res_mask = predictor.add_new_mask(
                                inference_state=inference_state,
                                frame_idx=out_frame_idx,
                                obj_id=object_id,
                                mask=input_gt_mask,
                                run_mem_encoder=True
                            )
                        else:
                            correct_result = predictor.correct_by_iou(
                                inference_state=inference_state,
                                frame_idx=out_frame_idx,
                                obj_id=object_id,
                                cur_iou=gt_iou,
                                gt_mask=gt_mask,
                                pred_mask_tensor=out_mask_logits>0,
                                correct_threshold=correct_threshold,
                                max_num_clicks=max_num_click_per_frame,
                            )
                            user_correct_type = correct_result['correct_type']
                            user_correct_num_clicks = correct_result['correct_num_clicks']
                            gt_iou = correct_result['correct_iou']
                            correct_res_mask = correct_result['correct_res_mask']
                        video_segments[out_frame_idx][object_id] = (correct_res_mask[0] > 0).cpu().numpy()
                        # predictor.train_lora(predictor.multi_lora[best_lora_idx], gt_mask, training_epoch=10)
                        correction_type = 'user_correction'
                        if 0 <= best_lora_iou <= 0.4 and len(predictor.multi_lora) < max_lora_num:
                            training_epoch = 100
                            torch.cuda.synchronize()
                            start_time = time.time()
                            mask_decoder_lora = copy.deepcopy(predictor.sam_mask_decoder)
                            predictor.convert_to_lora(mask_decoder_lora.transformer)
                            predictor.freeze_non_lora(mask_decoder_lora)
                            predictor.train_lora(
                                mask_decoder_lora, 
                                gt_mask, 
                                training_epoch=training_epoch, 
                                mode='init',
                            )
                            torch.cuda.synchronize()
                            end_time = time.time()
                            train_lora_time = end_time - start_time
                            train_lora_type = 'init'
                        else:
                            torch.cuda.synchronize()
                            start_time = time.time()
                            predictor.train_lora(
                                predictor.multi_lora[best_lora_idx], 
                                gt_mask, 
                                training_epoch=20, 
                                mode='finetune', 
                                model_idx=best_lora_idx,
                                replay=True,
                            )
                            torch.cuda.synchronize()
                            end_time = time.time()
                            train_lora_time = end_time - start_time
                            train_lora_type = 'finetune'
                            cur_correction_num += 1
                    elif successful_predict:
                        print(f"frame: {out_frame_idx}, original iou: {gt_iou}, lora iou: {best_lora_iou}")
                        correction_type = 'lora_corrected'
                        video_segments[out_frame_idx][object_id] = (best_lora_predicted_mask_score[0] > 0).cpu().numpy()

                # predictor.propagate_in_video_preflight(inference_state)
                output_prompt_per_object[out_frame_idx] = {
                    'prev_iou': float(gt_iou), 
                    'prev_mask': bmask_to_rle(mask.astype(np.bool)),
                    'type': correction_type, 
                    'train_lora_type': train_lora_type,
                    'train_lora_time': train_lora_time,
                    'lora_inference_time': lora_inference_time,
                    'correct_type': user_correct_type,
                    'correct_num_clicks': user_correct_num_clicks,
                }
                
            output_gt_iou_per_object.append(gt_iou)
    if min(output_gt_iou_per_object) >= correct_threshold:
        finish_process_obj = True

    return output_gt_iou_per_object, output_prompt_per_object, finish_process_obj

# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_online_evaluation_pass(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
    LIT_LoRA_mode=False,
    correct_method='center', # can be 'center' or 'random'
    max_num_click_per_frame=3, # number of corrections per frame,
    correct_threshold=0.75,
    correct_type='mask',
    track_object_appearing_later_in_video=False,
    pass_num=1,
    criteria='gt_iou',
    max_lora_num=3,
):
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png', '.bmp']
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].replace('frame', '').split('_')[-1]))
    gt_mask_dir = os.path.join(input_mask_dir, video_name)
    # inference_state = predictor.init_state(
    #     video_path=video_dir, async_loading_frames=False, gt_path=gt_mask_dir
    # )
    offload_video_to_cpu = True if len(frame_names) > 1500 else False
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False,
        offload_video_to_cpu=offload_video_to_cpu
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    
    input_palette, inputs_per_object = get_input_per_obj(
        base_video_dir=base_video_dir,
        input_mask_dir=input_mask_dir,
        video_name=video_name,
        use_all_masks=use_all_masks,
        per_obj_png_file=per_obj_png_file,
        track_object_appearing_later_in_video=track_object_appearing_later_in_video
    )

    # run inference separately for each object in the video
    object_ids = sorted(inputs_per_object)
    
    if not per_obj_png_file:
        gt_masks = [np.array(Image.open(os.path.join(gt_mask_dir, i))) for i in sorted(os.listdir(gt_mask_dir))]

        obj_gt_mask_per_object = {
            obj: [
                np.where(gt_array == obj, 1, np.where(gt_array == 255, 255, 0)).astype(gt_array.dtype)
                for gt_array in gt_masks[list(inputs_per_object[obj].keys())[0]: ]
            ]
            for obj in object_ids
        }
    else:
        obj_lists = sorted(os.listdir(os.path.join(input_mask_dir, video_name)))
        obj_gt_mask_per_object = {i: [] for i in object_ids}
        for obj in obj_lists:
            gt_mask_obj = os.path.join(input_mask_dir, video_name, obj)
            gt_mask_index = [int(i.replace('.png', '')) for i in  sorted(os.listdir(gt_mask_obj))]
            gt_masks = [np.array(Image.open(os.path.join(gt_mask_obj, i))) for i in sorted(os.listdir(gt_mask_obj))]
            cnt = 0
            for i in range(len(frame_names)):
                if i in gt_mask_index:
                    obj_gt_mask_per_object[int(obj)].append(gt_masks[cnt] > 0)
                    cnt += 1
                else:
                    obj_gt_mask_per_object[int(obj)].append(None)
    finish_process_obj = set()
                    
    cur_pass_num = 1 if pass_num is not None else None
    while True:
        output_prompt_per_object = {obj_idx:{} for obj_idx in object_ids}
        output_pred_iou_per_object = {obj_idx: [] for obj_idx in object_ids}
        output_gt_iou_per_object = {obj_idx: [] for obj_idx in object_ids}
        output_memory_allocated_per_object = {obj_idx: 0 for obj_idx in object_ids}
        output_memory_reserved_per_object = {obj_idx: 0 for obj_idx in object_ids}
        video_segments = defaultdict(dict)
        for object_idx, object_id in enumerate(object_ids):
            if object_id in finish_process_obj:
                continue
            if LIT_LoRA_mode:
                output_gt_iou_per_object[object_id], output_prompt_per_object[object_id], finish_process = vos_online_one_pass_lora_correction(
                    predictor,
                    inference_state,
                    inputs_per_object,
                    obj_gt_mask_per_object,
                    video_segments,
                    object_id,
                    score_thresh,
                    correct_threshold,
                    correct_type,
                    correction_num_limit=cur_pass_num,
                    max_num_click_per_frame=max_num_click_per_frame,
                    max_lora_num=max_lora_num,
                )
            else:
                output_gt_iou_per_object[object_id], output_prompt_per_object[object_id], finish_process = vos_online_one_pass_user_correction(
                    predictor,
                    inference_state,
                    inputs_per_object,
                    obj_gt_mask_per_object,
                    video_segments,
                    object_id,
                    score_thresh,
                    correct_threshold,
                    correct_type,
                    correction_num_limit=cur_pass_num,
                    max_num_click_per_frame=max_num_click_per_frame,
                )
            if finish_process:
                finish_process_obj.add(object_id)
        cache = {'pred_iou': output_pred_iou_per_object,
                'prompt': output_prompt_per_object, 'gt_iou': output_gt_iou_per_object,
                'memory_allocated': output_memory_allocated_per_object, 'memory_reserved': output_memory_reserved_per_object}
                #  'multimask': output_multimask_per_objects, 'multimask_pred_iou': output_multimask_pred_iou_per_objects}
        # post-processing: consolidate the per-object scores into per-frame masks
        # os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
        output_palette = input_palette or DAVIS_PALETTE
            
        if len(video_segments) == 0:
            break
        # write the output masks as palette PNG files to output_mask_dir
        if cur_pass_num != None:
            save_mask_dir = os.path.join(output_mask_dir, 'pred_mask', f'pass_{cur_pass_num}')
            if not os.path.exists(save_mask_dir):
                os.makedirs(save_mask_dir)
        else:
            save_mask_dir = os.path.join(output_mask_dir, 'pred_mask')
            if not os.path.exists(save_mask_dir):
                os.makedirs(save_mask_dir)
        for frame_idx, per_obj_output_mask in video_segments.items():
            save_masks_to_dir(
                output_mask_dir=save_mask_dir,
                video_name=video_name,
                frame_name=frame_names[frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
                per_obj_png_file=True,
                output_palette=output_palette,
            )
        if cur_pass_num != None:
            cache_path = (os.path.join(output_mask_dir, 'cache', f'pass_{cur_pass_num}'))
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
        else:
            cache_path = (os.path.join(output_mask_dir, 'cache'))
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
        cache_path = os.path.join(cache_path, video_name + '.jsonl')
        with open(cache_path, 'w') as f:
            f.write(str(cache))
        predictor.reset_state(inference_state)
        if len(finish_process_obj) == len(object_ids):
            break
        if cur_pass_num == None:
            break
        cur_pass_num += 1
        if cur_pass_num > pass_num:
            break
    gc.collect()
    del inference_state
    del output_prompt_per_object
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_base_plus.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        required=True,
        help="directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run VOS prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="whether to use all available PNG files in input_mask_dir "
        "(default without this flag: just the first PNG file as input to the SAM 2 model; "
        "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
    )
    parser.add_argument(
        "--per_obj_png_file",
        action="store_true",
        help="whether use separate per-object PNG files for input and output masks "
        "(default without this flag: all object masks are packed into a single PNG file on each frame following DAVIS format; "
        "note that the SA-V dataset stores each object mask as an individual PNG file and requires this flag)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--track_object_appearing_later_in_video",
        action="store_true",
        help="whether to track objects that appear later in the video (i.e. not on the first frame; "
        "some VOS datasets like LVOS or YouTube-VOS don't have all objects appearing in the first frame)",
    )
    parser.add_argument(
        "--use_vos_optimized_video_predictor",
        action="store_true",
        help="whether to use vos optimized video predictor with all modules compiled",
    )
    parser.add_argument(
        "--online_evaluation_unlimited",
        action="store_true",
        help="whether to use online evaluation with unlimited correction",
    )
    parser.add_argument(
        "--num_pass",
        type=int,
        default=1,
        help="number of passes for online evaluation",
    )
    parser.add_argument(
        "--correct_threshold",
        type=float,
        default=0.5,
        help="threshold for the correction point (default: 0.75)",
    )
    parser.add_argument(
        "--criteria",
        type=str,
        default='gt_iou',
        help="criteria for the evaluation (default: iou)",
    )
    parser.add_argument(
        "--correct_type",
        type=str,
        default='mask',
        help="threshold for the output mask logits (default: 0.0)",
    )

    parser.add_argument(
        "--LIT_LoRA_mode",
        action="store_true",
        help="whether to use LoRA",
    )

    parser.add_argument(
        "--max_num_click_per_frame",
        type=int,
        default=3,
        help="number of corrections per frame (default: 3)",
    )

    parser.add_argument(
        "--max_lora_num",
        type=int,
        default=3,
        help="number of LoRA models to train (default: 3)",
    )

    args = parser.parse_args()

    # if we use per-object PNG files, they could possibly overlap in inputs and outputs
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
        vos_optimized=args.use_vos_optimized_video_predictor,
        add_all_frames_to_correct_as_cond=False,
        use_mask_input_as_output_without_sam=True
    )
    if args.LIT_LoRA_mode:
        predictor.LIT_LoRA_mode = True
        predictor.reset_lora()
    else:
        predictor.LIT_LoRA_mode = False

    if args.online_evaluation_unlimited:
        args.num_pass = None

    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print(
            "using only the first frame's mask in input_mask_dir as input to the SAM 2 model"
        )
    # if a video list file is provided, read the video names from the file
    # (otherwise, we use all subdirectories in base_video_dir)
    if args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = [
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        ]
    print(f"running VOS prediction on {len(video_names)} videos:\n{video_names}")

    video_names = sorted(video_names)[148:]
    # processed = sorted(os.listdir(os.path.join(args.output_mask_dir, 'pred_mask', 'pass_1')))
    # video_names = [video_name for video_name in video_names if video_name not in processed]
    # save the args to a json file
    if not os.path.exists(args.output_mask_dir):
        os.makedirs(args.output_mask_dir)
    with open(os.path.join(args.output_mask_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)

    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        vos_online_evaluation_pass(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            input_mask_dir=args.input_mask_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            score_thresh=args.score_thresh,
            use_all_masks=args.use_all_masks,
            per_obj_png_file=args.per_obj_png_file,
            LIT_LoRA_mode=args.LIT_LoRA_mode,
            correct_threshold=args.correct_threshold,
            correct_type=args.correct_type,
            track_object_appearing_later_in_video=args.track_object_appearing_later_in_video,
            pass_num=args.num_pass,
            criteria=args.criteria,
            max_num_click_per_frame=args.max_num_click_per_frame,
            max_lora_num=args.max_lora_num,
        )
    print(
        f"completed VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )


if __name__ == "__main__":
    main()
