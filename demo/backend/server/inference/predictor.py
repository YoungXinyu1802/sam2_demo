# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Generator, List

import numpy as np
import torch
from app_conf import APP_ROOT, MODEL_SIZE
from inference.data_types import (
    AddMaskRequest,
    AddPointsRequest,
    CancelPorpagateResponse,
    CancelPropagateInVideoRequest,
    ClearPointsInFrameRequest,
    ClearPointsInVideoRequest,
    ClearPointsInVideoResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    GenerateLoraCandidatesRequest,
    GenerateLoraCandidatesResponse,
    LoRACandidateValue,
    Mask,
    PropagateDataResponse,
    PropagateDataValue,
    PropagateInVideoRequest,
    PropagateToFrameRequest,
    RemoveObjectRequest,
    RemoveObjectResponse,
    StartSessionRequest,
    StartSessionResponse,
    TrainLoRARequest,
    TrainLoRAResponse,
)
from pycocotools.mask import decode as decode_masks, encode as encode_masks
from sam2.build_sam import build_sam2_video_predictor
import copy
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class InferenceAPI:

    def __init__(self) -> None:
        super(InferenceAPI, self).__init__()

        self.session_states: Dict[str, Any] = {}
        self.score_thresh = 0
        # LoRA mode flag
        self.lit_lora_mode = False
        # LoRA training data: {session_id: {obj_id: [gt_masks]}}
        # Store GT masks for training, features are captured automatically by predictor
        self.lora_training_data: Dict[str, Dict[int, List]] = {}

        if MODEL_SIZE == "tiny":
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_tiny.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif MODEL_SIZE == "small":
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_small.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif MODEL_SIZE == "large":
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else:  # base_plus (default)
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        # select the device for computation
        force_cpu_device = os.environ.get("SAM2_DEMO_FORCE_CPU_DEVICE", "0") == "1"
        if force_cpu_device:
            logger.info("forcing CPU device for SAM 2 demo")
        if torch.cuda.is_available() and not force_cpu_device:
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and not force_cpu_device:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info(f"using device: {device}")

        if device.type == "cuda":
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            logging.warning(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        self.device = device
        self.predictor = build_sam2_video_predictor(
            model_cfg, checkpoint, device=device
        )
        self.inference_lock = Lock()

    def enable_lora_mode(self):
        """Enable LIT-LoRA mode in the predictor"""
        if not self.lit_lora_mode:
            logger.info("Enabling LIT-LoRA mode")
            self.predictor.LIT_LoRA_mode = True
            self.predictor.reset_lora()
            self.lit_lora_mode = True
            
    def disable_lora_mode(self):
        """Disable LIT-LoRA mode in the predictor"""
        if self.lit_lora_mode:
            logger.info("Disabling LIT-LoRA mode")
            self.predictor.LIT_LoRA_mode = False
            self.lit_lora_mode = False

    def enable_lora_mode_endpoint(self, request):
        """HTTP endpoint to enable LIT-LoRA mode"""
        from inference.data_types import EnableLoRAModeResponse
        try:
            self.enable_lora_mode()
            return EnableLoRAModeResponse(
                success=True,
                message="LIT-LoRA mode enabled"
            )
        except Exception as e:
            logger.error(f"Error enabling LoRA mode: {e}")
            return EnableLoRAModeResponse(
                success=False,
                message=f"Failed to enable LoRA mode: {str(e)}"
            )
    
    def disable_lora_mode_endpoint(self, request):
        """HTTP endpoint to disable LIT-LoRA mode"""
        from inference.data_types import DisableLoRAModeResponse
        try:
            self.disable_lora_mode()
            return DisableLoRAModeResponse(
                success=True,
                message="LIT-LoRA mode disabled"
            )
        except Exception as e:
            logger.error(f"Error disabling LoRA mode: {e}")
            return DisableLoRAModeResponse(
                success=False,
                message=f"Failed to disable LoRA mode: {str(e)}"
            )
    
    def start_over_endpoint(self, request):
        """HTTP endpoint to reset all states to original condition"""
        from inference.data_types import StartOverResponse
        try:
            session_id = request.session_id
            
            # Reset LoRA mode
            if self.lit_lora_mode:
                logger.info("Disabling LIT-LoRA mode for start over")
                self.predictor.LIT_LoRA_mode = False
                self.lit_lora_mode = False
            
            # Reset LoRA models if they exist
            if hasattr(self.predictor, 'trained_lora') and self.predictor.trained_lora:
                logger.info("Resetting LoRA models for start over")
                self.predictor.reset_lora()
            
            # Clear LoRA training data
            if hasattr(self, 'lora_training_data') and session_id in self.lora_training_data:
                logger.info("Clearing LoRA training data for start over")
                self.lora_training_data[session_id] = {}
            
            # Clear session state if it exists
            if session_id in self.session_states:
                logger.info("Clearing session state for start over")
                session = self.session_states[session_id]
                # Reset the inference state to initial state
                if 'state' in session:
                    # Reset the existing inference state
                    self.predictor.reset_state(session['state'])
            
            logger.info(f"Successfully reset all states for session {session_id}")
            return StartOverResponse(
                success=True,
                message="All states reset successfully"
            )
        except Exception as e:
            logger.error(f"Error in start over: {e}")
            import traceback
            traceback.print_exc()
            return StartOverResponse(
                success=False,
                message=f"Failed to reset states: {str(e)}"
            )

    def autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            return contextlib.nullcontext()

    def start_session(self, request: StartSessionRequest) -> StartSessionResponse:
        with self.autocast_context(), self.inference_lock:
            session_id = str(uuid.uuid4())
            # for MPS devices, we offload the video frames to CPU by default to avoid
            # memory fragmentation in MPS (which sometimes crashes the entire process)
            offload_video_to_cpu = self.device.type == "mps"
            inference_state = self.predictor.init_state(
                request.path,
                offload_video_to_cpu=offload_video_to_cpu,
            )
            self.session_states[session_id] = {
                "canceled": False,
                "state": inference_state,
            }
            return StartSessionResponse(session_id=session_id)

    def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        is_successful = self.__clear_session_state(request.session_id)
        return CloseSessionResponse(success=is_successful)

    def add_points(
        self, request: AddPointsRequest, test: str = ""
    ) -> PropagateDataResponse:
        with self.autocast_context(), self.inference_lock:
            session = self.__get_session(request.session_id)
            inference_state = session["state"]

            frame_idx = request.frame_index
            obj_id = request.object_id
            points = request.points
            labels = request.labels
            clear_old_points = request.clear_old_points

            # add new prompts and instantly get the output on the same frame
            frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                clear_old_points=clear_old_points,
                normalize_coords=False,
            )

            masks_binary = (masks > self.score_thresh)[:, 0].cpu().numpy()

            rle_mask_list = self.__get_rle_mask_list(
                object_ids=object_ids, masks=masks_binary
            )

            return PropagateDataResponse(
                frame_index=frame_idx,
                results=rle_mask_list,
            )

    def add_mask(self, request: AddMaskRequest) -> PropagateDataResponse:
        """
        Add new points on a specific video frame.
        - mask is a numpy array of shape [H_im, W_im] (containing 1 for foreground and 0 for background).
        Note: providing an input mask would overwrite any previous input points on this frame.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            frame_idx = request.frame_index
            obj_id = request.object_id
            rle_mask = {
                "counts": request.mask.counts,
                "size": request.mask.size,
            }

            mask = decode_masks(rle_mask)

            logger.info(
                f"add mask on frame {frame_idx} in session {session_id}: {obj_id=}, {mask.shape=}"
            )
            session = self.__get_session(session_id)
            inference_state = session["state"]

            frame_idx, obj_ids, video_res_masks = self.model.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=torch.tensor(mask > 0),
            )
            masks_binary = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()

            rle_mask_list = self.__get_rle_mask_list(
                object_ids=obj_ids, masks=masks_binary
            )

            return PropagateDataResponse(
                frame_index=frame_idx,
                results=rle_mask_list,
            )

    def clear_points_in_frame(
        self, request: ClearPointsInFrameRequest
    ) -> PropagateDataResponse:
        """
        Remove all input points in a specific frame.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            frame_idx = request.frame_index
            obj_id = request.object_id

            logger.info(
                f"clear inputs on frame {frame_idx} in session {session_id}: {obj_id=}"
            )
            session = self.__get_session(session_id)
            inference_state = session["state"]
            frame_idx, obj_ids, video_res_masks = (
                self.predictor.clear_all_prompts_in_frame(
                    inference_state, frame_idx, obj_id
                )
            )
            masks_binary = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()

            rle_mask_list = self.__get_rle_mask_list(
                object_ids=obj_ids, masks=masks_binary
            )

            return PropagateDataResponse(
                frame_index=frame_idx,
                results=rle_mask_list,
            )

    def clear_points_in_video(
        self, request: ClearPointsInVideoRequest
    ) -> ClearPointsInVideoResponse:
        """
        Remove all input points in all frames throughout the video.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            logger.info(f"clear all inputs across the video in session {session_id}")
            session = self.__get_session(session_id)
            inference_state = session["state"]
            self.predictor.reset_state(inference_state)
            return ClearPointsInVideoResponse(success=True)

    def remove_object(self, request: RemoveObjectRequest) -> RemoveObjectResponse:
        """
        Remove an object id from the tracking state.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            obj_id = request.object_id
            logger.info(f"remove object in session {session_id}: {obj_id=}")
            session = self.__get_session(session_id)
            inference_state = session["state"]
            new_obj_ids, updated_frames = self.predictor.remove_object(
                inference_state, obj_id
            )

            results = []
            for frame_index, video_res_masks in updated_frames:
                masks = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
                rle_mask_list = self.__get_rle_mask_list(
                    object_ids=new_obj_ids, masks=masks
                )
                results.append(
                    PropagateDataResponse(
                        frame_index=frame_index,
                        results=rle_mask_list,
                    )
                )

            return RemoveObjectResponse(results=results)

    def propagate_in_video(
        self, request: PropagateInVideoRequest
    ) -> Generator[PropagateDataResponse, None, None]:
        session_id = request.session_id
        start_frame_idx = request.start_frame_index
        propagation_direction = "both"
        max_frame_num_to_track = None

        """
        Propagate existing input points in all frames to track the object across video.
        """

        # Note that as this method is a generator, we also need to use autocast_context
        # in caller to this method to ensure that it's called under the correct context
        # (we've added `autocast_context` to `gen_track_with_mask_stream` in app.py).
        with self.autocast_context(), self.inference_lock:
            logger.info(
                f"propagate in video in session {session_id}: "
                f"{propagation_direction=}, {start_frame_idx=}, {max_frame_num_to_track=}"
            )

            try:
                session = self.__get_session(session_id)
                session["canceled"] = False

                inference_state = session["state"]
                if propagation_direction not in ["both", "forward", "backward"]:
                    raise ValueError(
                        f"invalid propagation direction: {propagation_direction}"
                    )

                # First doing the forward propagation
                if propagation_direction in ["both", "forward"]:
                    for outputs in self.predictor.propagate_in_video(
                        inference_state=inference_state,
                        start_frame_idx=start_frame_idx,
                        max_frame_num_to_track=max_frame_num_to_track,
                        reverse=False,
                    ):
                        if session["canceled"]:
                            return None

                        frame_idx, obj_ids, video_res_masks = outputs
                        masks_binary = (
                            (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
                        )

                        rle_mask_list = self.__get_rle_mask_list(
                            object_ids=obj_ids, masks=masks_binary
                        )

                        yield PropagateDataResponse(
                            frame_index=frame_idx,
                            results=rle_mask_list,
                        )

                # Then doing the backward propagation (reverse in time)
                if propagation_direction in ["both", "backward"]:
                    for outputs in self.predictor.propagate_in_video(
                        inference_state=inference_state,
                        start_frame_idx=start_frame_idx,
                        max_frame_num_to_track=max_frame_num_to_track,
                        reverse=True,
                    ):
                        if session["canceled"]:
                            return None

                        frame_idx, obj_ids, video_res_masks = outputs
                        masks_binary = (
                            (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
                        )

                        rle_mask_list = self.__get_rle_mask_list(
                            object_ids=obj_ids, masks=masks_binary
                        )

                        yield PropagateDataResponse(
                            frame_index=frame_idx,
                            results=rle_mask_list,
                        )
            finally:
                # Log upon completion (so that e.g. we can see if two propagations happen in parallel).
                # Using `finally` here to log even when the tracking is aborted with GeneratorExit.
                logger.info(
                    f"propagation ended in session {session_id}; {self.__get_session_stats()}"
                )

    def cancel_propagate_in_video(
        self, request: CancelPropagateInVideoRequest
    ) -> CancelPorpagateResponse:
        session = self.__get_session(request.session_id)
        session["canceled"] = True
        return CancelPorpagateResponse(success=True)

    def propagate_to_frame(
        self, request: PropagateToFrameRequest
    ) -> PropagateDataResponse:
        """
        Propagate tracking to a specific frame. This is used for frame-by-frame tracking.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            frame_idx = request.frame_index
            
            logger.info(
                f"[InferenceAPI] Starting frame propagation to frame {frame_idx} in session {session_id}"
            )
            
            session = self.__get_session(session_id)
            inference_state = session["state"]
            
            try:
                frame_idx, obj_ids, video_res_masks = self.predictor.propagate_to_frame(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                )
                
                masks_binary = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
                
                rle_mask_list = self.__get_rle_mask_list(
                    object_ids=obj_ids, masks=masks_binary
                )
                
                logger.info(
                    f"[InferenceAPI] Frame propagation completed for frame {frame_idx}, "
                    f"processed {len(obj_ids)} objects, generated {len(rle_mask_list)} masks"
                )
                
                return PropagateDataResponse(
                    frame_index=frame_idx,
                    results=rle_mask_list,
                )
            except Exception as e:
                logger.error(f"Error propagating to frame {frame_idx}: {e}")
                raise

    def train_lora(self, request: TrainLoRARequest) -> TrainLoRAResponse:
        """
        Add a training sample and train LoRA immediately (following online_eval.py workflow).
        This ensures features are fresh when training happens.
        """
        with self.autocast_context(), self.inference_lock:
            try:
                session_id = request.session_id
                obj_id = request.object_id
                frame_idx = request.frame_index
                
                logger.info(
                    f"Adding LoRA training sample for session {session_id}, obj {obj_id}, frame {frame_idx}"
                )
                
                # Enable LoRA mode if not already enabled
                if not self.lit_lora_mode:
                    self.enable_lora_mode()
                
                # Initialize storage if needed
                if session_id not in self.lora_training_data:
                    self.lora_training_data[session_id] = {}
                if obj_id not in self.lora_training_data[session_id]:
                    self.lora_training_data[session_id][obj_id] = []
                
                # Decode the mask
                rle_mask = {
                    "counts": request.mask.counts,
                    "size": request.mask.size,
                }
                mask = decode_masks(rle_mask)
                
                # Get the session and inference state
                session = self.__get_session(session_id)
                inference_state = session["state"]
                
                # First, propagate to this frame to ensure features are captured
                # This is critical: propagate_to_frame runs _track_step which populates temp_feat_for_lora
                try:
                    _, _, _ = self.predictor.propagate_to_frame(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                    )
                except Exception as e:
                    logger.warning(f"Could not propagate to frame {frame_idx}: {e}")
                
                # Add the mask - features should already be captured from propagation
                input_mask = ((mask > 0) & (mask != 255)).astype(np.uint8)
                _, _, _ = self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=input_mask,
                    run_mem_encoder=True
                )
                
                # Train LoRA IMMEDIATELY after adding mask (following online_eval.py workflow)
                # This ensures temp_feat_for_lora has the correct features
                if not hasattr(self.predictor, 'trained_lora') or not self.predictor.trained_lora:
                    # First correction: train initial LoRA model
                    logger.info("Training initial LoRA model")
                    
                    # Debug: Check if features were captured
                    if hasattr(self.predictor, 'temp_feat_for_lora'):
                        pix_feat = self.predictor.temp_feat_for_lora.get("pix_feat_with_mem")
                        if pix_feat is None:
                            logger.error(f"Features not captured! temp_feat_for_lora: {self.predictor.temp_feat_for_lora.keys()}")
                        else:
                            logger.info(f"Features captured successfully. Shape: {pix_feat.shape}")
                    
                    mask_decoder_lora = copy.deepcopy(self.predictor.sam_mask_decoder)
                    self.predictor.convert_to_lora(mask_decoder_lora.transformer)
                    self.predictor.freeze_non_lora(mask_decoder_lora)
                    
                    training_success = self.predictor.train_lora(
                        mask_decoder_lora, 
                        mask,  # Use the GT mask
                        training_epoch=100,
                        mode='init'
                    )
                    
                    if training_success:
                        logger.info("Initial LoRA model trained successfully")
                    else:
                        logger.warning("LoRA training failed - features may not have been captured")
                else:
                    # Subsequent corrections: could fine-tune existing LoRA
                    logger.info("LoRA already trained, could fine-tune in future")
                    # For now, we'll just add to training data for potential future fine-tuning
                
                # Store the GT mask for tracking
                self.lora_training_data[session_id][obj_id].append({
                    'frame_idx': frame_idx,
                    'gt_mask': mask
                })
                
                num_samples = len(self.lora_training_data[session_id][obj_id])
                logger.info(f"LoRA training sample added. Total samples: {num_samples}")
                
                return TrainLoRAResponse(
                    success=True,
                    message=f"Training sample added and LoRA trained. Total samples: {num_samples}"
                )
            except Exception as e:
                logger.error(f"Error adding LoRA training sample: {e}")
                import traceback
                traceback.print_exc()
                return TrainLoRAResponse(
                    success=False,
                    message=f"Error: {str(e)}"
                )

    def generate_lora_candidates(
        self, request: GenerateLoraCandidatesRequest
    ) -> GenerateLoraCandidatesResponse:
        """
        Generate mask candidates using already-trained LoRA models.
        LoRA should already be trained via train_lora() calls.
        """
        with self.autocast_context(), self.inference_lock:
            try:
                session_id = request.session_id
                obj_id = request.object_id
                frame_idx = request.frame_index
                
                logger.info(
                    f"Generating LoRA candidates for session {session_id}, obj {obj_id}, frame {frame_idx}"
                )
                
                # Check if LoRA has been trained
                if not hasattr(self.predictor, 'trained_lora') or not self.predictor.trained_lora:
                    logger.warning("No LoRA model trained yet. Please add corrections first via train_lora.")
                    return GenerateLoraCandidatesResponse(
                        frame_index=frame_idx,
                        object_id=obj_id,
                        candidates=[]
                    )
                
                session = self.__get_session(session_id)
                inference_state = session["state"]
                
                # First, propagate to frame to capture features
                try:
                    _, _, _ = self.predictor.propagate_to_frame(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                    )
                except Exception as e:
                    logger.warning(f"Could not propagate to frame {frame_idx}: {e}")
                
                # Generate candidates from ALL LoRA models
                logger.info(f"Generating predictions from all LoRA models for frame {frame_idx}")
                
                lora_candidates = self.predictor.lora_predict_all_candidates(
                    inference_state=inference_state,
                    frame_idx=frame_idx
                )
                
                if not lora_candidates:
                    logger.warning("No LoRA candidates generated")
                    return GenerateLoraCandidatesResponse(
                        frame_index=frame_idx,
                        object_id=obj_id,
                        candidates=[]
                    )
                
                # Convert each candidate to RLE format
                candidates = []
                for idx, candidate in enumerate(lora_candidates):
                    predicted_mask_score = candidate['predicted_mask_score']
                    
                    # Convert to binary mask and encode as RLE
                    lora_mask = (predicted_mask_score[0] > 0).squeeze().cpu().numpy().astype(np.uint8)
                    logger.info(f"[LoRA Candidate {idx}] Mask shape before RLE: {lora_mask.shape}")
                    mask_rle = encode_masks(np.array(lora_mask, dtype=np.uint8, order="F"))
                    mask_rle["counts"] = mask_rle["counts"].decode()
                    logger.info(f"[LoRA Candidate {idx}] RLE size: {mask_rle['size']}")
                    
                    candidates.append(
                        LoRACandidateValue(
                            mask=Mask(
                                size=mask_rle["size"],
                                counts=mask_rle["counts"],
                            ),
                            confidence=float(idx),  # Use index as identifier
                        )
                    )
                
                logger.info(f"Generated {len(candidates)} LoRA candidates for user selection")
                
                return GenerateLoraCandidatesResponse(
                    frame_index=frame_idx,
                    object_id=obj_id,
                    candidates=candidates,
                )
                
            except Exception as e:
                logger.error(f"Error generating LoRA candidates: {e}")
                import traceback
                traceback.print_exc()
                return GenerateLoraCandidatesResponse(
                    frame_index=frame_idx,
                    object_id=obj_id,
                    candidates=[],
                )

    def apply_lora_candidate(
        self, request
    ):
        """
        Apply a selected LoRA candidate to memory (like a mask correction).
        This is called when user chooses one of the displayed candidates.
        """
        from inference.data_types import ApplyLoraCandidateResponse
        
        with self.autocast_context(), self.inference_lock:
            try:
                session_id = request.session_id
                obj_id = request.object_id
                frame_idx = request.frame_index
                candidate_idx = request.candidate_index
                
                logger.info(
                    f"Applying LoRA candidate {candidate_idx} for session {session_id}, obj {obj_id}, frame {frame_idx}"
                )
                
                # Check if LoRA has been trained
                if not hasattr(self.predictor, 'trained_lora') or not self.predictor.trained_lora:
                    logger.warning("No LoRA model trained yet")
                    return ApplyLoraCandidateResponse(
                        success=False,
                        message="No LoRA model trained"
                    )
                
                session = self.__get_session(session_id)
                inference_state = session["state"]
                
                # Apply the selected candidate
                logger.info(f"Applying candidate {candidate_idx} to memory")
                
                predicted_mask_score = self.predictor.apply_lora_candidate(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    candidate_idx=candidate_idx
                )
                
                logger.info(f"Successfully applied LoRA candidate {candidate_idx}")
                
                return ApplyLoraCandidateResponse(
                    success=True,
                    message=f"Candidate {candidate_idx} applied successfully"
                )
                
            except Exception as e:
                logger.error(f"Error applying LoRA candidate: {e}")
                import traceback
                traceback.print_exc()
                return ApplyLoraCandidateResponse(
                    success=False,
                    message=f"Error: {str(e)}"
                )

    def __get_rle_mask_list(
        self, object_ids: List[int], masks: np.ndarray
    ) -> List[PropagateDataValue]:
        """
        Return a list of data values, i.e. list of object/mask combos.
        """
        return [
            self.__get_mask_for_object(object_id=object_id, mask=mask)
            for object_id, mask in zip(object_ids, masks)
        ]

    def __get_mask_for_object(
        self, object_id: int, mask: np.ndarray
    ) -> PropagateDataValue:
        """
        Create a data value for an object/mask combo.
        """
        mask_rle = encode_masks(np.array(mask, dtype=np.uint8, order="F"))
        mask_rle["counts"] = mask_rle["counts"].decode()
        return PropagateDataValue(
            object_id=object_id,
            mask=Mask(
                size=mask_rle["size"],
                counts=mask_rle["counts"],
            ),
        )

    def __get_session(self, session_id: str):
        session = self.session_states.get(session_id, None)
        if session is None:
            raise RuntimeError(
                f"Cannot find session {session_id}; it might have expired"
            )
        return session

    def __get_session_stats(self):
        """Get a statistics string for live sessions and their GPU usage."""
        # print both the session ids and their video frame numbers
        live_session_strs = [
            f"'{session_id}' ({session['state']['num_frames']} frames, "
            f"{len(session['state']['obj_ids'])} objects)"
            for session_id, session in self.session_states.items()
        ]
        session_stats_str = (
            "Test String Here - -"
            f"live sessions: [{', '.join(live_session_strs)}], GPU memory: "
            f"{torch.cuda.memory_allocated() // 1024**2} MiB used and "
            f"{torch.cuda.memory_reserved() // 1024**2} MiB reserved"
            f" (max over time: {torch.cuda.max_memory_allocated() // 1024**2} MiB used "
            f"and {torch.cuda.max_memory_reserved() // 1024**2} MiB reserved)"
        )
        return session_stats_str

    def __clear_session_state(self, session_id: str) -> bool:
        session = self.session_states.pop(session_id, None)
        if session is None:
            logger.warning(
                f"cannot close session {session_id} as it does not exist (it might have expired); "
                f"{self.__get_session_stats()}"
            )
            return False
        else:
            logger.info(f"removed session {session_id}; {self.__get_session_stats()}")
            return True
