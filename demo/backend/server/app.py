# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Generator

from app_conf import (
    GALLERY_PATH,
    GALLERY_PREFIX,
    POSTERS_PATH,
    POSTERS_PREFIX,
    UPLOADS_PATH,
    UPLOADS_PREFIX,
)
from data.loader import preload_data
from data.schema import schema
from data.store import set_videos
from flask import Flask, make_response, Request, request, Response, send_from_directory
from flask_cors import CORS
from inference.data_types import (
    ApplyLoraCandidateRequest,
    EnableLoRAModeRequest,
    DisableLoRAModeRequest,
    GenerateLoraCandidatesRequest,
    PropagateDataResponse,
    PropagateInVideoRequest,
    PropagateToFrameRequest,
    TrainLoRARequest,
)
from inference.multipart import MultipartResponseBuilder
from inference.predictor import InferenceAPI
from strawberry.flask.views import GraphQLView

logger = logging.getLogger(__name__)

app = Flask(__name__)
cors = CORS(app, supports_credentials=True)

videos = preload_data()
set_videos(videos)

inference_api = InferenceAPI()


@app.route("/healthy")
def healthy() -> Response:
    return make_response("OK", 200)


@app.route(f"/{GALLERY_PREFIX}/<path:path>", methods=["GET"])
def send_gallery_video(path: str) -> Response:
    try:
        return send_from_directory(
            GALLERY_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{POSTERS_PREFIX}/<path:path>", methods=["GET"])
def send_poster_image(path: str) -> Response:
    try:
        return send_from_directory(
            POSTERS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{UPLOADS_PREFIX}/<path:path>", methods=["GET"])
def send_uploaded_video(path: str):
    try:
        return send_from_directory(
            UPLOADS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


# TOOD: Protect route with ToS permission check
@app.route("/propagate_in_video", methods=["POST"])
def propagate_in_video() -> Response:
    data = request.json
    args = {
        "session_id": data["session_id"],
        "start_frame_index": data.get("start_frame_index", 0),
    }

    boundary = "frame"
    frame = gen_track_with_mask_stream(boundary, **args)
    return Response(frame, mimetype="multipart/x-savi-stream; boundary=" + boundary)


# TOOD: Protect route with ToS permission check
@app.route("/propagate_to_frame", methods=["POST"])
def propagate_to_frame() -> Response:
    data = request.json
    req = PropagateToFrameRequest(
        type="propagate_to_frame",
        session_id=data["session_id"],
        frame_index=data["frame_index"],
    )
    
    response = inference_api.propagate_to_frame(request=req)
    return make_response(response.to_json(), 200)


# TOOD: Protect route with ToS permission check
@app.route("/train_lora", methods=["POST"])
def train_lora() -> Response:
    try:
        data = request.json
        logger.info(f"Received train_lora request: session={data.get('session_id')}, obj={data.get('object_id')}, frame={data.get('frame_index')}")
        from inference.data_types import Mask
        mask_data = data["mask"]
        req = TrainLoRARequest(
            type="train_lora",
            session_id=data["session_id"],
            object_id=data["object_id"],
            frame_index=data["frame_index"],
            mask=Mask(size=mask_data["size"], counts=mask_data["counts"]),
        )
        
        response = inference_api.train_lora(request=req)
        return make_response(response.to_json(), 200)
    except Exception as e:
        logger.error(f"Error in train_lora: {e}", exc_info=True)
        return make_response({"error": str(e)}, 500)


# TOOD: Protect route with ToS permission check
@app.route("/generate_lora_candidates", methods=["POST"])
def generate_lora_candidates() -> Response:
    try:
        data = request.json
        logger.info(f"Received generate_lora_candidates request: {data}")
        req = GenerateLoraCandidatesRequest(
            type="generate_lora_candidates",
            session_id=data["session_id"],
            object_id=data["object_id"],
            frame_index=data["frame_index"],
        )
        
        response = inference_api.generate_lora_candidates(request=req)
        return make_response(response.to_json(), 200)
    except Exception as e:
        logger.error(f"Error in generate_lora_candidates: {e}", exc_info=True)
        return make_response({"error": str(e)}, 500)


# TOOD: Protect route with ToS permission check
@app.route("/enable_lora_mode", methods=["POST", "OPTIONS"])
def enable_lora_mode() -> Response:
    if request.method == "OPTIONS":
        # Handle CORS preflight
        response = make_response("", 200)
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
        
    try:
        data = request.json
        logger.info(f"Received enable_lora_mode request: session={data.get('session_id')}")
        req = EnableLoRAModeRequest(
            type="enable_lora_mode",
            session_id=data["session_id"],
        )
        
        response = inference_api.enable_lora_mode_endpoint(request=req)
        return make_response(response.to_json(), 200)
    except Exception as e:
        logger.error(f"Error in enable_lora_mode: {e}", exc_info=True)
        return make_response({"error": str(e)}, 500)


# TOOD: Protect route with ToS permission check
@app.route("/disable_lora_mode", methods=["POST", "OPTIONS"])
def disable_lora_mode() -> Response:
    if request.method == "OPTIONS":
        # Handle CORS preflight
        response = make_response("", 200)
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
        
    try:
        data = request.json
        logger.info(f"Received disable_lora_mode request: session={data.get('session_id')}")
        req = DisableLoRAModeRequest(
            type="disable_lora_mode",
            session_id=data["session_id"],
        )
        
        response = inference_api.disable_lora_mode_endpoint(request=req)
        return make_response(response.to_json(), 200)
    except Exception as e:
        logger.error(f"Error in disable_lora_mode: {e}", exc_info=True)
        return make_response({"error": str(e)}, 500)


# TOOD: Protect route with ToS permission check
@app.route("/apply_lora_candidate", methods=["POST", "OPTIONS"])
def apply_lora_candidate() -> Response:
    if request.method == "OPTIONS":
        # Handle CORS preflight
        response = make_response("", 200)
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
        
    try:
        data = request.json
        logger.info(f"Received apply_lora_candidate request: {data}")
        req = ApplyLoraCandidateRequest(
            type="apply_lora_candidate",
            session_id=data["session_id"],
            object_id=data["object_id"],
            frame_index=data["frame_index"],
            candidate_index=data["candidate_index"],
        )
        
        response = inference_api.apply_lora_candidate(request=req)
        return make_response(response.to_json(), 200)
    except Exception as e:
        logger.error(f"Error in apply_lora_candidate: {e}", exc_info=True)
        return make_response({"error": str(e)}, 500)


def gen_track_with_mask_stream(
    boundary: str,
    session_id: str,
    start_frame_index: int,
) -> Generator[bytes, None, None]:
    with inference_api.autocast_context():
        request = PropagateInVideoRequest(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=start_frame_index,
        )

        for chunk in inference_api.propagate_in_video(request=request):
            yield MultipartResponseBuilder.build(
                boundary=boundary,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Frame-Current": "-1",
                    # Total frames minus the reference frame
                    "Frame-Total": "-1",
                    "Mask-Type": "RLE[]",
                },
                body=chunk.to_json().encode("UTF-8"),
            ).get_message()


class MyGraphQLView(GraphQLView):
    def get_context(self, request: Request, response: Response) -> Any:
        return {"inference_api": inference_api}


# Add GraphQL route to Flask app.
app.add_url_rule(
    "/graphql",
    view_func=MyGraphQLView.as_view(
        "graphql_view",
        schema=schema,
        # Disable GET queries
        # https://strawberry.rocks/docs/operations/deployment
        # https://strawberry.rocks/docs/integrations/flask
        allow_queries_via_get=False,
        # Strawberry recently changed multipart request handling, which now
        # requires enabling support explicitly for views.
        # https://github.com/strawberry-graphql/strawberry/issues/3655
        multipart_uploads_enabled=True,
    ),
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
