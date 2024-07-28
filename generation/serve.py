from io import BytesIO

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
import argparse
import base64
from time import time
from omegaconf import OmegaConf

from DreamGaussianLib import GaussianProcessor, ModelsPreLoader, HDF5Loader
from utils.video_utils import VideoUtils

import kiui
from kiui.op import recenter

import requests, os
import numpy as np
from PIL import Image
from typing import Optional
from functools import lru_cache
import base64
import threading
from diffusers import DiffusionPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler
import torch
from pydantic import BaseModel
from io import BytesIO
from huggingface_hub import hf_hub_download

############# LGM #############
import tyro
from safetensors.torch import load_file
from LGM.core.models import LGM
from LGM.core.options import AllConfigs, Options

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument("--config", default="configs/image_sai.yaml")
    return parser.parse_args()

root_folder = '' # set the path to your root (TAO 17 folder) path here
temp_out_path = os.path.join(root_folder, 'temp')
os.makedirs(temp_out_path, exist_ok=True)
mesh_model_config_path = os.path.join(root_folder, 'InstantMesh/configs/instant-mesh-base.yaml')
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
base_model_id = "stabilityai/sdxl-turbo"
model_3d_view_id = 'sudo-ai/zero123plus-v1.2'
mesh_model_id = 'TencentARC/InstantMesh'
mv_dream_name = "ashawkey/imagedream-ipmv-diffusers"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-8steps-CFG-lora.safetensors"

# load rembg
bg_remover = rembg.new_session()


class SampleInput(BaseModel):
    prompt: str

class DiffUsers:
    def __init__(self):

        print("setting up model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        ## n step lora
        self.pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(self.device)
        self.pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipeline.fuse_lora()
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing="trailing")
        self.steps = 2
        self.guidance_scale = 6

        self._lock = threading.Lock()
        print("model setup done")

    def generate_image(self, prompt: str):
        generator = torch.Generator(self.device)
        seed = generator.seed()
        generator = generator.manual_seed(seed)
        image = self.pipeline(
            prompt="3d model of " + prompt + ", (white background), high quality, 4k, masterpiece, artistic",
            negative_prompt="worst quality, low quality",
            num_inference_steps=self.steps,
            generator=generator,
            guidance_scale=self.guidance_scale,
        ).images[0]
        buf = BytesIO()
        image.save(buf, format="png")
        buf.seek(0)
        image = base64.b64encode(buf.read()).decode()
        return {"image": image}
    
    def sample(self, input: SampleInput):
        try:
            with self._lock:
                return self.generate_image(input.prompt)
        except Exception as e:
            print(e)
            with self._lock:
                return self.generate_image(input.prompt)


class InstantMeshGenerator:
    def __init__(self, seed=0):
        print('setting up multi-view model')

        import rembg, imageio, tempfile
        from pytorch_lightning import seed_everything
        from einops import rearrange, repeat
        from torchvision.transforms import v2 
        from tqdm import tqdm 
        from InstantMesh.src.uitls.infer_util import remove_background, \
                                                     resize_foreground
        from InstantMesh.src.utils.train_util import instantiate_from_config
        from InstantMesh.src.utils.camera_util import FOV_to_intrinsics, \
                                                      get_zero123plus_input_cameras, \
                                                      get_circular_camera_poses
        from InstantMesh.src.utils.mesh_util import save_obj, \
                                                    save_obj_with_mtl

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        self.pipeline = DiffusionPipeline.from_pretrained(model_3d_view_id, custom_pipeline='zero123plus', torch_dtype=torch.float16)
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing='trailing')
        
        unet_ckpt_path = hf_hub_download(repo_id=mesh_model_id, filename='diffusion_pytorch_model.bin', repo_type='model')
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        self.pipeline.unet.load_state_dict(state_dict, strict=True)
        self.pipeline.to(self.device)
        seed_everything(seed)

        print('setting up instant mesh model')
        self.config = OmegaConf.load(mesh_model_config_path)
        config_name = os.path.basename(mesh_model_config_path).replace('.yaml', '')
        self.model_config = self.config.model_config
        self.infer_config = self.config.infer_config
        
        self.mesh_model = instantiate_from_config(self.model_config)
        
        model_ckpt_path = hf_hub_download(repo_id=mesh_model_id, filename='instant_mesh_base.ckpt', repo_type='model')
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        self.mesh_model.load_state_dict(state_dict, strict=True)
        self.mesh_model.to(self.device)
        self.IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False
        if self.IS_FLEXICUBES:
            self.mesh_model.init_flexicubes_geometry(self.device, fovy=30.0)
        self.mesh_model = self.mesh_model.eval()

    
    @staticmethod
    def preprocess(input_img, do_remove_bg):
        rembg_session = rembg.new_session() if do_remove_bg else None
        if do_remove_bg:
            input_img = remove_background(input_img, rembg_session)
            input_img = resize_foreground(input_img, 0.85)
        return input_img
    
    '''function to generate multi-view img'''
    def generate_mvs(self, input_img, steps, seed):
        seed_everything(seed)
        generator = torch.Generator(device=self.device)
        z123_img = self.pipeline(input_img, 
                                 num_inference_steps=steps, 
                                 generator=generator).images[0]
        # show_img = np.asarray(z123_img, dtype=np.uint8)
        # show_img = torch.from_numpy(show_img)
        # show_img = rearrange(show_img, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
        # show_img = rearrange(show_img, '(n m) h w c -> (n h) (m w), c', n=2, m=3)
        # show_img = Image.fromarray(show_img.numpy())
        return z123_img#, show_img

    '''function to generate mesh'''
    def make_mesh(self, mesh_fpath, planes):
        mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
        mesh_dirname = os.path.dirname(mesh_fpath)
        mesh_vis_fpath = os.path.join(mesh_dirname, f'{mesh_basename}.glb')
        
        with torch.no_grad():
            mesh_out = model.extract_mesh(planes, use_texture_map=True, **self.infer_config)
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out

            save_obj_with_mtl(vertices.data.cpu().numpy(), 
                              uvs.data.cpu().numpy(), 
                              faces.data.cpu().numpy(), 
                              mesh_tex_idx.data.cpu().numpy(), 
                              tex_map.permute(1, 2, 0).data.cpu().numpy(),
                              mesh_fpath,)
            print(f'mesh with texmap saved to {mesh_fpath}')
        return mesh_fpath
    
    '''function to generate mesh from multi-view img'''
    def generate_mesh_from_mvs(self, img: torch.Tensor):
        img = img.to(dtype=torch.float32)
        img /= 255.0
        img = img.permute(2, 0, 1).contiguous() # channel, height, width
        img = rearrange(img. 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
        img = img.unsqueeze(0).to(self.device)
        img = v2.functional.resize(img, (320, 320), interpolation=3, antilias=True).clamp(0, 1)

        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(self.device)

        with torch.no_grad():
            planes = model.forward_planes(img, input_cameras)
        
        tempfile.tempdir = temp_out_path
        mesh_fpath = tempfile.NamedTemporaryFile(suffix=f'.obj', delete=False).name
        
        # make mesh
        mesh_fpath = self.make_mesh(mesh_fpath, planes)
        return mesh_fpath
        




args = get_args()
app = FastAPI()
diffusers = DiffUsers()
instant_mesh = InstantMeshGenerator()


def get_config() -> OmegaConf:
    config = OmegaConf.load("configs/image_sai.yaml")
    return config


def get_models(config: OmegaConf = Depends(get_config)):
    return ModelsPreLoader.preload_model(config, "cuda")


@app.post("/generate/")
async def generate(
    prompt: str = Form(),
    config: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
):
    best_score = 0
    best_buffer = None

    for i in range(10):
        buffer = await _generate(models, config, prompt)
        buffer = base64.b64encode(buffer.getbuffer()).decode("utf-8")

        response = requests.post("http://localhost:8094/validate/", json={"prompt": prompt, "data": buffer})

        # Check if the request was successful
        if response.status_code == 200:
            print("Data sent successfully!")
            score = response.json().get("score", 0)
            print(f"Score: {score} in attempt {i + 1}")
            if score >= 0.8:
                print("Score is high enough, stopping")
                return buffer
            if score > best_score and score > 0.6:
                best_score = score
                best_buffer = buffer 
            elif score < 0.6:
                print("Score is too low, trying again")
        else:
            print(f"Failed to send data: {response.text}")

    # If the loop completes without returning, return the buffer with the best score
    if best_score > 0.6:
        print(f"Did not receive a high enough score after 10 attempts, returning buffer with best score: {best_score}")
        return best_buffer
    else:
        print("Did not receive a score greater than 0.6 after 10 attempts, returning empty buffer")
        return 


def get_img_from_prompt(prompt:str=""):
    data = diffusers.sample(SampleInput(prompt=prompt))
    return data["image"]

async def _generate(models: list, opt: OmegaConf, prompt: str) -> BytesIO:
    try:
        start_time = time()
        print("Trying to get image from diffusers")
        img = get_img_from_prompt(prompt)
        print("Got image from diffusers")

        # generate multi-view img for better 3d modeling
        mv_img = instant_mesh.generate_mvs(img, steps=20, seed=0)

        # get mesh from multi-view img
        mesh_path = instant_mesh.generate_mesh_from_mvs(mv_img)

        ##########################################
        # PUT YOUR CODE HERE
        ##########################################
        # -> The function generate_mesh_from_mvs return
        # a path to .obj file -> you can modify the function 
        # to return the correct input for the buffer below.

        # Below is the old pipeline for buffer (I leave it here for a reference).
        # gaussian_processor = GaussianProcessor.GaussianProcessor(opt, prompt="", base64_img = img)
        # processed_data = gaussian_processor.train(models, opt.iters)
        # hdf5_loader = HDF5Loader.HDF5Loader()
        # buffer = hdf5_loader.pack_point_cloud_to_io_buffer(*processed_data)
        print(f"[INFO] It took: {(time() - start_time) / 60.0} min")
        return buffer

    except Exception as e:
        print(e)
        return ""


@app.post("/generate_raw/")
async def generate_raw(
    prompt: str = Form(),
    opt: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
):
    buffer = await _generate(models, opt, prompt)
    return Response(content=buffer.getvalue(), media_type="application/octet-stream")


@app.post("/generate_model/")
async def generate_model(
    prompt: str = Form(),
    opt: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
) -> Response:
    start_time = time()
    gaussian_processor = GaussianProcessor.GaussianProcessor(opt, prompt)
    gaussian_processor.train(models, opt.iters)
    print(f"[INFO] It took: {(time() - start_time) / 60.0} min")

    buffer = BytesIO()
    gaussian_processor.get_gs_model().save_ply(buffer)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/octet-stream")


@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(),
    video_res: int = Form(1088),
    opt: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
):
    start_time = time()
    gaussian_processor = GaussianProcessor.GaussianProcessor(opt, prompt)
    processed_data = gaussian_processor.train(models, opt.iters)
    print(f"[INFO] It took: {(time() - start_time) / 60.0} min")

    video_utils = VideoUtils(video_res, video_res, 5, 5, 10, -30, 10)
    buffer = video_utils.render_video(*processed_data)

    return StreamingResponse(content=buffer, media_type="video/mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
