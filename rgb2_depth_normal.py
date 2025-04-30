
import argparse, os, pathlib, cv2, torch, numpy as np
from PIL import Image
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

#models used
DEPTH_MODEL_NAME  = "depth-anything/Depth-Anything-V2-Small-hf"   # depth (on peut aussi utiliser depth-anything/Depth-Anything-V2-Large-hf)
NORMAL_HUB_REPO   = "alexsax/omnidata_models"                     # normals (on peut aussi utiliser surface_normal_clip_vitl14)
NORMAL_ENTRYPOINT = "surface_normal_dpt_hybrid_384"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We activate depth anything v2
depth_proc  = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
depth_model = AutoModelForDepthEstimation.from_pretrained(
                 DEPTH_MODEL_NAME, torch_dtype=torch.float32
              ).to(device).eval()

# We activate Omnidata normals
normal_model = torch.hub.load(
                   NORMAL_HUB_REPO, NORMAL_ENTRYPOINT, pretrained=True
               ).to(device).eval()
normal_tf = T.Compose([
    T.Resize(384, interpolation=Image.BILINEAR),
    T.CenterCrop(384),
    T.ToTensor()                           
])


#Function to save the image for depth in 256x256 format
def save_depth(tensor, path, target_hw=(256,256)):
    d = torch.nn.functional.interpolate(
            tensor.unsqueeze(0).unsqueeze(0), size=target_hw,
            mode="bilinear", align_corners=False
        ).squeeze()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)   
    im = (d.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(im).save(path)

#Function to save the image for normals in 256x256 format
def save_normals(tensor, path, target_hw=(256,256)):
    n = tensor / (tensor.norm(dim=0, keepdim=True) + 1e-8)    
    n_vis = ((n + 1) * 127.5).clamp(0,255).byte()             
    n_vis = torch.nn.functional.interpolate(
                n_vis.unsqueeze(0), size=target_hw,
                mode="bilinear", align_corners=False
            )[0]
    img = n_vis.permute(1,2,0).cpu().numpy()
    Image.fromarray(img).save(path)

# function for the pipeline
def run(rgb_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    stem = pathlib.Path(rgb_path).stem
    # we load the image and resize it to 256x256
    rgb_pil = Image.open(rgb_path).convert("RGB").resize((256,256), Image.BILINEAR)

    #depth image computed
    with torch.no_grad():
        inputs = depth_proc(images=rgb_pil, return_tensors="pt").to(device)
        depth_pred = depth_model(**inputs).predicted_depth[0]
    save_depth(depth_pred, os.path.join(out_dir, f"{stem}_depth.png"))

    #normal image computed
    with torch.no_grad():
        n_in  = normal_tf(rgb_pil).unsqueeze(0).to(device)
        n_out = normal_model(n_in)[0]                  # shape 3×384×384
    save_normals(n_out, os.path.join(out_dir, f"{stem}_normal.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rgb")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    run(args.rgb, args.out_dir)

#RUN WITH : python rgb2_depth_normal.py rgb_image.jpg --out_dir outputs

#note : rajouter les dépendances suivant dans requirements.txt torch torchvision torchaudio transformers timm pillow opencv-python tqdm