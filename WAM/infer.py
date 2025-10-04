import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from glob import glob
import time
from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint, default_transform, unnormalize_img,
    create_random_mask, plot_outputs, msg2str
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model from the specified checkpoint
exp_dir = "checkpoints"
json_path = os.path.join(exp_dir, "params.json")
ckpt_path = os.path.join(exp_dir, 'checkpoint.pth')
wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
# Seed 
seed = 42
torch.manual_seed(seed)
# to load images
def load_img(path):
    img = Image.open(path).convert("RGB")
    img = default_transform(img).unsqueeze(0).to(device)
    return img

# Parameters
img_dir = "assets/images"  # Directory containing the original images
img_dir = "assets/WIT"  # Directory containing the original images
img_dir = "/home/lynn/lynn/three_bricks/WIT_dataset/WIT_data_organized"  # Directory containing the original images


num_imgs = 2  # Number of images to watermark from the folder
proportion_masked = 0.5  # Proportion of the image to be watermarked (0.5 means 50% of the image)

# create output folder
output_dir = "WIT_output_0519"
os.makedirs(output_dir, exist_ok=True)
# define a 32-bit message to be embedded into the images
wm_msg = wam.get_random_msg(1)  # [1, 32]
print(f"Original message to hide: {msg2str(wm_msg[0])}")

# Iterate over each image in the directory
# for img_ in os.listdir(img_dir)[:num_imgs]:
# for img_ in os.listdir(img_dir):     ################# get all images from the directory
for img_ in sorted(glob(img_dir+"/*.jpg"))[118:]:     ################# get all images from the directory
    # Load and preprocess the image
    
    img_ = img_.split("/")[-1]
    img_pt = load_img(os.path.join(img_dir, img_))  # [1, 3, H, W]
    # img_pt = load_img(os.path.join(img_))  # [1, 3, H, W] ############################ glob 

    # Embed the watermark message into the image
    outputs = wam.embed(img_pt, wm_msg)

    # Create a random mask to watermark only a part of the image
    mask = create_random_mask(img_pt, num_masks=1, mask_percentage=proportion_masked)  # [1, 1, H, W]
    img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)  # [1, 3, H, W]

    # Detect the watermark in the watermarked image
    preds = wam.detect(img_w)["preds"]  # [1, 33, 256, 256]
    mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, 256, 256], predicted mask
    bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits
    
    # Predict the embedded message and calculate bit accuracy
    pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()  # [1, 32]
    bit_acc = (pred_message == wm_msg).float().mean().item()

    # Save the watermarked image and the detection mask
    mask_preds_res = F.interpolate(mask_preds.unsqueeze(1), size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear", align_corners=False)  # [1, 1, H, W]
    ######################################################################   save watermark noise
    save_image(unnormalize_img(img_w), f"{output_dir}/{img_}_wm.png")

    ######################################################################   save watermark noise
    save_image(mask_preds_res, f"{output_dir}/{img_}_pred.png")
    save_image(mask, f"{output_dir}/{img_}_target.png")
    ######################################################################   save watermark noise
    
    plot_outputs(img_pt.detach(), img_w.detach(), mask.detach(), mask_preds_res.detach(), labels = None, centroids = None)
    
    # Print the predicted message and bit accuracy for each image
    print(f"Predicted message for image {img_}: {msg2str(pred_message[0])}")
    print(f"Bit accuracy for image {img_}: {bit_acc:.2f}")