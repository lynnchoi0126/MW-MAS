import os
import sys
import json
from glob import glob
from PIL import Image
from datetime import datetime, timedelta
import re

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'WAM')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'three_bricks')))

from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint, default_transform, unnormalize_img,
    create_random_mask, plot_outputs, msg2str
)
# from watermark_anything.augmentation.augmenter import Augmenter

### for image-watermarking evaluation
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import cv2
import time
from skimage.measure import shannon_entropy
from collections import defaultdict

def compute_metrics(img_orig, img_wm, msg_orig, msg_pred):
    # Unnormalize (assuming [-1,1] to [0,255])
    img_orig_np = ((img_orig.squeeze().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
    img_wm_np = ((img_wm.squeeze().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)

    # Convert to CHW → HWC
    img_orig_np = np.transpose(img_orig_np, (1, 2, 0))
    img_wm_np = np.transpose(img_wm_np, (1, 2, 0))

    # PSNR
    psnr_val = peak_signal_noise_ratio(img_orig_np, img_wm_np, data_range=255)

    # SSIM
    ssim_val = structural_similarity(img_orig_np, img_wm_np, multichannel=True, win_size = 3)

    # NC (Normalized Correlation)
    msg_orig_np = msg_orig.squeeze().cpu().numpy().astype(np.float32)
    msg_pred_np = msg_pred.squeeze().cpu().numpy().astype(np.float32)
    nc_val = np.sum(msg_orig_np * msg_pred_np) / (np.sqrt(np.sum(msg_orig_np**2)) * np.sqrt(np.sum(msg_pred_np**2)) + 1e-8)

    # BER (Bit Error Rate)
    bit_errors = np.sum(msg_orig_np != msg_pred_np)
    ber_val = bit_errors / len(msg_orig_np)

    return psnr_val, ssim_val, nc_val, ber_val
def posterior_watermarked(p_value, alpha=0.9, prior = 0.5):
    numerator = alpha * prior
    denominator = numerator + p_value * (1 - prior)
    return numerator / denominator if denominator != 0 else 0

from main_watermark import get_args_parser, WmGenerator, OpenaiGenerator, OpenaiDetector, OpenaiDetectorZ
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
import numpy as np
def natural_key(filename):
    # Extract numbers as integers from filename for sorting
    nums = re.findall(r'\d+', filename)
    return [int(num) for num in nums]
# ---------- Configs ----------
IMG_DIR = "../three_bricks/WIT_dataset/WIT_data_organized_resized_combined"  # Directory containing the original images
PROMPT_PATH = "../three_bricks/data/WIT_data_0623_combined.json"
OUTPUT_DIR = "./llama_noins_paired_output"
CHECKPOINT_DIR = "../WAM/checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load Models ----------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Image model
wam = load_model_from_checkpoint(
    os.path.join(CHECKPOINT_DIR, "params.json"),
    os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
).to(device).eval()

##### (Image Augmentor)
augs = {
    'rotate': 1,
    'jpeg': 1,
    'gaussian_blur': 1,
    'crop_resize_pad': 1,
    'identity': 1,
}
augs_params = {
    'rotate': {'min_angle': -10, 'max_angle': 10},
    'jpeg': {'min_quality': 50, 'max_quality': 90},
    'gaussian_blur': {'min_kernel_size': 3, 'max_kernel_size': 11},
    'crop_resize_pad': {
    'resize_min': 0.8,
    'resize_max': 1.2,
    'crop_min': 0.5,
    'crop_max': 0.7
    }
}
###################################param for orchestration
TEXT_MODEL_MB = 6898.32
IMAGE_MODEL_MB = 94.42
ALPHA = 0.001   # 모델 크기 가중치
LAMBDA = 0.5    # distortion weight
MU = 0.1        # cost weight
###################################param for orchestration

masks = {"kind": "full"}

# augmenter = Augmenter(masks=masks, augs=augs, augs_params=augs_params).eval()
# print('Image Augmentation Applied:')
# for aug, enabled in augs.items():
#     if enabled:
#         print(f"  - {aug}: {augs_params.get(aug, {})}")

# Text model
auth_token = None
### guanaco-7b:
model_name = "huggyllama/llama-7b"
adapters_name = 'timdettmers/guanaco-7b'

# ### GPT:  
# model_name = "gpt2" #"openai-community/gpt2"
# adapters_name = None

### Mistral-7B:
# auth_token = 'XXX'  # Replace with your Hugging Face token if needed
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# adapters_name = None

# ### llama-7b":
# model_name = "huggyllama/llama-7b"
# adapters_name = None
    
#### "guanaco-13b":
    #     model_name = "huggyllama/llama-13b"

# ### gemma 2
# auth_token = 'XXX'  # Replace with your Hugging Face token if needed
# model_name = "google/gemma-2-9b-it"
# adapters_name = None

print(f"Using model: {model_name}, adapters: {adapters_name}")    #     adapters_name = 'timdettmers/guanaco-13b'
OUTPUT_DIR = f"./orch/{model_name.split('/')[-1]}"
if adapters_name is not None:
    OUTPUT_DIR += f"_{adapters_name.split('/')[-1]}"
print(f"Output directory: {OUTPUT_DIR}_orch")

if auth_token is not None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        use_auth_token=auth_token
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        torch_dtype=torch.float16,
    )
from peft import PeftModel
if adapters_name is not None:
    model = PeftModel.from_pretrained(model, adapters_name)
model.eval()

# Generator
generator = OpenaiGenerator(
    model, tokenizer,
    1, 42,
    "hash", 35317, 0
)
# generator = OpenaiGenerator(model, tokenizer,ngram=1, seed=42,seeding="hash", 35317, payload=0)

# Detector
# detector = OpenaiDetector(tokenizer, ngram=1, seed=42, seeding="hash", hash_key=35317)
detector = OpenaiDetector(tokenizer, 1, 42, "hash", 35317)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

# ---------- Load Data ----------
with open(PROMPT_PATH, "r") as f:
    text_data = json.load(f)

image_paths = glob(os.path.join(IMG_DIR, "*.jpg"))
image_paths = sorted(image_paths, key=natural_key)

assert len(image_paths) == len(text_data), "Mismatch between image and text data lengths"

# ---------- Data Slicing ----------
ind_st = 0
ind_end = 250
text_data = text_data[ind_st:ind_end]
image_paths = image_paths[ind_st:ind_end]


# ---------- Process Each Pair ----------
log_stats = []
all_pvalues = []  # Store all p-values for AUC calculation
all_labels = []    # Store corresponding labels (1 for watermarked, 0 for non-watermarked
def extract_features(text, image):
    # Text features
    text = text["input"]
    text_len = len(text.split())
    avg_token_len = np.mean([len(w) for w in text.split()])
    num_sentences = text.count('.') + text.count('!') + text.count('?')

    # Image features
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    entropy = shannon_entropy(gray)
    entropy = np.mean(entropy)
    color_var = np.var(image)

    return {
        "text_len": text_len,
        "avg_token_len": avg_token_len,
        "num_sentences": num_sentences,
        "edge_density": edge_density,
        "entropy": entropy,
        "color_var": color_var
    }

# Region classification
def classify_region(text, image):
    feats = extract_features(text, image)
    if feats["text_len"] < 30 or feats["entropy"] < 5.0:
        return "easy"
    elif feats["text_len"] > 85 or feats["edge_density"] > 0.2:
        return "hard"
    else:
        return "medium"

# --- Reward for each region and agent
stats = defaultdict(lambda: {
    "text": {"score": [], "distortion": [], "cost": []},
    "image": {"score": [], "distortion": [], "cost": []}
})

def image_agent(img): 
    #### ----- Image inference
    
    img_tensor = default_transform(img).unsqueeze(0).to(device)
    wm_msg = wam.get_random_msg(1)
    outputs = wam.embed(img_tensor, wm_msg)
    mask = create_random_mask(img_tensor, num_masks=1, mask_percentage=0.5)
    img_w = outputs['imgs_w'] * mask + img_tensor * (1 - mask)

    #decode unattatacked image
    preds = wam.detect(img_w)["preds"]
    mask_preds = torch.sigmoid(preds[:, 0, :, :])
    bit_preds = preds[:, 1:, :, :]
    pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
    bit_acc = (pred_message == wm_msg).float().mean().item()
    psnr, ssim, nc, ber = compute_metrics(img_tensor, img_w, wm_msg, pred_message)
    return bit_acc, psnr, ssim, nc

def text_agent(text_obj): 
    #### ----- Text generation - watermarked
    ins = text_obj["instruction"]
    input = text_obj["input"]
    if model_name == "gpt2":
        prompt = f"{ins}\n{input}\n"
    if model_name == "huggyllama/llama-7b" and adapters_name == None:
        prompt = f"{ins}\n{input}\n"
    else:
        prompt = f"A chat between a human and an artificial intelligence assistant.\nThe assistant gives helpful answers to the user's questions.\n\n### Human: {ins}\n\n### Input:\n{input}\n\n### Assistant:"
    result = generator.generate([prompt],max_gen_len=120, temperature=1.0,top_p=0.95)
    print(f"Full Generated text: {result}")
    result = result[0]
    result = result[len(prompt):].strip()

    # For watermarked xtext
    scores_no_aggreg_wm = detector.get_scores_by_t([result], scoring_method="v2", payload_max=0)
    scores_wm = detector.aggregate_scores(scores_no_aggreg_wm)
    pvalues_wm = detector.get_pvalues(scores_no_aggreg_wm)
    pvalue_wm = float(pvalues_wm[0][0])
    
    # For original/non-watermarked text
    original_text = text_obj["output"]
    scores_no_aggreg_orig = detector.get_scores_by_t([original_text], scoring_method="v2", payload_max=0)
    scores_orig = detector.aggregate_scores(scores_no_aggreg_orig)
    pvalues_orig = detector.get_pvalues(scores_no_aggreg_orig)
    pvalue_orig = float(pvalues_orig[0][0])

    if pvalue_wm <= 1:
        prediction = 1
    else:
        prediction = 0
    probability = posterior_watermarked(pvalue_wm)
    xs = sbert_model.encode([result, text_obj["output"]], convert_to_tensor=True)
    score_sbert = cossim(xs[0], xs[1]).item()

    return result, probability, score_sbert, pvalue_orig, pvalue_wm
def run_and_update(img, text_obj, selection, region):
    result = {"agents": [], "score": None, "distortion": None, "cost": None}
    scores, dists, costs = [], [], []

    if selection in ["text", "both"]:
        start = time.time()
        gen_text, s, d, pvalue_orig, pvalue_wm = text_agent(text_obj)
        duration = time.time() - start
        c = duration + ALPHA * TEXT_MODEL_MB
        stats[region]["text"]["score"].append(s)
        stats[region]["text"]["distortion"].append(d)
        stats[region]["text"]["cost"].append(c)
        scores.append(s)
        dists.append(d)
        costs.append(c)
        result["agents"].append("text")
        result["text_probability"] = s
        result["score_sbert"] = d
        result["pvalue_orig"] = pvalue_orig
        result["pvalue_wm"] = pvalue_wm
        result["gen_text"] = gen_text

    if selection in ["image", "both"]:
        start = time.time()
        bit_acc, psnr, ssim, nc = image_agent(img)
        duration = time.time() - start
        c = duration + ALPHA * IMAGE_MODEL_MB
        stats[region]["image"]["score"].append(bit_acc)
        stats[region]["image"]["distortion"].append(ssim) 
        stats[region]["image"]["cost"].append(c)
        scores.append(bit_acc)
        dists.append(ssim)
        costs.append(c)
        result["agents"].append("image")
        result["bit_acc"] = bit_acc
        result["psnr"] = psnr
        result["ssim"] = ssim
        result["nc"] = nc

    # Aggregate
    result["score"] = np.mean(scores)
    result["distortion"] = np.mean(dists)
    result["cost"] = np.sum(costs)

    return result


def select_best_combination(region):
    candidates = []

    if stats[region] is None:
        return "both"  # Default to both if no stats available
    if stats[region]["text"] is None or stats[region]["image"] is None:
        return "both"  # Default to both if no stats available
    # Text only
    t = stats[region]["text"]
    if t["score"]: 
        score = np.mean(t["score"])
        dist = np.mean(t["distortion"])
        cost = np.mean(t["cost"])
        reward = score - LAMBDA * dist - MU * cost
        candidates.append(("text", reward))

    # Image only
    i = stats[region]["image"]
    if i["score"]:
        score = np.mean(i["score"])
        dist = np.mean(i["distortion"])
        cost = np.mean(i["cost"])
        reward = score - LAMBDA * dist - MU * cost
        candidates.append(("image", reward))

    # Both
    if t["score"] and i["score"]:
        score = (np.mean(t["score"]) + np.mean(i["score"])) / 2
        dist = (np.mean(t["distortion"]) + np.mean(i["distortion"])) / 2
        cost = np.mean(t["cost"]) + np.mean(i["cost"])
        reward = score - LAMBDA * dist - MU * cost
        candidates.append(("both", reward))

    if not candidates:
        return "both"

    names, rewards = zip(*candidates)
    rewards = list(rewards)
    names = list(names)
    best_idx = np.argmax(rewards)
    probs = [0.25] * len(names)
    probs[best_idx] = 0.5

    probs = np.array(probs)
    probs /= probs.sum()

    selected = np.random.choice(names, p=probs)
    return selected

agent_counts = {"text": 0, "image": 0, "both": 0}
region_counts = {"easy": 0, "medium": 0, "hard": 0}
time_per_samp = []
for idx, (img_path, text_obj) in enumerate(zip(image_paths, text_data)):
    start = datetime.now()
    print(f"Processing pair {idx+1}/{len(image_paths)}")
    idx += ind_st  # Adjust index for logging
    img = Image.open(img_path).convert("RGB")

    region = classify_region(text_obj, img)
    selected = select_best_combination(region)
    result = run_and_update(img, text_obj, selected, region)
    print(f"[{region.upper()}] 선택: {result['agents']} | "
          f"Score: {result['score']:.3f}, Distortion: {result['distortion']:.3f}, Cost: {result['cost']:.3f}")
    end = datetime.now()

    diff = end - start
    time_per_samp.append(diff)
    agent_counts[selected] += 1
    region_counts[region] += 1

    log_stat = {
        'region': region,
        'index': idx,
        'selected': result['agents'],
        'score': result['score'],
        'distortion': result['distortion'],
        'cost': result['cost'],
        'agents': result['agents'],
        'text_probability': result.get('text_probability', None),
        'score_sbert': result.get('score_sbert', None),
        'pvalue_orig': result.get('pvalue_orig', None),
        'pvalue_wm': result.get('pvalue_wm', None),
        'bit_acc': result.get('bit_acc', None),
        'psnr': result.get('psnr', None),
        'ssim': result.get('ssim', None),
        'nc': result.get('nc', None),
        }

    
    # Collect for AUC
    if 'pvalue_wm' in result:
        all_pvalues.extend([result['pvalue_wm'], result['pvalue_orig']])
        all_labels.extend([1, 0])  # 1=watermarked, 0=non-watermarked
    
    log_stats.append(log_stat)

    ## Save outputs
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    sample_dir = os.path.join(OUTPUT_DIR, f"sample_{idx:04d}")
    os.makedirs(sample_dir, exist_ok=True)

    # save_image(unnormalize_img(img_w), os.path.join(sample_dir, "watermarked.png"))
    # save_image(unnormalize_img(img_attacked), os.path.join(sample_dir, f"attacked_{selected_aug}.png"))
    # save_image(mask_preds.unsqueeze(1), os.path.join(sample_dir, "pred_mask.png"))
    # save_image(mask, os.path.join(sample_dir, "target_mask.png"))

    with open(os.path.join(sample_dir, "text_output.txt"), "w") as f:
        # f.write(f"Image Augmentation: {selected_aug}\n")
        if 'gen_text' in result.keys():
            f.write(f"Generated (Watermarked): {result['gen_text']}\n")
            # f.write(f"Original Text: {text_obj['output']}\n")
        # f.write(log_stat)
        for key in log_stat.keys():
            f.write(f"{key}: {log_stat[key]}\n")
    
# ---------- AUC Calculation ----------
probabilities = [posterior_watermarked(p) for p in all_pvalues]
auc_score = roc_auc_score(all_labels, probabilities)
print(f"AUC Score: {auc_score:.4f}")
print(f"Mean S-BERT Score: {np.mean([stat['score_sbert'] for stat in log_stats if stat['score_sbert'] is not None]):.4f}")
print(f"(non-attacked)Mean Bit Accuracy: {np.mean([stat['bit_acc'] for stat in log_stats if stat['bit_acc'] is not None]):.4f}")
print(f"(non-attacked) Mean PSNR: {np.mean([stat['psnr'] for stat in log_stats if stat['psnr'] is not None]):.2f}")
print(f"(non-attacked) Mean SSIM: {np.mean([stat['ssim'] for stat in log_stats if stat['ssim'] is not None]):.4f}")
print(f"(non-attacked) Mean NC: {np.mean([stat['nc'] for stat in log_stats if stat['nc'] is not None]):.4f}")
# print(f"(non-attacked) Mean BER: {np.mean([stat['ber'] for stat in log_stats if stat['ber'] is not None]):.4f}")
# print(f"(Attacked) Mean PSNR: {np.mean([stat['psnr_att'] for stat in log_stats]):.2f}, SSIM: {np.mean([stat['ssim_att'] for stat in log_stats]):.4f}, NC: {np.mean([stat['nc_att'] for stat in log_stats]):.4f}, BER: {np.mean([stat['ber_att'] for stat in log_stats]):.4f}")
print(f"Mean P-value (Original): {np.mean([stat['pvalue_orig'] for stat in log_stats if stat['pvalue_orig'] is not None]):.6f}")
print(f"Mean P-value (Watermarked): {np.mean([stat['pvalue_wm'] for stat in log_stats if stat['pvalue_wm'] is not None]):.6f}")
print("-----------------------------------------------------------------------")
print(f"Agent Counts: {agent_counts}")
print(f"Region Counts: {region_counts}")
print(f"Total pairs processed: {len(log_stats)}")


print("All pairs processed!")

print(f"Total Time: {sum(time_per_samp, timedelta())}")
print(f"Average Time per Sample: {sum(time_per_samp, timedelta()) / len(time_per_samp)}")
