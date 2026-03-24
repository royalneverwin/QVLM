import argparse
import os
import json
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image

def fast_map_dpp(kernel, k):
    """
    Fast MAP inference for DPP.
    kernel: (B, N, N)
    k: int, number of items to select
    Returns: select_idx (B, k)
    """
    B, N, _ = kernel.shape
    device = kernel.device
    cis = torch.zeros((k, B, N), device=device) # (T, B, N)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (B, N)
    select_idx = torch.empty((k, B), dtype=torch.long, device=device) # (T, B)
    
    for i in range(k):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j

        eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
            / (torch.sqrt(di2s[torch.arange(B), j] + 1e-8)).unsqueeze(-1)
        cis[i, :, :] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(B), j] = -float('inf')
    
    select_idx = torch.sort(select_idx.t()).values # (B, k)
    return select_idx

def create_mask_image(select_idx, num_patches_side=24, patch_size=14, target_size=256):
    """
    Creates a heatmap where selected tokens are highlighted.
    select_idx: (k,) tensor of selected indices
    """
    mask = torch.zeros((num_patches_side * num_patches_side,))
    mask[select_idx.cpu()] = 1.0
    mask = mask.reshape(1, 1, num_patches_side, num_patches_side)
    mask = F.interpolate(mask, size=(target_size, target_size), mode='nearest')
    mask = mask.squeeze().numpy()
    return mask

def visualize(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name, 
        load_8bit=False, 
        load_4bit=False,
        visual_token_num=args.visual_token_num
    )

    # Load ScienceQA questions
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))

    # Filter out questions without images and take the first 10
    image_samples = [q for q in questions if 'image' in q][:10]

    for sample in image_samples:
        image_file = sample["image"]
        question_id = sample["id"]
        
        # In ScienceQA, the query is the first conversation's value

        qs = sample['conversations'][0]['value'].replace('<image>', '').strip()
        cur_prompt = qs
        
        # Output directory for this specific question
        output_dir = args.output_folder
        os.makedirs(output_dir, exist_ok=True)

        image_path = os.path.join(args.image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping...")
            continue
            
        image = Image.open(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        images = image_tensor.unsqueeze(0).half().cuda()

        image_mean = image_processor.image_mean
        image_std = image_processor.image_std
        resized_image = image_tensor.permute(1, 2, 0)
        resized_image = resized_image * torch.tensor(image_std) + torch.tensor(image_mean)
        resized_image = (resized_image.numpy() * 255).astype(np.uint8)
        resized_image = Image.fromarray(resized_image)
        resized_image = resized_image.resize((256, 256))
        original_image = np.array(resized_image)

        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        cur_prompt = '<image>' + '\n' + cur_prompt

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.no_grad():
            image_features, image_embeds, text_embeds = model.get_model().get_vision_tower()(
                images, 
                texts=sample['conversations'][0]['value']
            )

            # --- 1. Compute Base Relevance (CDPruner style) ---
            # Relevance (cosine similarity as in CDPruner code)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) # (B, N, C)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) # (M, C)
            if text_embeds.shape[0] > 1: # 跳过文本太长的
                print(f"Skip text with text.embeds.length {text_embeds.shape[0]}")
                continue
            relevance = torch.matmul(image_embeds, text_embeds.t()) # (B, N, M)
            relevance_norm = relevance.mean(dim=-1) # (B, N)
            
            # Normalize relevance to [0, 1]
            relevance_min, relevance_max = relevance_norm.min(), relevance_norm.max()
            relevance_norm = (relevance_norm - relevance_min + 1e-8) / (relevance_max - relevance_min + 1e-8)
            
            # --- 2. Compute Quantization Sensitivity ---
            quant_sensitivity = image_features.norm(dim=-1) # (N,)
            quant_min, quant_max = quant_sensitivity.min(), quant_sensitivity.max()
            quant_norm = (quant_sensitivity - quant_min + 1e-8) / (quant_max - quant_min + 1e-8)
            
            # --- 3. Compute Similarity Matrix ---
            image_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, D)
            image_normalized = image_normalized.float() # (B, N, D)
            similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (B, N, N)
            
            # --- Selection 1: Standard CDPruner ---
            relevance_map = -relevance[0].reshape(16, 16).unsqueeze(0).unsqueeze(0).cpu()
            relevance_map = F.interpolate(relevance_map, size=(256, 256), mode='nearest')
            relevance_map = (relevance_map - relevance_map.min()) / (relevance_map.max() - relevance_map.min())
            relevance_map = np.uint8(relevance_map.squeeze(0).squeeze(0).detach() * 255)
            # kernel_base = relevance_norm.unsqueeze(2) * similarity * relevance_norm.unsqueeze(1)
            # select_base = fast_map_dpp(kernel_base, args.visual_token_num)[0]
            # mask_base = create_mask_image(select_base)
            
            # --- Selection 2: Quant-Aware CDPruner ---
            alpha = args.alpha
            fused_relevance = alpha * relevance_norm + (1 - alpha) * quant_norm
            kernel_quant = fused_relevance.unsqueeze(2) * similarity * fused_relevance.unsqueeze(1)
            select_quant = fast_map_dpp(kernel_quant, args.visual_token_num)[0]
            mask_quant = create_mask_image(select_quant)
            
            # --- Selection 3: Top Quantization Outliers (Magnitude only) ---
            # Select top-K highest magnitude tokens directly
            _, select_outliers = torch.topk(quant_sensitivity, args.visual_token_num)
            mask_outliers = create_mask_image(select_outliers)
            
            # --- Plotting ---
            fig = plt.figure(figsize=(15, 10))
            
            # Original
            ax1 = fig.add_subplot(2, 3, 2)
            ax1.imshow(original_image)
            ax1.axis('off')
            ax1.text(0.5, -0.1, "Original Image", size=16, ha="center", transform=ax1.transAxes)
            
            # Standard Pruning
            ax2 = fig.add_subplot(2, 3, 4)
            jet_colormap = plt.get_cmap('jet')
            relevance_map_colored = jet_colormap(relevance_map)
            relevance_map_colored = np.uint8(relevance_map_colored * 255)
            overlay_base = np.uint8(original_image * 0.5 + relevance_map_colored[:, :, :3] * 0.5)
            # overlay_base = np.uint8(original_image * 0.5 + plt.cm.jet(mask_base)[:, :, :3] * 255 * 0.5)
            ax2.imshow(overlay_base)
            ax2.axis('off')
            ax2.text(0.5, -0.1, "Standard Pruning", size=16, ha="center", transform=ax2.transAxes)
            
            # Top Outliers
            ax3 = fig.add_subplot(2, 3, 5)
            overlay_outliers = np.uint8(original_image * 0.5 + plt.cm.jet(mask_outliers)[:, :, :3] * 255 * 0.5)
            ax3.imshow(overlay_outliers)
            ax3.axis('off')
            ax3.text(0.5, -0.1, "Top Outliers", size=16, ha="center", transform=ax3.transAxes)
            
            # Quant-Aware Pruning
            ax4 = fig.add_subplot(2, 3, 6)
            overlay_quant = np.uint8(original_image * 0.5 + plt.cm.jet(mask_quant)[:, :, :3] * 255 * 0.5)
            ax4.imshow(overlay_quant)
            ax4.axis('off')
            ax4.text(0.5, -0.1, "Quant-Aware Pruning", size=16, ha="center", transform=ax4.transAxes)
            
            plt.subplots_adjust(hspace=0.3)
            plt.savefig(os.path.join(output_dir, f"{question_id}_comparison.png"), bbox_inches='tight')
            plt.close()
            
            print(f"Processed QID: {question_id}, text: {sample['conversations'][0]['value']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--question-file", type=str, required=True, help="Path to ScienceQA question JSON")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to ScienceQA images folder")
    parser.add_argument("--output-folder", type=str, default="playground/data/visualize")
    parser.add_argument("--visual_token_num", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.7, help="Fusion weight for Quant-Aware method")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    args = parser.parse_args()

    visualize(args)