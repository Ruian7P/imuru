import torch
from PIL import Image
from transformers import AutoModel
from huggingface_hub import hf_hub_download
from torchvision.transforms import functional as F


def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((img.width * 64 // img.height, 64))
    img = F.to_tensor(img)
    img = F.normalize(img, [0.5], [0.5])
    return img.unsqueeze(0)


MODEL_PATH = "./emuru_result/head_t5_small_2e-5_ech5"

# Load model (CPU)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
).cpu().eval()


print("Model + VAE moved to CUDA")


# Load sample style image
img_path = hf_hub_download(
    repo_id="blowing-up-groundhogs/emuru",
    filename="sample.png"
)

style_img = load_image(img_path).cpu()

# Generate
gen_text = "EMURU"

with torch.inference_mode():
    generated_pil_image = model.generate(
        gen_text=gen_text,
        style_img=style_img,
        max_new_tokens=64
    )

generated_pil_image.save("my_new_model_output.png")
print("Saved → my_new_model_output.png")
