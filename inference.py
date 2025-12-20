import argparse
import torch
from PIL import Image
from transformers import AutoModel
from torchvision.transforms import functional as F

def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((img.width * 64 // img.height, 64))
    img = F.to_tensor(img)
    img = F.normalize(img, [0.5], [0.5])
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Ruian7P/imuru_large")
    parser.add_argument("--target_text", type=str, default="Never gonna make you cry")
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--img_path", type=str, default="./dataset/sample.png")
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    style_img = load_image(args.img_path).to(device)

    generated_pil_image = model.generate(
        gen_text=args.target_text,
        style_img=style_img,
        max_new_tokens=256
    )

    out_path = args.save_path.rstrip("/\\") + "/generated_sample.png"
    generated_pil_image.save(out_path)

if __name__ == "__main__":
    main()
