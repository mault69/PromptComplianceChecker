import torch
import open_clip
import numpy as np
import openai
from openai import OpenAI, OpenAIError, AuthenticationError, RateLimitError  # ✅ Importation correcte des erreurs OpenAI
from PIL import Image
from pathlib import Path

# 📌 Dossier contenant les fichiers de configuration
CONFIG_PATH = Path(__file__).parent

# 📌 Chargement de la clé API OpenAI depuis un fichier
def load_api_key():
    api_file = CONFIG_PATH / "api_key.txt"
    if api_file.exists():
        return api_file.read_text().strip()
    return None  # ⚠️ Aucune clé API trouvée

# 📌 Cache pour éviter des appels répétés à OpenAI
GPT_MODELS_CACHE = None

# 📌 Récupération de la liste des modèles OpenAI disponibles
def get_openAI_models():
    global GPT_MODELS_CACHE
    if GPT_MODELS_CACHE is not None:
        return GPT_MODELS_CACHE  

    api_key = load_api_key()
    if not api_key:
        print("⚠️ OpenAI API Key not found! Using default GPT model.")
        GPT_MODELS_CACHE = ["gpt-4"]
        return GPT_MODELS_CACHE

    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        GPT_MODELS_CACHE = [model.id for model in models.data]  
        return GPT_MODELS_CACHE
    except Exception as e:
        print(f"⚠️ Error retrieving OpenAI models: {e}")
        GPT_MODELS_CACHE = ["gpt-4"]
        return GPT_MODELS_CACHE  

# 📌 Chargement dynamique des modèles OpenAI
GPT_MODELS = get_openAI_models()

# 📌 Détection des modèles CLIP disponibles
COMFYUI_MODEL_PATH = Path("ComfyUI/models/")
CLIP_PATH = COMFYUI_MODEL_PATH / "clip"

VALID_CLIP_MODELS = ["RN50", "ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336"]

def get_available_models(folder, valid_models=None):
    model_path = COMFYUI_MODEL_PATH / folder
    available = [f.name for f in model_path.iterdir() if f.is_dir() or f.suffix in [".pt", ".safetensors"]] if model_path.exists() else []
    return [m for m in available if m in valid_models] if valid_models else available

CLIP_MODELS = get_available_models("clip", VALID_CLIP_MODELS) or ["ViT-B-32"]
DEVICE_OPTIONS = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

def convert_comfyui_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 4:
        image = image.squeeze(0)
    if image.shape[0] in [1, 3]:  
        image = np.transpose(image, (1, 2, 0))
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image)

class PromptComplianceChecker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": ""}),
                "CLIP": (CLIP_MODELS, {"default": CLIP_MODELS[0]}),
                "device": (DEVICE_OPTIONS, {"default": DEVICE_OPTIONS[0]}),
                "gpt_model": (GPT_MODELS, {"default": GPT_MODELS[0]}),
                "enable_similarity": ("BOOLEAN", {"default": True}),
                "enable_gpt_correction": ("BOOLEAN", {"default": False}),
                "correction_threshold": ("FLOAT", {"default": 70.0, "min": 0.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = (
        "CLIP_Result",
        "CLIP_Similarity_Score",
        "Corrected_Prompt"
    )

    FUNCTION = "run"
    CATEGORY = "Analysis"

    def generate_gpt_prompt(self, prompt, similarity_score, correction_threshold, gpt_model):
        """Génère un prompt amélioré avec ChatGPT en utilisant l'API OpenAI v1.0.0+."""
        if similarity_score >= correction_threshold or similarity_score == -1:
            return "✅ Prompt is well-matched. No correction needed."

        gpt_prompt = f"""
        The following image does not fully match the prompt provided. Improve the prompt to be more detailed and precise.

        **Original Prompt:** {prompt}
        **Similarity Score:** {similarity_score:.2f} (Target: {correction_threshold:.2f})
        """

        try:
            client = OpenAI(api_key=load_api_key())  # ✅ Nouveau format pour OpenAI v1.0.0+
            response = client.chat.completions.create(
                model=gpt_model,
                messages=[{"role": "system", "content": gpt_prompt}]
            )
            return response.choices[0].message.content

        except AuthenticationError:
            return "⚠️ Invalid API Key. Check api_key.txt."

        except RateLimitError:
            return "⚠️ OpenAI Rate Limit Reached. Try again later."

        except OpenAIError as e:
            return f"⚠️ OpenAI Error: {str(e)}"

        except Exception as e:
            return f"⚠️ Unexpected Error: {str(e)}"

    def compute_clip_similarity(self, image, prompt, clip_model, device):
        """Compare l'image et le prompt avec CLIP pour obtenir un score de similarité."""
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained="openai")
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer(clip_model)

        image = preprocess(image).unsqueeze(0).to(device)
        text = tokenizer([prompt])

        with torch.no_grad():
            text_features = model.encode_text(text).to(device)
            image_features = model.encode_image(image).to(device)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).item() * 100

    def run(self, image, prompt, CLIP, device, gpt_model, enable_similarity, enable_gpt_correction, correction_threshold):
        """Exécute l'analyse de conformité du prompt avec l'image."""
        device = torch.device(device)
        image = convert_comfyui_image(image)

        similarity_score = self.compute_clip_similarity(image, prompt, CLIP, device) if enable_similarity else -1

        corrected_prompt = (
            self.generate_gpt_prompt(prompt, similarity_score, correction_threshold, gpt_model)
            if enable_gpt_correction and load_api_key() else
            "✅ Prompt is well-matched. No correction needed."
        )

        return (f"CLIP Model: {CLIP}, Score: {similarity_score:.2f}%", similarity_score, corrected_prompt)

# 📌 Enregistrement du Node
NODE_CLASS_MAPPINGS = {"PromptComplianceChecker": PromptComplianceChecker}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptComplianceChecker": "Prompt Compliance Checker"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
