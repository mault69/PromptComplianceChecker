import torch
import open_clip
import numpy as np
import openai
from openai import OpenAI
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

# 📌 Récupération des modèles OpenAI disponibles
GPT_MODELS_CACHE = None

def get_openAI_models():
    """Récupère uniquement les modèles GPT depuis l'API OpenAI."""
    global GPT_MODELS_CACHE
    if GPT_MODELS_CACHE is not None:
        return GPT_MODELS_CACHE  

    api_key = load_api_key()
    if not api_key:
        print("⚠️ OpenAI API Key not found! Using default GPT models.")
        GPT_MODELS_CACHE = ["gpt-4", "gpt-3.5-turbo"]
        return GPT_MODELS_CACHE

    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()

        # 📌 Filtrer uniquement les modèles GPT
        GPT_MODELS_CACHE = [model.id for model in models.data if model.id.startswith("gpt")]

        # 📌 Vérification : si aucun modèle GPT n'est trouvé, utiliser les valeurs par défaut
        if not GPT_MODELS_CACHE:
            print("⚠️ No GPT models found in OpenAI API. Using defaults.")
            GPT_MODELS_CACHE = ["gpt-4", "gpt-3.5-turbo"]

        return GPT_MODELS_CACHE

    except Exception as e:
        print(f"⚠️ Error retrieving OpenAI models: {e}")
        GPT_MODELS_CACHE = ["gpt-4", "gpt-3.5-turbo"]
        return GPT_MODELS_CACHE  

# 📌 Chargement des modèles GPT disponibles
GPT_MODELS = get_openAI_models()

# 📌 Détection des modèles CLIP disponibles
COMFYUI_MODEL_PATH = Path("ComfyUI/models/")
CLIP_PATH = COMFYUI_MODEL_PATH / "clip"
VALID_CLIP_MODELS = ["RN50", "ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336"]

def get_available_models(folder, valid_models=None):
    """Liste les modèles disponibles dans un dossier spécifique."""
    model_path = COMFYUI_MODEL_PATH / folder
    available = [
        f.name for f in model_path.iterdir()
        if f.is_dir() or f.suffix in [".pt", ".safetensors"]
    ] if model_path.exists() else []

    return [m for m in available if m in valid_models] if valid_models else available

CLIP_MODELS = get_available_models("clip", VALID_CLIP_MODELS) or ["ViT-B-32"]
DEVICE_OPTIONS = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

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
                "enable_gpt_correction": ("BOOLEAN", {"default": True}),
                "correction_threshold": ("FLOAT", {"default": 70.0, "min": 0.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "CLIP_Result",
        "CLIP_Similarity_Score",
        "Style_Corrected_Prompt",
        "Lighting_Corrected_Prompt",
        "Objects_Corrected_Prompt",
        "Final_Synthesized_Prompt"
    )

    FUNCTION = "run"
    CATEGORY = "Analysis"

    def run(self, image, prompt, CLIP, device, gpt_model, enable_similarity, enable_gpt_correction, correction_threshold):
        """Exécute l'analyse et génère les prompts corrigés."""
        device = torch.device(device)

        # 📌 Vérification et conversion de l'image
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.ndim == 4:
            image = image.squeeze(0)
        if image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

        # 📌 Chargement et exécution du modèle CLIP
        model, _, preprocess = open_clip.create_model_and_transforms(CLIP, pretrained="openai")
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer(CLIP)

        image = preprocess(image).unsqueeze(0).to(device)
        text = tokenizer([prompt])

        with torch.no_grad():
            text_features = model.encode_text(text).to(device)
            image_features = model.encode_image(image).to(device)

        # 📌 Normalisation et calcul du score de similarité
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_score = (image_features @ text_features.T).item() * 100

        # 📌 Génération des corrections de prompt avec GPT
        if enable_gpt_correction:
            corrected_prompts, final_prompt = self.generate_gpt_prompt(prompt, similarity_score, correction_threshold, gpt_model)
        else:
            corrected_prompts = ["✅ No correction needed."] * 3
            final_prompt = "✅ No synthesis needed."

        return (f"CLIP Model: {CLIP}, Score: {similarity_score:.2f}%", similarity_score, *corrected_prompts, final_prompt)


    def generate_gpt_prompt(self, prompt, similarity_score, correction_threshold, gpt_model):
        """Génère exactement 3 alternatives de prompt avec séparation stricte + une synthèse."""
        if similarity_score >= correction_threshold or similarity_score == -1:
            return ["✅ No correction needed."] * 3, "✅ No synthesis needed."

        gpt_prompt = f"""
        You will generate exactly **three** improved versions of the following prompt.
        Each version must start with the label **Style Correction:**, **Lighting Correction:**, or **Object & Scene Correction:**.

        **Original Prompt:** {prompt}

        **Expected format (use `###` as separator):**
        **Style Correction:** [Your improved prompt]
        ### 
        **Lighting Correction:** [Your improved prompt]
        ### 
        **Object & Scene Correction:** [Your improved prompt]
        """

        try:
            client = OpenAI(api_key=load_api_key())
            response = client.chat.completions.create(
                model=gpt_model,
                messages=[{"role": "system", "content": gpt_prompt}]
            )

            corrected_prompts = response.choices[0].message.content.strip().split("###")
            corrected_prompts = [p.strip() for p in corrected_prompts]

            # 📌 Vérification stricte pour s'assurer d'avoir exactement 3 corrections
            if len(corrected_prompts) < 3:
                corrected_prompts.extend(["⚠️ No correction generated."] * (3 - len(corrected_prompts)))

            elif len(corrected_prompts) > 3:
                corrected_prompts = corrected_prompts[:3]  # Prend uniquement les 3 premières corrections

            # 📌 Ajout de la synthèse des 3 prompts
            final_prompt = self.generate_synthesized_prompt(corrected_prompts, gpt_model)

            return corrected_prompts, final_prompt  

        except Exception as e:
            return [f"⚠️ OpenAI Error: {str(e)}"] * 3, f"⚠️ OpenAI Error: {str(e)}"


    def generate_synthesized_prompt(self, corrected_prompts, gpt_model):
        """Fusionne les 3 prompts en une seule version optimisée."""
        synthesis_prompt = f"""
        Take the following three improved versions of a prompt and create a single, optimized version that integrates the best aspects of all three.

        **Style Correction:** {corrected_prompts[0]}
        **Lighting Correction:** {corrected_prompts[1]}
        **Object & Scene Correction:** {corrected_prompts[2]}

        The final prompt should be well-structured, highly detailed, and artistically optimized.
        """

        try:
            client = OpenAI(api_key=load_api_key())
            response = client.chat.completions.create(
                model=gpt_model,
                messages=[{"role": "system", "content": synthesis_prompt}]
            )

            final_prompt = response.choices[0].message.content.strip()
            return final_prompt

        except Exception as e:
            return f"⚠️ OpenAI Error: {str(e)}"


# 📌 Enregistrement du Node
NODE_CLASS_MAPPINGS = {"PromptComplianceChecker": PromptComplianceChecker}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptComplianceChecker": "Prompt Compliance Checker"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
