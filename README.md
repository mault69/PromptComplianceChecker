📌 Description du Node : Prompt Compliance Checker
🚀 Objectif du Node
Le Prompt Compliance Checker est un module avancé pour ComfyUI, conçu pour évaluer et améliorer la conformité d’une image générée avec son prompt d’origine.
Il utilise CLIP pour mesurer la cohérence entre l’image et le prompt, et GPT pour corriger et améliorer le prompt en cas d’incohérence.

📌 Fonctionnalités Principales
1️⃣ Analyse de la Similarité (CLIP)
Utilisation de CLIP pour mesurer la similarité entre l’image et le prompt.
Score de similarité exprimé en pourcentage (0-100%).
Modèle CLIP sélectionnable dans les modèles disponibles de ComfyUI.
2️⃣ Correction Automatique du Prompt (GPT)
Génération de trois versions corrigées du prompt pour améliorer :
🎨 Le Style (ajustement artistique, respect des inspirations visuelles)
💡 L’Éclairage (amélioration des lumières, de l’ambiance visuelle)
🎭 Les Objets & Scène (ajout/suppression d’éléments pour respecter le prompt)
Séparation stricte des trois corrections avec ### pour éviter les décalages.
Utilisation du modèle GPT sélectionnable via un fichier de configuration.
3️⃣ Fusion des Corrections en un Prompt Optimisé
Synthèse des 3 corrections pour proposer un seul prompt ultra-optimisé.
La synthèse prend en compte les meilleurs aspects de chaque correction.
4️⃣ Paramètres Personnalisables
✅ Activation/Désactivation de la correction GPT (si désactivé, retourne le prompt d’origine).
✅ Seuil de correction personnalisable (si le score CLIP est supérieur au seuil, GPT ne modifie pas le prompt).
✅ Sélection du modèle CLIP et GPT directement dans ComfyUI.

📌 Résultats Produits
Nom de la Sortie	Description
CLIP_Result	Nom du modèle CLIP utilisé + Score de similarité (%)
CLIP_Similarity_Score	Score brut de similarité CLIP (0-100)
Style_Corrected_Prompt	Correction du style artistique
Lighting_Corrected_Prompt	Correction de l’éclairage et de l’ambiance
Objects_Corrected_Prompt	Correction des objets et de la mise en scène
Final_Synthesized_Prompt	Fusion des trois corrections en un prompt final amélioré

📌 Exemple de Fonctionnement
🎯 Prompt d’Entrée :
"A futuristic city under a red sunset."

🔍 Score de Similarité CLIP :
CLIP Model: ViT-B-32, Score: 68.4%

🛠️ Corrections Générées :
makefile
Copier
Style_Corrected_Prompt:
"A cyberpunk-inspired futuristic city glowing under a neon red sunset."

Lighting_Corrected_Prompt:
"A futuristic city bathed in warm red hues, with contrasting blue LED reflections."

Objects_Corrected_Prompt:
"A futuristic city with flying cars, towering skyscrapers, and detailed neon signs."
🟢 Synthèse Final_Synthesized_Prompt :
"A cyberpunk-inspired futuristic city glowing under a neon red sunset, baigné dans des teintes rouges chaudes avec des reflets LED bleus contrastants. La scène comprend des voitures volantes, des gratte-ciels imposants et des enseignes lumineuses détaillées."

📌 Configuration et Installation
1️⃣ Ajoutez le Node à custom_nodes/PromptComplianceChecker dans ComfyUI.
2️⃣ Placez votre clé API OpenAI dans api_key.txt dans le dossier du node.
3️⃣ Lancez ComfyUI et sélectionnez le Node.

🔥 Conclusion :
Le Prompt Compliance Checker est un outil puissant et flexible qui permet d’évaluer, de corriger et d’optimiser automatiquement les prompts en fonction de leur adéquation avec l’image générée. 🚀🎨
