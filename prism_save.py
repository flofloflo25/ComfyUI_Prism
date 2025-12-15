import os
import subprocess
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch
import folder_paths
import sys
from datetime import datetime
import getpass
import shutil
import tempfile
import cv2

# ===== CONFIGURATION ============================================
# Les paramètres sont maintenant chargés depuis config.json
# Copier config.example.json vers config.json et ajuster les chemins

def _load_config():
    """Charge la configuration depuis config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    # Valeurs par défaut
    default_config = {
        "prism_root": "C:\\Program Files\\Prism2",
        "ffmpeg_path": "bin\\ffmpeg.exe",
        "prism_projects_file": "prism_projects.json"
    }

    if not os.path.exists(config_path):
        print(f"[ComfyUI_prism] ⚠ config.json non trouvé, utilisation des valeurs par défaut")
        print(f"[ComfyUI_prism] Copier config.example.json vers config.json et ajuster les chemins")
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Valider et remplir les valeurs manquantes
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value

        return config
    except Exception as e:
        print(f"[ComfyUI_prism] ✗ Erreur lors du chargement de config.json: {e}")
        print(f"[ComfyUI_prism] Utilisation des valeurs par défaut")
        return default_config

# Charger la configuration
_CONFIG = _load_config()

PRISM_ROOT = _CONFIG["prism_root"]

# Chemin vers ffmpeg (peut être relatif au dossier du plugin ou absolu)
FFMPEG_PATH = _CONFIG["ffmpeg_path"]
if not os.path.isabs(FFMPEG_PATH):
    FFMPEG_PATH = os.path.join(os.path.dirname(__file__), FFMPEG_PATH)

# Fichier "cache" contenant l'arborescence des projets Prism
PRISM_PROJECTS_FILE = _CONFIG["prism_projects_file"]
if not os.path.isabs(PRISM_PROJECTS_FILE):
    PRISM_PROJECTS_FILE = os.path.join(os.path.dirname(__file__), PRISM_PROJECTS_FILE)

print(f"[ComfyUI_prism] Configuration chargée:")
print(f"[ComfyUI_prism]   - Prism Root: {PRISM_ROOT}")
print(f"[ComfyUI_prism]   - FFmpeg: {FFMPEG_PATH}")
print(f"[ComfyUI_prism]   - Projects File: {PRISM_PROJECTS_FILE}")
# Séparateur utilisé pour afficher les choix projet > séquence > shot
CHOICE_SEPARATOR = " > "
SEQUENCE_PLACEHOLDER = "[Séquence non requise (assets)]"
SHOT_PLACEHOLDER = "[Shot non requis tant que 'entity_type' != shot]"
# ========================================================================

class PrismSaveImage:
    _prism_cache = None
    _prism_cache_mtime = None
    _normalized_projects = None

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        
    @classmethod
    def INPUT_TYPES(cls):
        # L'utilisateur fournit directement le chemin du projet Prism
        # (ex: D:/_RnD/Prism_RND/Prism_RnD). À partir de ce chemin, le node
        # lit 00_Pipeline/pipeline.json et reconstruit l'arborescence.
        return {
            "required": {
                "images": ("IMAGE",),
                "project_path": ("STRING", {"default": "D:/_RnD/Prism_RND/Prism_RnD"}),
                "entity_type": (["asset", "shot"],),
                "sequence_name": ("STRING", {"default": ""}),
                "shot_name": ("STRING", {"default": ""}),
                "version": ("STRING", {"default": "auto"}),
                 # Nom du dossier dans Renders/2dRender (ex: "comfy", "paintover")
                "identifier": ("STRING", {"default": "comfy"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "optional": {
                # Format d'output pour les images
                "format": (["png", "jpeg", "exr"], {"default": "png"}),
                # Qualité pour JPEG (1-100)
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                # Colorspace pour la sauvegarde
                "colorspace": (["Linear", "sRGB", "ACEScg", "Rec709"], {"default": "sRGB"}),
                # Type de rendu Prism : contrôle le type logique (2D, 3D, etc.)
                "type": (["2D", "3D", "Playblast", "External"], {"default": "2D"}),
                "comment": ("STRING", {"default": ""}),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                # Permet d'ouvrir le Project Browser Prism sans sauver d'images
                "open_project_browser": ("BOOLEAN", {"default": False}),
                "save_versioninfo": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                # Nécessaire pour sauver le workflow dans les métadonnées PNG
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Prism"

    @classmethod
    def _init_prism(cls):
        """Initialize Prism paths.

        NOTE:
        Importing PrismCore inside the ComfyUI Python process seems to cause
        hard crashes (exit code 3221226505) on this setup. To keep ComfyUI
        stable we only prepare the paths here and avoid using PrismCore
        directly. Project Browser is launched via subprocess instead.
        """
        try:
            prism_scripts = os.path.join(PRISM_ROOT, "Scripts")

            if not os.path.exists(prism_scripts):
                print(f"✗ Prism Scripts not found at: {prism_scripts}")
                print(f"   Please check PRISM_ROOT path")
                return None

            if prism_scripts not in sys.path:
                sys.path.insert(0, prism_scripts)
                print(f"✓ Added Prism to path: {prism_scripts}")

            # We intentionally do NOT import PrismCore here.
            return None

        except Exception:
            import traceback
            print("✗ Error initializing Prism:")
            print(traceback.format_exc())
            return None

    @classmethod
    def _load_prism_cache(cls):
        """Load and cache the raw JSON data (with timestamp invalidation)."""
        try:
            if not os.path.exists(PRISM_PROJECTS_FILE):
                cls._prism_cache = None
                cls._normalized_projects = None
                cls._prism_cache_mtime = None
                return None

            current_mtime = os.path.getmtime(PRISM_PROJECTS_FILE)
            if (
                cls._prism_cache is not None
                and cls._prism_cache_mtime == current_mtime
            ):
                return cls._prism_cache

            with open(PRISM_PROJECTS_FILE, "r", encoding="utf-8") as f:
                cls._prism_cache = json.load(f)
                cls._prism_cache_mtime = current_mtime
                cls._normalized_projects = None  # force rebuild
                return cls._prism_cache

        except Exception:
            import traceback
            print("✗ Error loading prism project cache:")
            print(traceback.format_exc())
            cls._prism_cache = None
            cls._normalized_projects = None
            cls._prism_cache_mtime = None
            return None

    @classmethod
    def _get_normalized_projects(cls):
        """Return a normalized list of projects with sequences/shots expanded."""
        cache = cls._load_prism_cache()
        if cache is None:
            return []

        if cls._normalized_projects is not None:
            return cls._normalized_projects

        normalized = []
        projects = cache.get("projects", [])
        for project in projects:
            project_name = project.get("name", "UnnamedProject")

            normalized_sequences = []
            sequences = project.get("sequences", [])
            if isinstance(sequences, list):
                for seq in sequences:
                    if isinstance(seq, dict):
                        seq_name = seq.get("name", "UnnamedSequence")
                        seq_shots = seq.get("shots", [])
                    else:
                        seq_name = str(seq)
                        seq_shots = []
                    normalized_sequences.append(
                        {
                            "name": seq_name,
                            "shots": seq_shots or []
                        }
                    )

            normalized.append(
                {
                    "name": project_name,
                    "path": project.get("path"),
                    "assets": project.get("assets", []),
                    "sequences": normalized_sequences,
                    "shots": project.get("shots", []),
                }
            )

        cls._normalized_projects = normalized
        return normalized

    @classmethod
    def _get_prism_projects(cls):
        """Return project names for the dropdown."""
        projects = cls._get_normalized_projects()
        if not projects:
            return ["[No Prism project cache found - see prism_projects.json]"]
        return [project["name"] for project in projects]

    @classmethod
    def _get_sequence_choices(cls):
        """Return list of project > sequence combos for the dropdown."""
        projects = cls._get_normalized_projects()
        if not projects:
            return [SEQUENCE_PLACEHOLDER]

        choices = [SEQUENCE_PLACEHOLDER]
        for project in projects:
            project_name = project["name"]
            for sequence in project["sequences"]:
                label = f"{project_name}{CHOICE_SEPARATOR}{sequence['name']}"
                choices.append(label)
        if len(choices) == 1:
            choices.append("[Aucune séquence déclarée]")
        return choices

    @classmethod
    def _get_shot_choices(cls):
        """Return list of project > (sequence) > shot combos for dropdown."""
        projects = cls._get_normalized_projects()
        if not projects:
            return [SHOT_PLACEHOLDER]

        choices = [SHOT_PLACEHOLDER]
        for project in projects:
            project_name = project["name"]

            # Shots defined inside sequences
            for sequence in project["sequences"]:
                sequence_name = sequence["name"]
                for shot in sequence.get("shots", []):
                    label = (
                        f"{project_name}{CHOICE_SEPARATOR}"
                        f"{sequence_name}{CHOICE_SEPARATOR}{shot}"
                    )
                    choices.append(label)

            # Legacy per-project shots (without sequence info)
            for shot in project.get("shots", []):
                label = f"{project_name}{CHOICE_SEPARATOR}{shot}"
                choices.append(label)

        if len(choices) == 1:
            choices.append("[Aucun shot déclaré]")
        return choices

    @classmethod
    def _find_project(cls, project_name):
        """Return normalized project dict by name."""
        for project in cls._get_normalized_projects():
            if project["name"] == project_name:
                return project
        return None

    @staticmethod
    def _is_placeholder(value):
        return not value or value.startswith("[")

    @classmethod
    def _parse_sequence_choice(cls, project_name, sequence_choice):
        """Extract sequence name, ensuring it belongs to the selected project."""
        if cls._is_placeholder(sequence_choice):
            return None

        parts = [part.strip() for part in sequence_choice.split(CHOICE_SEPARATOR)]
        if len(parts) == 1:
            seq_project = project_name
            sequence_name = parts[0]
        else:
            seq_project = parts[0]
            sequence_name = parts[-1]

        if seq_project != project_name:
            print(
                f"⚠ Sequence '{sequence_choice}' does not belong to selected project "
                f"'{project_name}'. Using it anyway."
            )

        return sequence_name or None

    @classmethod
    def _parse_shot_choice(cls, project_name, shot_choice):
        """Extract (sequence_name, shot_name) tuple from dropdown label."""
        if cls._is_placeholder(shot_choice):
            return (None, None)

        parts = [part.strip() for part in shot_choice.split(CHOICE_SEPARATOR)]
        shot_project = project_name
        sequence_name = None
        shot_name = None

        if len(parts) == 1:
            shot_name = parts[0]
        elif len(parts) == 2:
            shot_project, shot_name = parts
        else:
            shot_project = parts[0]
            shot_name = parts[-1]
            sequence_name = parts[-2]

        if shot_project != project_name:
            print(
                f"⚠ Shot '{shot_choice}' does not belong to selected project "
                f"'{project_name}'. Using it anyway."
            )

        return (sequence_name, shot_name)

    def save_images(self, images, project_path, entity_type, sequence_name,
                   shot_name, version, identifier,
                   filename_prefix, format="png", jpeg_quality=95, colorspace="sRGB", type="2D", comment="",
                   prompt_text="", open_project_browser=False, save_versioninfo=True,
                   prompt=None, extra_pnginfo=None):

        # Mode bouton : ouvrir Prism / Project Browser via un process externe,
        # sans importer PrismCore dans le process de ComfyUI (sinon crash).
        if open_project_browser:
            try:
                prism_scripts = os.path.join(PRISM_ROOT, "Scripts")
                tray_path = os.path.join(prism_scripts, "PrismTray.py")

                if os.path.exists(tray_path):
                    print(f"✓ Launching PrismTray: {tray_path}")
                    subprocess.Popen([sys.executable, tray_path])
                    # UI values must be iterables (lists), not bare bools
                    return {"ui": {"project_browser_opened": [True], "images": []}, "result": (images,)}
                else:
                    print(f"✗ PrismTray.py not found at: {tray_path}")
                    return {"ui": {"project_browser_opened": [False], "images": []}, "result": (images,)}
            except Exception:
                import traceback
                print("✗ Erreur lors du lancement de PrismTray / Project Browser :")
                print(traceback.format_exc())
                return {"ui": {"project_browser_opened": [False], "images": []}, "result": (images,)}

        # Les noms de séquence et de shot sont fournis tels quels par l'utilisateur
        selected_sequence = (sequence_name or "").strip() or None
        selected_shot = (shot_name or "").strip() or None

        project_data = {
            "name": os.path.basename(os.path.normpath(project_path)),
            "path": project_path,
        }

        if entity_type == "shot":
            if not selected_sequence:
                raise Exception(
                    "Veuillez renseigner une séquence valide pour publier un shot "
                    "(champ 'sequence_name' dans le node PrismSaveImage)."
                )
            if not selected_shot:
                raise Exception(
                    "Veuillez renseigner un nom de shot valide (champ 'shot_name' "
                    "dans le node PrismSaveImage)."
                )
        # Calcul du chemin de rendu 2D (sans PrismCore) en suivant pipeline.json
        output_dir = self._build_2d_render_path(
            project_data=project_data,
            entity_type=entity_type,
            sequence_name=selected_sequence,
            shot_name=selected_shot,
            version=version,
            identifier=identifier,
        )

        # Sauvegarde des images ComfyUI dans ce dossier
        frame_count, results = self._save_tensor_images(
            images=images,
            target_dir=output_dir,
            filename_prefix=filename_prefix,
            format=format,
            jpeg_quality=jpeg_quality,
            colorspace=colorspace,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )

        # Copier les images dans le dossier output de ComfyUI pour le preview
        preview_results = []
        for result in results:
            src_path = os.path.join(output_dir, result["filename"])
            dst_path = os.path.join(self.output_dir, result["filename"])

            try:
                shutil.copy2(src_path, dst_path)
                preview_results.append({
                    "filename": result["filename"],
                    "subfolder": "",
                    "type": "output"
                })
            except Exception as e:
                print(f"⚠ Impossible de copier l'image pour le preview: {e}")

        # Sauvegarde du versioninfo.json si demandé
        if save_versioninfo:
            # Récupérer la version normalisée du dossier de sortie
            version_folder = os.path.basename(output_dir)

            # Obtenir le nom d'utilisateur
            try:
                username = getpass.getuser()
            except Exception:
                username = os.getenv('USERNAME', 'unknown')

            # Obtenir le nom du projet
            project_name = os.path.basename(os.path.normpath(project_path))

            # Format de la date comme dans Prism (DD.MM.YY HH:MM:SS)
            date_str = datetime.now().strftime("%d.%m.%y %H:%M:%S")

            # Construire le chemin de sortie des fichiers
            file_extension = "jpg" if format == "jpeg" else format
            output_pattern = os.path.join(
                output_dir,
                f"{filename_prefix}_####.{file_extension}"
            )

            # Construire les données de version enrichies
            try:
                info_path = os.path.join(output_dir, "versioninfo.json")

                # Construire la hiérarchie (sequence/shot pour les shots)
                hierarchy = ""
                if entity_type == "shot" and selected_sequence and selected_shot:
                    hierarchy = f"{selected_sequence}/{selected_shot}"

                versioninfo_data = {
                    "hierarchy": hierarchy,
                    "itemType": entity_type,
                    "sequence": selected_sequence or "",
                    "shot": selected_shot or "",
                    "type": entity_type,
                    "identifier": identifier,
                    "user": username.lower(),
                    "version": version_folder,
                    "comment": comment or "",
                    "extension": f".{file_extension}",
                    "mediaType": "2drenders",
                    "username": username,
                    "date": date_str,
                    "colorspace": colorspace,
                    "dependencies": [],
                    "externalFiles": []
                }

                # Ajouter le prompt si fourni
                if prompt_text and prompt_text.strip():
                    versioninfo_data["prompt"] = prompt_text.strip()

                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(versioninfo_data, f, indent=4, ensure_ascii=False)
                print(f"✓ versioninfo.json sauvegardé : {info_path}")
                print(f"  Commentaire : '{comment}'")
                if prompt_text and prompt_text.strip():
                    print(f"  Prompt inclus dans le JSON")
            except Exception:
                import traceback
                print("⚠ Impossible d'écrire versioninfo.json :")
                print(traceback.format_exc())

        print(f"✓ Images sauvegardées dans Prism (2D) : {output_dir}")
        print(f"  Format: {format}, Colorspace: {colorspace}")

        return {"ui": {"images": preview_results}, "result": (images,)}

    # ------------------------------------------------------------------
    # Helpers pour construire les chemins 2D en lisant pipeline.json
    # ------------------------------------------------------------------
    @staticmethod
    def _read_pipeline_config(project_data):
        """Read 00_Pipeline/pipeline.json for a given project."""
        project_path = project_data.get("path")
        if not project_path:
            raise Exception(
                f"Aucun 'path' défini pour le projet '{project_data.get('name')}'."
            )

        pipeline_path = os.path.join(project_path, "00_Pipeline", "pipeline.json")
        if not os.path.exists(pipeline_path):
            raise Exception(
                f"pipeline.json introuvable pour le projet : {pipeline_path}"
            )

        with open(pipeline_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize_version_string(raw_version, padding):
        """Return a version string comme 'v0001' à partir de 'auto', '1', 'v0010', etc."""
        if not raw_version or raw_version.lower() == "auto":
            return None  # géré plus tard

        v = str(raw_version).strip()
        if v.startswith("v") or v.startswith("V"):
            num = v[1:]
        else:
            num = v

        try:
            num_int = int(num)
        except ValueError:
            # On garde tel quel si ce n'est pas un entier
            return v

        return "v" + str(num_int).zfill(padding)

    @staticmethod
    def _next_version_in_dir(render_root, padding):
        """Retourne la prochaine version style 'v0001' en scannant un dossier."""
        if not os.path.exists(render_root):
            return "v" + "1".zfill(padding)

        versions = [
            d for d in os.listdir(render_root)
            if os.path.isdir(os.path.join(render_root, d)) and d.lower().startswith("v")
        ]
        if not versions:
            return "v" + "1".zfill(padding)

        max_num = 0
        for v in versions:
            try:
                num = int(v[1:])
                max_num = max(max_num, num)
            except Exception:
                continue

        return "v" + str(max_num + 1).zfill(padding)

    @staticmethod
    def _latest_version_in_dir(render_root):
        """Retourne la dernière version existante (ex: 'v0004')."""
        if not os.path.exists(render_root):
            return None

        versions = [
            d for d in os.listdir(render_root)
            if os.path.isdir(os.path.join(render_root, d)) and d.lower().startswith("v")
        ]
        if not versions:
            return None

        max_num = 0
        latest = None
        for v in versions:
            try:
                num = int(v[1:])
                if num >= max_num:
                    max_num = num
                    latest = v
            except Exception:
                continue
        return latest

    def _build_2d_render_path(
        self,
        project_data,
        entity_type,
        sequence_name,
        shot_name,
        version,
        identifier,
    ):
        """Construit le chemin de rendu 2D à partir de pipeline.json."""
        cfg = self._read_pipeline_config(project_data)
        folder_structure = cfg.get("folder_structure", {})
        globals_cfg = cfg.get("globals", {})

        project_path = project_data.get("path")
        if not project_path:
            raise Exception(
                f"Aucun 'path' défini pour le projet '{project_data.get('name')}'."
            )

        # --------------------------------------------------------------
        # 1) Chemin de l'entité (asset ou shot)
        # --------------------------------------------------------------
        if entity_type == "asset":
            assets_tpl = folder_structure.get("assets", {}).get("value")
            if not assets_tpl:
                raise Exception("Template 'assets' manquant dans pipeline.json.")

            # Sans champ dédié dans l'UI, on utilise l'identifier comme nom d'asset
            asset_path = identifier or "Asset"
            entity_path = assets_tpl.replace("@project_path@", project_path)
            entity_path = entity_path.replace("@asset_path@", asset_path)

        elif entity_type == "shot":
            if not sequence_name or not shot_name:
                raise Exception(
                    "Séquence et shot requis pour publier un type 'shot'."
                )

            sequences_tpl = folder_structure.get("sequences", {}).get("value")
            shots_tpl = folder_structure.get("shots", {}).get("value")

            if not sequences_tpl or not shots_tpl:
                raise Exception(
                    "Templates 'sequences' ou 'shots' manquants dans pipeline.json."
                )

            sequence_path = sequences_tpl.replace("@project_path@", project_path)
            sequence_path = sequence_path.replace("@sequence@", sequence_name)

            entity_path = shots_tpl.replace("@sequence_path@", sequence_path)
            entity_path = entity_path.replace("@shot@", shot_name)
        else:
            raise Exception(f"Type d'entité inconnu : {entity_type}")

        # --------------------------------------------------------------
        # 2) Chemin des 2d renders + identifier
        # --------------------------------------------------------------
        renders2d_tpl = folder_structure.get("2drenders", {}).get("value")
        if not renders2d_tpl:
            raise Exception("Template '2drenders' manquant dans pipeline.json.")

        render_root = renders2d_tpl.replace("@entity_path@", entity_path)
        render_root = render_root.replace("@identifier@", identifier)

        # --------------------------------------------------------------
        # 3) Versioning : @render_path@/@version@
        # --------------------------------------------------------------
        render_versions_tpl = folder_structure.get("renderVersions", {}).get("value")
        if not render_versions_tpl:
            raise Exception("Template 'renderVersions' manquant dans pipeline.json.")

        version_padding = int(globals_cfg.get("versionPadding", 4))
        normalized_version = self._normalize_version_string(version, version_padding)

        if normalized_version is None:
            # auto → calculer la prochaine version
            normalized_version = self._next_version_in_dir(render_root, version_padding)

        render_version_path = render_versions_tpl.replace("@render_path@", render_root)
        render_version_path = render_version_path.replace("@version@", normalized_version)

        os.makedirs(render_version_path, exist_ok=True)
        return render_version_path

    @staticmethod
    def _convert_colorspace(img, colorspace):
        """Convertit une image depuis sRGB (input ComfyUI) vers le colorspace cible.

        Args:
            img: Image numpy array en float32 [0, 1], assumée en sRGB (comportement par défaut ComfyUI)
            colorspace: Colorspace cible ("Linear", "sRGB", "ACEScg", "Rec709")

        Returns:
            Image convertie en float32 [0, 1]
        """
        if colorspace == "sRGB":
            # Pas de conversion nécessaire, l'image est déjà en sRGB
            return img

        elif colorspace == "Linear":
            # Conversion sRGB -> Linear (inverse du gamma sRGB)
            def srgb_to_linear(srgb):
                linear = np.where(
                    srgb <= 0.04045,
                    srgb / 12.92,
                    np.power(np.clip((srgb + 0.055) / 1.055, 0, 1), 2.4)
                )
                return linear

            return srgb_to_linear(img)

        elif colorspace == "Rec709":
            # Rec709 utilise un gamma similaire à sRGB
            # Pour une conversion précise, on devrait passer par Linear, mais comme sRGB et Rec709
            # sont très proches, on peut garder l'image telle quelle ou faire un gamma simple
            # Conversion sRGB -> Linear -> Rec709
            linear = PrismSaveImage._convert_colorspace(img, "Linear")
            # Linear -> Rec709 (gamma 2.2)
            return np.power(np.clip(linear, 0, 1), 1.0/2.2)

        elif colorspace == "ACEScg":
            # ACEScg est un colorspace linéaire avec une gamut spécifique
            # Conversion: sRGB -> Linear sRGB -> ACEScg

            # Étape 1: sRGB -> Linear sRGB
            linear = PrismSaveImage._convert_colorspace(img, "Linear")

            # Étape 2: Linear sRGB -> ACEScg (matrice de conversion)
            # Matrice de conversion sRGB Linear -> ACEScg (approximation)
            matrix = np.array([
                [ 0.613097,  0.339523,  0.047379],
                [ 0.070194,  0.916354,  0.013452],
                [ 0.020616,  0.109570,  0.869815]
            ], dtype=np.float32)

            # Appliquer la matrice
            original_shape = linear.shape
            if linear.ndim == 2:
                # Image en niveaux de gris, dupliquer sur 3 canaux
                img_rgb = np.stack([linear, linear, linear], axis=-1)
            else:
                img_rgb = linear

            img_flat = img_rgb.reshape(-1, 3)
            img_converted = np.dot(img_flat, matrix.T)
            img_converted = img_converted.reshape(img_rgb.shape)

            # Retourner avec la forme originale si c'était en niveaux de gris
            if linear.ndim == 2:
                # Prendre le premier canal
                img_converted = img_converted[..., 0]

            return np.clip(img_converted, 0, 65504)  # Clamp pour EXR half float

        else:
            # Colorspace non reconnu, retourner tel quel
            print(f"⚠ Colorspace '{colorspace}' non reconnu, aucune conversion appliquée")
            return img

    @staticmethod
    def _save_tensor_images(images, target_dir, filename_prefix,
                            format="png", jpeg_quality=95, colorspace="sRGB",
                            prompt=None, extra_pnginfo=None):
        """Sauvegarde un batch d'images ComfyUI dans un dossier cible,
        avec métadonnées (workflow) compatibles avec le node Save Image.
        Retourne le nombre d'images sauvegardées et les infos pour le preview UI."""
        if isinstance(images, torch.Tensor):
            imgs = images.detach().cpu().numpy()
        else:
            imgs = np.array(images)

        # BxHxWxC
        if imgs.ndim == 3:
            imgs = imgs[None, ...]

        frame_count = 0
        results = []
        for i, img in enumerate(imgs):
            # Appliquer la conversion de colorspace
            img_converted = PrismSaveImage._convert_colorspace(img, colorspace)

            # Pour EXR, on garde les valeurs float32 [0, 1]
            # Pour PNG/JPEG, on convertit en uint8 [0, 255]
            if format == "exr":
                img_np = np.clip(img_converted, 0, 65504).astype(np.float32)
                file_extension = "exr"

                # Sauvegarder en EXR avec OpenEXR
                filename = f"{filename_prefix}_{i:04d}.{file_extension}"
                out_path = os.path.join(target_dir, filename)

                try:
                    import OpenEXR
                    import Imath

                    height, width = img_np.shape[:2]
                    channels = img_np.shape[2] if img_np.ndim == 3 else 1

                    # Préparer les canaux
                    if channels == 1:
                        # Image en niveaux de gris
                        r_channel = img_np[:, :].tobytes()
                        g_channel = r_channel
                        b_channel = r_channel
                    else:
                        # Image RGB
                        r_channel = img_np[:, :, 0].tobytes()
                        g_channel = img_np[:, :, 1].tobytes()
                        b_channel = img_np[:, :, 2].tobytes()

                    # Créer le header EXR
                    header = OpenEXR.Header(width, height)
                    header['channels'] = {
                        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                    }

                    # Écrire le fichier EXR
                    exr_file = OpenEXR.OutputFile(out_path, header)
                    exr_file.writePixels({'R': r_channel, 'G': g_channel, 'B': b_channel})
                    exr_file.close()

                except ImportError:
                    print("⚠ OpenEXR non disponible, tentative avec OpenCV...")
                    try:
                        # Fallback avec OpenCV si disponible
                        # OpenCV attend BGR, pas RGB
                        if img_np.ndim == 3 and img_np.shape[2] == 3:
                            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        else:
                            img_bgr = img_np
                        cv2.imwrite(out_path, img_bgr, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
                    except Exception as e:
                        print(f"✗ Impossible de sauvegarder en EXR: {e}")
                        print("  Installez OpenEXR: pip install OpenEXR")
                        raise

            else:
                # PNG ou JPEG
                img_np = np.clip(img_converted * 255.0, 0, 255).astype(np.uint8)
                if img_np.shape[-1] == 1:
                    img_np = img_np[..., 0]
                pil_img = Image.fromarray(img_np)

                # Construction des métadonnées PNG (seulement pour PNG)
                if format == "png":
                    pnginfo = PngInfo()
                    if prompt is not None:
                        try:
                            pnginfo.add_text("prompt", json.dumps(prompt))
                        except Exception:
                            pass

                    if isinstance(extra_pnginfo, dict):
                        for k, v in extra_pnginfo.items():
                            try:
                                pnginfo.add_text(k, json.dumps(v))
                            except Exception:
                                continue

                    file_extension = "png"
                    filename = f"{filename_prefix}_{i:04d}.{file_extension}"
                    out_path = os.path.join(target_dir, filename)
                    pil_img.save(out_path, pnginfo=pnginfo, compress_level=4)

                elif format == "jpeg":
                    file_extension = "jpg"
                    filename = f"{filename_prefix}_{i:04d}.{file_extension}"
                    out_path = os.path.join(target_dir, filename)

                    # Convertir en RGB si nécessaire (JPEG ne supporte pas RGBA)
                    if pil_img.mode == "RGBA":
                        pil_img = pil_img.convert("RGB")

                    pil_img.save(out_path, quality=jpeg_quality, optimize=True)

            frame_count += 1

            # Ajouter les infos pour le preview UI avec chemin absolu
            results.append({
                "filename": filename,
                "subfolder": target_dir,
                "type": "temp"
            })

        return frame_count, results


class PrismLoadImage:
    """Node utilitaire pour relire les rendus Prism (2D) depuis un projet."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_path": ("STRING", {"default": "D:/_RnD/Prism_RND/Prism_RnD"}),
                "sequence_name": ("STRING", {"default": ""}),
                "shot_name": ("STRING", {"default": ""}),
                "identifier": ("STRING", {"default": "comfy"}),
            },
            "optional": {
                # Spécifier une version (ex: v0003). Si "latest" ou vide -> dernière.
                "version": ("STRING", {"default": "latest"}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 9999}),
                # Champ factice pour forcer le recalcul (à connecter à un seed, timestamp, etc.)
                "force_refresh": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load_images"
    CATEGORY = "Prism"

    def load_images(
        self,
        project_path,
        sequence_name,
        shot_name,
        identifier,
        version="latest",
        limit=0,
        force_refresh="",
    ):
        project_data = {
            "name": os.path.basename(os.path.normpath(project_path)),
            "path": project_path,
        }

        sequence_name = (sequence_name or "").strip()
        shot_name = (shot_name or "").strip()
        identifier = (identifier or "").strip()

        if not sequence_name:
            raise Exception("sequence_name est requis pour charger un shot Prism.")
        if not shot_name:
            raise Exception("shot_name est requis pour charger un shot Prism.")
        if not identifier:
            raise Exception("identifier est requis (ex: comfy, paintover, etc.).")

        cfg = PrismSaveImage._read_pipeline_config(project_data)
        folder_structure = cfg.get("folder_structure", {})
        globals_cfg = cfg.get("globals", {})
        project_dir = project_data["path"]

        sequences_tpl = folder_structure.get("sequences", {}).get("value")
        shots_tpl = folder_structure.get("shots", {}).get("value")
        renders2d_tpl = folder_structure.get("2drenders", {}).get("value")
        render_versions_tpl = folder_structure.get("renderVersions", {}).get("value")

        if not (sequences_tpl and shots_tpl and renders2d_tpl and render_versions_tpl):
            raise Exception(
                "Templates 'sequences', 'shots', '2drenders' ou 'renderVersions' "
                "manquants dans pipeline.json."
            )

        sequence_path = sequences_tpl.replace("@project_path@", project_dir)
        sequence_path = sequence_path.replace("@sequence@", sequence_name)

        entity_path = shots_tpl.replace("@sequence_path@", sequence_path)
        entity_path = entity_path.replace("@shot@", shot_name)

        render_root = renders2d_tpl.replace("@entity_path@", entity_path)
        render_root = render_root.replace("@identifier@", identifier)

        render_versions_dir = render_versions_tpl.replace("@render_path@", render_root)
        version_base_dir = render_versions_dir.replace("@version@", "")
        version_base_dir = version_base_dir.rstrip("/\\") or render_root
        version_padding = int(globals_cfg.get("versionPadding", 4))

        if not version or version.lower() == "latest":
            version_folder = PrismSaveImage._latest_version_in_dir(version_base_dir)
            if not version_folder:
                raise Exception(
                    f"Aucune version trouvée dans {render_root}. Vérifie le chemin."
                )
        else:
            version_folder = PrismSaveImage._normalize_version_string(
                version, version_padding
            )

        version_path = render_versions_dir.replace("@version@", version_folder)
        if not os.path.exists(version_path):
            raise Exception(f"Dossier de version introuvable : {version_path}")

        image_files = [
            os.path.join(version_path, f)
            for f in sorted(os.listdir(version_path))
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".exr"))
        ]

        if not image_files:
            raise Exception(f"Aucune image trouvée dans {version_path}")

        if limit > 0:
            image_files = image_files[:limit]

        images = []
        for path in image_files:
            # Charger les fichiers EXR avec OpenCV ou OpenEXR
            if path.lower().endswith(".exr"):
                try:
                    # Tenter avec OpenCV d'abord (plus simple)
                    img_bgr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if img_bgr is None:
                        raise Exception(f"Impossible de charger {path} avec OpenCV")

                    # Convertir BGR vers RGB
                    if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        img_rgb = img_bgr

                    # Les données EXR sont déjà en float32, on s'assure qu'elles sont dans [0, 1]
                    arr = np.clip(img_rgb, 0, 1).astype(np.float32)
                    images.append(arr)

                except Exception:
                    # Fallback sur OpenEXR
                    try:
                        import OpenEXR
                        import Imath

                        exr_file = OpenEXR.InputFile(path)
                        header = exr_file.header()

                        dw = header['dataWindow']
                        width = dw.max.x - dw.min.x + 1
                        height = dw.max.y - dw.min.y + 1

                        # Lire les canaux RGB
                        pt = Imath.PixelType(Imath.PixelType.FLOAT)
                        r_str = exr_file.channel('R', pt)
                        g_str = exr_file.channel('G', pt)
                        b_str = exr_file.channel('B', pt)

                        # Convertir en numpy
                        r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
                        g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
                        b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)

                        # Combiner les canaux
                        arr = np.stack([r, g, b], axis=-1)
                        arr = np.clip(arr, 0, 1).astype(np.float32)
                        images.append(arr)

                    except Exception as e:
                        print(f"⚠ Impossible de charger {path}: {e}")
                        print("  Installez OpenCV ou OpenEXR: pip install opencv-python ou pip install OpenEXR")
                        raise
            else:
                # Charger avec PIL pour les autres formats
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    arr = np.array(img).astype(np.float32) / 255.0
                    images.append(arr)

        images_np = np.stack(images, axis=0)
        images_tensor = torch.from_numpy(images_np)
        return (images_tensor,)


class PrismLoadVideo:
    """Node utilitaire pour charger les vidéos Prism (2D) depuis un projet."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_path": ("STRING", {"default": "D:/_RnD/Prism_RND/Prism_RnD"}),
                "sequence_name": ("STRING", {"default": ""}),
                "shot_name": ("STRING", {"default": ""}),
                "identifier": ("STRING", {"default": "comfy"}),
            },
            "optional": {
                # Spécifier une version (ex: v0003). Si "latest" ou vide -> dernière.
                "version": ("STRING", {"default": "latest"}),
                # Extension vidéo à rechercher
                "video_format": (["mp4", "mov", "avi", "webm"], {"default": "mp4"}),
                # Limiter le nombre de frames chargées (0 = toutes)
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 9999}),
                # Sauter des frames (1 = toutes, 2 = une sur deux, etc.)
                "skip_frames": ("INT", {"default": 1, "min": 1, "max": 100}),
                # Champ factice pour forcer le recalcul
                "force_refresh": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("images", "frame_count", "fps")
    FUNCTION = "load_video"
    CATEGORY = "Prism"

    def load_video(
        self,
        project_path,
        sequence_name,
        shot_name,
        identifier,
        version="latest",
        video_format="mp4",
        max_frames=0,
        skip_frames=1,
        force_refresh="",
    ):
        project_data = {
            "name": os.path.basename(os.path.normpath(project_path)),
            "path": project_path,
        }

        sequence_name = (sequence_name or "").strip()
        shot_name = (shot_name or "").strip()
        identifier = (identifier or "").strip()

        if not sequence_name:
            raise Exception("sequence_name est requis pour charger un shot Prism.")
        if not shot_name:
            raise Exception("shot_name est requis pour charger un shot Prism.")
        if not identifier:
            raise Exception("identifier est requis (ex: comfy, paintover, etc.).")

        cfg = PrismSaveImage._read_pipeline_config(project_data)
        folder_structure = cfg.get("folder_structure", {})
        globals_cfg = cfg.get("globals", {})
        project_dir = project_data["path"]

        sequences_tpl = folder_structure.get("sequences", {}).get("value")
        shots_tpl = folder_structure.get("shots", {}).get("value")
        renders2d_tpl = folder_structure.get("2drenders", {}).get("value")
        render_versions_tpl = folder_structure.get("renderVersions", {}).get("value")

        if not (sequences_tpl and shots_tpl and renders2d_tpl and render_versions_tpl):
            raise Exception(
                "Templates 'sequences', 'shots', '2drenders' ou 'renderVersions' "
                "manquants dans pipeline.json."
            )

        sequence_path = sequences_tpl.replace("@project_path@", project_dir)
        sequence_path = sequence_path.replace("@sequence@", sequence_name)

        entity_path = shots_tpl.replace("@sequence_path@", sequence_path)
        entity_path = entity_path.replace("@shot@", shot_name)

        render_root = renders2d_tpl.replace("@entity_path@", entity_path)
        render_root = render_root.replace("@identifier@", identifier)

        render_versions_dir = render_versions_tpl.replace("@render_path@", render_root)
        version_base_dir = render_versions_dir.replace("@version@", "")
        version_base_dir = version_base_dir.rstrip("/\\") or render_root
        version_padding = int(globals_cfg.get("versionPadding", 4))

        if not version or version.lower() == "latest":
            version_folder = PrismSaveImage._latest_version_in_dir(version_base_dir)
            if not version_folder:
                raise Exception(
                    f"Aucune version trouvée dans {render_root}. Vérifie le chemin."
                )
        else:
            version_folder = PrismSaveImage._normalize_version_string(
                version, version_padding
            )

        version_path = render_versions_dir.replace("@version@", version_folder)
        if not os.path.exists(version_path):
            raise Exception(f"Dossier de version introuvable : {version_path}")

        # Chercher le fichier vidéo
        video_files = [
            os.path.join(version_path, f)
            for f in os.listdir(version_path)
            if f.lower().endswith(f".{video_format}")
        ]

        if not video_files:
            raise Exception(
                f"Aucune vidéo .{video_format} trouvée dans {version_path}"
            )

        # Prendre la première vidéo trouvée
        video_path = video_files[0]
        print(f"✓ Chargement de la vidéo : {video_path}")

        # Charger la vidéo avec OpenCV
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(f"Impossible d'ouvrir la vidéo : {video_path}")

        # Récupérer les informations de la vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"✓ Vidéo : {total_frames} frames @ {fps} fps")

        frames = []
        frame_idx = 0
        loaded_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Appliquer le skip_frames
            if frame_idx % skip_frames == 0:
                # Convertir BGR (OpenCV) vers RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convertir en float32 normalisé [0, 1]
                frame_normalized = frame_rgb.astype(np.float32) / 255.0
                frames.append(frame_normalized)
                loaded_count += 1

                # Vérifier la limite max_frames
                if max_frames > 0 and loaded_count >= max_frames:
                    break

            frame_idx += 1

        cap.release()

        if not frames:
            raise Exception(f"Aucune frame chargée depuis {video_path}")

        print(f"✓ {len(frames)} frames chargées")

        # Convertir en tensor
        images_np = np.stack(frames, axis=0)
        images_tensor = torch.from_numpy(images_np)

        return (images_tensor, len(frames), fps)


class PrismScanProject:
    """Node utilitaire qui scanne un projet Prism et retourne les listes
    d'assets, séquences et shots disponibles sur disque.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_path": ("STRING", {"default": "D:/_RnD/Prism_RND/Prism_RnD"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("assets", "sequences", "shots")
    FUNCTION = "scan_project"
    CATEGORY = "Prism"

    def scan_project(self, project_path):
        project_data = {
            "name": os.path.basename(os.path.normpath(project_path)),
            "path": project_path,
        }

        cfg = PrismSaveImage._read_pipeline_config(project_data)
        folder_structure = cfg.get("folder_structure", {})

        project_dir = project_data["path"]

        # ------------------------------------------------------------------
        # Assets
        # ------------------------------------------------------------------
        assets_tpl = folder_structure.get("assets", {}).get("value")
        assets = []
        if assets_tpl:
            assets_path_tpl = assets_tpl.replace("@project_path@", project_dir)
            # On enlève la partie variable (@asset_path@) pour trouver la racine
            root_assets = assets_path_tpl.split("@asset_path@")[0].rstrip("/\\")

            if os.path.isdir(root_assets):
                # Liste uniquement les dossiers de premier niveau (les noms d'assets)
                for d in sorted(os.listdir(root_assets)):
                    full = os.path.join(root_assets, d)
                    if os.path.isdir(full):
                        assets.append(d)

        # ------------------------------------------------------------------
        # Séquences / shots
        # ------------------------------------------------------------------
        sequences_tpl = folder_structure.get("sequences", {}).get("value")
        shots_tpl = folder_structure.get("shots", {}).get("value")

        sequences = []
        shots = []

        if sequences_tpl and shots_tpl:
            sequences_path_tpl = sequences_tpl.replace("@project_path@", project_dir)
            sequences_root = sequences_path_tpl.split("@sequence@")[0].rstrip("/\\")

            if os.path.isdir(sequences_root):
                for seq_name in sorted(
                    d for d in os.listdir(sequences_root)
                    if os.path.isdir(os.path.join(sequences_root, d))
                ):
                    sequences.append(seq_name)

                    sequence_path = os.path.join(sequences_root, seq_name)
                    # Shots = sous-dossiers directs de la séquence
                    for shot_name in sorted(
                        d for d in os.listdir(sequence_path)
                        if os.path.isdir(os.path.join(sequence_path, d))
                    ):
                        shots.append(f"{seq_name}/{shot_name}")

        print(f"[PrismScanProject] Assets: {len(assets)}, "
              f"Sequences: {len(sequences)}, Shots: {len(shots)}")

        return (assets, sequences, shots)


class PrismSaveVideo:
    """Node pour sauvegarder des vidéos dans l'arborescence Prism."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "project_path": ("STRING", {"default": "D:/_RnD/Prism_RND/Prism_RnD"}),
                "entity_type": (["asset", "shot"],),
                "sequence_name": ("STRING", {"default": ""}),
                "shot_name": ("STRING", {"default": ""}),
                "version": ("STRING", {"default": "auto"}),
                "identifier": ("STRING", {"default": "comfy"}),
                "filename": ("STRING", {"default": "render"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            },
            "optional": {
                "format": (["mp4", "mov", "avi", "webm"], {"default": "mp4"}),
                "quality": (["high", "medium", "low"], {"default": "high"}),
                "comment": ("STRING", {"default": ""}),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "save_versioninfo": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "Prism"

    def save_video(self, images, project_path, entity_type, sequence_name,
                   shot_name, version, identifier, filename, fps,
                   format="mp4", quality="high", comment="",
                   prompt_text="", save_versioninfo=True,
                   prompt=None, extra_pnginfo=None):

        selected_sequence = (sequence_name or "").strip() or None
        selected_shot = (shot_name or "").strip() or None

        project_data = {
            "name": os.path.basename(os.path.normpath(project_path)),
            "path": project_path,
        }

        if entity_type == "shot":
            if not selected_sequence:
                raise Exception(
                    "Veuillez renseigner une séquence valide pour publier un shot."
                )
            if not selected_shot:
                raise Exception(
                    "Veuillez renseigner un nom de shot valide."
                )

        # Calcul du chemin de rendu 2D
        output_dir = self._build_2d_render_path(
            project_data=project_data,
            entity_type=entity_type,
            sequence_name=selected_sequence,
            shot_name=selected_shot,
            version=version,
            identifier=identifier,
        )

        # Conversion des images en vidéo
        video_path = self._images_to_video(
            images=images,
            target_dir=output_dir,
            filename=filename,
            fps=fps,
            format=format,
            quality=quality,
        )

        # Sauvegarde du versioninfo.json si demandé
        if save_versioninfo:
            version_folder = os.path.basename(output_dir)

            try:
                username = getpass.getuser()
            except Exception:
                username = os.getenv('USERNAME', 'unknown')

            project_name = os.path.basename(os.path.normpath(project_path))
            date_str = datetime.now().strftime("%d.%m.%y %H:%M:%S")

            # Construire la hiérarchie
            hierarchy = ""
            if entity_type == "shot" and selected_sequence and selected_shot:
                hierarchy = f"{selected_sequence}/{selected_shot}"

            try:
                info_path = os.path.join(output_dir, "versioninfo.json")

                # Compter le nombre de frames
                frame_count = images.shape[0] if isinstance(images, torch.Tensor) else len(images)

                versioninfo_data = {
                    "hierarchy": hierarchy,
                    "itemType": entity_type,
                    "sequence": selected_sequence or "",
                    "shot": selected_shot or "",
                    "type": entity_type,
                    "identifier": identifier,
                    "user": username.lower(),
                    "version": version_folder,
                    "comment": comment or "",
                    "extension": f".{format}",
                    "mediaType": "2drenders",
                    "username": username,
                    "date": date_str,
                    "fps": fps,
                    "frame_count": frame_count,
                    "dependencies": [],
                    "externalFiles": []
                }

                if prompt_text and prompt_text.strip():
                    versioninfo_data["prompt"] = prompt_text.strip()

                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(versioninfo_data, f, indent=4, ensure_ascii=False)
                print(f"✓ versioninfo.json sauvegardé : {info_path}")
                print(f"  Commentaire : '{comment}'")
                if prompt_text and prompt_text.strip():
                    print(f"  Prompt inclus dans le JSON")
            except Exception:
                import traceback
                print("⚠ Impossible d'écrire versioninfo.json :")
                print(traceback.format_exc())

        print(f"✓ Vidéo sauvegardée dans Prism : {video_path}")

        # Créer les infos pour le preview UI
        results = [{
            "filename": os.path.basename(video_path),
            "subfolder": "",
            "type": "output"
        }]

        return {"ui": {"images": results}, "result": (images,)}

    def _build_2d_render_path(self, project_data, entity_type, sequence_name,
                             shot_name, version, identifier):
        """Utilise la même logique que PrismSaveImage pour construire le chemin."""
        return PrismSaveImage()._build_2d_render_path(
            project_data=project_data,
            entity_type=entity_type,
            sequence_name=sequence_name,
            shot_name=shot_name,
            version=version,
            identifier=identifier,
        )

    @staticmethod
    def _images_to_video(images, target_dir, filename, fps, format, quality):
        """Convertit un batch d'images en vidéo en utilisant ffmpeg."""
        if isinstance(images, torch.Tensor):
            imgs = images.detach().cpu().numpy()
        else:
            imgs = np.array(images)

        if imgs.ndim == 3:
            imgs = imgs[None, ...]

        # Créer un dossier temporaire pour les frames
        temp_dir = tempfile.mkdtemp()

        try:
            # Sauvegarder chaque frame en PNG
            for i, img in enumerate(imgs):
                img_np = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                if img_np.shape[-1] == 1:
                    img_np = img_np[..., 0]
                pil_img = Image.fromarray(img_np)
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                pil_img.save(frame_path)

            # Définir la qualité vidéo
            quality_settings = {
                "high": "18",    # CRF pour haute qualité
                "medium": "23",  # CRF pour qualité moyenne
                "low": "28"      # CRF pour basse qualité
            }
            crf = quality_settings.get(quality, "23")

            # Nom du fichier de sortie
            output_filename = f"{filename}.{format}"
            output_path = os.path.join(target_dir, output_filename)

            # Construire la commande ffmpeg
            input_pattern = os.path.join(temp_dir, "frame_%06d.png")

            if format == "mp4":
                codec = "libx264"
                extra_args = ["-pix_fmt", "yuv420p"]
            elif format == "mov":
                codec = "libx264"
                extra_args = ["-pix_fmt", "yuv420p"]
            elif format == "webm":
                codec = "libvpx-vp9"
                extra_args = []
            elif format == "avi":
                codec = "libx264"
                extra_args = []
            else:
                codec = "libx264"
                extra_args = []

            # Ajouter un filtre pour assurer des dimensions paires (requis pour H264)
            vf_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

            ffmpeg_cmd = [
                FFMPEG_PATH,
                "-y",  # Overwrite output file
                "-framerate", str(fps),
                "-i", input_pattern,
                "-vf", vf_filter,  # Filtre pour dimensions paires
                "-c:v", codec,
                "-crf", crf,
                *extra_args,
                output_path
            ]

            # Exécuter ffmpeg
            print(f"✓ Conversion en vidéo ({format}, {quality}, {fps}fps)...")
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                print(f"✗ Erreur ffmpeg: {result.stderr}")
                raise Exception(f"Échec de la conversion vidéo avec ffmpeg: {result.stderr}")

            return output_path

        finally:
            # Nettoyer le dossier temporaire
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


# Node registration
NODE_CLASS_MAPPINGS = {
    "PrismSaveImage": PrismSaveImage,
    "PrismLoadImage": PrismLoadImage,
    "PrismScanProject": PrismScanProject,
    "PrismSaveVideo": PrismSaveVideo,
    "PrismLoadVideo": PrismLoadVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrismSaveImage": "Save Image (Prism Pipeline)",
    "PrismLoadImage": "Load Image (Prism Pipeline)",
    "PrismScanProject": "Scan Project (Prism Pipeline)",
    "PrismSaveVideo": "Save Video (Prism Pipeline)",
    "PrismLoadVideo": "Load Video (Prism Pipeline)",
}