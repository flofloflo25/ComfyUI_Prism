# ComfyUI_prism

Nodes ComfyUI pour l'intégration avec le pipeline Prism.

## Installation

### Installation automatique (Recommandée)

1. **Cloner ou copier ce dossier** dans le répertoire `custom_nodes` de ComfyUI :
   ```bash
   cd ComfyUI/custom_nodes
   git clone <votre-repo> ComfyUI_prism
   # ou copier le dossier manuellement
   ```

2. **Installer les dépendances** :
   ```bash
   cd ComfyUI_prism
   pip install -r requirements.txt
   ```

3. **Télécharger et installer ffmpeg automatiquement** :
   ```bash
   python download_ffmpeg.py
   ```
   Ce script télécharge automatiquement ffmpeg et l'installe dans le dossier `bin/`.

4. **Configurer les chemins** :
   - Copier `config.example.json` vers `config.json`
   - Modifier `config.json` selon votre configuration :
     ```json
     {
       "prism_root": "C:\\Program Files\\Prism2",
       "ffmpeg_path": "bin\\ffmpeg.exe",
       "prism_projects_file": "prism_projects.json"
     }
     ```

5. **Redémarrer ComfyUI**

### Installation manuelle

Si vous préférez installer ffmpeg manuellement ou utiliser une installation existante :

1. **Télécharger ffmpeg** :
   - Windows : https://www.gyan.dev/ffmpeg/builds/
   - Linux : `sudo apt install ffmpeg` ou télécharger depuis https://johnvansickle.com/ffmpeg/
   - macOS : `brew install ffmpeg`

2. **Configurer le chemin** dans `config.json` :
   - Chemin relatif : `"ffmpeg_path": "bin\\ffmpeg.exe"` (si dans le dossier bin/)
   - Chemin absolu : `"ffmpeg_path": "C:\\chemin\\vers\\ffmpeg.exe"`
   - Utiliser ffmpeg Prism : `"ffmpeg_path": "C:\\Users\\USERNAME\\AppData\\Local\\Temp\\Prism\\Update\\extract\\Tools\\FFmpeg\\bin\\ffmpeg.exe"`

## Nodes disponibles

### 1. **PrismSaveImage**
Sauvegarde des images dans l'arborescence Prism (pour shots et assets).

**Paramètres :**
- `project_path` : Chemin du projet Prism
- `entity_type` : Type (asset ou shot)
- `sequence_name` : Nom de la séquence (requis pour shots)
- `shot_name` : Nom du shot (requis pour shots)
- `version` : Version (auto ou manuelle)
- `identifier` : Nom du dossier de rendu
- `filename_prefix` : Préfixe des fichiers
- `format` : Format d'image (png, jpeg, exr)
- `colorspace` : Colorspace (Linear, sRGB, ACEScg, Rec709)

### 2. **PrismSaveAsset**
Sauvegarde des images en tant qu'asset Prism (workflow simplifié pour assets).

**Paramètres :**
- `project_path` : Chemin du projet Prism
- `asset_name` : Nom de l'asset
- `version` : Version (auto ou manuelle)
- `identifier` : Nom du dossier de rendu
- `filename_prefix` : Préfixe des fichiers
- `format` : Format d'image (png, jpeg, exr)
- `colorspace` : Colorspace (Linear, sRGB, ACEScg, Rec709)

### 3. **PrismLoadImage**
Charge des images (shots) depuis l'arborescence Prism.

**Paramètres :**
- `project_path` : Chemin du projet Prism
- `sequence_name` : Nom de la séquence
- `shot_name` : Nom du shot
- `identifier` : Identifiant du rendu
- `version` : Version à charger (latest par défaut)
- `limit` : Limiter le nombre d'images
- `show_preview` : Afficher preview des images chargées

**Note :** Bouton "Update Preview" disponible pour recharger les images.

### 4. **PrismLoadAsset**
Charge des images (assets) depuis l'arborescence Prism.

**Paramètres :**
- `project_path` : Chemin du projet Prism
- `asset_name` : Nom de l'asset
- `identifier` : Identifiant du rendu
- `version` : Version à charger (latest par défaut)
- `limit` : Limiter le nombre d'images
- `show_preview` : Afficher preview des images chargées

**Note :** Bouton "Update Preview" disponible pour recharger les images.

### 5. **PrismSaveVideo**
Sauvegarde des séquences d'images en vidéo dans Prism.

**Paramètres :**
- Mêmes paramètres que PrismSaveImage
- `filename` : Nom du fichier vidéo
- `fps` : Frame rate (1-120)
- `format` : Format vidéo (mp4, mov, avi, webm)
- `quality` : Qualité (high, medium, low)

**Note :** Nécessite ffmpeg configuré dans config.json.

### 6. **PrismLoadVideo**
Charge des vidéos depuis l'arborescence Prism et les convertit en séquence d'images.

**Paramètres :**
- Mêmes paramètres que PrismLoadImage
- `video_format` : Format de la vidéo (mp4, mov, avi, webm)
- `max_frames` : Limiter le nombre de frames (0 = toutes)
- `skip_frames` : Charger 1 frame sur N

**Sorties :**
- `images` : Tensor IMAGE
- `frame_count` : Nombre de frames
- `fps` : Frame rate

### 7. **PrismScanProject**
Scanne un projet Prism et liste les assets, séquences et shots disponibles.

## Configuration requise

- ComfyUI
- Python 3.9+
- OpenCV (pour le support vidéo)
- FFmpeg (pour PrismSaveVideo)
- Prism Pipeline Manager

## Structure du projet Prism

Le plugin s'attend à une structure Prism standard avec un fichier `00_Pipeline/pipeline.json` définissant l'arborescence du projet.

## Format versioninfo.json

Les nodes génèrent automatiquement un fichier `versioninfo.json` compatible avec Prism contenant :
- Métadonnées du shot/asset
- Informations de version
- Utilisateur et date
- Commentaires et prompts
- Hiérarchie du projet

## Configuration

Le plugin utilise un fichier `config.json` pour stocker les chemins :

```json
{
    "prism_root": "C:\\Program Files\\Prism2",
    "ffmpeg_path": "bin\\ffmpeg.exe",
    "prism_projects_file": "prism_projects.json"
}
```

### Options de configuration

- **prism_root** : Chemin vers l'installation de Prism
- **ffmpeg_path** : Chemin vers ffmpeg (relatif ou absolu)
- **prism_projects_file** : Fichier contenant la liste des projets

### Chemins relatifs vs absolus

Les chemins peuvent être :
- **Relatifs** au dossier du plugin : `"bin\\ffmpeg.exe"` → `ComfyUI_prism/bin/ffmpeg.exe`
- **Absolus** : `"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"`

## Dépannage

### Erreur "config.json non trouvé"
- Copier `config.example.json` vers `config.json`
- Ajuster les chemins selon votre installation

### Erreur "ffmpeg not found"
- Vérifier que le chemin dans `config.json` est correct
- Exécuter `python download_ffmpeg.py` pour installer ffmpeg automatiquement
- Ou télécharger manuellement et mettre à jour `config.json`

### Erreur "width not divisible by 2"
- ✓ Corrigé automatiquement : le node ajuste les dimensions pour la compatibilité H264

### OpenCV non trouvé
```bash
pip install opencv-python
```

### Logs de configuration
Au démarrage de ComfyUI, vérifiez les logs pour voir la configuration chargée :
```
[ComfyUI_prism] Configuration chargée:
[ComfyUI_prism]   - Prism Root: C:\Program Files\Prism2
[ComfyUI_prism]   - FFmpeg: C:\ComfyUI\custom_nodes\ComfyUI_prism\bin\ffmpeg.exe
[ComfyUI_prism]   - Projects File: C:\ComfyUI\custom_nodes\ComfyUI_prism\prism_projects.json
```

## License

À définir selon votre projet.
