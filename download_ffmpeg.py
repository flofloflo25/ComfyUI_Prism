#!/usr/bin/env python3
"""
Script pour télécharger et installer ffmpeg dans le dossier bin/
Supporte Windows, Linux et macOS
"""

import os
import sys
import platform
import urllib.request
import zipfile
import tarfile
import shutil

def download_file(url, destination):
    """Télécharge un fichier avec barre de progression."""
    print(f"Téléchargement depuis {url}...")

    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rProgression: {percent}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, destination, reporthook)
    print("\n✓ Téléchargement terminé")

def extract_archive(archive_path, extract_to):
    """Extrait une archive zip ou tar.gz."""
    print(f"Extraction de {archive_path}...")

    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tar.xz'):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)

    print("✓ Extraction terminée")

def download_ffmpeg():
    """Télécharge et installe ffmpeg selon le système d'exploitation."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, "bin")
    temp_dir = os.path.join(script_dir, "temp_ffmpeg")

    # Créer les dossiers
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    system = platform.system()

    try:
        if system == "Windows":
            print("Système détecté: Windows")
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            archive_path = os.path.join(temp_dir, "ffmpeg.zip")

            download_file(url, archive_path)
            extract_archive(archive_path, temp_dir)

            # Trouver le dossier extrait
            extracted_folders = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
            if extracted_folders:
                ffmpeg_folder = os.path.join(temp_dir, extracted_folders[0], "bin")
                ffmpeg_exe = os.path.join(ffmpeg_folder, "ffmpeg.exe")

                if os.path.exists(ffmpeg_exe):
                    # Copier ffmpeg.exe vers bin/
                    dest = os.path.join(bin_dir, "ffmpeg.exe")
                    shutil.copy2(ffmpeg_exe, dest)
                    print(f"✓ ffmpeg.exe installé dans: {dest}")
                else:
                    print("✗ ffmpeg.exe non trouvé dans l'archive")

        elif system == "Linux":
            print("Système détecté: Linux")
            url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
            archive_path = os.path.join(temp_dir, "ffmpeg.tar.xz")

            download_file(url, archive_path)
            extract_archive(archive_path, temp_dir)

            # Trouver ffmpeg
            for root, dirs, files in os.walk(temp_dir):
                if "ffmpeg" in files:
                    ffmpeg_file = os.path.join(root, "ffmpeg")
                    dest = os.path.join(bin_dir, "ffmpeg")
                    shutil.copy2(ffmpeg_file, dest)
                    os.chmod(dest, 0o755)  # Rendre exécutable
                    print(f"✓ ffmpeg installé dans: {dest}")
                    break

        elif system == "Darwin":  # macOS
            print("Système détecté: macOS")
            print("Pour macOS, veuillez installer ffmpeg avec Homebrew:")
            print("  brew install ffmpeg")
            print("Puis configurez le chemin dans config.json")
            return

        else:
            print(f"✗ Système non supporté: {system}")
            return

        print("\n✓ Installation terminée!")
        print(f"ffmpeg est maintenant disponible dans: {bin_dir}")
        print("\nLe fichier config.json est déjà configuré pour utiliser bin/ffmpeg.exe")

    finally:
        # Nettoyer le dossier temporaire
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("✓ Fichiers temporaires nettoyés")

if __name__ == "__main__":
    print("=" * 60)
    print("Installation de ffmpeg pour ComfyUI_prism")
    print("=" * 60)
    print()

    download_ffmpeg()

    print()
    print("=" * 60)
    print("Installation terminée!")
    print("=" * 60)
