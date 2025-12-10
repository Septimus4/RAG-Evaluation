# utils/data_loader.py
import os
import requests
import zipfile
import io
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
import numpy as np
from tqdm import tqdm # Ajout de tqdm

# --- OCR: chargement différé pour éviter les erreurs à l'import ---
fitz = None
Image = None
easyocr = None
reader = None
tesseract = None

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Fonctions d'extraction de texte ---

def extract_text_from_pdf_with_ocr(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier PDF en utilisant l'OCR (EasyOCR)."""
    global fitz, Image, easyocr, reader
    # Chargement paresseux des dépendances OCR
    if fitz is None or Image is None or easyocr is None or reader is None:
        try:
            import fitz as _fitz  # PyMuPDF
            from PIL import Image as _Image
            import easyocr as _easyocr
            fitz, Image, easyocr = _fitz, _Image, _easyocr
            logging.info("Initialisation du lecteur EasyOCR (lazy)...")
            # Désactive le GPU par défaut pour la compatibilité en conteneur
            reader = easyocr.Reader(['en', 'fr'], gpu=False, download_enabled=True)
            logging.info("Lecteur EasyOCR initialisé.")
        except ImportError as e:
            logging.warning(f"Modules OCR (PyMuPDF, Pillow, easyocr) non installés: {e}. L'OCR n'est pas disponible.")
            return None
        except Exception as e:
            logging.error(f"Erreur inattendue lors de l'initialisation OCR: {e}")
            return None

    text_content = []
    try:
        doc = fitz.open(file_path)
        # Utiliser tqdm pour la barre de progression
        for page_num in tqdm(range(len(doc)), desc=f"OCR de {os.path.basename(file_path)}"):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Augmenter la résolution pour l'OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            try:
                img_np = np.array(img)
                results = reader.readtext(img_np)
                page_text = "\n".join([res[1] for res in results])
                text_content.append(page_text)
                # logging.info(f"OCR effectuée sur la page {page_num + 1} de {file_path} avec EasyOCR") # Commenté pour éviter le spam de logs avec tqdm
            except Exception as ocr_e:
                logging.error(f"Erreur lors de l'OCR de la page {page_num + 1} de {file_path} avec EasyOCR: {ocr_e}")
                continue

        doc.close()
        full_text = "\n".join(text_content).strip()
        if full_text:
            logging.info(f"Texte extrait via OCR de PDF: {file_path} ({len(full_text)} caractères)")
            return full_text
        else:
            logging.warning(f"Aucun texte significatif extrait via OCR de {file_path}.")
            return None
    except Exception as e:
        logging.error(f"Erreur lors de l'ouverture ou du traitement OCR du PDF {file_path}: {e}")
        return None

def extract_text_from_pdf_with_pymupdf(file_path: str) -> Optional[str]:
    """Extrait le texte via PyMuPDF (sans OCR), utile sur certains PDFs.
    Retourne None si PyMuPDF n'est pas disponible ou si aucun texte utile n'est trouvé.
    """
    try:
        import fitz  # lazy import
    except Exception as e:
        logging.debug(f"PyMuPDF indisponible pour {file_path}: {e}")
        return None

def extract_text_from_pdf_with_tesseract(file_path: str) -> Optional[str]:
    """OCR via Tesseract (pytesseract) sur rendu image PyMuPDF, CPU-only.
    Nécessite le binaire système `tesseract-ocr` et le paquet Python `pytesseract`.
    """
    global fitz, Image, tesseract
    try:
        if fitz is None:
            import fitz as _fitz
            fitz = _fitz
        if Image is None:
            from PIL import Image as _Image
            Image = _Image
        if tesseract is None:
            import pytesseract as _pytesseract
            tesseract = _pytesseract
    except Exception as e:
        logging.warning(f"Dépendances Tesseract non disponibles: {e}")
        return None
    try:
        doc = fitz.open(file_path)
        texts = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            try:
                txt = tesseract.image_to_string(img, lang="eng+fra")
            except Exception:
                txt = tesseract.image_to_string(img)
            if txt and txt.strip():
                texts.append(txt)
        doc.close()
        full = "\n".join(texts).strip()
        if full:
            logging.info(f"Texte extrait via Tesseract pour {file_path} ({len(full)} caractères)")
            return full
        logging.warning(f"Tesseract n'a pas produit de texte significatif pour {file_path}.")
        return None
    except Exception as e:
        logging.error(f"Erreur OCR Tesseract pour {file_path}: {e}")
        return None
    try:
        doc = fitz.open(file_path)
        texts = []
        for page in doc:
            try:
                t = page.get_text("text") or ""
            except Exception:
                t = ""
            if t.strip():
                texts.append(t)
        doc.close()
        full = "\n".join(texts).strip()
        return full if full else None
    except Exception as e:
        logging.debug(f"PyMuPDF extraction a échoué pour {file_path}: {e}")
        return None

def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier PDF, avec fallback OCR si peu de texte est trouvé."""
    try:
        from PyPDF2 import PdfReader
        _reader = PdfReader(file_path)
        parts = []
        for page in _reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                parts.append(t + "\n")
        text = "".join(parts)
        
        if len(text.strip()) < 100: # Si très peu de texte est extrait, tenter PyMuPDF puis OCR
            logging.info(f"Peu de texte trouvé dans {file_path} via extraction standard ({len(text.strip())} caractères). Tentative PyMuPDF...")
            mu_text = extract_text_from_pdf_with_pymupdf(file_path)
            if mu_text and len(mu_text.strip()) >= 100:
                logging.info(f"Texte extrait via PyMuPDF pour {file_path} ({len(mu_text)} caractères)")
                return mu_text
            logging.info("Tentative d'OCR EasyOCR en dernier recours...")
            ocr_text = extract_text_from_pdf_with_ocr(file_path)
            if ocr_text:
                return ocr_text
            else:
                logging.info("Tentative OCR Tesseract en dernier recours...")
                tess_text = extract_text_from_pdf_with_tesseract(file_path)
                if tess_text:
                    return tess_text
                logging.warning(f"L'OCR n'a pas produit de texte significatif pour {file_path}.")
                return text # Retourne le peu de texte trouvé ou vide
        
        logging.info(f"Texte extrait de PDF: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction PDF {file_path}: {e}. Tentative PyMuPDF puis OCR...")
        mu_text = extract_text_from_pdf_with_pymupdf(file_path)
        if mu_text:
            return mu_text
        ocr_text = extract_text_from_pdf_with_ocr(file_path)
        if ocr_text:
            return ocr_text
        tess_text = extract_text_from_pdf_with_tesseract(file_path)
        if tess_text:
            return tess_text
        logging.warning(f"Impossible d'extraire du texte pour {file_path}.")
        return None


def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier Word DOCX."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        logging.info(f"Texte extrait de DOCX: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction DOCX {file_path}: {e}")
        return None

def extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier texte brut."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        logging.info(f"Texte extrait de TXT: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction TXT {file_path}: {e}")
        return None

def extract_text_from_csv(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier CSV (convertit en string)."""
    try:
        import pandas as pd
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1') # Essayer un autre encodage courant
        except Exception as read_e:
             logging.warning(f"Erreur lecture CSV {file_path}: {read_e}. Tentative avec séparateur ';'")
             try:
                 df = pd.read_csv(file_path, sep=';')
             except UnicodeDecodeError:
                  df = pd.read_csv(file_path, sep=';', encoding='latin1')
             except Exception as read_e2:
                   logging.error(f"Impossible de lire le CSV {file_path}: {read_e2}")
                   return None

        text = df.to_string()
        logging.info(f"Texte extrait de CSV: {file_path} ({len(text)} caractères)")
        return text
    except ImportError:
        logging.warning("Pandas non installé. Impossible de lire les fichiers CSV.")
        return None
    except Exception as e:
        logging.error(f"Erreur extraction CSV {file_path}: {e}")
        return None

def extract_text_from_excel(file_path: str) -> Optional[Union[str, Dict[str, str]]]:
    """Extrait le texte de chaque feuille d'un fichier Excel."""
    try:
        import pandas as pd
        # Lire toutes les feuilles dans un dictionnaire de DataFrames
        excel_file = pd.ExcelFile(file_path)
        sheets_data = {}
        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            sheets_data[sheet_name] = df.to_string()
        
        logging.info(f"Texte extrait de {len(sheets_data)} feuille(s) dans Excel: {file_path}")
        # Si une seule feuille, retourne directement le texte pour la compatibilité
        if len(sheets_data) == 1:
            return list(sheets_data.values())[0]
        return sheets_data
    except ImportError:
        logging.warning("Pandas ou openpyxl non installé. Impossible de lire les fichiers Excel.")
        return None
    except Exception as e:
        logging.error(f"Erreur extraction Excel {file_path}: {e}")
        return None

# --- Fonctions de chargement ---

def download_and_extract_zip(url: str, output_dir: str) -> bool:
    """Télécharge un fichier ZIP depuis une URL et l'extrait."""
    if not url:
        logging.warning("Aucune URL fournie pour le téléchargement.")
        return False
    try:
        logging.info(f"Téléchargement des données depuis {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            logging.info(f"Extraction du contenu dans {output_dir}...")
            z.extractall(output_dir)
        logging.info("Téléchargement et extraction terminés.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de téléchargement: {e}")
        return False
    except zipfile.BadZipFile:
        logging.error("Le fichier téléchargé n'est pas un ZIP valide.")
        return False
    except Exception as e:
        logging.error(f"Erreur inattendue lors du téléchargement/extraction: {e}")
        return False

def load_and_parse_files(input_dir: str) -> List[Dict[str, any]]:
    """
    Charge et parse récursivement les fichiers d'un répertoire.
    Retourne une liste de dictionnaires, chacun représentant un document.
    """
    documents = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logging.error(f"Le répertoire d'entrée '{input_dir}' n'existe pas.")
        return []

    logging.info(f"Parcours du répertoire source: {input_dir}")
    for file_path in input_path.rglob("*.*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(input_path)
            source_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
            ext = file_path.suffix.lower()
            
            logging.debug(f"Traitement du fichier: {relative_path} (Dossier source: {source_folder})")

            extracted_content = None
            if ext == ".pdf":
                extracted_content = extract_text_from_pdf(str(file_path))
            elif ext == ".docx":
                extracted_content = extract_text_from_docx(str(file_path))
            elif ext == ".txt":
                extracted_content = extract_text_from_txt(str(file_path))
            elif ext == ".csv":
                extracted_content = extract_text_from_csv(str(file_path))
            elif ext in [".xlsx", ".xls"]:
                extracted_content = extract_text_from_excel(str(file_path))
            # Suppression de la gestion des fichiers HTML
            else:
                logging.warning(f"Type de fichier non supporté ignoré: {relative_path}")
                continue

            if not extracted_content:
                logging.warning(f"Aucun contenu n'a pu être extrait de {relative_path}")
                continue
            
            # Si c'est un dictionnaire (plusieurs feuilles Excel), créer un doc par feuille
            if isinstance(extracted_content, dict):
                for sheet_name, text in extracted_content.items():
                    documents.append({
                        "page_content": text,
                        "metadata": {
                            "source": f"{str(relative_path)} (Feuille: {sheet_name})",
                            "filename": file_path.name,
                            "sheet": sheet_name,
                            "category": source_folder,
                            "full_path": str(file_path.resolve())
                        }
                    })
            else: # Pour tous les autres types de fichiers
                 documents.append({
                    "page_content": extracted_content,
                    "metadata": {
                        "source": str(relative_path),
                        "filename": file_path.name,
                        "category": source_folder,
                        "full_path": str(file_path.resolve())
                    }
                })

    logging.info(f"{len(documents)} documents chargés et parsés.")
    return documents