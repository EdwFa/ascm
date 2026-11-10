#!/usr/bin/env python3
"""PDF -> cleaned text for data/raw_en/<slug>.txt (needs poppler's pdftotext, or the pypdf package as a fallback).
Put the PDFs in data/pdf/ and list them in data/manifest.csv (column 'pdf'). The PDFs and the extracted text are NOT
committed (publisher copyright); only the manifest and this script are."""
import argparse, csv, os, shutil, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ascm.clean import clean_pdf_text

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', default='data/manifest.csv'); ap.add_argument('--pdf-dir', default='data/pdf')
ap.add_argument('--out', default='data/raw_en')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for row in csv.DictReader(open(a.manifest, encoding='utf-8')):
    if not row.get('pdf'):
        continue
    src = os.path.join(a.pdf_dir, row['pdf'])
    if not os.path.exists(src):
        print('MISSING', src); continue
    if shutil.which('pdftotext'):
        raw = subprocess.run(['pdftotext', '-enc', 'UTF-8', src, '-'], capture_output=True, text=True, check=True).stdout
    else:
        from pypdf import PdfReader
        raw = '\n\n'.join(p.extract_text() or '' for p in PdfReader(src).pages)
    txt = clean_pdf_text(raw)
    open(os.path.join(a.out, row['slug'] + '.txt'), 'w', encoding='utf-8').write(txt)
    print(f"{row['slug']:16} {len(txt.split()):6} words")
