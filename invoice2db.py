"""
Script that given an invoice number identifies such invoice from multiple PDF files extracts its details.
- Runs as: python3 invoice2db.py
- Extracts invoice ID from factura_a_header, then searches among two other files for matching invoices and stores contents in db.
- Assumes the invoice number is indeed present in factura_a_header
- Hardcoded files path to ./archivos/
"""

import pdfplumber
import re
import json
import os
import io
import fitz
import pytesseract
import sqlite3
from PIL import Image
import logging
logging.basicConfig(level=logging.CRITICAL)

def create_db(json, invoice_number, filename):

    conn = sqlite3.connect('./invoice_details.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            invoice_number TEXT,
            details TEXT  -- Storing JSON as text
        )
    ''')

    cursor.execute('''
        INSERT INTO invoices (invoice_number, filename, details)
        VALUES (?, ?, ?)
    ''', (invoice_number, filename, json))  # Convert dictionary to JSON string

    # Commit and close the connection
    conn.commit()
    conn.close()


def extract_invoice_number_from_a(pdf):
    """Extract invoice number from pdf by regexp search."""
    with pdfplumber.open(pdf) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            match = re.search(r'Numero Factura o Ticker (\w+)', text)
            if match:
                logging.info(f"Found invoice number: {match.group(1)}")
                return match.group(1)
    return None

def extract_text_from_pdfs(pdf_file, invoice_number):
    """Extracts text from PDF files, in case it comes from an image parse it with OCR."""
    
    # Assume pdf
    with fitz.open(pdf_file) as pdf:
        full_text = ""
        for page in pdf:
            full_text += page.get_text("text")
        
    if(len(full_text) > 0):
        logging.debug("pdf case", pdf_file, ":", len(full_text))

    # If extraction comes out empty, we assume it's an image and process it with OCR
    else: 
        doc = fitz.open(pdf_file)
    
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            pix = page.get_pixmap(dpi=198) # hardcoded dpi, tuned for specifically for factura_c_detalle.pdf
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img = img.convert('L')  # grayscale
            
            full_text = pytesseract.image_to_string(img)        
            logging.debug(full_text)

    return full_text

def text_to_json(text, invoice_number, pdf_file):
    """Converts text to JSON format."""
    purchase_details = []
    details_match = re.findall(r'(\d{3})\s+(\d{2}-\d{2}-\d{4})\s+(.+)', text)

    for detail in details_match:
        codigo, fecha, producto = detail
        purchase_details.append({
            "codigo": int(codigo),
            "fecha": fecha,
            "producto": producto
        })

    output = {
        "Documento_detalle": pdf_file,
        "NumeroFactura": invoice_number,
        "detalles": purchase_details
    }
    json_output = json.dumps(output, indent=4, ensure_ascii=False)
    logging.debug(f"JSON output: {json_output}")
    return(json_output)

def process_invoices(directory):
    """
        Main function to run the extraction process and store in db as JSON
        - Extracts invoice number from factura_a_header
        - Extracts text from factura_b_detalle and factura_c_detalle
        - For each checks if it contains the invoice number and in that case stores the details in a JSON
        - Also stores the details in a db, as asked in the exercise
    """
    factura_a = os.path.join(directory, 'factura_a_header.pdf')
    factura_b = os.path.join(directory, 'factura_b_detalle.pdf')
    factura_c = os.path.join(directory, 'factura_c_detalle.pdf')

    invoice_number = extract_invoice_number_from_a(factura_a)
    logging.debug(f"el ticker que queremos es: {invoice_number}")

    if not invoice_number:
        logging.debug(f"No encontré el ticker en la factura A.")
        return(0)

    for pdf in [factura_b, factura_c]:
        output_text = extract_text_from_pdfs(pdf, invoice_number)
        logging.debug(f"output_text: {output_text}")

        if invoice_number in output_text:
            logging.info(f"Encontré el ticker en la factura {pdf}.")
            output_json = text_to_json(output_text, invoice_number, pdf)
            create_db(output_json, invoice_number, pdf)
            with open(f'{invoice_number}.json', 'w') as json_file:
                json_file.write(output_json + '\n')
        else:
            logging.debug(f"No encontré el ticker en la factura B o C.")            
    
if __name__ == "__main__":
    directory = "./archivos/"
    process_invoices(directory)
    
