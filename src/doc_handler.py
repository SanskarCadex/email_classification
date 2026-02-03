"""
doc_handler.py - Extract text from email attachments for classification.

This module extracts text from ALL types of attachments (PDFs, images, text files, etc.)
and makes it available to the classification model. The extracted text is appended to
the email body before being sent to the model API for classification.

Supports:
- PDF files: Text extraction using PyPDF2/pdfplumber
- Image files: OCR using Tesseract (PNG, JPG, JPEG, GIF, BMP, TIFF)
- Text files: Direct text extraction (.txt, .csv, .log)
- Other formats: Attempts OCR for image-like files

The extracted text helps the model better classify emails, especially for:
- claims_paid_with_proof vs claims_paid_no_proof (when payment proof is in attachments)
- Better understanding of email context from attached documents
"""

import os
import io
import base64
import tempfile
import httpx
from typing import Optional, Dict, List, Tuple
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Initialize PDF library flags
PDF_AVAILABLE = False
USE_PDFPLUMBER = False

# Try PyPDF2 first
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    pass

# Prefer pdfplumber if available (more accurate)
try:
    import pdfplumber
    PDF_AVAILABLE = True
    USE_PDFPLUMBER = True
except ImportError:
    pass

from src.log_config import logger


class DocumentReader:
    """Extract text from various document formats."""
    
    def __init__(self):
        """Initialize document reader with available libraries."""
        self.tesseract_available = TESSERACT_AVAILABLE
        self.pdf_available = PDF_AVAILABLE
        # Check if pdfplumber is available (preferred over PyPDF2)
        try:
            import pdfplumber
            self.use_pdfplumber = True
        except ImportError:
            self.use_pdfplumber = False
        
        if not self.tesseract_available:
            logger.warning("Tesseract OCR not available. Image OCR will be disabled.")
        if not self.pdf_available:
            logger.warning("PDF libraries not available. PDF extraction will be disabled.")
    
    def extract_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes.
        Tries multiple methods: pdfplumber -> PyPDF2 -> OCR fallback
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Extracted text string
        """
        if not self.pdf_available:
            logger.warning("PDF extraction not available - install PyPDF2 or pdfplumber")
            return ""
        
        # Track the best non-empty extraction across all methods
        best_text = ""
        
        # Method 1: Try pdfplumber first (most accurate)
        if self.use_pdfplumber:
            try:
                candidate_text = self._extract_with_pdfplumber(pdf_bytes)
                if candidate_text and len(candidate_text.strip()) > len(best_text.strip()):
                    best_text = candidate_text
                if best_text and len(best_text.strip()) > 50:
                    logger.info(f"Successfully extracted {len(best_text)} characters using pdfplumber")
                    return best_text
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Method 2: Try PyPDF2
        try:
            candidate_text = self._extract_with_pypdf2(pdf_bytes)
            if candidate_text and len(candidate_text.strip()) > len(best_text.strip()):
                best_text = candidate_text
            if best_text and len(best_text.strip()) > 50:
                logger.info(f"Successfully extracted {len(best_text)} characters using PyPDF2")
                return best_text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Method 3: OCR fallback - convert PDF pages to images and use OCR
        if self.tesseract_available and (not best_text or len(best_text.strip()) < 50):
            try:
                logger.info("Text extraction failed or minimal - trying OCR fallback on PDF pages")
                candidate_text = self._extract_pdf_with_ocr(pdf_bytes)
                if candidate_text and len(candidate_text.strip()) > len(best_text.strip()):
                    best_text = candidate_text
                if best_text and len(best_text.strip()) > 50:
                    logger.info(f"Successfully extracted {len(best_text)} characters using OCR")
                    return best_text
            except Exception as e:
                logger.warning(f"OCR fallback failed: {e}")
        
        # Return whatever best text we got (even if minimal)
        if best_text:
            logger.warning(f"Only extracted {len(best_text)} characters from PDF - may need manual review")
            return best_text
        
        logger.error("Failed to extract any text from PDF using all methods")
        return ""
    
    def _extract_with_pypdf2(self, pdf_bytes: bytes) -> str:
        """Extract text using PyPDF2."""
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting text from PDF page {page_num}: {e}")
                    continue
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"PyPDF2 extraction error: {e}")
            return ""
    
    def _extract_with_pdfplumber(self, pdf_bytes: bytes) -> str:
        """Extract text using pdfplumber (more accurate)."""
        try:
            import pdfplumber
            pdf_file = io.BytesIO(pdf_bytes)
            text_parts = []
            
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting text from PDF page {page_num}: {e}")
                        continue
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"pdfplumber extraction error: {e}")
            return ""
    
    def _extract_pdf_with_ocr(self, pdf_bytes: bytes) -> str:
        """
        Fallback: Convert PDF pages to images and use OCR.
        This is slower but works for scanned PDFs or PDFs with embedded images.
        """
        try:
            from pdf2image import convert_from_bytes
        except ImportError:
            logger.warning("pdf2image not available - cannot use OCR fallback for PDFs")
            return ""
        
        try:
            # Convert PDF pages to images
            images = convert_from_bytes(pdf_bytes, dpi=300)
            text_parts = []
            
            for page_num, image in enumerate(images):
                try:
                    # Convert PIL image to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Perform OCR on the image
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    if page_text.strip():
                        text_parts.append(page_text.strip())
                except Exception as e:
                    logger.warning(f"Error performing OCR on PDF page {page_num}: {e}")
                    continue
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF OCR extraction error: {e}")
            return ""
    
    def extract_from_image(self, image_bytes: bytes) -> str:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image_bytes: Image file content as bytes
            
        Returns:
            Extracted text string
        """
        if not self.tesseract_available:
            logger.warning("Tesseract OCR not available - cannot extract text from images")
            return ""
        
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary (Tesseract requires RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang='eng')
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def extract_from_text_file(self, text_bytes: bytes, encoding: str = 'utf-8') -> str:
        """
        Extract text from plain text file.
        
        Args:
            text_bytes: Text file content as bytes
            encoding: Text encoding (default: utf-8)
            
        Returns:
            Extracted text string
        """
        try:
            # Try specified encoding first
            try:
                return text_bytes.decode(encoding)
            except UnicodeDecodeError:
                # Fallback to utf-8
                try:
                    return text_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    # Fallback to latin-1
                    try:
                        return text_bytes.decode('latin-1')
                    except UnicodeDecodeError:
                        # Final fallback: use specified encoding with errors='ignore'
                        return text_bytes.decode(encoding, errors='ignore')
        except Exception as e:
            logger.error(f"Error extracting text from text file: {e}")
            return ""
    
    def extract_text(self, file_bytes: bytes, content_type: str, filename: Optional[str] = None) -> str:
        """
        Extract text from file based on content type.
        
        Args:
            file_bytes: File content as bytes
            content_type: MIME content type (e.g., 'application/pdf', 'image/png')
            filename: Optional filename for format detection
            
        Returns:
            Extracted text string
        """
        if not file_bytes:
            return ""
        
        # Normalize content type
        content_type_lower = content_type.lower() if content_type else ""
        
        # PDF files
        if 'pdf' in content_type_lower or (filename and filename.lower().endswith('.pdf')):
            logger.info(f"Extracting text from PDF: {filename or 'unknown'}")
            return self.extract_from_pdf(file_bytes)
        
        # Image files
        if any(img_type in content_type_lower for img_type in ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff']):
            logger.info(f"Extracting text from image using OCR: {filename or 'unknown'}")
            return self.extract_from_image(file_bytes)
        
        # Text files
        if 'text' in content_type_lower or (filename and filename.lower().endswith(('.txt', '.csv', '.log'))):
            logger.info(f"Extracting text from text file: {filename or 'unknown'}")
            return self.extract_from_text_file(file_bytes)
        
        # Try image extraction for unknown types if filename suggests image
        if filename:
            image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg']
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                logger.info(f"Attempting OCR extraction for image file: {filename}")
                return self.extract_from_image(file_bytes)
        
        logger.warning(f"Unsupported content type for text extraction: {content_type} (filename: {filename})")
        logger.info("Supported types: PDF, images (PNG/JPG/JPEG/GIF/BMP/TIFF), text files (TXT/CSV/LOG)")
        return ""


def extract_text_from_attachment(attachment_data: Dict, access_token: str, 
                                 message_id: str, email_address: str, 
                                 base_url: str = "https://graph.microsoft.com/v1.0") -> Optional[str]:
    """
    Download attachment from Microsoft Graph API and extract text.
    
    Args:
        attachment_data: Attachment metadata dict with 'id', 'name', 'contentType'
        access_token: Microsoft Graph API access token
        message_id: Email message ID
        email_address: Email account address
        base_url: Microsoft Graph API base URL
        
    Returns:
        Extracted text string or None if extraction fails
    """
    
    attachment_id = attachment_data.get("id", "")
    attachment_name = attachment_data.get("name", "unknown")
    content_type = attachment_data.get("contentType", "")
    
    if not attachment_id:
        logger.warning("No attachment ID provided")
        return None
    
    try:
        # Download attachment
        headers = {"Authorization": f"Bearer {access_token}"}
        attachment_url = f"{base_url}/users/{email_address}/messages/{message_id}/attachments/{attachment_id}"
        
        logger.info(f"Downloading attachment: {attachment_name} ({content_type})")
        response = httpx.get(attachment_url, headers=headers, timeout=60)
        response.raise_for_status()
        
        attachment_info = response.json()
        content_bytes_b64 = attachment_info.get("contentBytes", "")

        # Guardrail: prevent memory spikes for large attachments before base64 decode
        # Graph attachment payload includes "size" (bytes)
        # NOTE: Hardcoded to 5 MB so production does not depend on .env config.
        max_bytes = 5 * 1024 * 1024  # 5 MB
        size = attachment_info.get("size")
        if isinstance(size, int) and size > max_bytes:
            logger.warning(
                f"Attachment too large to decode ({size} bytes > {max_bytes}). "
                f"Skipping: {attachment_name} (attachment_id={attachment_id})"
            )
            return None
        
        if not content_bytes_b64:
            logger.warning(f"No content bytes found in attachment: {attachment_name}")
            return None
        
        # Decode base64 content
        try:
            file_bytes = base64.b64decode(content_bytes_b64)
        except Exception as e:
            logger.error(f"Error decoding base64 attachment content: {e}")
            return None
        
        # Extract text using DocumentReader
        reader = DocumentReader()
        extracted_text = reader.extract_text(file_bytes, content_type, attachment_name)
        
        if extracted_text:
            logger.info(f"Successfully extracted {len(extracted_text)} characters from {attachment_name}")
        else:
            logger.warning(f"No text extracted from {attachment_name}")
        
        return extracted_text
        
    except Exception as e:
        logger.error(f"Error processing attachment {attachment_name}: {e}")
        return None


def extract_text_from_all_attachments(message_id: str, email_address: str, 
                                     access_token: str, 
                                     base_url: str = "https://graph.microsoft.com/v1.0") -> Dict[str, str]:
    """
    Extract text from all attachments in an email.
    
    Args:
        message_id: Email message ID
        email_address: Email account address
        access_token: Microsoft Graph API access token
        base_url: Microsoft Graph API base URL
        
    Returns:
        Dictionary mapping attachment names to extracted text
    """
    
    results = {}
    
    try:
        # Get list of attachments
        headers = {"Authorization": f"Bearer {access_token}"}
        attachments_url = f"{base_url}/users/{email_address}/messages/{message_id}/attachments"
        
        response = httpx.get(attachments_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        attachments = response.json().get("value", [])
        
        if not attachments:
            logger.info(f"No attachments found in email {message_id}")
            return results
        
        logger.info(f"Found {len(attachments)} attachment(s) in email {message_id}")
        
        # Extract text from each attachment
        for attachment in attachments:
            attachment_name = attachment.get("name", "unknown")
            attachment_id = attachment.get("id", "")
            extracted_text = extract_text_from_attachment(
                attachment, access_token, message_id, email_address, base_url
            )
            
            if extracted_text:
                key = attachment_name
                # Avoid overwriting when multiple attachments share the same name
                if key in results and attachment_id:
                    key = f"{attachment_name} ({attachment_id})"
                results[key] = extracted_text
        
        return results
        
    except Exception as e:
        logger.error(f"Error extracting text from attachments: {e}")
        return results


def get_formatted_attachment_content_for_classification(message_id: str, email_address: str, 
                                                       access_token: str, 
                                                       base_url: str = "https://graph.microsoft.com/v1.0") -> Tuple[str, int]:
    """
    Extract text from all attachments and format it for model classification.
    
    This function is specifically designed for claims_paid classification where attachment
    content (PDFs, images with OCR, text files) needs to be appended to the email body
    with the exact marker format that the classifier expects.
    
    Args:
        message_id: Email message ID
        email_address: Email account address
        access_token: Microsoft Graph API access token
        base_url: Microsoft Graph API base URL
    
    Returns:
        Tuple of (formatted_attachment_content, attachment_count):
        - formatted_attachment_content: Formatted attachment content string with marker "--- ATTACHMENT CONTENT ---"
          Returns empty string if no attachments or no text extracted.
        - attachment_count: Number of attachments successfully processed (0 if none)
        
    Example output:
        ("\n\n--- ATTACHMENT CONTENT ---\n\n--- Content from receipt.pdf ---\nCheck #12345...\n\n", 1)
    """
    try:
        # Extract text from all attachments
        attachment_texts = extract_text_from_all_attachments(
            message_id=message_id,
            email_address=email_address,
            access_token=access_token,
            base_url=base_url
        )
        
        if not attachment_texts:
            logger.info(f"No text extracted from attachments in email {message_id}")
            return "", 0
        
        # Format attachment content with exact marker format (classifier expects this)
        attachment_content = "\n\n--- ATTACHMENT CONTENT ---\n\n"
        # Guardrails: prevent unbounded text from attachments (hardcoded limits)
        # NOTE: These are intentionally hardcoded so production does not depend on .env.
        max_total_chars = 20000
        max_per_attachment_chars = 10000
        total_chars = 0
        processed_count = 0
        
        for attachment_name, extracted_text in attachment_texts.items():
            if extracted_text:
                original_len = len(extracted_text)
                # Per-attachment cap
                if original_len > max_per_attachment_chars:
                    extracted_text = extracted_text[:max_per_attachment_chars] + "\n...[truncated]"
                    logger.info(
                        f"Attachment text truncated for {attachment_name}: "
                        f"{original_len} -> {len(extracted_text)} characters"
                    )
                # Global cap
                if total_chars + len(extracted_text) > max_total_chars:
                    remaining = max_total_chars - total_chars
                    if remaining <= 0:
                        logger.warning(
                            f"Total attachment text limit reached ({max_total_chars} chars) "
                            f"while processing {attachment_name}; remaining attachments skipped"
                        )
                        break
                    # Truncate to remaining budget
                    truncated_text = extracted_text[:remaining] + "\n...[truncated]"
                    attachment_content += f"--- Content from {attachment_name} ---\n{truncated_text}\n\n"
                    total_chars += len(truncated_text)
                    processed_count += 1
                    logger.warning(
                        f"Total attachment text truncated at {max_total_chars} chars "
                        f"while processing {attachment_name}; remaining attachments skipped"
                    )
                    break

                attachment_content += f"--- Content from {attachment_name} ---\n{extracted_text}\n\n"
                total_chars += len(extracted_text)
                processed_count += 1
        
        logger.info(f"Extracted {total_chars} characters from {processed_count} attachment(s) in email {message_id}")
        return attachment_content, processed_count
        
    except Exception as e:
        logger.warning(f"Error formatting attachment content for email {message_id}: {e}")
        return "", 0
