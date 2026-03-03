"""
Invoice handler: talks to the file_mover (VDI) document retrieval service.

Responsibilities:
- Build the request payload from company / ABCFN / optional invoice number.
- Call the VDI file_mover at POST /retrieve-document.
- Handle responses:
    - On success: return file bytes or list of files, filename(s), mime_type.
    - On error (JSON from API): return structured error for caller to decide.

NOTE:
- ABCFN numbers may come with or without a leading underscore.
- file_mover returns AES-encrypted ZIP; we decrypt it.
- If ZIP has <= 5 files: attach each file individually to the email.
- If ZIP has > 5 files: re-zip without password and attach the ZIP.
- VDI_URL points to file_mover (e.g. http://20.102.88.158:5000/retrieve-document).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import io

import pyzipper
import requests
from dotenv import load_dotenv

from src.log_config import logger  


load_dotenv()


# ============================================================================
# Config: env vars with hardcoded fallbacks (no Kubernetes/GCP config access)
# ============================================================================
INVOICE_HANDLER_ENABLED = True

# VDI file_mover URL - POST /retrieve-document; fallback when env not set
VDI_URL = (os.getenv("VDI_URL") or os.getenv("INVOICE_FETCH_URL") or "").strip() or "http://20.102.88.158:5000/retrieve-document"
INVOICE_FETCH_URL = VDI_URL

# Password used by file_mover to encrypt ZIP; fallback when env not set
_zip_password = os.getenv("ZIP_PASSWORD", "abccollect")
ZIP_PASSWORD = _zip_password.encode("utf-8") if _zip_password else b""

# If ZIP contains <= this many files, attach each file individually; otherwise re-zip and attach ZIP
MAX_FILES_ATTACH_INDIVIDUALLY = 5

# Check if handler should be enabled (single flag)
if INVOICE_HANDLER_ENABLED and not ZIP_PASSWORD:
    logger.warning(
        "ZIP_PASSWORD not set in environment; "
        "encrypted ZIPs will fail to decrypt."
    )

if not INVOICE_FETCH_URL:
    logger.warning(
        "VDI_URL/INVOICE_FETCH_URL not set; invoice_handler will be disabled until configured."
    )


def _is_url_or_link(token: str) -> bool:
    """Return True if the token looks like a URL or link, not an invoice number."""
    if not token or len(token) < 2:
        return False
    t = token.strip().lower()
    if t.startswith("["):
        t = t[1:]
    if t.endswith("]"):
        t = t[:-1]
    return (
        t.startswith("http://")
        or t.startswith("https://")
        or t.startswith("http:")
        or "://" in t
        or (t.startswith("www.") and "." in t[4:])
        or ("?" in t and "=" in t and "." in t)
    )


def extract_invoice_numbers_from_text(text: str) -> List[str]:
    """
    Try to extract invoice number(s) from the given email text.
    This is independent of classification and only focuses on parsing the body text.
    Excludes URLs and links (e.g. Salesforce, image servers) that may contain digits.
    """
    if not text:
        return []

    lines = text.splitlines()
    invoice_numbers: List[str] = []

    # Strategy 1: Look for the common "Inv#" table pattern and then read the
    # following data lines, capturing the first token on each row as the invoice
    # number (e.g. "1-136279281556" or "24550741").
    for idx, line in enumerate(lines):
        if "Inv#" in line:
            for j in range(idx + 1, min(idx + 15, len(lines))):
                candidate = lines[j].strip()
                if not candidate:
                    continue
                if not re.search(r"\d", candidate):
                    if invoice_numbers:
                        break
                    continue

                parts = candidate.split()
                if not parts:
                    continue
                first_token = parts[0]
                # Require at least 5 digits and exclude URL-like tokens
                if re.search(r"\d{5,}", first_token) and not _is_url_or_link(first_token):
                    if first_token not in invoice_numbers:
                        invoice_numbers.append(first_token)
            if invoice_numbers:
                break

    # Strategy 2: Look for sentences like "invoice number is 2004004015" or "invoice # 2004004015"
    # Enforce same minimum-digit validation as Strategy 1 (>= 5 digits)
    MIN_INVOICE_DIGITS = 5
    pattern = re.compile(
        r"invoice\s*(?:number|#)\s*(?:is|:)?\s*([A-Za-z0-9\-]+)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        val = m.group(1)
        digit_count = len(re.findall(r"\d", val or ""))
        if (
            val
            and digit_count >= MIN_INVOICE_DIGITS
            and val not in invoice_numbers
            and not _is_url_or_link(val)
        ):
            invoice_numbers.append(val)

    return invoice_numbers


@dataclass
class InvoiceFetchResult:
    success: bool
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    content: Optional[bytes] = None
    # When ZIP has <= MAX_FILES_ATTACH_INDIVIDUALLY files: list of (filename, bytes) for each file
    content_files: Optional[List[Tuple[str, bytes]]] = None
    error: Optional[str] = None
    details: Optional[Any] = None
    request_payload: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None


class InvoiceHandler:
    """
    Client for the VDI file_mover /retrieve-document endpoint.

    Sends POST with {company_name, abcfn_number, invoice_number?}.
    Receives encrypted ZIP, decrypts it. If <= 5 files, returns each for individual
    attachment; if > 5 files, re-zips without password. Callers attach to draft emails.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 180,
        session: Optional[requests.Session] = None,
    ):
        # Check hardcoded flag first
        if not INVOICE_HANDLER_ENABLED:
            logger.info(
                "InvoiceHandler initialized but disabled by INVOICE_HANDLER_ENABLED=False. "
                "All methods will return early."
            )
        
        self.base_url = base_url or INVOICE_FETCH_URL
        self.timeout = timeout
        self.session = session or requests.Session()

        if not self.base_url and INVOICE_HANDLER_ENABLED:
            logger.error(
                "InvoiceHandler initialised without VDI_URL/INVOICE_FETCH_URL. "
                "Set VDI_URL in your .env or it will use hardcoded default."
            )

    @staticmethod
    def _normalize_abcfn(abcfn_number: str) -> str:
        """
        Normalize ABCFN number:
        - Strip whitespace
        - Remove a single leading underscore if present

        External callers may pass with or without underscore; the VM API
        expects the clean numeric/string ID.
        """
        if not abcfn_number:
            return abcfn_number
        abcfn_number = abcfn_number.strip()
        # Remove a single leading underscore, if present
        if abcfn_number.startswith("_"):
            return abcfn_number[1:]
        return abcfn_number

    def _process_encrypted_zip(
        self, encrypted_zip_bytes: bytes
    ) -> Tuple[str, Any]:
        """
        Decrypt the password-protected ZIP, then either attach files individually or re-zip.

        - If ZIP contains <= MAX_FILES_ATTACH_INDIVIDUALLY files: return ("files", [(filename, bytes), ...])
        - If ZIP contains > MAX_FILES_ATTACH_INDIVIDUALLY files: return ("zip", unencrypted_zip_bytes)

        Args:
            encrypted_zip_bytes: Password-protected ZIP file bytes

        Returns:
            ("files", list of (filename, bytes)) or ("zip", zip_bytes)

        Raises:
            Exception: If decryption fails
        """
        temp_extract_dir = None
        temp_zip_path = None

        try:
            # Step 1: Decrypt and extract ZIP
            temp_extract_dir = tempfile.mkdtemp(prefix="invoice_extract_")
            logger.info(f"Decrypting ZIP ({len(encrypted_zip_bytes)} bytes) to {temp_extract_dir}")

            with pyzipper.AESZipFile(
                io.BytesIO(encrypted_zip_bytes),
                "r",
                compression=pyzipper.ZIP_DEFLATED,
                encryption=pyzipper.WZ_AES,
            ) as zf:
                zf.setpassword(ZIP_PASSWORD)
                zf.extractall(path=temp_extract_dir)

            logger.info(f"Successfully decrypted and extracted ZIP to {temp_extract_dir}")

            # Step 2: Collect all files (relative paths)
            all_files: List[Tuple[str, str]] = []  # (arcname, full_path)
            for root, _dirs, files in os.walk(temp_extract_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_extract_dir)
                    all_files.append((arcname, file_path))

            file_count = len(all_files)
            logger.info(f"ZIP contains {file_count} file(s) (threshold: {MAX_FILES_ATTACH_INDIVIDUALLY})")

            if file_count == 0:
                # Empty ZIP: do not attach anything to the email
                logger.info("ZIP is empty - nothing to attach")
                return ("empty", None)

            if file_count <= MAX_FILES_ATTACH_INDIVIDUALLY:
                # Attach each file individually (no re-zipping)
                content_files: List[Tuple[str, bytes]] = []
                for arcname, file_path in all_files:
                    # Use arcname with path separators replaced to avoid overwriting same basenames
                    safe_fname = arcname.replace("/", "_").replace("\\", "_") or "invoice"
                    with open(file_path, "rb") as f:
                        content_files.append((safe_fname, f.read()))
                logger.info(f"Returning {len(content_files)} file(s) for individual attachment")
                return ("files", content_files)

            # Step 3: Re-zip without password (> 5 files)
            temp_zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            temp_zip_path.close()

            logger.info(f"Re-zipping {file_count} file(s) (no password)")

            with zipfile.ZipFile(temp_zip_path.name, "w", zipfile.ZIP_DEFLATED) as zf:
                for arcname, file_path in all_files:
                    zf.write(file_path, arcname)

            with open(temp_zip_path.name, "rb") as f:
                unencrypted_zip_bytes = f.read()

            logger.info(f"Successfully created unencrypted ZIP ({len(unencrypted_zip_bytes)} bytes)")
            return ("zip", unencrypted_zip_bytes)

        except Exception as e:
            logger.error(f"Failed to decrypt/process ZIP: {e}", exc_info=True)
            raise
        finally:
            if temp_extract_dir and os.path.exists(temp_extract_dir):
                try:
                    shutil.rmtree(temp_extract_dir)
                except Exception as cleanup_err:
                    logger.warning(f"Failed to cleanup extract dir: {cleanup_err}")

            if temp_zip_path and os.path.exists(temp_zip_path.name):
                try:
                    os.remove(temp_zip_path.name)
                except Exception as cleanup_err:
                    logger.warning(f"Failed to cleanup temp ZIP: {cleanup_err}")

    def _build_payload(
        self,
        company_name: str,
        abcfn_number: str,
        invoice_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_abcfn = self._normalize_abcfn(abcfn_number)

        payload: Dict[str, Any] = {
            "company_name": company_name,
            "abcfn_number": normalized_abcfn,
        }

        if invoice_number:
            payload["invoice_number"] = str(invoice_number).strip()

        return payload

    def fetch_invoices(
        self,
        company_name: str,
        abcfn_number: str,
        invoice_number: Optional[str] = None,
    ) -> InvoiceFetchResult:
        """
        Fetch invoices from the VM API.

        - If invoice_number is None:
            Returns ZIP with all invoices for that ABCFN.
        - If invoice_number is provided:
            Returns PDF (single invoice) or ZIP (multiple matches),
            depending on what the VM returns.
        """
        # Redacted view of identifiers for safe logging / error payloads
        redacted_base = {
            "company_name": "***" if company_name else None,
            "abcfn_number": "***" if abcfn_number else None,
            "invoice_number": "***" if invoice_number else None,
        }

        # Global kill switch: when disabled, we do NOT call the VM and simply return a disabled result.
        if not INVOICE_HANDLER_ENABLED:
            logger.info(
                "InvoiceHandler is disabled by hardcoded flag INVOICE_HANDLER_ENABLED=False; "
                "skipping invoice fetch."
            )
            error_msg = "Invoice handler disabled by hardcoded flag"

            return InvoiceFetchResult(
                success=False,
                error=error_msg,
                request_payload={
                    **redacted_base,
                },
            )

        if not self.base_url:
            return InvoiceFetchResult(
                success=False,
                error="VDI_URL/INVOICE_FETCH_URL is not configured",
            )

        payload = self._build_payload(company_name, abcfn_number, invoice_number)
        redacted_payload = {
            **payload,
            "company_name": "***" if payload.get("company_name") else None,
            "abcfn_number": "***" if payload.get("abcfn_number") else None,
            "invoice_number": "***" if payload.get("invoice_number") else None,
        }

        logger.info(
            "Calling invoice VM API",
            extra={"url": self.base_url, "payload": redacted_payload},
        )

        try:
            resp = self.session.post(
                self.base_url,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            logger.error(f"Invoice VM API request failed: {e}")
            return InvoiceFetchResult(
                success=False,
                error="VM invoice API unreachable",
                details=str(e),
                request_payload=redacted_payload,
            )

        content_type = resp.headers.get("Content-Type", "")

        # If the VM API returns JSON, it's usually an error payload
        if content_type.startswith("application/json"):
            try:
                data = resp.json()
            except Exception:
                data = {"raw": resp.text[:500]}

            # Enhanced error logging matching VM proxy improvements
            error_msg = data.get("error") or "Invoice VM API error"
            logger.warning(
                "Invoice VM API returned JSON error response",
                extra={"status": resp.status_code, "data": data},
            )
            logger.warning(f"  Status: {resp.status_code}")
            logger.warning(f"  Error: {error_msg}")
            logger.warning(f"  Success: {data.get('success', False)}")
            
            # Provide helpful context for common errors
            if resp.status_code == 404:
                logger.warning("  This usually means:")
                logger.warning("    - Company folder not found in CSV or filesystem")
                logger.warning("    - ABCFN folder not found (tried with/without underscore)")
                logger.warning("    - Invoices folder not found inside ABCFN folder")
                logger.warning("  Requested: company='***', abcfn='***'")
            elif resp.status_code == 503:
                logger.warning("  VDI service is unreachable (connection error)")
            elif resp.status_code == 504:
                logger.warning("  VDI service timeout (request took too long)")

            return InvoiceFetchResult(
                success=False,
                error=error_msg,
                details=data,
                request_payload=redacted_payload,
                status_code=resp.status_code,
            )

        # For non-200 statuses with non-JSON body, treat as generic error
        if resp.status_code != 200:
            logger.error(
                "Invoice VM API returned non-200 without JSON",
                extra={
                    "status": resp.status_code,
                    "content_type": content_type,
                    "preview": resp.text[:500],
                },
            )
            return InvoiceFetchResult(
                success=False,
                error=f"Invoice VM API returned status {resp.status_code}",
                details=resp.text[:500],
                request_payload=redacted_payload,
                status_code=resp.status_code,
            )

        # At this point we assume we got a file (ZIP / PDF etc.)
        file_bytes = resp.content

        # Get Content-Disposition header once for both ZIP detection and filename extraction
        disposition = resp.headers.get("Content-Disposition", "")
        
        # Check if it's a ZIP file (by content-type or filename)
        is_zip = content_type.startswith("application/zip")
        if not is_zip:
            # Fallback: check filename extension
            if ".zip" in disposition.lower():
                is_zip = True
        
        # If it's a ZIP file, decrypt and process (attach individually if <=5 files, else re-zip)
        if is_zip:
            try:
                logger.info("Received encrypted ZIP, decrypting and processing...")
                result_type, result_data = self._process_encrypted_zip(file_bytes)
                if result_type == "empty":
                    # Empty ZIP: nothing to attach
                    return InvoiceFetchResult(
                        success=True,
                        filename=None,
                        mime_type=content_type or "application/octet-stream",
                        content=None,
                        content_files=None,
                        request_payload=redacted_payload,
                        status_code=resp.status_code,
                    )
                if result_type == "files":
                    # <= 5 files: return for individual attachment
                    return InvoiceFetchResult(
                        success=True,
                        filename=None,
                        mime_type=content_type or "application/octet-stream",
                        content=None,
                        content_files=result_data,
                        request_payload=redacted_payload,
                        status_code=resp.status_code,
                    )
                # > 5 files: re-zipped bytes
                file_bytes = result_data
                logger.info("Successfully decrypted and re-zipped invoice file")
            except Exception as e:
                logger.error(f"Failed to decrypt/process invoice ZIP: {e}", exc_info=True)
                return InvoiceFetchResult(
                    success=False,
                    error=f"Failed to decrypt invoice ZIP: {str(e)}",
                    request_payload=redacted_payload,
                    status_code=resp.status_code,
                )

        # Try to extract filename from Content-Disposition, else generate one
        filename = None
        if "filename=" in disposition:
            # Robust parser for Content-Disposition format
            # Handles: filename="file.zip" and filename*=UTF-8''file.zip
            try:
                # First try RFC 2231 encoding (filename*=UTF-8''...)
                if "filename*=" in disposition:
                    part = disposition.split("filename*=", 1)[1]
                    # Extract after UTF-8''
                    if "UTF-8''" in part:
                        filename = part.split("UTF-8''", 1)[1].split(";")[0].strip()
                    else:
                        filename = part.split(";")[0].strip().strip('"')
                else:
                    # Standard filename="..." format
                    part = disposition.split("filename=", 1)[1]
                    # Remove quotes and semicolons, stop at first semicolon or end
                    filename = part.split(";")[0].strip().strip('"').strip("'")
                
                # Decode URL encoding if present (from RFC 2231)
                if filename and "%" in filename:
                    import urllib.parse
                    filename = urllib.parse.unquote(filename)
            except Exception:
                filename = None

        if not filename:
            # Fallback name based on inputs
            suffix = ""
            if invoice_number:
                suffix = f"_{invoice_number}"
            # Guess extension from content type
            ext = ".bin"
            if "pdf" in content_type.lower():
                ext = ".pdf"
            elif "zip" in content_type.lower():
                ext = ".zip"
            filename = f"{company_name}_{abcfn_number}{suffix}{ext}"

        logger.info(
            "Invoice VM API file fetched successfully",
            extra={
                "file_name": filename,
                "mime_type": content_type,
                "size_bytes": len(file_bytes),
            },
        )

        return InvoiceFetchResult(
            success=True,
            filename=filename,
            mime_type=content_type or "application/octet-stream",
            content=file_bytes,
            request_payload=redacted_payload,
            status_code=resp.status_code,
        )


__all__ = ["InvoiceFetchResult", "InvoiceHandler", "extract_invoice_numbers_from_text"]


