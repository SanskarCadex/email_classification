"""
Invoice handler: talks to the VM / VDI invoice fetch service.

Responsibilities:
- Build the request payload from company / ABCFN / optional invoice number.
- Call the VM API (fetch-invoices) whose URL comes from .env.
- Handle responses:
    - On success: return file bytes, filename, mime_type and request metadata.
    - On error (JSON from API): return structured error for caller to decide.

NOTE:
- ABCFN numbers may come with or without a leading underscore.
- The external contract we aim for is:
    - Our pipeline can pass either form.
    - This handler always sends the *normalized* form (without leading "_")
      to the VM API, which can handle mapping to the underlying folder.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Optional, Dict, Any
import io

import pyzipper
import requests
from dotenv import load_dotenv

from src.log_config import logger  


load_dotenv()


# ============================================================================
# HARDCODED FLAG: Set to False to completely disable invoice handler
# ============================================================================
# This flag is hardcoded in the file to allow quick disable without env changes
# When False, all invoice handler methods will return early without processing
INVOICE_HANDLER_ENABLED = False  # Set to False to disable entire invoice handler

# URL of the VM invoice fetch API (Docker service on the VM)
# Example: http://34.170.59.164:5001/fetch-invoices
INVOICE_FETCH_URL = os.getenv("INVOICE_FETCH_URL")

# Enable/disable invoice handler via environment variable (default: false)
# This works in conjunction with INVOICE_HANDLER_ENABLED flag above
ENABLE_INVOICE_HANDLER = os.getenv("ENABLE_INVOICE_HANDLER", "false").lower() == "true"

# Password used by VDI to decrypt the AES-encrypted ZIP
# Must be set via .env for security (no hardcoded default)
ZIP_PASSWORD = os.getenv("ZIP_PASSWORD", "").encode("utf-8")

# Check if handler should be enabled (both flags must be True)
HANDLER_ACTIVE = INVOICE_HANDLER_ENABLED and ENABLE_INVOICE_HANDLER

if HANDLER_ACTIVE and not ZIP_PASSWORD:
    logger.warning(
        "ZIP_PASSWORD not set in environment; "
        "encrypted ZIPs will fail to decrypt."
    )

if not INVOICE_HANDLER_ENABLED:
    logger.info(
        "Invoice handler is DISABLED by hardcoded flag INVOICE_HANDLER_ENABLED=False. "
        "Set INVOICE_HANDLER_ENABLED = True in invoice_handler.py to enable."
    )

if not INVOICE_FETCH_URL:
    logger.warning(
        "INVOICE_FETCH_URL not set in environment; "
        "invoice_handler will be disabled until configured."
    )



@dataclass
class InvoiceFetchResult:
    success: bool
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    content: Optional[bytes] = None
    error: Optional[str] = None
    details: Optional[Any] = None
    request_payload: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None


class InvoiceHandler:
    """
    Simple client for the VM /fetch-invoices endpoint.

    This module is intentionally decoupled from email sending:
    callers (e.g. fetch_reply.py) can use it to fetch files and then
    decide whether to draft or send the second invoice email.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 60,
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
                "InvoiceHandler initialised without INVOICE_FETCH_URL. "
                "Set INVOICE_FETCH_URL in your .env file."
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

    def _decrypt_and_rezip(self, encrypted_zip_bytes: bytes) -> bytes:
        """
        Decrypt the password-protected ZIP and re-zip without password.
        
        Flow:
        1. Decrypt AES-encrypted ZIP using pyzipper
        2. Extract all files to temp directory
        3. Re-zip without password using standard zipfile
        4. Return unencrypted ZIP bytes
        
        Args:
            encrypted_zip_bytes: Password-protected ZIP file bytes
            
        Returns:
            Unencrypted ZIP file bytes
            
        Raises:
            Exception: If decryption or re-zipping fails
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
            
            # Step 2: Re-zip without password
            temp_zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            temp_zip_path.close()
            
            logger.info(f"Re-zipping extracted files to {temp_zip_path.name} (no password)")
            
            with zipfile.ZipFile(temp_zip_path.name, "w", zipfile.ZIP_DEFLATED) as zf:
                # Walk through extracted directory and add all files
                for root, _dirs, files in os.walk(temp_extract_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Calculate relative path from extract directory
                        arcname = os.path.relpath(file_path, temp_extract_dir)
                        zf.write(file_path, arcname)
            
            # Step 3: Read the unencrypted ZIP bytes
            with open(temp_zip_path.name, "rb") as f:
                unencrypted_zip_bytes = f.read()
            
            logger.info(f"Successfully created unencrypted ZIP ({len(unencrypted_zip_bytes)} bytes)")
            
            return unencrypted_zip_bytes
            
        except Exception as e:
            logger.error(f"Failed to decrypt/re-zip: {e}", exc_info=True)
            raise
        finally:
            # Cleanup temp files
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

        # Global kill switch: check hardcoded flag first, then env flag
        # When disabled, we do NOT call the VM and simply return a disabled result.
        if not HANDLER_ACTIVE:
            if not INVOICE_HANDLER_ENABLED:
                logger.info(
                    "InvoiceHandler is disabled by hardcoded flag INVOICE_HANDLER_ENABLED=False; "
                    "skipping invoice fetch."
                )
                error_msg = "Invoice handler disabled by hardcoded flag"
            elif not ENABLE_INVOICE_HANDLER:
                logger.info(
                    "InvoiceHandler is disabled by ENABLE_INVOICE_HANDLER env flag; "
                    "skipping invoice fetch."
                )
                error_msg = "Invoice handler disabled by environment configuration"
            else:
                error_msg = "Invoice handler disabled"
            
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
                error="INVOICE_FETCH_URL is not configured",
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
        
        # If it's a ZIP file, decrypt and re-zip without password
        if is_zip:
            try:
                logger.info("Received encrypted ZIP, decrypting and re-zipping without password...")
                file_bytes = self._decrypt_and_rezip(file_bytes)
                logger.info("Successfully decrypted and re-zipped invoice file")
            except Exception as e:
                logger.error(f"Failed to decrypt/re-zip invoice file: {e}", exc_info=True)
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


__all__ = ["InvoiceFetchResult", "InvoiceHandler"]


