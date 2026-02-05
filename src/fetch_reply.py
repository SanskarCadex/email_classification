"""
fetch_reply.py - Module for fetching, classifying, and moving emails.

Updated with UNIFIED signal handling for Issue #6:
1. Email fetching from multiple accounts
2. Model API integration with retry logic (Issue #2 fixed)
3. Email classification and folder organization
4. Clean text extraction without threads
5. Reply generation with threaded drafts (during processing)
6. Complete email data structure for MongoDB
7. Proper message ID handling after folder moves
8. UNIFIED threading-based stop mechanism
"""

import os
import time
import threading
import httpx
import msal
import requests
import re
import tempfile
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple
from dotenv import load_dotenv
from src.db import get_mongo, PostgresHelper
from src.log_config import logger
from src.doc_handler import get_formatted_attachment_content_for_classification
from src.invoice_handler import InvoiceHandler

load_dotenv()

# Configuration
MS_GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TENANT_ID = os.getenv("TENANT_ID")
EMAIL_ADDRESS = "ABCcollectionsteamD@abc-amega.com,ABCcollectionsteamI@abc-amega.com,ABCcollectionsteamJ@abc-amega.com"
YOUR_DOMAIN = os.getenv("YOUR_DOMAIN", "abc-amega.com")
MODEL_API_URL = "http://104.197.197.76:8000"
COMPANY_NAME = os.getenv("COMPANY_NAME", "ABC/AMEGA")
SEND_INVOICE_REQUEST_NO_INFO = False  # Set to True to send directly, False for draft
SEND_CLAIMS_PAID_NO_PROOF = False     # Set to True to send directly, False for draft
SEND_INVOICE_WITH_ATTACHMENT = False  # Set to True to send the SECOND invoice email directly, False for draft (only runs if first was sent)

# Updated list of allowed labels
ALLOWED_LABELS = [
    "no_reply_no_info",
    "no_reply_with_info", 
    "auto_reply_no_info",
    "auto_reply_with_info",
    "invoice_request_no_info",
    "invoice_request_with_info",  
    "claims_paid_no_proof",
    "claims_paid_with_proof",
    "manual_review",
    "uncategorised"
]
# Labels that should receive responses
# NOTE:
# - Classification event_type will be "invoice_request_no_info" or "claims_paid_no_proof".
# - "invoice_request_acknowledgment" is included so ModelAPIClient.generate_reply()
#   is allowed to call /generate_reply for the ACK template, even though it is
#   not a classification label returned by the model.
RESPONSE_LABELS = [
    "invoice_request_no_info",
    "invoice_request_acknowledgment",
    "claims_paid_no_proof"
]

def validate_config():
    """Validate required environment variables."""
    required_vars = ["CLIENT_ID", "CLIENT_SECRET", "TENANT_ID", "EMAIL_ADDRESS"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")
    logger.info("Configuration validation passed")

class ModelAPIClient:
    """Client for model API calls - 180 second timeout with manual review fallback."""
    
    def __init__(self):
        self.base_url = MODEL_API_URL
        
    def health_check(self):
        """Quick health check with timeout."""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            return response.status_code == 200
        except:
            return False

    def process_email_complete(self, subject, body, headers=None, sender_email=None, 
                             recipient_emails=None, has_attachments=False, had_threads=False):
        """Process email with model API - 180 second timeout, immediate response."""
        payload = {
            "subject": subject,
            "body": body,
            "headers": headers or [],
            "sender_email": sender_email,
            "recipient_emails": recipient_emails or [],
            "has_attachments": has_attachments,
            "had_threads": had_threads
        }
        
        try:
            logger.info("Calling model API (180s timeout)...")
            start_time = time.time()
            
            # Single 180-second timeout - no retries needed
            response = requests.post(
                f"{self.base_url}/api/process_email_complete", 
                json=payload, 
                timeout=180  # 180 seconds max
            )
            response.raise_for_status()
            result = response.json()
            
            # Log actual processing time
            elapsed = time.time() - start_time
            logger.info(f"Model API completed in {elapsed:.1f}s: {result.get('event_type', 'unknown')}")
            
            return result
            
        except requests.exceptions.Timeout:
            # This call uses a 180s timeout above; reflect that in logs.
            logger.warning("Model API timeout after 180s - classifying as manual_review")
            return self._get_manual_review_fallback()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Model API request error: {e} - classifying as manual_review")
            return self._get_manual_review_fallback()
            
        except Exception as e:
            logger.error(f"Unexpected model API error: {e} - classifying as manual_review")
            return self._get_manual_review_fallback()

    def generate_reply(self, subject, body, label, sender_name=None, sender_email=None, 
                       recipient_email=None, entities=None, max_retries=3):
        """Generate reply with retry logic - 60 second timeout per attempt, up to 3 retries."""
        if label not in RESPONSE_LABELS:
            logger.warning(f"Label {label} not in RESPONSE_LABELS - skipping reply generation")
            return ""
            
        payload = {
            "subject": subject,
            "body": body,
            "label": label,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "recipient_email": recipient_email,
            "entities": entities or {}
        }
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Generating reply for {label} (attempt {attempt}/{max_retries}, 60s timeout)...")
                start_time = time.time()
                
                # Single 60-second timeout for reply generation
                response = requests.post(
                    f"{self.base_url}/api/generate_reply", 
                    json=payload, 
                    timeout=60  # 60 seconds max
                )
                response.raise_for_status()
                result = response.json()
                reply = result.get("reply", "")
                
                # Log actual processing time
                elapsed = time.time() - start_time
                
                if reply and reply.strip():
                    logger.info(f"Reply generated successfully in {elapsed:.1f}s for {label}: {len(reply)} chars (attempt {attempt})")
                    return reply
                else:
                    logger.warning(f"Reply generation returned empty reply for {label} (attempt {attempt}/{max_retries})")
                    if attempt < max_retries:
                        logger.info(f"Retrying reply generation for {label}...")
                        time.sleep(2)  # Brief delay before retry
                        continue
                    else:
                        logger.error(f"Reply generation failed after {max_retries} attempts - returned empty reply for {label}")
                        return ""
                
            except requests.exceptions.Timeout:
                logger.warning(f"Reply generation timeout for {label} (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    logger.info(f"Retrying reply generation for {label} after timeout...")
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"Reply generation failed after {max_retries} attempts due to timeout for {label}")
                    return ""
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Reply generation error for {label} (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying reply generation for {label} after error...")
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"Reply generation failed after {max_retries} attempts due to request error for {label}: {e}")
                    return ""
                
            except Exception as e:
                logger.error(f"Unexpected reply error for {label} (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying reply generation for {label} after unexpected error...")
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"Reply generation failed after {max_retries} attempts due to unexpected error for {label}: {e}")
                    return ""

    def _get_manual_review_fallback(self):
        """Fallback response for failed API calls - sends to manual review."""
        logger.info("Using manual_review fallback - email will need human review")
        return {
            "debtor_number": "",
            "event_type": "manual_review",
            "target_folder": "manual_review",
            "reply_sent": "no_response",
            "new_contact_email": "",
            "new_contact_phone": "",
            "contact_status": "active",
            "cleaned_body": ""
        }

def html_to_text(html_content):
    """Convert HTML to clean text - exactly like test code."""
    if not html_content:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html_content)
    
    # Clean up HTML entities
    entities = {
        '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&#39;': "'", '\r\n': '\n', '\r': '\n'
    }
    
    for entity, replacement in entities.items():
        text = text.replace(entity, replacement)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def extract_clean_email_content(msg):
    """Extract clean email content without HTML and threads - exactly like test code."""
    clean_body = ""
    data_source = ""
    had_threads = False
    
    # Get both uniqueBody and fullBody
    unique_body = msg.get("uniqueBody", {})
    full_body = msg.get("body", {})
    
    # Check if email has threads (compare lengths)
    if unique_body and unique_body.get("content") and full_body and full_body.get("content"):
        unique_content = unique_body.get("content", "").strip()
        full_content = full_body.get("content", "").strip()
        
        if len(unique_content) > 0 and len(full_content) > len(unique_content) * 1.2:
            had_threads = True
    
    # FIX: Match test.py behavior - when had_threads=False, use FULL body (like test.py uses full .eml)
    # This ensures the model sees the same content as test.py for non-threaded emails
    if not had_threads and full_body and full_body.get("content"):
        # Use full body for non-threaded emails (matches test.py behavior)
        content = full_body.get("content", "").strip()
        content_type = full_body.get("contentType", "").lower()
        
        if content_type == "text":
            clean_body = content
            data_source = "body_text"
        elif content_type == "html":
            clean_body = html_to_text(content)
            data_source = "body_html"
    elif had_threads and unique_body and unique_body.get("content"):
        # For threaded emails, use uniqueBody (current reply only)
        content = unique_body.get("content", "").strip()
        content_type = unique_body.get("contentType", "").lower()
        
        if content_type == "text":
            clean_body = content
            data_source = "uniqueBody_text"
        elif content_type == "html":
            clean_body = html_to_text(content)
            data_source = "uniqueBody_html"
    
    # Fallback to full body if uniqueBody didn't work for threaded emails
    if not clean_body and full_body and full_body.get("content"):
        content = full_body.get("content", "").strip()
        content_type = full_body.get("contentType", "").lower()
        
        if content_type == "text":
            clean_body = content
            data_source = "body_text"
        elif content_type == "html":
            clean_body = html_to_text(content)
            data_source = "body_html"
    
    # Last resort: bodyPreview
    if not clean_body:
        clean_body = msg.get("bodyPreview", "").strip()
        data_source = "bodyPreview"
    
    return clean_body, data_source, had_threads

class MSGraphClient:
    """Microsoft Graph API client for email operations."""
    
    def __init__(self):
        self.base_url = MS_GRAPH_BASE_URL
        self.client_id = CLIENT_ID
        self.client_secret = CLIENT_SECRET
        self.tenant_id = TENANT_ID
        
        # Parse multiple email addresses
        email_env = EMAIL_ADDRESS or ""
        if "," in email_env:
            self.email_addresses = [email.strip() for email in email_env.split(",")]
        else:
            self.email_addresses = [email_env] if email_env else []
        
        self._token_cache = {"token": None, "expires_at": 0}
        
        logger.info(f"MSGraphClient initialized for {len(self.email_addresses)} email(s)")
        
    def get_access_token(self):
        """Get a valid access token."""
        current_time = time.time()
        if self._token_cache.get("token") and self._token_cache.get("expires_at", 0) > current_time + 60:
            return self._token_cache["token"]
        
        validate_config()
        
        app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}"
        )
        
        result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        
        if "access_token" in result:
            self._token_cache = {
                "token": result["access_token"],
                "expires_at": current_time + result.get("expires_in", 3600)
            }
            return result["access_token"]
        else:
            raise Exception(f"Failed to acquire token: {result.get('error_description')}")

    def fetch_unread_emails_from_account(self, email_address, max_emails):
        """Fetch unread emails from a single account - OLDEST FIRST like test code."""
        params = {
            "$orderby": "receivedDateTime asc",  # OLDEST FIRST (like test code)
            "$filter": "isRead eq false and isDraft eq false",
            "$select": "id,subject,from,body,bodyPreview,uniqueBody,receivedDateTime,hasAttachments,toRecipients,ccRecipients,internetMessageHeaders,conversationId",
            "$top": max_emails
        }
        
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        url = f"{self.base_url}/users/{email_address}/mailFolders/inbox/messages"
        
        response = httpx.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        
        emails = response.json().get("value", [])
        
        # Add account info to each email
        for email in emails:
            email["source_account"] = email_address
        
        logger.info(f"Fetched {len(emails)} unread emails from {email_address}")
        return emails

    def fetch_unread_emails(self, batch_size):
        """Fetch unread emails from all accounts with simple distribution."""
        all_emails = []
        
        # Step 1: Check availability in each account
        account_availability = {}
        total_available = 0
        
        for email_address in self.email_addresses:
            access_token = self.get_access_token()
            count_headers = {
                "Authorization": f"Bearer {access_token}",
                "ConsistencyLevel": "eventual"
            }
            count_url = f"{self.base_url}/users/{email_address}/mailFolders/inbox/messages/$count"
            count_params = {"$filter": "isRead eq false and isDraft eq false"}
            
            response = httpx.get(count_url, headers=count_headers, params=count_params, timeout=60)
            available_count = int(response.text) if response.status_code == 200 else 0
            
            account_availability[email_address] = available_count
            total_available += available_count
            
            logger.info(f"Account {email_address}: {available_count} unread emails")
        
        if total_available == 0:
            logger.info("No unread emails found")
            return all_emails
        
        # Step 2: Calculate emails to process
        emails_to_process = min(batch_size, total_available)
        logger.info(f"Processing {emails_to_process} emails from {total_available} available")
        
        # Step 3: Simple distribution - priority to accounts with most emails
        sorted_accounts = sorted(account_availability.items(), key=lambda x: x[1], reverse=True)
        
        distribution_plan = {}
        remaining_to_fetch = emails_to_process
        
        for email_address, available_count in sorted_accounts:
            if remaining_to_fetch <= 0:
                distribution_plan[email_address] = 0
                continue
            
            to_fetch = min(available_count, remaining_to_fetch)
            distribution_plan[email_address] = to_fetch
            remaining_to_fetch -= to_fetch
        
        # Log the distribution plan
        logger.info("Distribution plan:")
        for email_address, to_fetch in distribution_plan.items():
            available = account_availability[email_address]
            logger.info(f"  {email_address}: {to_fetch}/{available} emails")
        
        # Step 4: Fetch emails according to the distribution plan
        total_fetched = 0
        for email_address in self.email_addresses:
            emails_to_fetch = distribution_plan.get(email_address, 0)
            
            if emails_to_fetch <= 0:
                continue
                
            emails = self.fetch_unread_emails_from_account(email_address, emails_to_fetch)
            all_emails.extend(emails)
            total_fetched += len(emails)
            
            logger.info(f"Collected {len(emails)}/{emails_to_fetch} emails from {email_address}")
        
        logger.info(f"Batch complete: {total_fetched} emails collected")
        return all_emails

    def move_email_to_folder(self, message_id, folder_id, email_address):
        """Move email to folder."""
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        endpoint = f"{self.base_url}/users/{email_address}/messages/{message_id}/move"
        payload = {"destinationId": folder_id}
        
        response = httpx.post(endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        new_id = response.json().get("id", message_id)
        logger.info(f"Moved email {message_id} to folder {folder_id}")
        return True, new_id

    def mark_email_read(self, message_id, email_address, is_read=True):
        """Mark email as read/unread."""
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        endpoint = f"{self.base_url}/users/{email_address}/messages/{message_id}"
        payload = {"isRead": is_read}
        
        response = httpx.patch(endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return True

    def create_threaded_reply_draft(self, original_message_id, reply_text, from_account):
        """Create threaded reply draft using createReply method - EXACTLY like test code."""
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Add footer
        footer = f"\n\n---\nThis email was generated automatically by {COMPANY_NAME} System.\nSent on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
        body_with_footer = reply_text + footer
        
        try:
            # STEP 1: Create reply draft using createReply (this is what worked in test!)
            create_reply_endpoint = f"{self.base_url}/users/{from_account}/messages/{original_message_id}/createReply"
            
            response = httpx.post(create_reply_endpoint, headers=headers, json={}, timeout=30)
            
            if response.status_code in [200, 201]:
                draft_data = response.json()
                draft_id = draft_data.get("id")
                
                if draft_id:
                    logger.info(f"Reply draft created: {draft_id}")
                    
                    # STEP 2: Update the draft with our content
                    update_payload = {
                        "body": {
                            "contentType": "Text",
                            "content": body_with_footer
                        }
                    }
                    
                    update_endpoint = f"{self.base_url}/users/{from_account}/messages/{draft_id}"
                    update_response = httpx.patch(update_endpoint, headers=headers, json=update_payload, timeout=30)
                    
                    if update_response.status_code in [200, 204]:
                        logger.info("Threaded draft body updated successfully")
                        
                        # STEP 3: Verify draft actually exists before returning (safety check)
                        # If verification fails, we still return draft_id since creation succeeded
                        # This prevents losing drafts due to temporary network/API issues
                        verify_endpoint = f"{self.base_url}/users/{from_account}/messages/{draft_id}"
                        try:
                            verify_response = httpx.get(verify_endpoint, headers=headers, timeout=10)  # Shorter timeout for verification
                            
                            if verify_response.status_code == 200:
                                verify_data = verify_response.json()
                                if verify_data.get("isDraft", False):
                                    logger.info(f"Draft verified and exists in mailbox: {draft_id}")
                                    return draft_id
                                else:
                                    logger.warning(f"Draft ID returned but message is not a draft (might have been sent): {draft_id}")
                                    # Still return draft_id since creation succeeded - let caller decide
                                    return draft_id
                            elif verify_response.status_code == 404:
                                logger.error(f"Draft created but NOT FOUND in Graph API (404) - draft may have been deleted: {draft_id}")
                                # Draft doesn't exist - return None to prevent storing invalid draft_id
                                return None
                            else:
                                logger.warning(f"Draft created but verification returned {verify_response.status_code} - returning draft_id anyway (may be temporary issue)")
                                # Return draft_id since creation succeeded - verification failure might be temporary
                                return draft_id
                        except httpx.TimeoutException:
                            logger.warning("Draft created but verification timed out - returning draft_id anyway (may be temporary network issue)")
                            return draft_id
                        except Exception as verify_error:
                            logger.warning(f"Draft created but verification failed with error: {verify_error} - returning draft_id anyway")
                            return draft_id
                    else:
                        logger.warning(f"Draft created but body update failed: {update_response.status_code}")
                        # Still verify the draft exists even if update failed
                        verify_endpoint = f"{self.base_url}/users/{from_account}/messages/{draft_id}"
                        try:
                            verify_response = httpx.get(verify_endpoint, headers=headers, timeout=10)
                            if verify_response.status_code == 200:
                                verify_data = verify_response.json()
                                if verify_data.get("isDraft", False):
                                    logger.warning("Draft exists but body update failed - returning draft_id anyway")
                                    return draft_id
                                else:
                                    logger.warning("Draft exists but is not a draft (might have been sent) - returning draft_id anyway")
                                    return draft_id
                            elif verify_response.status_code == 404:
                                logger.error("Draft created but NOT FOUND in Graph API (404) after body update failed")
                                return None
                            else:
                                logger.warning(f"Draft created but verification returned {verify_response.status_code} - returning draft_id anyway")
                                return draft_id
                        except Exception as verify_error:
                            logger.warning(f"Draft created but verification failed: {verify_error} - returning draft_id anyway")
                            return draft_id
                else:
                    logger.error("No draft ID returned from createReply")
            else:
                logger.error(f"createReply failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error creating threaded reply: {e}")
        
        return None

    def send_threaded_reply_directly(self, original_message_id, reply_text, from_account):
        """Send threaded reply directly (not as draft)."""
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Add footer
        footer = f"\n\n---\nThis email was generated automatically by {COMPANY_NAME} System.\nSent on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
        body_with_footer = reply_text + footer
        
        try:
            # STEP 1: Create reply draft first
            create_reply_endpoint = f"{self.base_url}/users/{from_account}/messages/{original_message_id}/createReply"
            
            response = httpx.post(create_reply_endpoint, headers=headers, json={}, timeout=30)
            
            if response.status_code in [200, 201]:
                draft_data = response.json()
                draft_id = draft_data.get("id")
                
                if draft_id:
                    # STEP 2: Update the draft with our content
                    update_payload = {
                        "body": {
                            "contentType": "Text",
                            "content": body_with_footer
                        }
                    }
                    
                    update_endpoint = f"{self.base_url}/users/{from_account}/messages/{draft_id}"
                    update_response = httpx.patch(update_endpoint, headers=headers, json=update_payload, timeout=30)
                    
                    if update_response.status_code in [200, 204]:
                        # STEP 3: Send the draft
                        send_endpoint = f"{self.base_url}/users/{from_account}/messages/{draft_id}/send"
                        send_response = httpx.post(send_endpoint, headers=headers, json={}, timeout=30)
                        
                        if send_response.status_code in [200, 202]:
                            logger.info(f"Threaded reply sent directly: {draft_id}")
                            return True
                        else:
                            logger.error(f"Failed to send reply: {send_response.status_code}")
                            return False
                    else:
                        logger.error(f"Failed to update draft: {update_response.status_code}")
                        return False
                else:
                    logger.error("No draft ID returned from createReply")
                    return False
            else:
                logger.error(f"createReply failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending threaded reply directly: {e}")
            return False

    def send_existing_draft(self, draft_id: str, from_account: str) -> bool:
        """Send an existing draft message (used when attachments were added to a draft)."""
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        try:
            send_endpoint = f"{self.base_url}/users/{from_account}/messages/{draft_id}/send"
            send_response = httpx.post(send_endpoint, headers=headers, json={}, timeout=30)
            if send_response.status_code in [200, 202]:
                logger.info(f"Existing draft sent: {draft_id}")
                return True
            logger.error(f"Failed to send existing draft {draft_id}: {send_response.status_code} - {send_response.text}")
            return False
        except Exception as e:
            logger.error(f"Error sending existing draft {draft_id}: {e}")
            return False

    def attach_files_to_draft(self, draft_id: str, from_account: str, file_paths: List[str]) -> int:
        """
        Attach files to an existing draft email
        
        Args:
            draft_id: Draft message ID
            from_account: Email account
            file_paths: List of file paths to attach
            
        Returns:
            Number of successfully attached files
        """
        import base64
        
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        success_count = 0
        
        for file_path in file_paths:
            try:
                # Read file and encode to base64
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                file_base64 = base64.b64encode(file_content).decode('utf-8')
                filename = os.path.basename(file_path)
                
                # Check file size (Microsoft Graph limit is 3MB for inline attachments)
                file_size = len(file_content)
                if file_size > 3 * 1024 * 1024:  # 3MB
                    logger.warning(f"File too large to attach: {filename} ({file_size} bytes)")
                    continue
                
                # Create attachment
                attachment_payload = {
                    "@odata.type": "#microsoft.graph.fileAttachment",
                    "name": filename,
                    "contentBytes": file_base64
                }
                
                endpoint = f"{self.base_url}/users/{from_account}/messages/{draft_id}/attachments"
                response = httpx.post(endpoint, headers=headers, json=attachment_payload, timeout=60)
                
                if response.status_code in [200, 201]:
                    logger.info(f"Attached file: {filename}")
                    success_count += 1
                else:
                    logger.error(f"Failed to attach {filename}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error attaching file {file_path}: {e}")
        
        return success_count

    def ensure_classification_folders(self, email_address):
        """Ensure classification folders exist."""
        def _normalize_folder_name(name):
            return re.sub(r"\s+", " ", name.strip().lower())

        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

        parent_name = "Email Classifications"
        folder_map = {}

        # Get all folders
        response = httpx.get(f"{self.base_url}/users/{email_address}/mailFolders?$top=1000",
                             headers=headers, timeout=30)
        response.raise_for_status()
        all_folders = response.json()["value"]

        # Find or create parent folder
        parent_id = None
        normalized_parent = _normalize_folder_name(parent_name)
        for folder in all_folders:
            if _normalize_folder_name(folder["displayName"]) == normalized_parent:
                parent_id = folder["id"]
                break

        if not parent_id:
            response = httpx.post(f"{self.base_url}/users/{email_address}/mailFolders",
                                  headers=headers, json={"displayName": parent_name}, timeout=30)
            response.raise_for_status()
            parent_id = response.json()["id"]
            logger.info(f"Created parent folder: {parent_name}")

        # Get child folders
        response = httpx.get(f"{self.base_url}/users/{email_address}/mailFolders/{parent_id}/childFolders",
                             headers=headers, timeout=30)
        response.raise_for_status()
        child_folders = response.json()["value"]

        # Create missing child folders
        for label in ALLOWED_LABELS:
            display_name = label.replace("_", " ").title()
            normalized_display_name = _normalize_folder_name(display_name)

            folder_id = None
            for folder in child_folders:
                if _normalize_folder_name(folder["displayName"]) == normalized_display_name:
                    folder_id = folder["id"]
                    break

            if not folder_id:
                response = httpx.post(
                    f"{self.base_url}/users/{email_address}/mailFolders/{parent_id}/childFolders",
                    headers=headers,
                    json={"displayName": display_name},
                    timeout=30
                )
                if response.status_code in [200, 201]:
                    folder_id = response.json()["id"]
                    logger.info(f"Created folder: {display_name}")

            if folder_id:
                folder_map[label] = folder_id

        logger.info(f"Folder mapping ready: {len(folder_map)} folders")
        return folder_map

class EmailProcessor:
    """Main email processing logic with UNIFIED stop signal."""
    
    def __init__(self, batch_id, stop_event=None):
        self.batch_id = batch_id
        self.stop_event = stop_event or threading.Event()
        self.mongo = get_mongo()
        self.model_api = ModelAPIClient()
        self.graph_client = MSGraphClient()
        self.invoice_handler = InvoiceHandler()
        self.folder_mappings = {}
        
        # Set batch in MongoDB
        if self.mongo:
            self.mongo.set_batch_id(batch_id)
    
    def _create_draft_with_retry(self, message_id: str, reply_text: str, source_account: str, 
                                  context: str = "draft") -> str:
        """Create threaded reply draft with retry logic.
        
        Args:
            message_id: Original message ID to reply to
            reply_text: Reply content
            source_account: Email account
            context: Description for logging (e.g., "invoice ack", "fallback", "threaded")
        
        Returns:
            Draft ID if successful, None otherwise
        """
        for attempt in range(1, 4):
            draft_id = self.graph_client.create_threaded_reply_draft(
                message_id, reply_text, source_account
            )
            if draft_id:
                logger.info(f"{context.capitalize()} draft created successfully (attempt {attempt}/3): {draft_id}")
                return draft_id
            
            if attempt < 3:
                logger.warning(f"{context.capitalize()} draft creation failed (attempt {attempt}/3) - retrying...")
                time.sleep(2)  # Brief delay before retry
            else:
                logger.error(f"❌ CRITICAL: {context.capitalize()} draft creation FAILED after 3 attempts for email {message_id}")
                if reply_text:
                    logger.error(f"   Reply text was generated ({len(reply_text)} chars) but draft could not be created")
        
        return None
    
    def _process_single_email(self, msg):
        """Process a single email with UNIFIED stop signal and enhanced attachment detection."""
        # CHECK UNIFIED STOP BEFORE PROCESSING EACH EMAIL
        if self.stop_event.is_set():
            logger.info("STOP: Stop signal detected during email processing - stopping NOW")
            return False
        
        message_id = msg.get("id", "unknown")
        source_account = msg.get("source_account", "")
        
        # Extract basic email data
        sender_info = msg.get("from", {}).get("emailAddress", {})
        sender = sender_info.get("address", "")
        sender_name = sender_info.get("name", sender)
        subject = msg.get("subject", "")
        received = msg.get("receivedDateTime", "")
        conversation_id = msg.get("conversationId", "")
        
        # Enhanced attachment detection - matching test.py logic
        # Step 1: Start with Graph API value (actual attachments)
        has_attachments = msg.get("hasAttachments", False)
        proof_attachments = []
        inline_images_non_proof = []
        
        # Step 2: Check for inline attachments in HTML content, but filter out logos/signatures
        # Only count inline images as proof if they're PDFs or have proof-related filenames
        if not has_attachments:  # Only check if no traditional attachments found
            try:
                unique_body = msg.get("uniqueBody", {})
                full_body = msg.get("body", {})
                
                for body_obj in [unique_body, full_body]:
                    if body_obj and body_obj.get("contentType", "").lower() == "html":
                        html_content = body_obj.get("content", "")
                        if html_content:
                            # Extract filenames from CID references (e.g., cid:image001.png@01DC7FB8.F63AF040)
                            cid_filename_pattern = r'cid:([^@\s"\'<>]+)'
                            cid_filenames = re.findall(cid_filename_pattern, html_content, re.IGNORECASE)
                            
                            # Extract filenames from img src attributes
                            img_src_pattern = r'<img[^>]*src=["\']([^"\']+)["\']'
                            img_srcs = re.findall(img_src_pattern, html_content, re.IGNORECASE)
                            
                            # Extract filenames from object/embed data attributes
                            object_data_pattern = r'<object[^>]*data=["\']([^"\']+)["\']'
                            object_data = re.findall(object_data_pattern, html_content, re.IGNORECASE)
                            
                            # Combine all extracted filenames
                            all_filenames = cid_filenames + img_srcs + object_data
                            
                            # Check for base64 encoded images (often screenshots of receipts - could be proof)
                            base64_patterns = [
                                r'data:image/[^;]+;base64,',
                                r'src="data:image/',
                                r'<img[^>]*data:image'
                            ]
                            has_base64_image = any(re.search(pattern, html_content, re.IGNORECASE) for pattern in base64_patterns)
                            
                            # Process each filename found
                            for filename in all_filenames:
                                filename_lower = filename.lower()
                                
                                # Filter out common logo/signature filenames
                                is_logo_or_signature = any(pattern in filename_lower for pattern in [
                                    "logo", "signature", "sig", "banner", "header", "footer",
                                    "image001", "image002", "image003", "~wrd", "cid:",
                                    "company", "brand", "icon"
                                ])
                                
                                # Check if it's a PDF or has proof-related filename
                                is_pdf = filename_lower.endswith('.pdf')
                                has_proof_keyword = any(keyword in filename_lower for keyword in [
                                    "receipt", "proof", "payment", "statement", "invoice",
                                    "confirmation", "transaction", "check", "wire", "ach"
                                ])
                                
                                # Count as proof attachment if:
                                # 1. It's a PDF (likely proof document)
                                # 2. Filename suggests proof (receipt, proof, payment, etc.)
                                # 3. NOT a logo/signature
                                is_proof_likely = (is_pdf or has_proof_keyword) and not is_logo_or_signature
                                
                                if is_proof_likely:
                                    has_attachments = True
                                    proof_attachments.append({
                                        "filename": filename,
                                        "type": "inline_proof"
                                    })
                                    logger.info(f"Detected proof inline attachment in email {message_id}: {filename}")
                                else:
                                    inline_images_non_proof.append({
                                        "filename": filename,
                                        "is_logo": is_logo_or_signature
                                    })
                            
                            # Base64 images (screenshots) - could be proof, but be conservative
                            # Only count if we haven't found any proof attachments yet
                            if has_base64_image and not has_attachments:
                                # Base64 images are often screenshots of receipts, so count as potential proof
                                has_attachments = True
                                proof_attachments.append({
                                    "filename": "base64_screenshot",
                                    "type": "base64_image"
                                })
                                logger.info(f"Detected base64 encoded image in email {message_id} - treating as potential proof")
                            
                            if has_attachments:
                                break
                            
                            # Log non-proof inline images for debugging
                            if inline_images_non_proof:
                                logger.debug(f"Found {len(inline_images_non_proof)} inline images (non-proof) in email {message_id}")
            except Exception as e:
                logger.warning(f"Error during inline attachment detection for {message_id}: {e}")
        
        # Log final attachment status
        if has_attachments:
            logger.info(f"Email {message_id} has proof attachments: {len(proof_attachments)} proof attachment(s)")
        elif inline_images_non_proof:
            logger.debug(f"Email {message_id} has {len(inline_images_non_proof)} inline images (non-proof, filtered out)")
        
        # ✅ Extract headers for model API
        headers = msg.get("internetMessageHeaders", [])
        
        # Extract clean content
        clean_body, data_source, had_threads = extract_clean_email_content(msg)
        
        # Extract recipients  
        to_recipients = msg.get("toRecipients", [])
        recipient = ""
        recipient_emails = []
        if to_recipients:
            recipient = to_recipients[0].get("emailAddress", {}).get("address", "")
            recipient_emails = [r.get("emailAddress", {}).get("address", "") for r in to_recipients]
        
        # Extract CC recipients
        cc_recipients = msg.get("ccRecipients", [])
        cc_emails = []
        if cc_recipients:
            cc_emails = [cc.get("emailAddress", {}).get("address", "") for cc in cc_recipients]
        cc_string = ", ".join(cc_emails) if cc_emails else None
        
        # Determine receiver type
        receiver_type = "external"
        if recipient and "@abc-amega.com" in recipient.lower():
            receiver_type = "internal"
        
        logger.info(f"Processing email {message_id} from {sender} | Account: {source_account} | Attachments: {has_attachments}")
        
        # Check for duplicates
        if self.mongo and self.mongo.email_exists(message_id):
            logger.info(f"Skipping duplicate email by message_id: {message_id}")
            return True
        
        # CHECK UNIFIED STOP BEFORE MODEL API CALL
        if self.stop_event.is_set():
            logger.info("STOP: Stop signal detected before model API call - stopping NOW")
            return False
        
        # ✅ Extract and append attachment text for claims_paid classification
        # This is critical for distinguishing claims_paid_with_proof vs claims_paid_no_proof
        attachments_processed_count = 0
        if has_attachments:
            try:
                access_token = self.graph_client.get_access_token()
                attachment_content, attachments_processed_count = get_formatted_attachment_content_for_classification(
                    message_id=message_id,
                    email_address=source_account,
                    access_token=access_token,
                    base_url=MS_GRAPH_BASE_URL
                )
                
                if attachment_content:
                    clean_body = clean_body + attachment_content
                    logger.info(f"Appended attachment text to email body for classification ({attachments_processed_count} attachment(s) processed)")
                else:
                    logger.info("Attachments detected but no text extracted - classification will use email body only")
            except (SystemExit, KeyboardInterrupt) as e:
                # CRITICAL: Log context but let control exceptions propagate for clean shutdown
                logger.error(f"CRITICAL: SystemExit/KeyboardInterrupt raised during attachment processing for email {message_id}: {e}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                # Re-raise to allow clean application shutdown
                raise
            except ImportError as e:
                logger.error(f"CRITICAL: Missing dependency for attachment processing: {e}")
                logger.error("Please ensure pytesseract, pdfplumber, or PyPDF2 are installed")
                logger.warning("Continuing email processing without attachment text")
                attachments_processed_count = 0
            except Exception as e:
                logger.error(f"CRITICAL: Error extracting attachment text for email {message_id}: {e}", exc_info=True)
                logger.warning("Continuing email processing without attachment text - this email will be classified using body only")
                attachments_processed_count = 0
        
        # ✅ Call model API with enhanced attachment detection
        # Log metadata only (no PII/sensitive content)
        body_length = len(clean_body)
        logger.info(f"Sending to model API: body_length={body_length}, has_attachments={has_attachments}, attachments_processed={attachments_processed_count}")
        if "--- ATTACHMENT CONTENT ---" in clean_body:
            logger.info("✅ Attachment content included in body for model classification")
        
        model_response = self.model_api.process_email_complete(
            subject=subject,
            body=clean_body,
            headers=headers,
            sender_email=sender,
            recipient_emails=recipient_emails,
            has_attachments=has_attachments,  # Enhanced detection included
            had_threads=had_threads
        )
        
        # CHECK UNIFIED STOP AFTER MODEL API CALL
        if self.stop_event.is_set():
            logger.info("STOP: Stop signal detected after model API call - stopping NOW")
            return False
        
        # Get model fields
        debtor_number = model_response.get("debtor_number", "")
        company_name = model_response.get("company_name", "")
        invoice_number = model_response.get("invoice_number", "")
        debtor_id = None  # Always null
        event_type = model_response.get("event_type", "uncategorised") 
        target_folder = model_response.get("target_folder", "uncategorised")
        reply_sent = model_response.get("reply_sent", "no_response")
        new_contact_email = model_response.get("new_contact_email", "")
        new_contact_phone = model_response.get("new_contact_phone", "")
        contact_status = model_response.get("contact_status", "active")
        cleaned_body = model_response.get("cleaned_body", "")
        
        # FEATURE 1: Add sender email to cleaned_body (first line in double quotes)
        if sender and cleaned_body:
            cleaned_body = f'sender mail : "{sender}"\n\n{cleaned_body}'
            logger.info(f"Added sender email to cleaned_body: {sender}")
        elif sender and not cleaned_body:
            cleaned_body = f'sender mail : "{sender}"'
            logger.info(f"Set cleaned_body to sender email only: {sender}")
        
        # FEATURE 2: Pad cleaned_body to minimum 256 characters for SSIS compatibility
        # SSIS samples first 8 rows - if all are < 256 chars, it assumes DT_WSTR (255 limit) instead of DT_NTEXT
        if len(cleaned_body) < 256:
            padding_needed = 256 - len(cleaned_body)
            cleaned_body = cleaned_body + " " * padding_needed
            logger.info(f"Padded cleaned_body to 256 characters (added {padding_needed} spaces) for SSIS compatibility")
        
        # Validate event type
        if event_type not in ALLOWED_LABELS:
            event_type = "uncategorised"
            target_folder = "uncategorised"
        
        logger.info(f"Model classified email as: {event_type}")
        
        # ✅ VALIDATION: Check if reply is needed and log if missing
        if event_type in RESPONSE_LABELS:
            logger.info(f"Email requires reply (event_type: {event_type} is in RESPONSE_LABELS)")
        else:
            logger.info(f"Email does not require reply (event_type: {event_type} not in RESPONSE_LABELS)")
        
        # Generate threaded reply if needed
        reply_text = ""
        draft_created = False
        draft_id = None
        invoices_attached = False
        invoice_count = 0
        email_sent_directly = False  # Track if email was sent directly
        
        if event_type in RESPONSE_LABELS:
            # CHECK UNIFIED STOP BEFORE REPLY GENERATION
            if self.stop_event.is_set():
                logger.info("STOP: Stop signal detected before reply generation - stopping NOW")
                return False
            
            # Build entities dict from model response for better context
            entities = {}
            if debtor_number:
                entities["debtor_number"] = debtor_number
            if company_name:
                entities["company_name"] = company_name
            if invoice_number:
                entities["invoice_number"] = invoice_number
            if sender:
                entities["sender_email"] = sender
            if recipient_emails:
                entities["emails"] = recipient_emails
                entities["email"] = recipient_emails[0] if recipient_emails else None
            
            # Determine recipient_email (where reply will be sent - usually the sender)
            reply_recipient_email = recipient_emails[0] if recipient_emails else sender
            
            # Initialize invoice_flow_handled flag
            invoice_flow_handled = False
            
            # SPECIAL HANDLING:
            # invoice_request_no_info can generate TWO replies:
            # 1) acknowledgment (draft OR send)
            # 2) main invoice reply (ONLY created if #1 was actually SENT)
            if event_type == "invoice_request_no_info":
                invoice_flow_handled = True
                # STEP 1: Generate FIRST reply text (acknowledgment template)
                first_reply_text = self.model_api.generate_reply(
                    subject=subject,
                    body=clean_body,
                    label="invoice_request_acknowledgment",
                    sender_name=sender_name,
                    sender_email=sender,
                    recipient_email=reply_recipient_email,
                    entities=entities
                )
                
                # CHECK UNIFIED STOP AFTER REPLY GENERATION
                if self.stop_event.is_set():
                    logger.info("STOP: Stop signal detected after reply generation - stopping NOW")
                    return False
                
                # ✅ VALIDATION: Check if reply was actually generated
                if not first_reply_text or not first_reply_text.strip():
                    logger.error("❌ CRITICAL: Reply generation FAILED for invoice_request_no_info (acknowledgment) - no reply text generated after retries")
                    logger.error(f"   Email: {message_id}, Event Type: {event_type}, Label: invoice_request_acknowledgment")
                    logger.error("   This email requires a reply but none was generated - processing will continue without reply")
                    # Continue processing but without reply
                    first_reply_text = ""
                
                if first_reply_text:
                    logger.info(f"Invoice ack reply generated: {len(first_reply_text)} chars")
                
                # If we're drafting the first email, DO NOT generate the second email
                if not SEND_INVOICE_REQUEST_NO_INFO:
                    reply_text = first_reply_text or ""
                    if reply_text:
                        # ✅ RETRY: Try to create draft up to 3 times
                        draft_id = self._create_draft_with_retry(message_id, reply_text, source_account, "invoice ack")
                        if draft_id:
                            draft_created = True
                            logger.info(f"Invoice ack draft saved (no second email will be created): {draft_id}")
                        else:
                            logger.error(f"❌ CRITICAL: Invoice ack draft NOT created after 3 attempts - email {message_id} will be processed without draft")
                    else:
                        logger.error(f"❌ CRITICAL: No reply text available to create draft for email {message_id}")
                    # Stop here (no second email)
                else:
                    # Send FIRST email directly
                    first_sent = False
                    if first_reply_text:
                        try:
                            success = self.graph_client.send_threaded_reply_directly(
                                message_id, first_reply_text, source_account
                            )
                            if success:
                                first_sent = True
                                # Persist ACK as the response when sent directly,
                                # even if second reply generation fails later.
                                email_sent_directly = True
                                reply_text = first_reply_text or ""
                                logger.info("Invoice acknowledgment email sent directly")
                            else:
                                logger.warning("Invoice ack direct send failed; creating draft and skipping second email")
                                draft_id = self.graph_client.create_threaded_reply_draft(
                                    message_id, first_reply_text, source_account
                                )
                                if draft_id:
                                    draft_created = True
                                    reply_text = first_reply_text
                                    logger.info(f"Invoice ack fallback draft saved (no second email): {draft_id}")
                        except Exception as e:
                            logger.error(f"Error sending invoice ack directly: {e}; creating draft and skipping second email")
                            draft_id = self.graph_client.create_threaded_reply_draft(
                                message_id, first_reply_text, source_account
                            )
                            if draft_id:
                                draft_created = True
                                reply_text = first_reply_text
                                logger.info(f"Invoice ack fallback draft saved (no second email): {draft_id}")
                    
                    # STEP 2: ONLY if first email was SENT, generate SECOND email (main invoice template)
                    # ✅ WAIT: Ensure first draft was created successfully before proceeding
                    if first_sent:
                        # Wait a brief moment to ensure first email is fully processed
                        time.sleep(1)
                        
                        second_reply_text = self.model_api.generate_reply(
                            subject=subject,
                            body=clean_body,
                            label="invoice_request_no_info",
                            sender_name=sender_name,
                            sender_email=sender,
                            recipient_email=reply_recipient_email,
                            entities=entities
                        )
                        
                        # CHECK UNIFIED STOP AFTER REPLY GENERATION
                        if self.stop_event.is_set():
                            logger.info("STOP: Stop signal detected after reply generation - stopping NOW")
                            return False
                        
                        # ✅ VALIDATION: Check if second reply was actually generated
                        if not second_reply_text or not second_reply_text.strip():
                            logger.error("❌ CRITICAL: Second reply generation FAILED for invoice_request_no_info (main) - no reply text generated after retries")
                            logger.error(f"   Email: {message_id}, Event Type: {event_type}, Label: invoice_request_no_info")
                            logger.error("   First email was sent, but second reply could not be generated - will not create second draft")
                            second_reply_text = ""
                        
                        if second_reply_text:
                            logger.info(f"Invoice main reply generated: {len(second_reply_text)} chars")
                            # Store main reply text in MongoDB (most useful one)
                            reply_text = second_reply_text
                            
                            # Always create the second as a draft first (so we can attach invoice if available)
                            # ✅ RETRY: Try to create draft up to 3 times
                            second_draft_id = self._create_draft_with_retry(message_id, second_reply_text, source_account, "invoice main")
                            if second_draft_id:
                                draft_created = True
                                draft_id = second_draft_id  # Persist latest draft id to MongoDB
                                logger.info(f"Invoice main draft created: {second_draft_id}")
                                
                                # Try to fetch invoice(s) and attach (non-blocking if fails)
                                invoice_file_paths: List[str] = []
                                temp_dir = None
                                try:
                                    if company_name and debtor_number:
                                        fetch_result = self.invoice_handler.fetch_invoices(
                                            company_name=company_name,
                                            abcfn_number=debtor_number,
                                            invoice_number=invoice_number or None
                                        )
                                        if fetch_result.success and fetch_result.content and fetch_result.filename:
                                            temp_dir = tempfile.mkdtemp(prefix="invoice_")
                                            safe_name = os.path.basename(fetch_result.filename)
                                            if safe_name in ("", ".", ".."):
                                                safe_name = "invoice_file"
                                            invoice_path = os.path.join(temp_dir, safe_name)
                                            with open(invoice_path, "wb") as f:
                                                f.write(fetch_result.content)
                                            invoice_file_paths = [invoice_path]
                                            logger.info(f"Invoice file prepared for attachment: {fetch_result.filename} ({len(fetch_result.content)} bytes)")
                                        else:
                                            logger.info(f"Invoice fetch skipped/failed (non-blocking): {fetch_result.error}")
                                    else:
                                        logger.info("Invoice fetch skipped: missing company_name or debtor_number")
                                    
                                    if invoice_file_paths:
                                        attached_count = self.graph_client.attach_files_to_draft(
                                            second_draft_id,
                                            from_account=source_account,
                                            file_paths=invoice_file_paths
                                        )
                                        if attached_count > 0:
                                            invoices_attached = True
                                            invoice_count = attached_count
                                            logger.info(f"Attached {attached_count} invoice file(s) to draft {second_draft_id}")
                                except Exception as e:
                                    logger.warning(f"Invoice fetch/attach failed (non-blocking): {e}")
                                finally:
                                    # cleanup temp files
                                    try:
                                        if temp_dir:
                                            for p in invoice_file_paths:
                                                try:
                                                    os.remove(p)
                                                except Exception as e:
                                                    logger.debug(f"Failed to remove temp file {p}: {e}")
                                            try:
                                                os.rmdir(temp_dir)
                                            except Exception as e:
                                                logger.debug(f"Failed to remove temp dir {temp_dir}: {e}")
                                    except Exception as e:
                                        logger.debug(f"Cleanup error: {e}")
                                
                                # If configured, send the second draft (with or without attachment)
                                if SEND_INVOICE_WITH_ATTACHMENT:
                                    try:
                                        sent = self.graph_client.send_existing_draft(
                                            second_draft_id, from_account=source_account
                                        )
                                        if sent:
                                            email_sent_directly = True
                                            logger.info("Invoice main draft sent (after ack sent)")
                                        else:
                                            logger.warning("Failed to send invoice main draft; leaving as draft")
                                    except Exception as e:
                                        logger.error(f"Error sending invoice main draft; leaving as draft: {e}")
                            else:
                                logger.error(f"❌ CRITICAL: Invoice main draft NOT created after 3 attempts - email {message_id} will be processed without second draft")
            else:
                # Default single-reply behavior (e.g. claims_paid_no_proof)
                # invoice_flow_handled is already False (initialized above)
                reply_text = self.model_api.generate_reply(
                    subject=subject,
                    body=clean_body,
                    label=event_type,
                    sender_name=sender_name,
                    sender_email=sender,
                    recipient_email=reply_recipient_email,
                    entities=entities
                )
            
            # CHECK UNIFIED STOP AFTER REPLY GENERATION
            if self.stop_event.is_set():
                logger.info("STOP: Stop signal detected after reply generation - stopping NOW")
                return False
            
            # ✅ VALIDATION: Check if reply was actually generated for non-invoice flows
            if not invoice_flow_handled:
                if not reply_text or not reply_text.strip():
                    logger.error(f"❌ CRITICAL: Reply generation FAILED for {event_type} - no reply text generated after retries")
                    logger.error(f"   Email: {message_id}, Event Type: {event_type}, Label: {event_type}")
                    logger.error("   This email requires a reply but none was generated - processing will continue without reply")
                    reply_text = ""
                
            if reply_text:
                logger.info(f"Reply generated: {len(reply_text)} chars")
                
                # invoice_request_no_info is fully handled above (ack + optional main),
                # so we MUST NOT run the default send/draft logic again.
                if event_type == "invoice_request_no_info" and invoice_flow_handled:
                    pass
                else:
                    # FEATURE 2: Check per-response-type sending configuration - ONLY 2 TYPES
                    should_send_directly = False
                    
                    if event_type == "invoice_request_no_info":
                        should_send_directly = SEND_INVOICE_REQUEST_NO_INFO
                        logger.info(f"Invoice request (no info) - Send directly: {should_send_directly}")
                    elif event_type == "claims_paid_no_proof":
                        should_send_directly = SEND_CLAIMS_PAID_NO_PROOF
                        logger.info(f"Claims paid (no proof) - Send directly: {should_send_directly}")
                    
                    if should_send_directly:
                        # Send email directly using Graph API
                        try:
                            success = self.graph_client.send_threaded_reply_directly(
                                message_id, reply_text, source_account
                            )
                            if success:
                                logger.info(f"Email sent directly for {event_type}")
                                email_sent_directly = True
                            else:
                                logger.warning("Direct send failed, creating draft instead")
                                # ✅ RETRY: Try to create draft up to 3 times
                                draft_id = self._create_draft_with_retry(message_id, reply_text, source_account, "fallback")
                                if draft_id:
                                    draft_created = True
                                else:
                                    logger.error(f"❌ CRITICAL: Fallback draft NOT created after 3 attempts - email {message_id} will be processed without draft")
                        except Exception as e:
                            logger.error(f"Error sending email directly: {e}, creating draft instead")
                            # ✅ RETRY: Try to create draft up to 3 times
                            draft_id = self._create_draft_with_retry(message_id, reply_text, source_account, "fallback")
                            if draft_id:
                                draft_created = True
                            else:
                                logger.error(f"❌ CRITICAL: Fallback draft NOT created after 3 attempts - email {message_id} will be processed without draft")
                    else:
                        # Create draft as usual
                        # ✅ RETRY: Try to create draft up to 3 times
                        draft_id = self._create_draft_with_retry(message_id, reply_text, source_account, "threaded")
                        if draft_id:
                            draft_created = True
                        else:
                            logger.error(f"❌ CRITICAL: Draft NOT created after 3 attempts - email {message_id} will be processed without draft")
                            # draft_created remains False (default)
        
        # CHECK UNIFIED STOP BEFORE MONGODB STORAGE
        if self.stop_event.is_set():
            logger.info("STOP: Stop signal detected before MongoDB storage - stopping NOW")
            return False
        
        # Build email data for MongoDB
        email_data = {
            # Graph API fields
            "message_id": message_id,
            "sender": sender,
            "sender_name": sender_name,
            "recipient": recipient,
            "subject": subject,
            "body": clean_body,
            "received_at": received,
            "has_attachments": has_attachments,  # Enhanced attachment detection included
            "source_account": source_account,
            "conversation_id": conversation_id,
            "receiver_type": receiver_type,
            "cc": cc_string,
            
            # Model API fields
            "debtor_number": debtor_number,
            "company_name": company_name,
            "invoice_number": invoice_number,
            "event_type": event_type,
            "target_folder": target_folder,
            "reply_sent": reply_sent,
            "new_contact_email": new_contact_email,
            "new_contact_phone": new_contact_phone,
            "contact_status": contact_status,
            "cleaned_body": cleaned_body,  # Now includes sender email
            
            # Client fields
            "debtor_id": debtor_id,
            "classification": event_type,
            "prediction": event_type,
            "response": reply_text,
            "response_sent": email_sent_directly,  # True if sent directly, False if draft
            "draft_created": draft_created,
            "draft_id": draft_id,
            "invoices_attached": invoices_attached,
            "invoice_count": invoice_count,
            "attachments_processed_count": attachments_processed_count,  # Number of attachments successfully processed/extracted
            "batch_id": self.batch_id,
            "data_source": data_source,
            "had_threads": had_threads,
            "processed_at": datetime.utcnow().isoformat(),
            "batch_complete": False
        }
        
        # Store in MongoDB
        if self.mongo:
            result = self.mongo.insert_email(email_data)
            if result:
                logger.info(f"Email {message_id} stored in MongoDB")
        
        # CHECK UNIFIED STOP BEFORE FOLDER OPERATIONS
        if self.stop_event.is_set():
            logger.info("STOP: Stop signal detected before folder operations - stopping NOW")
            return False
        
        # Move to folder and track message ID properly
        folder_mapping = self.folder_mappings.get(source_account, {})
        folder_id = folder_mapping.get(event_type)  # Uses event_type (not target_folder)
        msg_id_for_read = message_id  # Start with original ID
        
        if folder_id:
            try:
                success, new_id = self.graph_client.move_email_to_folder(message_id, folder_id, source_account)
                if success and new_id != message_id:
                    msg_id_for_read = new_id  # Use new ID for subsequent operations
                    # Update MongoDB with new ID
                    if self.mongo:
                        self.mongo.update_message_id(message_id, new_id)
            except Exception as e:
                logger.warning(f"Failed to move email: {e}")
        
        return True
            
    def process_batch(self, batch_size):
        """Process a batch of emails - handle ANY number of emails like test code."""
        try:
            # Setup folders for all accounts
            logger.info("Setting up classification folders...")
            for email_address in self.graph_client.email_addresses:
                try:
                    folder_mapping = self.graph_client.ensure_classification_folders(email_address)
                    if folder_mapping:
                        self.folder_mappings[email_address] = folder_mapping
                except httpx.HTTPError as e:
                    logger.error(f"HTTP error setting up folders for {email_address}: {e}", exc_info=True)
                except httpx.TimeoutException as e:
                    logger.error(f"Timeout setting up folders for {email_address}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Failed to setup folders for {email_address}: {e}", exc_info=True)
            
            if not self.folder_mappings:
                logger.error("Could not create folder mappings for any account")
                return False, 0, 0
            
            # Fetch emails
            logger.info(f"Fetching up to {batch_size} emails...")
            try:
                emails = self.graph_client.fetch_unread_emails(batch_size)
            except Exception as e:
                logger.error(f"Failed to fetch emails: {e}", exc_info=True)
                return False, 0, 0
            
            if not emails:
                logger.info("No emails to process - this is normal, not an error")
                # STILL sync to PostgreSQL even with 0 emails (batch tracking)
                if self.mongo:
                    try:
                        synced = self.mongo.sync_batch_emails_to_postgres(self.batch_id)
                        logger.info(f"Synced {synced} emails to PostgreSQL")
                    except Exception as e:
                        logger.error(f"Failed to sync to PostgreSQL: {e}", exc_info=True)
                return True, 0, 0  # Success with 0 emails processed
            
            # Process whatever emails we have (1, 5, 30, 120 - doesn't matter)
            processed = 0
            failed = 0
            
            logger.info(f"Processing {len(emails)} emails...")
            
            for email in emails:
                # CHECK UNIFIED STOP BEFORE PROCESSING EACH EMAIL IN BATCH
                if self.stop_event.is_set():
                    logger.info("STOP: Stop signal detected during batch processing - stopping NOW")
                    break
                    
                try:
                    if self._process_single_email(email):
                        processed += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Error processing email {email.get('id', 'unknown')}: {e}", exc_info=True)
                    failed += 1
                    # CONTINUE PROCESSING OTHER EMAILS - like test code
                    continue
            
            # ALWAYS sync to PostgreSQL (whether we have 1 email or 100)
            if self.mongo:
                try:
                    synced = self.mongo.sync_batch_emails_to_postgres(self.batch_id)
                    logger.info(f"Synced {synced} emails to PostgreSQL")
                except Exception as e:
                    logger.error(f"Failed to sync to PostgreSQL: {e}", exc_info=True)
            
            logger.info(f"Batch {self.batch_id} complete: {processed} processed, {failed} failed")
            return True, processed, failed
        except Exception as e:
            logger.exception(f"CRITICAL: process_batch failed with unhandled exception: {e}")
            return False, 0, 0

def process_unread_emails(batch_id, batch_size=30, stop_event=None):
    """Process unread emails with UNIFIED stop signal."""
    processor = EmailProcessor(batch_id, stop_event)
    success, processed, failed = processor.process_batch(batch_size)
    
    return {
        "success": success,
        "emails_processed": processed,
        "emails_errored": failed,
        "batch_id": batch_id
    }

def get_failed_batches():
    """Get list of failed batch IDs that need retry."""
    mongo = get_mongo()
    if not mongo:
        return []
        
    # Find failed or incomplete batches
    failed_query = {
        "$or": [
            {"status": "failed"},
            {"status": "in_progress", "created_at": {"$lt": datetime.utcnow().replace(hour=datetime.utcnow().hour-1)}}
        ],
        "permanently_failed": {"$ne": True}
    }
    
    failed_batches = list(mongo.batch_runs_collection.find(failed_query, {"id": 1}))
    batch_ids = [batch.get("id") for batch in failed_batches if batch.get("id")]
    
    logger.info(f"Found {len(batch_ids)} failed batches to retry")
    return batch_ids

def retry_failed_batch(batch_id, batch_size=30):
    """Retry a specific failed batch."""
    logger.info(f"Retrying failed batch: {batch_id}")
    
    # Process the batch
    result = process_unread_emails(batch_id, batch_size)
    
    if result["success"]:
        # Update batch status
        mongo = get_mongo()
        if mongo:
            mongo.update_batch_result(
                batch_id,
                result["emails_processed"],
                result["emails_errored"],
                0,
                "success"
            )
        
        PostgresHelper.update_batch_result(
            batch_id,
            result["emails_processed"],
            result["emails_errored"],
            "success"
        )
        
        logger.info(f"Successfully retried batch {batch_id}")
        return True
    else:
        logger.warning(f"Retry failed for batch {batch_id}")
        return False

def generate_daily_report():
    """
    Generate daily report at 12:00 AM ET for the previous day.
    Shows ALL configured mailboxes + misclassification count from 'AI Agent Issues' folder.
    """
    from datetime import datetime, timedelta
    import pytz
    import httpx

    # ADD TIME CHECK FIRST - Only run between 12:00 AM and 12:59 AM EST
    et_tz = pytz.timezone("US/Eastern")
    now_et = datetime.now(et_tz)
    
    # Only run between 12:00 AM and 12:59 AM EST
    if now_et.hour != 0:
        logger.info(f"Report generation skipped - current time: {now_et.strftime('%I:%M %p %Z')} (only runs at 12 AM EST)")
        return False
    
    logger.info(f"Report generation starting at: {now_et.strftime('%I:%M %p %Z')}")

    # Report recipients
    report_emails = [
        "sanskar.gawande@cadex-solutions.com",
        "yogesh.patel@cadex-solutions.com",
        "susan.orzech@abc-amega.com"
    ]

    # Misclassification email configuration - Updated to use correct mailbox
    MISCLASSIFICATION_MAILBOX = "ABCCollectionsTeamDTest@abc-amega.com"

    try:
        # Timezone setup (now_et already defined above)
        yesterday_et = now_et - timedelta(days=1)

        start_date_et = yesterday_et.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date_et = start_date_et + timedelta(days=1)

        start_date_utc = start_date_et.astimezone(pytz.UTC).replace(tzinfo=None)
        end_date_utc = end_date_et.astimezone(pytz.UTC).replace(tzinfo=None)

        logger.info(f"Generating daily report for: {yesterday_et.strftime('%Y-%m-%d')} ET")
        logger.info(f"UTC query range: {start_date_utc} to {end_date_utc}")

        # Get MongoDB connection
        mongo = get_mongo()
        if not mongo:
            logger.error("No MongoDB connection for report")
            return False

        # Query emails processed yesterday
        query = {
            "processed_at": {
                "$gte": start_date_utc.isoformat(),
                "$lt": end_date_utc.isoformat()
            }
        }

        emails = list(mongo.collection.find(query))
        total_emails = len(emails)

        # Get ALL configured mailboxes (even if no emails)
        configured_mailboxes = EMAIL_ADDRESS.split(",")
        configured_mailboxes = [email.strip() for email in configured_mailboxes]
        
        logger.info(f"Configured mailboxes: {len(configured_mailboxes)}")
        for mailbox in configured_mailboxes:
            logger.info(f"  - {mailbox}")

        # FETCH MISCLASSIFICATION COUNT - From 'AI Agent Issues' folder
        logger.info("Fetching misclassification count from 'AI Agent Issues' folder...")
        misclassification_count = fetch_misclassification_count(
            MISCLASSIFICATION_MAILBOX,
            start_date_utc,
            end_date_utc
        )

        # Separate emails by source account
        emails_by_account = {}
        
        # Initialize ALL configured mailboxes with empty lists
        for mailbox in configured_mailboxes:
            emails_by_account[mailbox] = []
        
        # Add emails to their respective mailboxes
        for email in emails:
            account = email.get("source_account", "unknown")
            if account in emails_by_account:
                emails_by_account[account].append(email)
            else:
                # Handle emails from unconfigured mailboxes (legacy data)
                if account not in emails_by_account:
                    emails_by_account[account] = []
                emails_by_account[account].append(email)

        # Start building HTML TABLE report
        report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .total-row {{ background-color: #f9f9f9; font-weight: bold; }}
        .center {{ text-align: center; }}
        .number {{ text-align: right; }}
        h1, h2 {{ color: #333; }}
        .summary-box {{ 
            background-color: #f8f9fa; 
            border: 1px solid #dee2e6; 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 5px; 
        }}
    </style>
</head>
<body>

<h1>Daily Email Processing Report - {yesterday_et.strftime('%Y-%m-%d')} (Eastern Time)</h1>

<div class="summary-box">
    <h2>📊 OVERALL SUMMARY</h2>
    <table>
        <tr><th>Metric</th><th class="number">Value</th></tr>
        <tr><td>Total Emails Processed</td><td class="number">{total_emails}</td></tr>
        <tr><td>Configured Mailboxes</td><td class="number">{len(configured_mailboxes)}</td></tr>
        <tr><td>Misclassification Reports</td><td class="number">{misclassification_count}</td></tr>
        <tr><td>Report Coverage</td><td class="center">12:00 AM - 11:59 PM ET</td></tr>
    </table>
</div>

<h2>📧 EMAIL PROCESSING BY MAILBOX</h2>
<table>
    <thead>
        <tr>
            <th>Mailbox</th>
            <th class="center">Total Emails</th>
            <th class="center">Replies Generated</th>
            <th class="center">Drafts Created</th>
            <th class="center">Response Rate</th>
            <th class="center">Draft Rate</th>
        </tr>
    </thead>
    <tbody>
"""

        # Process each mailbox for summary table
        total_all_emails = 0
        total_all_replies = 0
        total_all_drafts = 0

        for account in sorted(emails_by_account.keys()):
            account_emails = emails_by_account[account]
            account_total = len(account_emails)
            
            # Count statistics for this mailbox
            replies_generated = 0
            drafts_created = 0
            
            for email in account_emails:
                if email.get("response", ""):
                    replies_generated += 1
                if email.get("draft_created", False):
                    drafts_created += 1
            
            # Calculate rates
            response_rate = (replies_generated / account_total * 100) if account_total > 0 else 0
            draft_rate = (drafts_created / account_total * 100) if account_total > 0 else 0
            
            # Add to totals
            total_all_emails += account_total
            total_all_replies += replies_generated
            total_all_drafts += drafts_created
            
            # Add row to table
            report_html += f"""
        <tr>
            <td>{account}</td>
            <td class="number">{account_total}</td>
            <td class="number">{replies_generated}</td>
            <td class="number">{drafts_created}</td>
            <td class="number">{response_rate:.1f}%</td>
            <td class="number">{draft_rate:.1f}%</td>
        </tr>"""

        # Add total row
        total_response_rate = (total_all_replies / total_all_emails * 100) if total_all_emails > 0 else 0
        total_draft_rate = (total_all_drafts / total_all_emails * 100) if total_all_emails > 0 else 0

        report_html += f"""
        <tr class="total-row">
            <td><strong>TOTAL</strong></td>
            <td class="number"><strong>{total_all_emails}</strong></td>
            <td class="number"><strong>{total_all_replies}</strong></td>
            <td class="number"><strong>{total_all_drafts}</strong></td>
            <td class="number"><strong>{total_response_rate:.1f}%</strong></td>
            <td class="number"><strong>{total_draft_rate:.1f}%</strong></td>
        </tr>
    </tbody>
</table>
"""

        # Add detailed classification breakdown TABLE
        report_html += """
<h2>🏷️ DETAILED CLASSIFICATION BREAKDOWN</h2>
<table>
    <thead>
        <tr>
            <th>Mailbox</th>
            <th>Classification</th>
            <th class="center">Count</th>
            <th class="center">Percentage</th>
        </tr>
    </thead>
    <tbody>
"""

        # Process detailed classification for each mailbox
        for account in sorted(emails_by_account.keys()):
            account_emails = emails_by_account[account]
            account_total = len(account_emails)
            
            if account_total == 0:
                report_html += f"""
        <tr>
            <td>{account}</td>
            <td colspan="3" class="center"><em>No emails processed</em></td>
        </tr>"""
                continue
            
            # Count classifications for this mailbox
            label_counts = {}
            for email in account_emails:
                label = email.get("event_type", "unknown")
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # Add classification rows
            first_row = True
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / account_total * 100) if account_total > 0 else 0
                label_display = label.replace('_', ' ').title()
                
                mailbox_cell = account if first_row else ""
                first_row = False
                
                report_html += f"""
        <tr>
            <td>{mailbox_cell}</td>
            <td>{label_display}</td>
            <td class="number">{count}</td>
            <td class="number">{percentage:.1f}%</td>
        </tr>"""

        report_html += """
    </tbody>
</table>
"""

        # Add footer
        report_html += f"""
<div class="summary-box">
    <h2>📋 REPORT INFORMATION</h2>
    <table>
        <tr><th>Report Date</th><td>{yesterday_et.strftime('%Y-%m-%d')} (Eastern Time)</td></tr>
        <tr><th>Report Generated</th><td>{now_et.strftime('%Y-%m-%d %I:%M:%S %p %Z')}</td></tr>
        <tr><th>Coverage Period</th><td>12:00 AM - 11:59 PM ET</td></tr>
        <tr><th>Total Processing Mailboxes</th><td>{len(configured_mailboxes)}</td></tr>
        <tr><th>Misclassification Monitoring</th><td>{MISCLASSIFICATION_MAILBOX} (AI Agent Issues folder)</td></tr>
    </table>
</div>

</body>
</html>
"""

        logger.info(f"Report ready: {total_emails} emails processed, {misclassification_count} misclassifications")

        # Send HTML email using Graph API
        return send_enhanced_report_email(report_html, yesterday_et.strftime('%Y-%m-%d'), report_emails)

    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        return False

def fetch_misclassification_count(misclassification_mailbox, start_date_utc, end_date_utc):
    """
    Fetch total count of emails in the 'AI Agent Issues' folder during the time period.
    Accesses ABCCollectionsTeamDTest@abc-amega.com mailbox and finds the specific folder.
    """
    try:
        # Create MSGraphClient to access the misclassification mailbox
        graph_client = MSGraphClient()
        access_token = graph_client.get_access_token()
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Step 1: Find the "AI Agent Issues" folder
        logger.info(f"Finding 'AI Agent Issues' folder in {misclassification_mailbox}")
        folders_url = f"{MS_GRAPH_BASE_URL}/users/{misclassification_mailbox}/mailFolders"
        folders_response = httpx.get(folders_url, headers=headers, timeout=60)
        
        if folders_response.status_code != 200:
            logger.error(f"Failed to fetch folders: {folders_response.status_code}")
            return 0
        
        folders = folders_response.json().get("value", [])
        ai_agent_folder_id = None
        
        # Search for "AI Agent Issues" folder (case-insensitive)
        for folder in folders:
            folder_name = folder.get("displayName", "").strip()
            if folder_name.lower() == "ai agent issues":
                ai_agent_folder_id = folder.get("id")
                logger.info(f"Found 'AI Agent Issues' folder with ID: {ai_agent_folder_id}")
                break
        
        if not ai_agent_folder_id:
            logger.warning(f"'AI Agent Issues' folder not found in {misclassification_mailbox}")
            return 0
        
        # Step 2: Convert UTC dates to ISO format for Graph API
        start_iso = start_date_utc.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        end_iso = end_date_utc.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        
        # Step 3: Fetch emails from the specific folder with date filtering
        params = {
            "$filter": f"receivedDateTime ge {start_iso} and receivedDateTime lt {end_iso}",
            "$select": "id",  # Only need count, not content
            "$top": 1000  # Limit to prevent timeout
        }
        
        folder_emails_url = f"{MS_GRAPH_BASE_URL}/users/{misclassification_mailbox}/mailFolders/{ai_agent_folder_id}/messages"
        
        logger.info("Fetching emails from 'AI Agent Issues' folder")
        logger.info(f"Time range: {start_iso} to {end_iso}")
        
        response = httpx.get(folder_emails_url, headers=headers, params=params, timeout=60)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch misclassification emails from folder: {response.status_code} - {response.text}")
            return 0
        
        emails = response.json().get("value", [])
        total_count = len(emails)
        
        logger.info(f"Found {total_count} misclassification emails in 'AI Agent Issues' folder")
        
        return total_count
        
    except Exception as e:
        logger.error(f"Error fetching misclassification count: {e}")
        return 0

def send_enhanced_report_email(report_html, report_date, recipients):
    """Send enhanced daily report email with HTML table format."""
    try:
        # Create MSGraphClient to get access token
        graph_client = MSGraphClient()
        access_token = graph_client.get_access_token()
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Enhanced email formatting
        to_recipients = [{"emailAddress": {"address": email}} for email in recipients]
        subject = f"📊 Daily Email Processing Report - {report_date} (ET) - Table Format"
        
        message = {
            "subject": subject,
            "body": {
                "contentType": "HTML", 
                "content": report_html
            },
            "toRecipients": to_recipients,
            "importance": "Normal"
        }
        
        payload = {"message": message, "saveToSentItems": "true"}
        endpoint = f"{MS_GRAPH_BASE_URL}/users/{EMAIL_ADDRESS.split(',')[0].strip()}/sendMail"
        
        response = httpx.post(endpoint, headers=headers, json=payload, timeout=30)
        
        if response.status_code in [200, 202]:
            logger.info(f"Enhanced HTML table report sent to {len(recipients)} recipients")            
            return True
        else:
            logger.error(f"Failed to send report: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending report email: {e}")
        return False