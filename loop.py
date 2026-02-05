"""
loop.py - Batch processing and orchestration for email classification system

Updated with UNIFIED signal handling for Issue #6:
1. Orchestration and scheduling
2. Excel export using model data  
3. Database batch management
4. No email_sender dependencies (replies handled in fetch_reply.py)
5. Simple draft counting from MongoDB
6. UNIFIED threading-based stop mechanism
"""

import os
import sys
import time
import uuid
import io
import httpx
import msal
import pytz
import threading
import pandas as pd
import requests 
import paramiko 
from paramiko import SSHClient
from scp import SCPClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Tuple, Dict, List, Optional, Any

# Import from refactored modules
from src.fetch_reply import process_unread_emails, generate_daily_report, SEND_INVOICE_REQUEST_NO_INFO, SEND_CLAIMS_PAID_NO_PROOF
from src.db import get_mongo, get_postgres, PostgresConnector
from src.log_config import logger

# Load environment variables
load_dotenv()

# Configuration - Hardcoded settings (no DevOps dependency)
_last_report_date = None

# Export format configuration: "csv" (default) or "excel"
EXPORT_FORMAT = "csv"
# Outbound event rows configuration: Only create duplicate "_with_reply" rows if email was actually sent
# Set to True to enable outbound event rows (only for actually sent emails), False to disable completely
ENABLE_OUTBOUND_EVENT_ROWS = False
# SFTP Configuration
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_PORT = int(os.getenv("SFTP_PORT", "22"))
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
SFTP_ENABLED = os.getenv("SFTP_ENABLED", "False").lower() in ["true", "yes", "1"]
# Environment variables for Microsoft Graph API (needed for alerts)
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TENANT_ID = os.getenv("TENANT_ID")
EMAIL_ADDRESS = "ABCcollectionsteamD@abc-amega.com,ABCcollectionsteamI@abc-amega.com,ABCcollectionsteamJ@abc-amega.com"

def check_model_health(max_retries: int = 3, base_timeout: int = 60) -> bool:
    """Robust model health check with retries and backoff."""
    model_url = "http://104.197.197.76:8000/api/health"

    for attempt in range(max_retries):
        try:
            # Increase timeout per retry: 60s → 120s → 180s
            timeout = base_timeout * (attempt + 1)
            response = requests.get(model_url, timeout=timeout)

            if response.status_code == 200:
                logger.info(f"Model health check passed on attempt {attempt+1}")
                return True
            else:
                logger.warning(f"Model health check failed (HTTP {response.status_code})")

        except requests.exceptions.Timeout:
            logger.warning(f"Model health check timeout (attempt {attempt+1}/{max_retries}, timeout={timeout}s)")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Model health check error (attempt {attempt+1}/{max_retries}): {str(e)}")

        # Exponential backoff: 5s, 10s, 20s
        wait_time = 5 * (2 ** attempt)
        logger.info(f"Retrying health check in {wait_time}s...")
        time.sleep(wait_time)

    logger.error("Model health check failed after all retries")
    return False


def get_batch_size():
    """Get batch size - hardcoded to 50"""
    return 50

def get_batch_interval():
    """Get batch interval in seconds (convert minutes to seconds)"""
    minutes = 60
    return minutes * 60  # Convert to seconds

def wait_for_model_recovery(stop_event, max_retries=3):
    """Wait for model to come back online - UNIFIED stop signal."""
    logger.info(f"Waiting for model to recover (max {max_retries} attempts)...")
    
    for attempt in range(max_retries):
        # Check unified stop signal
        if stop_event.is_set():
            logger.info("STOP: Stop signal detected during model recovery - stopping NOW")
            return False
            
        if check_model_health():
            logger.info(f"Model recovered after {attempt + 1} attempts - resuming processing")
            return True
        else:
            # Exponential backoff: 60s, 120s, 240s
            wait_time = 60 * (2 ** attempt)
            logger.info(f"Model still down - waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})")
            
            # Check stop signal during wait with 1-second intervals
            for i in range(wait_time):
                if stop_event.is_set():
                    logger.info("STOP: Stop signal detected during model recovery wait - stopping NOW")
                    return False
                time.sleep(1)
    
    logger.error(f"Model failed to recover after {max_retries} attempts - giving up")
    return False

def check_incomplete_batch():
    """Check if there's an incomplete batch to resume"""
    mongo = get_mongo()
    if not mongo:
        return None
        
    # Find batch that has emails but isn't marked as complete
    pipeline = [
        {
            "$group": {
                "_id": "$batch_id",
                "total_emails": {"$sum": 1},
                "batch_complete": {"$first": "$batch_complete"}
            }
        },
        {
            "$match": {
                "_id": {"$ne": None},
                "batch_complete": {"$ne": True}
            }
        },
        {"$sort": {"_id": -1}},
        {"$limit": 1}
    ]
    
    result = list(mongo.collection.aggregate(pipeline))
    if result:
        batch_id = result[0]["_id"]
        email_count = result[0]["total_emails"]
        logger.info(f"Found incomplete batch {batch_id} with {email_count} emails")
        return batch_id
        
    return None

def mark_batch_complete(batch_id: str):
    """Mark a batch as complete for Excel generation"""
    mongo = get_mongo()
    if mongo:
        # Mark all emails in this batch as batch_complete
        result = mongo.collection.update_many(
            {"batch_id": batch_id},
            {"$set": {"batch_complete": True}}
        )
        logger.info(f"Marked batch {batch_id} complete: {result.modified_count} emails")

def get_batch_email_count(batch_id: str) -> int:
    """Get current email count in batch"""
    mongo = get_mongo()
    if mongo:
        count = mongo.collection.count_documents({"batch_id": batch_id})
        return count
    return 0

def count_drafts_created(batch_id: str) -> int:
    """Count how many drafts were created for this batch - like test code"""
    mongo = get_mongo()
    if mongo:
        draft_count = mongo.collection.count_documents({
            "batch_id": batch_id,
            "draft_created": True
        })
        return draft_count
    return 0

def should_continue_batch(batch_id: str, target_size: int = 120) -> bool:
    """Check if batch should continue or is complete"""
    current_count = get_batch_email_count(batch_id)
    
    # If we have target_size emails, batch is full
    if current_count >= target_size:
        logger.info(f"Batch {batch_id} is full: {current_count}/{target_size} emails")
        mark_batch_complete(batch_id)
        return False
    
    logger.info(f"Batch {batch_id} continuing: {current_count}/{target_size} emails")
    return True

def ensure_batch_record_exists(batch_id: str) -> bool:
    """Ensure batch record exists in both PostgreSQL and MongoDB."""
    if not batch_id:
        return False
        
    # Check PostgreSQL
    pg_conn = get_postgres()
    if not pg_conn:
        return False
        
    try:
        with pg_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM core.batch_runs WHERE id = %s", (batch_id,))
            if cur.fetchone() is None:
                # Insert if not exists
                cur.execute(
                    """
                    INSERT INTO core.batch_runs (id, status, created_at, response_processed, mail_sent)
                    VALUES (%s, %s, NOW(), %s, %s)
                    """,
                    (batch_id, "in_progress", False, False)
                )
                pg_conn.commit()
                logger.info(f"Created missing batch record in PostgreSQL: {batch_id}")
    finally:
        if pg_conn:
            PostgresConnector.return_connection(pg_conn)
    
    # Check MongoDB
    mongo = get_mongo()
    if not mongo:
        return False
        
    batch = mongo.batch_runs_collection.find_one({"id": batch_id})
    if not batch:
        mongo.batch_runs_collection.insert_one({
            "id": batch_id,
            "status": "in_progress",
            "created_at": datetime.utcnow(),
            "retry_count": 0,
            "processed_count": 0,
            "failed_count": 0,
            "draft_count": 0,
            "permanently_failed": False
        })
        logger.info(f"Created missing batch record in MongoDB: {batch_id}")
        
    return True

def update_batch_id_only(batch_id, email_data=None):
    """Insert a record with batch_id and real email data in PostgreSQL."""
    if not batch_id:
        return 0

    conn = None
    try:
        conn = get_postgres()
        if not conn:
            return 0
            
        conn.autocommit = True
        with conn.cursor() as cur:
            # Use real email data when available
            if email_data and isinstance(email_data, dict):
                to_email = email_data.get('recipient', '')
                from_email = email_data.get('sender', '')
                email_subject = email_data.get('subject', '')
                is_sent = email_data.get('is_sent', False)
                debtor_number = email_data.get('debtor_number', '')
                debtor_id = email_data.get('debtor_id', None)
            else:
                # If no email_data is provided, use minimal values
                to_email = ''
                from_email = 'system@abc-amega.com'
                email_subject = ''
                is_sent = False
                debtor_number = ''
                debtor_id = None
            
            # Insert with appropriate values including model data
            cur.execute(
                """
                INSERT INTO core.account_email
                       (batch_id, to_email, from_email, email_subject, is_sent, debtor_number, debtor_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (batch_id, to_email, from_email, email_subject, is_sent, debtor_number, debtor_id)
            )
            
            logger.info(f"Successfully inserted batch tracking record: batch_id={batch_id}")
            return 1
            
    except Exception as exc:
        logger.error(f"Error in update_batch_id_only: {str(exc)}")
        return 0
    finally:
        if conn:
            PostgresConnector.return_connection(conn)

def upload_to_sftp(filename: str, file_content: Optional[bytes] = None, 
                  max_retries: int = 3, retry_delay: int = 5, stop_event=None) -> bool:
    """Upload file to SFTP server with UNIFIED stop signal checks."""
    if not SFTP_ENABLED:
        logger.info(f"SFTP disabled - skipping upload of {filename}")
        return False
    
    # Create default stop event if not provided
    if stop_event is None:
        stop_event = threading.Event()
    
    # CHECK UNIFIED STOP BEFORE SFTP
    if stop_event.is_set():
        logger.info("STOP: Stop signal detected before SFTP upload - stopping NOW")
        return False
    
    # Create unique filename to prevent overwrites
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uuid_str = uuid.uuid4().hex[:8]
    base_name, extension = os.path.splitext(filename)
    remote_name = f"{base_name}_{timestamp}_{uuid_str}{extension}"
    
    logger.info(f"Generated unique filename: {remote_name}")
    
    # Create temp file if content is provided
    temp_filename = None
    if file_content:
        temp_filename = f"/tmp/{remote_name}"
        with open(temp_filename, "wb") as f:
            f.write(file_content)
    
    retries = 0
    while retries < max_retries:
        # CHECK UNIFIED STOP DURING SFTP RETRIES
        if stop_event.is_set():
            logger.info("STOP: Stop signal detected during SFTP retry - stopping NOW")
            # Clean up temp file
            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)
            return False
        
        ssh = None
        sftp = None
        try:
            ssh = SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            logger.info(f"Connecting to SFTP server {SFTP_HOST}:{SFTP_PORT}")
            ssh.connect(
                SFTP_HOST,
                port=SFTP_PORT,
                username=SFTP_USERNAME,
                password=SFTP_PASSWORD,
                timeout=60,
                look_for_keys=False,
                allow_agent=False
            )
            
            # CHECK UNIFIED STOP AFTER SFTP CONNECTION
            if stop_event.is_set():
                logger.info("STOP: Stop signal detected during SFTP connection - stopping NOW")
                if temp_filename and os.path.exists(temp_filename):
                    os.remove(temp_filename)
                return False
            
            # Try SFTP first
            try:
                transport = ssh.get_transport()
                transport.set_keepalive(30)
                sftp = transport.open_sftp_client()
                
                source_path = temp_filename if temp_filename else filename
                logger.info(f"Uploading file {filename} to {remote_name} via SFTP")
                sftp.put(source_path, remote_name)
                logger.info(f"Successfully uploaded {filename} to SFTP server as {remote_name}")
                
                # Clean up temp file
                if temp_filename and os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    
                return True
                
            except Exception as sftp_err:
                logger.info(f"SFTP failed: {str(sftp_err)}, trying SCP")
                # Fall back to SCP
                with SCPClient(ssh.get_transport(), socket_timeout=60) as scp:
                    source_path = temp_filename if temp_filename else filename
                    scp.put(source_path, remote_name)
                    logger.info(f"Successfully uploaded {filename} via SCP")
                    
                    if temp_filename and os.path.exists(temp_filename):
                        os.remove(temp_filename)
                        
                    return True
                
        except paramiko.AuthenticationException as e:
            logger.error(f"Authentication failed: {str(e)}")
            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)
            return False
            
        except Exception as e:
            logger.info(f"SFTP upload error (attempt {retries+1}/{max_retries}): {str(e)}")
            
        finally:
            if sftp:
                sftp.close()
            if ssh:
                ssh.close()
        
        retries += 1
        if retries < max_retries:
            current_delay = retry_delay * (2 ** (retries - 1))
            logger.info(f"Retrying SFTP upload in {current_delay} seconds...")
            
            # CHECK UNIFIED STOP DURING SFTP RETRY DELAY
            for i in range(current_delay):
                if stop_event.is_set():
                    logger.info("STOP: Stop signal detected during SFTP retry delay - stopping NOW")
                    if temp_filename and os.path.exists(temp_filename):
                        os.remove(temp_filename)
                    return False
                time.sleep(1)
    
    # Clean up temp file if it still exists
    if temp_filename and os.path.exists(temp_filename):
        os.remove(temp_filename)
            
    logger.info(f"Failed to upload {filename} to SFTP after {max_retries} attempts")
    return False

def export_processed_emails_to_excel(batch_id: str, stop_event=None) -> Optional[str]:
    """Export processed emails to CSV or Excel using MODEL DATA with UNIFIED stop signal and formula injection protection.

    NOTE: Actual format is controlled by EXPORT_FORMAT config ("csv" or "excel").
    """
    if not batch_id:
        return None
    
    # Create default stop event if not provided
    if stop_event is None:
        stop_event = threading.Event()

    mongo = get_mongo()
    if not mongo:
        return None
        
    emails = list(mongo.collection.find({"batch_id": batch_id}))
    
    if not emails:
        logger.info(f"No emails for batch {batch_id} to export")
        return None
    
    if not SFTP_ENABLED:
        logger.info("SFTP disabled - export not generated")
        return None

    rows = []
    
    for e in emails:
        # Get email body (no Excel cell length limit enforced here)
        email_body = e.get("body", "") or e.get("text", "")
        
        # Get cleaned body from model (like test code, no Excel cell length limit enforced here)
        cleaned_body = e.get("cleaned_body", "")
        
        # Pad cleaned_body to minimum 256 characters for SSIS compatibility
        # SSIS samples first 8 rows - if all are < 256 chars, it assumes DT_WSTR (255 limit) instead of DT_NTEXT
        if len(cleaned_body) < 256:
            padding_needed = 256 - len(cleaned_body)
            cleaned_body = cleaned_body + " " * padding_needed
        
        # Get RAW values directly
        event_type_raw = e.get("event_type", "")
        target_folder_raw = e.get("target_folder", "")
        reply_sent = e.get("reply_sent", "no_response")
        
        # Get AI generated reply text
        reply_text = e.get("response", "")
        
        # SECURITY FIX: Excel formula injection protection - sanitize all string fields
        sender = e.get("sender", "")
        if sender and sender.startswith(('=', '@', '+', '-', '\t', '\r')):
            sender = "'" + sender
            
        recipient = e.get("recipient", e.get("to", ""))
        if recipient and recipient.startswith(('=', '@', '+', '-', '\t', '\r')):
            recipient = "'" + recipient
            
        subject = e.get("subject", "")
        if subject and subject.startswith(('=', '@', '+', '-', '\t', '\r')):
            subject = "'" + subject
            
        if email_body and email_body.startswith(('=', '@', '+', '-', '\t', '\r')):
            email_body = "'" + email_body
            
        if event_type_raw and event_type_raw.startswith(('=', '@', '+', '-', '\t', '\r')):
            event_type_raw = "'" + event_type_raw
            
        if target_folder_raw and target_folder_raw.startswith(('=', '@', '+', '-', '\t', '\r')):
            target_folder_raw = "'" + target_folder_raw
            
        if reply_sent and reply_sent.startswith(('=', '@', '+', '-', '\t', '\r')):
            reply_sent = "'" + reply_sent
            
        if cleaned_body and cleaned_body.startswith(('=', '@', '+', '-', '\t', '\r')):
            cleaned_body = "'" + cleaned_body
            
        # Sanitize reply text
        if reply_text and reply_text.startswith(('=', '@', '+', '-', '\t', '\r')):
            reply_text = "'" + reply_text
            
        debtor_number = e.get("debtor_number", "")
        # Force debtor_number (ABCFN) as STRING to prevent numeric conversion (e.g. 2877928.0)
        if debtor_number:
            debtor_number = str(debtor_number).rstrip('.0').rstrip('.')
        else:
            debtor_number = ""
        # Protect against formula injection even in CSV (Excel import)
        if debtor_number and debtor_number.startswith(('=', '@', '+', '-', '\t', '\r')):
            debtor_number = "'" + debtor_number
            
        new_contact_email = e.get("new_contact_email", "")
        if new_contact_email and new_contact_email.startswith(('=', '@', '+', '-', '\t', '\r')):
            new_contact_email = "'" + new_contact_email
            
        new_contact_phone = e.get("new_contact_phone", "")
        if new_contact_phone and new_contact_phone.startswith(('=', '@', '+', '-', '\t', '\r')):
            new_contact_phone = "'" + new_contact_phone
            
        contact_status = e.get("contact_status", "active")
        if contact_status and contact_status.startswith(('=', '@', '+', '-', '\t', '\r')):
            contact_status = "'" + contact_status
        
        # Create original row
        row = {
            # CLIENT DATA - Basic email metadata (SANITIZED)
            "EmailFrom": sender,
            "EmailTo": recipient,
            "SubjectLine": subject,
            "Date": e.get("received_at", e.get("date", "")),  # Date fields are safe
            "Body": email_body,
            
            # MODEL DATA - RAW format (SANITIZED)
            "Event Type": event_type_raw,
            "TargetFolder": target_folder_raw,
            "ReplySent": reply_sent,
            "CleanedBody": cleaned_body,
            "PrimaryFileNumber": debtor_number,
            "NewContactEmail": new_contact_email,
            "NewContactPhone": new_contact_phone,
            "ContactStatus": contact_status
        }
        
        # Add original row
        rows.append(row)
        
        # Check if this email needs a duplicate row with reply
        has_ai_reply = bool(reply_text.strip())
        needs_reply_duplicate = event_type_raw in ["invoice_request_no_info", "claims_paid_no_proof"]
        
        # Only create duplicate "_with_reply" row if:
        # 1. Outbound event rows are enabled (ENABLE_OUTBOUND_EVENT_ROWS = True)
        # 2. Email has AI reply
        # 3. Event type needs duplicate
        # 4. Email was actually sent (response_sent = True), not just a draft
        email_was_sent = e.get("response_sent", False)
        
        if ENABLE_OUTBOUND_EVENT_ROWS and has_ai_reply and needs_reply_duplicate and email_was_sent:
            # Email was sent - create duplicate row for outbound event
            # Create cleaned body with only reply text
            enhanced_cleaned_body = f"Reply:\n{reply_text}"
            
            # Truncate if too long for Excel
            if len(enhanced_cleaned_body) > 32767:
                enhanced_cleaned_body = enhanced_cleaned_body[:32764] + "..."
            
            # Pad enhanced_cleaned_body to minimum 256 characters for SSIS compatibility
            if len(enhanced_cleaned_body) < 256:
                padding_needed = 256 - len(enhanced_cleaned_body)
                enhanced_cleaned_body = enhanced_cleaned_body + " " * padding_needed
            
            # Sanitize enhanced cleaned body
            if enhanced_cleaned_body and enhanced_cleaned_body.startswith(('=', '@', '+', '-', '\t', '\r')):
                enhanced_cleaned_body = "'" + enhanced_cleaned_body
            
            # Create duplicate row with modified fields
            duplicate_row = row.copy()  # Copy all fields from original
            duplicate_row["Event Type"] = f"{event_type_raw}_with_reply"  # Add _with_reply suffix
            duplicate_row["CleanedBody"] = enhanced_cleaned_body  # Enhanced cleaned body
            
            # Add duplicate row
            rows.append(duplicate_row)
            
            logger.info(f"Created duplicate entry for {event_type_raw} with AI reply")

    df = pd.DataFrame(rows)
    
    # Ensure PrimaryFileNumber column remains STRING and strip any trailing ".0"
    if "PrimaryFileNumber" in df.columns:
        df["PrimaryFileNumber"] = df["PrimaryFileNumber"].astype(str).str.rstrip('.0').str.rstrip('.')
        # Replace string "nan" that can appear after astype(str)
        df["PrimaryFileNumber"] = df["PrimaryFileNumber"].replace('nan', '')
    
    # Column order - 13 columns (added CleanedBody like test code)
    cols = [
        "EmailFrom", "EmailTo", "SubjectLine", "Date",
        "Event Type", "TargetFolder", "ReplySent", "Body", "CleanedBody",
        "PrimaryFileNumber", "NewContactEmail", "NewContactPhone", "ContactStatus"
    ]
    
    # Ensure all columns exist
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    
    df = df[cols]
    
    # Generate filename based on configured export format
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_suffix = batch_id[-8:] if batch_id else uuid.uuid4().hex[:8]
    use_excel = EXPORT_FORMAT == "excel"
    extension = "xlsx" if use_excel else "csv"
    fname = f"AI_Agent_Data_Load_{ts}_{batch_suffix}.{extension}"
    
    # Log export format being used
    logger.info(f"Export format configured: {EXPORT_FORMAT.upper()} - Generating {extension.upper()} file: {fname}")
    logger.info(f"Total rows to export: {len(rows)} (from {len(emails)} emails)")
    
    # Write to desired format
    if use_excel:
        logger.info("Using Excel export format (legacy mode)")
        # Excel export (legacy behavior) - use BytesIO
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)
        file_bytes = buf.getvalue()
    else:
        # CSV export with CRLF line endings for SSIS compatibility
        logger.info("Using CSV export format with CRLF line endings for SSIS compatibility")
        logger.info("PrimaryFileNumber column forced as STRING to prevent numeric conversion")
        # Use StringIO and then encode to UTF-8 bytes for SFTP upload
        buf = io.StringIO()
        df.to_csv(
            buf,
            index=False,
            lineterminator='\r\n'
        )
        file_bytes = buf.getvalue().encode('utf-8')
        logger.info(f"CSV file generated in memory: {len(file_bytes):,} bytes")
    
    # Update logging to reflect potential duplicates
    original_email_count = len(emails)
    total_rows = len(rows)
    duplicate_count = total_rows - original_email_count
    
    format_name = 'Excel' if use_excel else 'CSV'
    logger.info(f"{format_name} export complete: {original_email_count} emails processed for batch {batch_id}")
    logger.info(f"  - Total rows: {total_rows} ({duplicate_count} duplicate entries with replies)")
    logger.info(f"  - Columns: 13")
    logger.info(f"  - Formula injection protection: Enabled")
    if not use_excel:
        logger.info(f"  - Line endings: CRLF (\\r\\n) for SSIS compatibility")
        logger.info(f"  - PrimaryFileNumber: Forced as STRING type")
    
    # Upload to SFTP with stop signal
    logger.info(f"Uploading {format_name} file to SFTP: {fname}")
    upload_success = upload_to_sftp(fname, file_bytes, stop_event=stop_event)
    
    if upload_success:
        logger.info(f"✅ {format_name} file successfully exported and uploaded: {fname}")
    else:
        logger.warning(f"❌ {format_name} file generation completed but SFTP upload failed: {fname}")
    
    return fname if upload_success else None

def process_batch(batch_id: Optional[str] = None, stop_event=None) -> Tuple[bool, int, int, int]:
    """Process one batch of emails with UNIFIED stop signal."""
    # Create default stop event if not provided
    if stop_event is None:
        stop_event = threading.Event()
        
    start_time = time.time()
    
    # Get current batch size
    batch_size = get_batch_size()
    
    # Generate batch ID if not provided
    if not batch_id:
        batch_id = PostgresConnector.insert_batch_run()
        if not batch_id:
            return False, 0, 0, 0
    
    logger.info(f"Processing batch {batch_id} (max {batch_size} emails)")
    logger.info(f"Invoice requests send directly: {SEND_INVOICE_REQUEST_NO_INFO}, Claims paid send directly: {SEND_CLAIMS_PAID_NO_PROOF}")
    # Ensure batch record exists
    ensure_batch_record_exists(batch_id)

    # Process emails with stop signal
    email_result = process_unread_emails(batch_id, batch_size, stop_event)
    
    if not email_result["success"]:
        # Update batch status to failed
        mongo = get_mongo()
        if mongo:
            mongo.update_batch_result(
                batch_id, 
                email_result.get("emails_processed", 0), 
                email_result.get("emails_errored", 0), 
                0,
                "failed"
            )
        
        PostgresConnector.update_batch_result(
            batch_id,
            email_result.get("emails_processed", 0),
            email_result.get("emails_errored", 0),
            "failed"
        )
        
        return False, email_result.get("emails_processed", 0), email_result.get("emails_errored", 0), 0

    # Store processing results
    processed_count = email_result.get("emails_processed", 0)
    failed_count = email_result.get("emails_errored", 0)
    batch_id = email_result.get("batch_id", batch_id)

    # Count drafts created during processing
    draft_count = count_drafts_created(batch_id)
    logger.info(f"Drafts created during processing: {draft_count}")
    
    # Calculate totals
    total_processed = processed_count
    total_failed = failed_count
    total_draft_count = draft_count
    
    # Determine status
    status = "success" if total_failed == 0 else "partial"
    
    # Update batch status in both databases
    ensure_batch_record_exists(batch_id)
    
    PostgresConnector.update_batch_result(
        batch_id, 
        total_processed, 
        total_failed, 
        status, 
        total_draft_count
    )
    
    mongo = get_mongo()
    if mongo:
        mongo.update_batch_result(
            batch_id, 
            total_processed, 
            total_failed, 
            total_draft_count, 
            status
        )

    # Update tracking record for reporting
    if processed_count > 0:
        mongo = get_mongo()
        emails = list(mongo.collection.find({"batch_id": batch_id}).limit(1))
        email_data = emails[0] if emails else None
        
        update_batch_id_only(batch_id, email_data)
        logger.info(f"Inserted batch tracking record with batch_id={batch_id}")

    # Log completion
    elapsed = time.time() - start_time
    logger.info(f"Batch {batch_id} completed in {elapsed:.1f}s: {total_processed} processed, "
               f"{total_failed} failed, {total_draft_count} drafts created")
    
    return True, total_processed, total_failed, total_draft_count

def check_and_send_daily_report():
    """Check if it's time to send daily report and send it."""
    global _last_report_date
    
    try:
        # Get current ET time
        et_tz = pytz.timezone("US/Eastern")
        now_et = datetime.now(et_tz)
        today = now_et.date()
        
        # Send report if we haven't sent today's report yet and it's after midnight
        should_send = (_last_report_date != today and now_et.hour >= 0)
        
        if should_send:
            logger.info(f"Sending daily report for {today} at {now_et.strftime('%I:%M %p %Z')}")
            
            try:
                success = generate_daily_report()
                if success:
                    logger.info("Daily report sent successfully")
                    _last_report_date = today
                else:
                    logger.error("Daily report failed to send")
            except Exception as e:
                logger.error(f"Error generating daily report: {e}")
                
    except Exception as e:
        logger.error(f"Error checking daily report: {e}")

def run_batch_processor(stop_event=None) -> bool:
    """Run a single batch processing cycle with UNIFIED stop signal."""
    # Create default stop event if not provided
    if stop_event is None:
        stop_event = threading.Event()
    
    logger.info(f"Starting batch processor run")
    
    # Get current batch size
    batch_size = get_batch_size()
    
    # Check for incomplete batch first
    batch_id = check_incomplete_batch()
    
    if batch_id:
        logger.info(f"Resuming incomplete batch: {batch_id}")
        current_count = get_batch_email_count(batch_id)
        logger.info(f"Current batch has {current_count} emails, targeting {batch_size}")
        
        # Check if this batch should continue
        if not should_continue_batch(batch_id, batch_size):
            logger.info(f"Batch {batch_id} is complete, starting new batch")
            batch_id = PostgresConnector.insert_batch_run()
    else:        
        logger.info("No incomplete batch found, starting new batch")
        batch_id = PostgresConnector.insert_batch_run()
    
    if not batch_id:
        return False
   
    logger.info(f"Processing with batch ID: {batch_id}")
   
    ensure_batch_record_exists(batch_id)
   
    # Pass stop_event to process_batch
    success, processed_count, failed_count, draft_count = process_batch(batch_id, stop_event)
    
    if success:
        # Always check total count and handle appropriately
        total_count = get_batch_email_count(batch_id)
        
        if total_count > 0:
            logger.info(f"Batch {batch_id} has {total_count} emails - generating Excel")
            mark_batch_complete(batch_id)
            
            # Check stop before Excel generation
            if not stop_event.is_set():
                excel_file = export_processed_emails_to_excel(batch_id, stop_event)
                if excel_file:
                    logger.info(f"Excel file generated: {excel_file}")
                else:
                    logger.info(f"Excel generation completed (SFTP disabled or failed)")
        else:
            logger.info(f"Batch {batch_id} has no emails - marking complete, no Excel needed")
            mark_batch_complete(batch_id)
        
        # NEW: Check for daily report after successful batch
        check_and_send_daily_report()
   
    return success

ALERT_EMAILS = [
    "sanskar.gawande@cadex-solutions.com"
    # Add more alert emails here as needed
]

# Global variables for monitoring
last_activity_time = datetime.now()
last_alert_time = None
alert_count = 0
manual_shutdown_flag = False  

def get_alert_access_token():
    """Get Microsoft Graph API access token for alert emails."""
    try:
        app = msal.ConfidentialClientApplication(
            client_id=CLIENT_ID,
            client_credential=CLIENT_SECRET,
            authority=f"https://login.microsoftonline.com/{TENANT_ID}"
        )
        
        result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        
        if "access_token" in result:
            return result["access_token"]
        else:
            logger.error(f"Failed to acquire alert token: {result.get('error_description')}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting alert access token: {e}")
        return None

def send_stuck_alert():
    """Send alert email that system appears stuck."""
    if not ALERT_EMAILS:
        logger.warning("No alert emails configured")
        return False
    
    try:
        # Get access token
        access_token = get_alert_access_token()
        if not access_token:
            logger.error("Could not get access token for alert")
            return False
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Calculate how long system has been stuck
        hours_stuck = (datetime.now() - last_activity_time).total_seconds() / 3600
        
        # Simple alert message
        alert_text = f"""ALERT: Email Processing System Stuck

The email processing system appears to be stuck for {hours_stuck:.1f} hours.

Please check the system status.

Alert sent: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Prepare email
        to_recipients = [{"emailAddress": {"address": email}} for email in ALERT_EMAILS]
        subject = "ALERT: Email Processing System Stuck"
        
        message = {
            "subject": subject,
            "body": {"contentType": "Text", "content": alert_text},
            "toRecipients": to_recipients,
            "importance": "High"
        }
        
        payload = {"message": message, "saveToSentItems": "true"}
        from_email = EMAIL_ADDRESS.split(',')[0].strip() if EMAIL_ADDRESS else ""
        endpoint = f"https://graph.microsoft.com/v1.0/users/{from_email}/sendMail"
        
        response = httpx.post(endpoint, headers=headers, json=payload, timeout=30)
        
        if response.status_code in [200, 202]:
            logger.warning(f"STUCK ALERT sent to {len(ALERT_EMAILS)} recipients - system stuck for {hours_stuck:.1f} hours")
            return True
        else:
            logger.error(f"Failed to send stuck alert: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending stuck alert: {e}")
        return False

def check_and_send_stuck_alert():
    """Check if system is stuck and send alerts - IGNORES manual shutdowns."""
    global last_alert_time, alert_count, manual_shutdown_flag
    
    # CRITICAL FIX: Don't send alerts if shutdown was manual
    if manual_shutdown_flag:
        return  # Exit early - no alerts for manual shutdowns
    
    # Calculate time since last activity
    time_since_activity = (datetime.now() - last_activity_time).total_seconds() / 3600  # hours
    
    # First alert after 3.5 hours of inactivity (only if not manual shutdown)
    if time_since_activity >= 3.5 and last_alert_time is None:
        logger.warning(f"System stuck for {time_since_activity:.1f} hours - sending first alert")
        if send_stuck_alert():
            last_alert_time = datetime.now()
            alert_count = 1
    
    # Follow-up alerts every 2 hours after first alert
    elif last_alert_time and time_since_activity >= (3.5 + alert_count * 2.0):
        logger.warning(f"System stuck for {time_since_activity:.1f} hours - sending follow-up alert #{alert_count + 1}")
        if send_stuck_alert():
            last_alert_time = datetime.now()
            alert_count += 1

def update_activity_timestamp():
    """Reset activity tracking - call this when system is working normally."""
    global last_activity_time, last_alert_time, alert_count, manual_shutdown_flag
    
    # Check if we're recovering from a stuck state
    was_stuck = last_alert_time is not None
    
    # Reset all monitoring variables
    last_activity_time = datetime.now()
    last_alert_time = None
    alert_count = 0
    manual_shutdown_flag = False  # Clear manual shutdown flag on activity
    
    if was_stuck:
        logger.info("System activity detected - monitoring reset, alerts cleared")

def mark_manual_shutdown():
    """Mark that the system is being shut down manually - prevents false stuck alerts."""
    global manual_shutdown_flag
    manual_shutdown_flag = True
    logger.info("Manual shutdown marked - stuck detection disabled")

def run_email_processor(stop_event=None):
    """Main batch processing loop with UNIFIED stop signal and stuck detection alerts."""
    # Create default stop event if not provided (for backward compatibility)
    if stop_event is None:
        stop_event = threading.Event()
    
    # Get current settings
    batch_size = get_batch_size()
    batch_interval = get_batch_interval()

    logger.info(f"Starting email batch processor (batch size: {batch_size}, interval: {batch_interval}s)")
    logger.info(f"Invoice requests - Send directly: {SEND_INVOICE_REQUEST_NO_INFO}")
    logger.info(f"Claims paid - Send directly: {SEND_CLAIMS_PAID_NO_PROOF}")
    logger.info(f"SFTP export is {'ENABLED' if SFTP_ENABLED else 'DISABLED'}")
    
    model_url = "http://104.197.197.76:8000"
    logger.info(f"Model API URL: {model_url}")
    
    consecutive_failures = 0
    max_failures = 3
   
    while True:
        # === STUCK ALERT CHECK ===
        check_and_send_stuck_alert()
        
        # CHECK UNIFIED STOP SIGNAL
        if stop_event.is_set():
            logger.info("STOP: Stop signal detected - shutting down processor NOW")
            mark_manual_shutdown()
            break
            
        start_time = datetime.now()
        logger.info(f"Starting batch at {start_time.isoformat()}")
       
        # === RESET ACTIVITY TIMESTAMP (system is active again) ===
        update_activity_timestamp()
        
        try:
            # Check model health before processing
            if not check_model_health():
                logger.info("Model is not available")
                consecutive_failures += 1
                
                if consecutive_failures >= max_failures:
                    logger.info(f"Model down for {max_failures} consecutive attempts - attempting recovery")
                    
                    recovery_success = wait_for_model_recovery(stop_event, max_retries=3)
                    if recovery_success:
                        consecutive_failures = 0
                        logger.info("Model recovery successful - continuing processing")
                        continue
                    else:
                        logger.error("Model recovery failed after maximum attempts - shutting down processor")
                        mark_manual_shutdown()
                        break
                else:
                    logger.info(f"Waiting 60 seconds before retry (failure {consecutive_failures}/{max_failures})")
                    for i in range(60):
                        if stop_event.is_set():
                            logger.info("STOP: Stop signal detected during wait - shutting down NOW")
                            mark_manual_shutdown()
                            return
                        time.sleep(1)
                continue
            
            # Reset failure counter on successful health check
            consecutive_failures = 0
            
            # Run main batch processor with stop signal - wrap in try-except to catch any silent failures
            try:
                run_batch_processor(stop_event)
            except Exception as e:
                logger.exception(f"CRITICAL: run_batch_processor crashed with exception: {e}")
                logger.error("This should not happen - batch processor should handle all exceptions internally")
                # Don't break - let the outer exception handler deal with it
                raise
           
            # Calculate time to next batch
            elapsed = (datetime.now() - start_time).total_seconds()
            wait_time = max(0, batch_interval - elapsed)
           
            logger.info(f"Batch complete. Next batch in {wait_time:.1f} seconds")
            
            for i in range(int(wait_time)):
                # Keep monitoring stuck status during idle wait
                check_and_send_stuck_alert()
                
                if stop_event.is_set():
                    logger.info("STOP: Stop signal detected during batch interval - shutting down NOW")
                    mark_manual_shutdown()
                    return
                time.sleep(1)
            
            remaining = wait_time - int(wait_time)
            if remaining > 0:
                time.sleep(remaining)
            
        except KeyboardInterrupt:
            logger.info("Batch processor interrupted by user")
            mark_manual_shutdown()
            break
        except Exception as e:
            logger.exception(f"Unhandled error in batch processor: {str(e)}")
            consecutive_failures += 1
            logger.info(f"Waiting 60 seconds before retry due to error")
            for i in range(60):
                check_and_send_stuck_alert()
                
                if stop_event.is_set():
                    logger.info("STOP: Stop signal detected during error wait - shutting down NOW")
                    mark_manual_shutdown()
                    return
                time.sleep(1)
    
    logger.info("Email processor stopped")

if __name__ == "__main__":
    try:
        stop_event = threading.Event()

        # Start batch processor in background
        threading.Thread(target=run_email_processor, args=(stop_event,), daemon=True).start()

        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Unhandled exception in main:")
        sys.exit(1)

