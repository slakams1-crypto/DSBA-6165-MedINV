import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import logging
logging.getLogger("asyncio").setLevel(logging.ERROR)
import asyncio, sys, signal, os, gc, base64, io, re, uuid, spaces, math
import threading, base64, uuid, unicodedata, subprocess, psutil, uvicorn
import time, datetime
import json
from PIL import Image
import pandas as pd
import gradio as gr
import torch
import spaces
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
import mysql.connector
from mysql.connector import Error
from huggingface_hub import login
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_community.tools import YouTubeSearchTool
from ddgs import DDGS
from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_model, AgentState, AgentMiddleware, LLMToolSelectorMiddleware, ToolRetryMiddleware,
    TodoListMiddleware, SummarizationMiddleware, HumanInTheLoopMiddleware
)
from langgraph.runtime import Runtime
from langsmith import traceable, Client, get_current_run_tree
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import html
ls_client = Client()
from supabase import create_client
from dotenv import load_dotenv
load_dotenv()
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.server.fastmcp import FastMCP
from huggingface_hub import hf_hub_download, HfApi
# LangGraph Implementation
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_core.documents import Document
import cv2
sys.path.insert(0, './MedInv')

# ============================================
# Logging Configuration
# ============================================
# ── Writable path outside Git tree ──
LOG_DIR = os.path.expanduser("~/logs")  # /home/user/logs
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Create directory exactly once, with explicit check
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print(f"📁 Created logs directory: {LOG_DIR}")
else:
    print(f"📁 Logs directory exists: {LOG_DIR}")

# ── Clear pre-existing handlers ──
root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)

# ── Configure logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logger = logging.getLogger(__name__)

# ── Verify ──
logger.info("=" * 60)
logger.info("Logging initialized")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Log file size: {os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0} bytes")
logger.info("=" * 60)

# ============================================
# LangSmith Tracing Set to False on StartUp
# ============================================
os.environ.setdefault("LANGSMITH_TRACING_V2", "false")

# Check versions
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"🔥 CUDA available: {torch.cuda.is_available()}")
print(f"🔥 CUDA device count: {torch.cuda.device_count()}")

# ============================================
# Gradio - Create a custom theme with medical blue
# ============================================
medical_theme = gr.themes.Soft(
    primary_hue="blue",  # This controls the primary button color
    secondary_hue="gray",
).set(
    button_primary_background_fill="#0066cc",
    button_primary_background_fill_dark="#004d99",
    button_primary_text_color="white",
    button_primary_border_color="#0066cc",
    button_primary_border_color_dark="#004d99",
)

# ============================================
# CSS for Gradio UI elements
# ============================================
custom_css = """
body { font-size: 16px !important; }
.gr-box { font-size: 16px !important; }
.file-btn-pair button { min-height: 44px !important; }
.file-btn-pair .file-preview { min-height: 44px !important; display: flex; align-items: center; }
/* Target the specific block wrapper Gradio creates */
.block.prompt-box {
    --body-text-color-subdued: #374151 !important;
}
.block.prompt-box textarea::placeholder,
.block.prompt-box input::placeholder {
    color: #6b7280 !important; 
    #font-weight: 600 !important;
    opacity: 1 !important;       /* browsers default to ~0.5 */
}
/* Also darken what the user types so it matches */
.block.prompt-box textarea,
.block.prompt-box input {
    color: #374151 !important; 
    #font-weight: 500 !important;
}
/* Rate this response */
.main-col {
    gap: 8px !important;
}
/* Panel cards */
.panel-card-title {
    background: var(--c-card-bg) !important;
    border-radius: 12px !important;
    box-shadow: var(--c-card-shadow) !important;
    border: 1px solid var(--c-card-border) !important;
    padding-top: 0px !important; /* Changed from 18px */
    padding-right: 18px !important; /* Explicitly set if needed, but usually inherited */
    padding-bottom: 18px !important; /* Explicitly set if needed, but usually inherited */
    padding-left: 18px !important;  /* Explicitly set if needed, but usually inherited */
}
.panel-card {
    background: var(--c-card-bg) !important;
    border-radius: 12px !important;
    box-shadow: var(--c-card-shadow) !important;
    border: 1px solid var(--c-card-border) !important;
    padding: 18px !important;
}
.panel-card > .form { gap: 12px !important; }
/* Pill badge title */
.panel-badge {
    display: inline-block !important;
    background: #eff6ff !important;
    color: #2563eb !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    padding: 6px 6px !important;
    border-radius: 999px !important;
    margin-bottom: 4px !important;   /* ← was 16px */
    margin-top: 1px !important;
    line-height: 1 !important;
}
.panel-badge-mod {
    display: inline-block !important;
    background: #eff6ff !important;
    color: #2563eb !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    padding-top: 6px !important;
    padding-bottom: 1px !important;
    padding-left: 6px !important;
    padding-right: 6px !important;
    border-radius: 999px !important;
    margin-bottom: 1px !important;   /* ← was 16px */
    margin-top: 1px !important;
    line-height: 1 !important;
}
.panel-badge-mod2 {
    display: inline-block !important;
    background: #eff6ff !important;
    color: #2563eb !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: lowercase !important;
    letter-spacing: 0.6px !important;
    padding-top: 6px !important;
    padding-bottom: 1px !important;
    padding-left: 6px !important;
    padding-right: 6px !important;
    border-radius: 999px !important;
    margin-bottom: 1px !important;   /* ← was 16px */
    margin-top: 1px !important;
    line-height: 1 !important;
}
/* Card panels */
.input-panel {
    background: #ffffff !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
    padding: 16px !important;        /* ← was 24px */
    margin-top: 10px !important;    /* default for text-input panel */
    margin-bottom: 10px !important;
}
.input-panel-mod {
    background: #ffffff !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
    padding: 4px !important;        /* ← was 24px */
    margin-top: 10px !important;    /* default for text-input panel */
    margin-bottom: 10px !important;
}
/* Multimodal cards only: slam top margin to 0 so they sit tight under the badge */
.multimodal-card {
    margin-top: 0 !important;
}
/* Equal height columns in multimodal row */
.equal-height {
    align-items: stretch !important;
}
.equal-height > .input-panel {
    display: flex !important;
    flex-direction: column !important;
}
/* Buttons */
.btn-primary {
    background: linear-gradient(135deg, #2563eb, #6d28d9) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: .2px !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.30) !important;
    transition: all 0.18s ease !important;
    color: #fff !important;
    padding: 10px 0 !important;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(37,99,235,0.40) !important;
}
.btn-secondary {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border: 1px solid var(--c-btn2-border) !important;
    background: var(--c-btn2-bg) !important;
    color: var(--c-btn2-text) !important;
    transition: all 0.15s ease !important;
}
.btn-secondary:hover {
    background: var(--c-chip-bg) !important;
    border-color: var(--c-tab-sel) !important;
}
/* Tab styling */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--c-tab-text) !important;
}
.tab-nav button.selected {
    color: var(--c-tab-sel) !important;
    border-bottom: 2px solid var(--c-tab-sel) !important;
}
"""

# For Testing Purposes -----------------------
data_url = "invoices/Medical_Invoice_01_01.png"

# ============================================
# Fix for the asyncio cleanup error
# ============================================
if sys.platform == "linux":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# ============================================
# Model configurations
# ============================================
# Invoice model
Invmodel = {
    "vqa-model": "lightonai/LightOnOCR-2-1B",
    "embed-model": "sentence-transformers/all-MiniLM-L6-v2"
}

# ============================================
# Embedding Model
# ============================================
# This will be pre-downloaded by preload_from_hub
embedding_model = HuggingFaceEmbeddings(
    model_name=Invmodel["embed-model"],
    cache_folder="/tmp/.cache"  # Use tmp for Spaces
)
logger.info("✓ Embedding model loaded")
print("✓ Embedding model loaded")

# =======================================
# Load model ONCE at module level
# =======================================
# Change to "CPU"
device = "cpu"

# device = "cuda"  # ZeroGPU emulation handles this at module level
dtype = torch.bfloat16

model = LightOnOcrForConditionalGeneration.from_pretrained(
    Invmodel["vqa-model"],
    torch_dtype=dtype
).to(device)

processor = LightOnOcrProcessor.from_pretrained(Invmodel["vqa-model"])
logger.info("✓ Invoice model loaded")
print("✓ Invoice model loaded")

# ============================================
# Get Batch Size from the configuration/environment variable
# ============================================
invoice_batch_size = int(os.getenv('INVOICE_BATCH_SIZE',"2"))
logger.info(f"DEBUG: Invoice Batch Size: {invoice_batch_size}")
print(f"DEBUG: Invoice Batch Size: {invoice_batch_size}")

# Initialize a global variable to store extracted invoice data
global all_processed_invoices
# Global flags to track state across agent invocations
global _kb_had_results
_kb_had_results = False

# ============================================
# Get API Keys & Tokens
# ============================================
# Get huggingface api token from environment variable (from Space secret)
hf_token = os.getenv('HUGGINGFACE_API_KEY')

if hf_token:
    # Authenticate silently without prompting
    login(token=hf_token, add_to_git_credential=True)
    logger.info("✓ Successfully authenticated with Hugging Face Hub")
    print("✓ Successfully authenticated with Hugging Face Hub")
else:
    logger.info("Warning: HF_TOKEN not found. Some features may be limited.")
    print("Warning: HF_TOKEN not found. Some features may be limited.")

# ============================================
# System prompt
# ============================================
SYSTEM_PROMPT = (
    "You are a helpful Medical Invoice Intelligence assistant. "
)    

# ============================================
# Vision prompt
# ============================================
VISION_PROMPT = (
    "You are an expert medical invoice assistant.\n\n"
)

#    "You are an expert medical invoice assistant.\n\n"
#    "Summarize the contents of the invoice in the following format:\n"
#    "Summary of Medical Invoice\n"
#    "1.Provider Information:\n"
#    "2.Patient Information:\n"
#    "3.Services Rendered:\n"
#    "4.Financial Summary:\n"
#    "If you have any specific questions about the services listed or need further information, feel free to ask!"

# ============================================
# Get Database Environment Variables
# ============================================
POSTGRES_URL = os.getenv("POSTGRES_URL", "https://gttssffdfdfdfdffsyleksbc.supabase.co")
POSTGRES_KEY = os.getenv("POSTGRES_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCOiJzdXBh...")
if not POSTGRES_KEY:
    print("❌ POSTGRES_KEY not set in environment")
print("✅ Loaded Database Environment variables...")

# ============================================
# Load existing vector store
# ============================================
logger.info("🔄 Loading Chroma vector store...")
print("🔄 Loading Chroma vector store...")
persist_directory = './data/vectorstore/InvoiceVStore'

# Verify the directory exists
if not os.path.exists(persist_directory):
    raise FileNotFoundError(
        f"Vector store not found at {persist_directory}. "
        f"Available: {os.listdir('.')}"
    )

# Load the vector store
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

doc_count = vectorstore._collection.count()
logger.info(f"✓ Loaded vector store with {doc_count} documents using HuggingFace embeddings")
print(f"✓ Loaded vector store with {doc_count} documents using HuggingFace embeddings")

# =========================================
# Create MCP server and add a test tool
# =========================================
mcp = FastMCP("MCPServer")
print("✓ MCP Server Created Successfully.")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# =========================================
# Connect to Dataset Repo and count PDFs in the dataset
# =========================================
# Huggingface Hub Dataset Repo
DATASET_REPO_ID = "slakams1/InvoiceDataset"

# Connect to dataset repo
api = HfApi()
files = api.list_repo_files(repo_id=DATASET_REPO_ID, repo_type="dataset")
pdf_files_in_dataset_repo = [f for f in files if f.endswith(".pdf")]
png_files_in_dataset_repo = [f for f in files if f.endswith(".png")]

# Check if pdf files exist or not.
if pdf_files_in_dataset_repo:
    logger.info(f"Found {len(pdf_files_in_dataset_repo)} PDF files in the dataset repo.")
    print(f"Found {len(pdf_files_in_dataset_repo)} PDF files in the dataset repo.")
    print(f"First PDF file in the dataset repo list: {pdf_files_in_dataset_repo[0]}")

# Check if png files exist or not.
if png_files_in_dataset_repo:
    logger.info(f"Found {len(png_files_in_dataset_repo)} PNG files in the dataset repo.")
    print(f"Found {len(png_files_in_dataset_repo)} PNG files in the dataset repo.")
    print(f"First PNG file in the dataset repo list: {png_files_in_dataset_repo[0]}")

# Download each PNG to local temp and collect full paths
local_png_paths = []
for png_file in png_files_in_dataset_repo[:invoice_batch_size]:
    local_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=png_file,
        repo_type="dataset",
        local_dir="/tmp/hf_invoices"  # or any local folder
    )
    local_png_paths.append(local_path)    

# ==========================================
# Image Encoder Helper Function
# ==========================================
# @spaces.GPU
def _encode_image_to_base64(image_path: str, max_dimension: int = None) -> tuple[str, str]:
    try:
        with Image.open(image_path) as img:
            if max_dimension:
                width, height = img.size
                if max(width, height) > max_dimension:
                    if width > height:
                        new_width = max_dimension
                        new_height = int(height * (max_dimension / width))
                    else:
                        new_height = max_dimension
                        new_width = int(width * (max_dimension / height))
                    try:
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                    except Exception as e:
                        print(f"Error resizing image {image_path}: {e}")
                        return "", ""

            buffered = io.BytesIO()
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = 'image/jpeg'
            format = 'JPEG'

            if ext in ['.png', '.gif', '.bmp', '.webp']:
                if img.mode in ('RGBA', 'P'):
                    mime_type = 'image/png'
                    format = 'PNG'
                else:
                    img = img.convert('RGB')
                    mime_type = 'image/jpeg'
                    format = 'JPEG'

            try:
                img.save(buffered, format=format)
                encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return encoded, mime_type
            except Exception as e:
                logger.error(f"Error saving or encoding image {image_path}: {e}")
                print(f"Error saving or encoding image {image_path}: {e}")
                return "", ""
    except FileNotFoundError:
        logger.error(f"❌ Error: Image file not found at {image_path}")
        print(f"❌ Error: Image file not found at {image_path}")
        return "", ""
    except Exception as e:
        logger.error(f"Error opening image {image_path}: {e}")
        print(f"Error opening image {image_path}: {e}")
        return "", ""

# ===========================================
# Helper Functions
# ===========================================

# Extract a value based on a key
def get_value(key, regex, text_content, default=None):
    match = re.search(regex, text_content)
    if match:
        # Remove leading/trailing spaces and bold markdown
        return match.group(1).strip().replace('**', '')
    return default

# ===========================================
# Convert currency string to float
# ===========================================
def currency_to_float(currency_str):
    if currency_str is None:
        return None
    try:
        # Handle negative values for insurance adjustment
        is_negative = False
        if '-' in str(currency_str):
            is_negative = True
        cleaned_str = str(currency_str).replace('$', '').replace(',', '').replace('-', '').strip()
        if cleaned_str:
            value = float(cleaned_str)
            return -value if is_negative else value
        return None
    except ValueError:
        logger.error(f"Warning: Could not convert '{currency_str}' to float.")
        print(f"Warning: Could not convert '{currency_str}' to float.")
        return None
        
# ===========================================
# Parse address into components
# ===========================================
def parse_address(address_str):
    if not address_str:
        return {"address": None, "city": None, "state": None, "zipcode": None}

    # This regex attempts to capture common address patterns.
    # It assumes: Street Address, City, State ZIP
    # Example: 123 Main St, Anytown, CA 90210
    match = re.search(r"^(.*?),\s*([A-Za-z\s]+),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)$", address_str)
    if match:
        return {
            "address": match.group(1).strip(),
            "city": match.group(2).strip(),
            "state": match.group(3).strip(),
            "zipcode": match.group(4).strip()
        }
    # If the above doesn't match, return the original string as address and None for others
    return {"address": address_str.strip(), "city": None, "state": None, "zipcode": None}

def clean_floats(obj):
    """Recursively replace NaN / Inf / -Inf with None. Handles numpy + pandas."""
    if isinstance(obj, dict):
        return {k: clean_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_floats(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    # Handle numpy float scalars that don't pass isinstance(..., float)
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
    except ImportError:
        logger.error("clean_floats()-Import Error")
        pass
    return obj    

# ===========================================
# Misc. Function to extract plain text from Gradio's Textbox format
# ===========================================
def extract_gradio_text(value):
    """Extract plain text from Gradio's Textbox format."""
    
    if isinstance(value, list) and len(value) > 0:
        if isinstance(value[0], dict) and 'text' in value[0]:
            return value[0]['text']
    return str(value) if value else ""

# ===========================================
# Misc. Function to deep clean chat history
# ===========================================
def clean_chat_history(history):
    """Ensure all messages are plain dicts with string content."""
    
    clean = []
    for msg in history:
        if isinstance(msg, dict):
            # Clean the content field
            raw_content = msg.get("content", "")
            cleaned_content = clean_text(raw_content)
            
            clean.append({
                "role": str(msg.get("role", "user")),
                "content": cleaned_content
            })
    return clean 

# ===========================================
# Misc. Function to extract plain text from Gradio's wrapped format
# ===========================================
def clean_text(value):
    """Extract plain text from Gradio's wrapped format.
    Handles: 
    - String values
    - Lists containing dicts with 'text' key: [{'text': '...', 'type': 'text'}]
    - Direct dicts with 'text' or 'content' keys
    """
    
    if value is None:
        return ""
    
    # Handle list of dicts (Gradio's new format)
    if isinstance(value, list) and len(value) > 0:
        if isinstance(value[0], dict):
            # Extract text from each dict and join
            texts = [item.get('text', '') for item in value if isinstance(item, dict)]
            return ' '.join(filter(None, texts))
        # If list of strings
        return ' '.join(str(item) for item in value if item is not None)
    
    # Handle direct dict
    elif isinstance(value, dict):
        return value.get('text', value.get('content', str(value)))
    
    # Already a string
    return str(value)    

# ============================================
# Helper to keep history code 
# ============================================
def _append_turn(history, user_text, assistant_text):
    out = []
    for msg in history:
        if isinstance(msg, dict):
            out.append({
                "role": msg.get("role", "user"),
                "content": str(msg.get("content", ""))
            })
    out.append({"role": "user", "content": user_text})
    out.append({"role": "assistant", "content": assistant_text})
    return out  

# ===========================================
# Helper to generate client batch id for processing a single invoice or a batch
# ===========================================
def get_client_batch_id():
    # Generate a single client_batch_id for this batch processing run
    client_batch_id = uuid.uuid4()
    return client_batch_id
    logger.info(f"Processing batch with client_batch_id: {client_batch_id}")
    print(f"Processing batch with client_batch_id: {client_batch_id}")

# ==========================================
# Store embeddings into vector store
# ==========================================
def split_document_into_chunks(document, chunk_size=100, chunk_overlap=10, add_start_index=True):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      add_start_index=add_start_index,
      separators=["\n\n", "\n", " ", ""]
  )

  return text_splitter.split_documents(document)

# =========================================
# Function to download pdfs from HF dataset repo to local cache and process one by one
# =========================================
def download_and_process(pdf_path_in_repo):
    local_file_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=pdf_path_in_repo,
        repo_type="dataset",
        local_dir="/home/user/app/invoices"  # cache locally
    )
    # Return the pdf file for further processing
    return local_file_path

# ===========================================
# Invoice related
# Helper: Extract invoice numbers from text
# ===========================================
def extract_invoice_number(query: str) -> Optional[str]:
    """Pull out 6+ digit numbers that look like invoice IDs."""
    numbers = re.findall(r'\b\d{6,}\b', query)
    return numbers[0] if numbers else None

# ===========================================
# Helper: Lookup invoice in PostgreSQL
# ===========================================
def lookup_invoice_in_database(invoice_no: str) -> Optional[str]:
    """
    Query PostgreSQL by exact invoice_number.
    Returns formatted text if found, None if not found.
    """    
    try:
        postgres = create_client(POSTGRES_URL, POSTGRES_KEY)
        result = postgres.rpc("get_invoice_by_number", {
            "p_invoice_number": invoice_no
        }).execute()
        
        print(f"DEBUG RPC raw response: {json.dumps(result.data, indent=2, default=str)}")
        
        if result.data is None:
            logger.info("DEBUG: result.data is None")
            print("DEBUG: result.data is None")
            return None
        
        # ── Handle BOTH dict and list response shapes ──
        if isinstance(result.data, dict):
            # Direct dict response (most common for JSONB scalar)
            raw = result.data
        elif isinstance(result.data, list) and len(result.data) > 0:
            # Wrapped in list
            raw = result.data[0]
        else:
            logger.info(f"DEBUG: Unexpected result.data type: {type(result.data)}")
            print(f"DEBUG: Unexpected result.data type: {type(result.data)}")
            return None
        
        # Unwrap if nested inside function name key
        if isinstance(raw, dict) and "get_invoice_by_number" in raw:
            invoice_data = raw["get_invoice_by_number"]
        else:
            invoice_data = raw
        
        if invoice_data is None:
            return None
        
        return format_invoice_from_json(invoice_data)
    
    except Exception as e:
        logger.error(f"⚠️ Database lookup error for {invoice_no}: {e}")
        print(f"⚠️ Database lookup error for {invoice_no}: {e}")
        import traceback
        print(f"🔥 Exception type: {type(e).__name__}")
        print(f"🔥 Exception: {e}")
        traceback.print_exc()
        return None      

def sanitize_for_json(obj):
    """Recursively replace NaN / Inf / -Inf with None so JSON/PostgreSQL accepts it."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    else:
        try:
            if math.isnan(obj) or math.isinf(obj):
                return None
        except TypeError:
            pass
        return obj        

def format_invoice_from_json(invoice_json: dict) -> str:
    """Convert the JSONB RPC result into human-readable chat text."""
    lines = []
    
    provider = invoice_json.get('provider') or {}
    if provider:
        lines.extend([
            f"**Provider:** {provider.get('provider_name', '')}",
            f"**Provider Address:** {provider.get('provider_address', '')}, {provider.get('provider_city', '')}, {provider.get('provider_state', '')}",
        ])
    else:
        print(f" Issue with retrieving provider details")
    
    patient = invoice_json.get('patient') or {}
    if patient:
        # Use patient_first_name (your actual column)
        first_name = patient.get('patient_first_name', '')
        last_name = patient.get('patient_last_name', '')
        full_name = f"{first_name} {last_name}".strip() if last_name else first_name
        # f"**Patient:** {full_name}",
        lines.extend([
            f"**Patient:** {patient.get('patient_first_name', '')} {patient.get('patient_last_name', '')}",        
            f"**Patient Address:** {patient.get('patient_address', '')}, {patient.get('patient_city', '')}, {patient.get('patient_state', '')}",            
            f"**DOB:** {patient.get('dob', '')}",
            f"**Insurance Policy #:** {patient.get('insurance_policy', '')}",
        ])
    
    # Invoice details
    lines.extend([
        f"**Invoice #:** {invoice_json.get('invoice_number', '')}",
        f"**Invoice Date:** {invoice_json.get('invoice_date', '')}",
    ])
    
    items = invoice_json.get('items') or []
    if items:
        lines.extend(["", "**Services:**"])
        for item in items:
            lines.append(
                f"- {item.get('line_item_description', 'Service')} "
                f"(Qty: {item.get('line_item_quantity', 1)}) — "
                f"${item.get('line_item_amount', '')}"
            )
    
    # Financials (handle zero/null gracefully)
    invoice_total = invoice_json.get('invoice_total')
    payment_total = invoice_json.get('payment_total')
    
    if not invoice_total and items:
        total = sum(float(item.get('line_item_amount', 0) or 0) * int(item.get('line_item_quantity', 1)) for item in items)
        lines.append(f"\n**Subtotal:** ${total:.2f}")
    
    amount_due = invoice_json.get('amount_due')
    insurance_adjustment = invoice_json.get('insurance_adjustment', '')
    
    if invoice_total and invoice_total != 0:
        lines.extend([f"**Total:** ${invoice_total}"])
    if payment_total and payment_total != 0:
        lines.extend([f"**Payment:** ${payment_total}"])
    if insurance_adjustment and insurance_adjustment != 0:
        lines.extend([f"**Insurance Adjustment:** ${insurance_adjustment}"])
    if amount_due and amount_due != 0:
        lines.extend([f"**Patient Responsibility:** ${amount_due}"])
        lines.extend([f"**Total Due:** ${amount_due}"])
    
    return "\n".join(lines)  

# ===========================================
# Helper: Format DB result into readable text
# ===========================================
def format_invoice_from_db(invoice: dict) -> str:
    """Convert PostgreSQL record to human-readable text for chat response."""
    inv = invoice.get("invoice_details", {}) if "invoice_details" in invoice else invoice
    prov = invoice.get("provider", {}) if "provider" in invoice else {}
    pat = invoice.get("patient", {}) if "patient" in invoice else {}
    items = invoice.get("items", []) if "items" in invoice else []
    
    lines = [
        f"**Invoice #{inv.get('invoice_no', 'N/A')}**",
        f"**Date:** {inv.get('invoice_date', 'N/A')}",
        f"**Provider:** {prov.get('provider_name', prov.get('name', 'N/A'))}",
        f"**Patient:** {pat.get('patient_name', pat.get('patient', 'N/A'))} | DOB: {pat.get('dob', 'N/A')}",
        "",
        "**Services:**",
    ]
    
    for item in items:
        lines.append(f"- {item.get('description', 'Service')}: ${item.get('amount', 'N/A')}")
    
    lines.extend([
        "",
        f"**Subtotal:** ${inv.get('subtotal', 'N/A')}",
        f"**Insurance Adjustment:** ${inv.get('insurance_adjustment', 'N/A')}",
        f"**Patient Responsibility:** ${inv.get('patient_responsibility', 'N/A')}",
        f"**💰 Total Amount Due:** ${inv.get('total_due', 'N/A')}",
    ])
    
    return "\n".join(lines)    

# =========================================
# Function to add documents to Vector Store
# =========================================
def add_documents_to_vectorstore(vectorstore, input_pdffile_path, output_vectorstore_path):
  """
  Adds PDF documents from a specified folder to an existing Chroma vector store.

  This function iterates through all PDF files in `input_pdffile_path`,
  loads each PDF, splits it into chunks using `RecursiveCharacterTextSplitter`,
  and then adds these chunks to the `vectorstore` (which is loaded from
  `output_vectorstore_path`). The vector store is persisted after all
  documents are processed.

  Args:
      vectorstore: An existing Chroma vector store object. (Though it's re-loaded
                   inside the function from `output_vectorstore_path`)
      input_pdffile_path (str): The path to the directory containing the PDF files.
      output_vectorstore_path (str): The path where the Chroma vector store
                                     is persisted and reloaded from.

  Returns:
      None. Prints status messages to the console.
  """
  print(f"Checking contents of: {input_pdffile_path}")
  if os.path.exists(input_pdffile_path):
      contents = os.listdir(input_pdffile_path)

      if contents:
        # print(f"Contents found: {contents}")
        logger.info(f"Number of contents found: {len(contents)}")
        print(f"Number of contents found: {len(contents)}")

        # Load an existing vector store
        vectorstore_loaded = Chroma(
            persist_directory=output_vectorstore_path,
            embedding_function=embedding_model
        )

        # print(f"✓ Vector store loaded from: {vectorstore_loaded.persist_directory}")
        logger.info(f"✓ Loaded vector store with {vectorstore_loaded._collection.count()} documents")
        print(f"✓ Loaded vector store with {vectorstore_loaded._collection.count()} documents")

        contents.sort()
        logger.info(f"Sorted contents: {contents}")
        print(f"Sorted contents: {contents}")

        pdf_files_in_contents = [f for f in contents if f.endswith('.pdf')]

        if pdf_files_in_contents:
            logger.info(f"Found {len(pdf_files_in_contents)} PDF files in the directory list.")
            print(f"Found {len(pdf_files_in_contents)} PDF files in the directory list.")
            print(f"First PDF file in the list: {pdf_files_in_contents[0]}")

            for pdf_file in pdf_files_in_contents:
                logger.info(f"Processing: {pdf_file}")
                print(f"Processing: {pdf_file}")

                try:
                  loader = PyPDFLoader(os.path.join(input_pdf_folder, pdf_file))
                  document = loader.load()

                  # print(f"Loaded {len(document)} pages")
                  # print(f"Page 0 metadata: {document[0].metadata}")
                  # print(f"document 0 content preview: {document[0].page_content[:600]}...")

                  # Split the loaded documents
                  all_splits = split_document_into_chunks(document)
                  logger.info(f"Split each document into {len(all_splits)} chunks")
                  print(f"Split each document into {len(all_splits)} chunks")

                  vectorstore_loaded.add_documents(all_splits)
                  logger.info(f"✓ Added {len(all_splits)} splits to the vector store")
                  print(f"✓ Added {len(all_splits)} splits to the vector store")
                  print(f"✓ Vector store now has {vectorstore_loaded._collection.count()} documents")

                except Exception as e:
                  logger.error(f"Error loading PDF: {str(e)}")
                  error_message = f"Error loading PDF: {str(e)}"
                  print(error_message)

            vectorstore_loaded.persist()
            logger.info(f"✓ Vector store persisted to: {vectorstore_loaded}")
            print(f"✓ Vector store persisted to: {vectorstore_loaded}")

        else:
            logger.info("No PDF files explicitly ending with '.pdf' were found in the directory list.")
            print("No PDF files explicitly ending with '.pdf' were found in the directory list.")
      else:
          logger.info("No contents found in the directory.")
          print("No contents found in the directory.")
  else:
      logger.info(f"Error: The directory '{input_pdffile_path}' does not exist or is not accessible.")
      print(f"Error: The directory '{input_pdffile_path}' does not exist or is not accessible.")    
      print(f"Checking contents of: {input_pdffile_path}")
      ROOT_DIR = "/home/user/app/invoices"
      if os.path.exists(input_pdffile_path):
          # contents = os.listdir(input_pdffile_path)
          contents = [f for f in os.listdir(ROOT_DIR) if f.endswith('.pdf')]
          
          if contents:
              logger.info(f"Contents found: {contents}")
              print(f"Contents found: {contents}")
              print(f"Number of contents found: {len(contents)}")
              
              contents.sort()
              logger.info(f"Sorted contents: {contents}")
              print(f"Sorted contents: {contents}")
    
              pdf_files_in_contents = [f for f in contents if f.endswith('.pdf')]
            
              if pdf_files_in_contents:
                  logger.info(f"Found {len(pdf_files_in_contents)} PDF files in the directory list.")
                  print(f"Found {len(pdf_files_in_contents)} PDF files in the directory list.")
                  print(f"First PDF file in the list: {pdf_files_in_contents[0]}")
            
                  for pdf_file in pdf_files_in_contents:
                      logger.info(f"Processing: {pdf_file}")
                      print(f"Processing: {pdf_file}")
        
                      try:
                          # loader = PyPDFLoader(os.path.join(input_pdffile_path, pdf_file))
                          loader = PyPDFLoader(os.path.join(ROOT_DIR, pdf_file))
                          document = loader.load()
        
                          # print(f"Loaded {len(document)} pages")
                          # print(f"Page 0 metadata: {document[0].metadata}")
                          # print(f"document 0 content preview: {document[0].page_content[:600]}...")
        
                          # Split the loaded documents
                          all_splits = split_document_into_chunks(document)
                          logger.info(f"Split each document into {len(all_splits)} chunks")
                          print(f"Split each document into {len(all_splits)} chunks")
        
                          vectorstore.add_documents(all_splits)
                          logger.info(f"✓ Added {len(all_splits)} splits to the vector store")
                          print(f"✓ Added {len(all_splits)} splits to the vector store")
                          print(f"✓ Vector store now has {vectorstore._collection.count()} documents")
        
                      except Exception as e:
                          logger.error(f"Error loading PDF: {str(e)}")
                          error_message = f"Error loading PDF: {str(e)}"
                          print(error_message)
    
                  vectorstore.persist()
                  logger.info(f"✓ Vector store persisted to: {vectorstore}")
                  print(f"✓ Vector store persisted to: {vectorstore}")
    
              else:
                  logger.info("No PDF files explicitly ending with '.pdf' were found in the directory list.")
                  print("No PDF files explicitly ending with '.pdf' were found in the directory list.")
          else:
              logger.info("No contents found in the directory.")
              print("No contents found in the directory.")
      else:
          logger.info(f"❌ Error: The directory '{input_pdffile_path}' does not exist or is not accessible.")
          print(f"❌ Error: The directory '{input_pdffile_path}' does not exist or is not accessible.")

# ===========================================
# Retrieve relevant context from knowledge base
# ===========================================
@tool
def retrieve_context(query: str) -> str:
    """Retrieve relevant context from knowledge base for a question."""
    global _kb_had_results
    
    try:
        # ── Strategy 1: Exact invoice number match via metadata ──
        # Extract numbers that look like invoice numbers (6+ digits)
        invoice_numbers = re.findall(r'\b\d{6,}\b', query)
        
        if invoice_numbers:
            # Search with relaxed semantic constraint but filter by invoice number presence
            docs = vectorstore.similarity_search(query, k=10)
            
            for doc in docs:
                doc_text = doc.page_content
                
                # Check if any extracted invoice number appears in the document
                for inv_num in invoice_numbers:
                    if inv_num in doc_text:
                        _kb_had_results = True
                        print(f"🔍 Found exact invoice match for {inv_num}")
                        return f"Invoice found in knowledge base:\n\n{doc_text}"
        
        # ── Strategy 2: Semantic search with lower threshold ──
        docs = vectorstore.similarity_search_with_relevance_scores(query, k=5)
        
        if not docs:
            _kb_had_results = False
            return "No relevant information found in knowledge base."
        
        # Lower threshold from 0.5 → 0.3 for better recall
        score_threshold = 0.30
        
        filtered_docs = [(doc, score) for doc, score in docs if score >= score_threshold]
        
        if not filtered_docs:
            _kb_had_results = False
            logger.info(f"DEBUG: Top semantic scores were: {[score for _, score in docs]}")
            print(f"DEBUG: Top semantic scores were: {[score for _, score in docs]}")
            return "No relevant information found in knowledge base."
        
        _kb_had_results = True
        logger.info(f"🔍 Retrieved {len(filtered_docs)} document(s) via semantic search")
        print(f"🔍 Retrieved {len(filtered_docs)} document(s) via semantic search")
        
        return "\n\n".join([
            f"Source {i+1}:\n{doc.page_content[:800]}..." 
            for i, (doc, score) in enumerate(filtered_docs)
        ])
    
    except Exception as e:
        _kb_had_results = False
        logger.error(f"Error in retrieve_context: {e}")
        print(f"Error in retrieve_context: {e}")
        return f"Error retrieving context: {e}"

# ===========================================
# GENERATE-CHAT FUNCTION for Q&A (Invokes the LangChain agent)
# ===========================================
def generate_chat(user_input, chat_history):
    """Generate response via agent. Retrieval is forced before agent invocation."""
    
    user_input = extract_gradio_text(user_input)
    if not user_input:
        return chat_history if chat_history else []
    
    if chat_history is None:
        chat_history = []

    # 1. Moderation
    #if check_moderation_flag(user_input):
    #    updated = list(chat_history)
    #    updated.append({"role": "user", "content": user_input})
    #    updated.append({"role": "assistant", "content": "I apologize, I am programmed to answer medical questions only."})
    #    return clean_chat_history(updated)
    
    try:
        # FORCE retrieval before agent invocation
        kb_raw = retrieve_context.invoke({"query": user_input})
        logger.info(f"[KB] Retrieved: '{kb_raw[:200]}...'")
        print(f"[KB] Retrieved: '{kb_raw[:200]}...'")
        
        # Parse retrieved content
        kb_answer = None
        if kb_raw and not kb_raw.lower().startswith("no relevant"):
            # Try "answer:" line first
            for line in kb_raw.splitlines():
                if line.strip().lower().startswith("answer:"):
                    ans = line.strip().split(":", 1)[1].strip()
                    if len(ans) > 2 and ans[1] == "." and ans[0].isalpha():
                        ans = ans[2:].strip()
                    kb_answer = ans if ans else None
                    break
            
            # Fallback: use full content
            if not kb_answer:
                stripped = kb_raw.strip()
                kb_answer = stripped if stripped else None
        
        ## Decide whether to use KB answer directly or fall through to LLM
        #use_direct_kb = False

        use_direct_kb = True
        if use_direct_kb:
            # Factual/definition with good KB answer → return directly
            final_ai_message = kb_answer
            logger.info(f"[KB Direct] Factual answer: '{final_ai_message[:200]}...'")
            print(f"[KB Direct] Factual answer: '{final_ai_message[:200]}...'")
        else:
            
            #agent_messages.append({"role": "user", "content": user_input})
            #response = agent.invoke({"messages": agent_messages})
            
            final_ai_message = "I cannot find the best answer. Please consult with a Doctor."
            for msg in reversed(response.get('messages', [])):
                if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
                    final_ai_message = msg.content
                    break
        
        # Build history for Gradio
        updated = []
        for msg in chat_history:
            if isinstance(msg, dict):
                updated.append({
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", ""))
                })
        updated.append({"role": "user", "content": user_input})
        updated.append({"role": "assistant", "content": final_ai_message})

        return clean_chat_history(updated)

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"Error in generate_chat: {e}")
        print(f"Error in generate_chat: {e}")
        updated = []
        if chat_history:
            for msg in chat_history:
                if isinstance(msg, dict):
                    updated.append({
                        "role": msg.get("role", "user"),
                        "content": str(msg.get("content", ""))
                    })
        updated.append({"role": "user", "content": user_input})
        updated.append({"role": "assistant", "content": error_msg})
        return clean_chat_history(updated)        

# =========================================
# Copy all pdf files from dataset store repo to local cache
# =========================================
# Application path where images are stored
output_vectorstore_path = './invoices/output-images'

# Copy all pdf files from dataset store repo to local cache
for pdf_file in pdf_files_in_dataset_repo[:invoice_batch_size]:
    pdf_local_file_path = download_and_process(pdf_file)
    print(f"Processing: {pdf_local_file_path}")

for png_file in png_files_in_dataset_repo[:invoice_batch_size]:
    png_local_file_path = download_and_process(png_file)
    print(f"Processing: {png_local_file_path}")    

# Add document(pdf) embeddings into chroma store
# add_documents_to_vectorstore(vectorstore, pdf_local_file_path, output_vectorstore_path)

# =========================================
# Function to process OCR output text and turn it into a dictionary 
# containing all extracted invoice details in JSON format
# =========================================
@traceable(run_type="chain", name="Medical Document Processing Platform")
def process_invoice_ocr_output(output_text, invoice_ord: int = None, client_batch_id: str = None):
    """
    Parses the OCR output text to extract invoice table data into a pandas DataFrame
    and other invoice details into a structured dictionary.

    Args:
        output_text (str): The OCR extracted text from an invoice image.
        invoice_ord (int, optional): The order of the invoice in the batch. Defaults to None.
        client_batch_id (str, optional): A unique identifier for the batch. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - df_invoice_items (pd.DataFrame): DataFrame of invoice line items.
            - full_invoice_data (dict): A dictionary containing all extracted invoice details.
    """
    df_invoice_items = pd.DataFrame()
    invoice_details = {}
    provider_details = {}
    patient_details = {}
    full_invoice_data = {}

    # Extract the table part from the output_text
    table_start_marker = "<table>"
    table_end_marker = "</table>"

    # Find the starting and ending indices of the table
    table_start_index = output_text.find(table_start_marker)
    table_end_index = output_text.find(table_end_marker)

    if table_start_index != -1 and table_end_index != -1:
        # Extract the HTML table content
        html_table = output_text[table_start_index : table_end_index + len(table_end_marker)]

        # Use pandas to read the HTML table
        try:
            df_invoice_items = pd.read_html(io.StringIO(html_table))[0]
            if 'Amount' in df_invoice_items.columns:
                # Strip currency symbols → coerce bad values to NaN → replace NaN with None
                df_invoice_items['Amount'] = (
                    df_invoice_items['Amount']
                    .replace({r'[$,]': ''}, regex=True)
                    .replace(['', '--', '-', 'null', 'None'], pd.NA)
                )
                df_invoice_items['Amount'] = pd.to_numeric(df_invoice_items['Amount'], errors='coerce')
                df_invoice_items['Amount'] = df_invoice_items['Amount'].where(pd.notnull(df_invoice_items['Amount']), None)            
            # print("Parsed Invoice Items DataFrame:")
            # display(df_invoice_items.head())
        except ValueError as e:
            logger.error(f"Could not parse table data from OCR output: {e}")
            print(f"Could not parse table data from OCR output: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing the table: {e}")
            print(f"An unexpected error occurred while parsing the table: {e}")
    else:
        print("No table found in the OCR output.")

    # Parse non-table data - Extract Invoice Details in the specified order
    invoice_details["invoice_no"] = get_value("Invoice #", r"\*\*Invoice #:\*\*\s*(.*?)(?:\s*\*\*Invoice Date:|$)", output_text)
    invoice_date_str = get_value("Invoice Date", r"\*\*Invoice Date:\*\*\s*(.*?)(?:\s*<table>|$)", output_text)

    # Convert invoice_date to ISO 8601 string for JSON serialization
    if invoice_date_str:
        try:
            invoice_details["invoice_date"] = pd.to_datetime(invoice_date_str).isoformat() # Convert to ISO format string
        except Exception:
            invoice_details["invoice_date"] = invoice_date_str # Fallback to raw string if conversion fails
    else:
        invoice_details["invoice_date"] = None

    # Add the new fields to invoice_details in the specified order
    if invoice_ord is not None:
        invoice_details["invoice_ord"] = invoice_ord
    if client_batch_id is not None:
        invoice_details["client_batch_id"] = str(client_batch_id)
    
    # Extract Summary Information (handling potential variations in markdown list) in the specified order
    invoice_details["subtotal"] = currency_to_float(get_value("Subtotal", r"(?:- )?\*\*Subtotal:\*\*\s*(.*?)(?:\s*(?:- )?\*\*Insurance Adjustment:|$)", output_text))
    invoice_details["insurance_adjustment"] = currency_to_float(get_value("Insurance Adjustment", r"(?:- )?\*\*Insurance Adjustment:\*\*\s*(.*?)(?:\s*(?:- )?\*\*Patient Responsibility:|$)", output_text))
    invoice_details["patient_responsibility"] = currency_to_float(get_value("Patient Responsibility", r"(?:- )?\*\*Patient Responsibility:\*\*\s*(.*?)(?:\s*(?:- )?\*\*Total Due:|$)", output_text))
    invoice_details["amount_due"] = currency_to_float(get_value("Total Due", r"(?:- )?\*\*Total Due:\*\*\s*(.*?)(?:\n|$)", output_text))

    # Extract Provider Details
    provider_details["provider"] = get_value("Provider", r"\*\*Provider:\*\*\s*(.*?)(?:\s*\*\*Provider Address:|$)", output_text)
    provider_address_raw = get_value("Provider Address", r"\*\*Provider Address:\*\*\s*(.*?)(?:\s*\*\*Patient:|$)", output_text)
    provider_address_parsed = parse_address(provider_address_raw)
    provider_details.update(provider_address_parsed)

    # Extract Patient Details
    patient_details["patient"] = get_value("Patient", r"\*\*Patient:\*\*\s*(.*?)(?:\s*\*\*Patient Address:|$)", output_text)
    patient_address_raw = get_value("Patient Address", r"\*\*Patient Address:\*\*\s*(.*?)(?:\s*\*\*DOB:|$)", output_text)
    patient_address_parsed = parse_address(patient_address_raw)
    patient_details.update(patient_address_parsed)
    patient_details["dob"] = get_value("DOB", r"\*\*DOB:\*\*\s*(.*?)(?:\s*\*\*Insurance Policy #:|$)", output_text)
    patient_details["insurance_policy"] = get_value("Insurance Policy #", r"\*\*Insurance Policy #:\*\*\s*(.*?)(?:\s*\*\*Invoice #:|​\s*$)", output_text)
    
    print("\nParsed Invoice Details:")
    print(json.dumps(invoice_details, indent=4, default=str)) # Use default=str for datetime objects
    print("\nParsed Provider Details:")
    print(json.dumps(provider_details, indent=4))
    print("\nParsed Patient Details:")
    print(json.dumps(patient_details, indent=4))

    # Combine all extracted data into the new structure
    full_invoice_data = {
        "invoice_details": invoice_details,
        "provider_details": provider_details,
        "patient_details": patient_details,
        "invoice_line_items": df_invoice_items.to_dict(orient="records") if not df_invoice_items.empty else []
    }

    logger.info("\nFull Invoice Data (as dict) with new structure: ")
    print("\nFull Invoice Data (as dict) with new structure: ")
    print(json.dumps(full_invoice_data, indent=4, default=str))

    # 🩹 CRITICAL: strip NaN/Inf before it enters LangGraph state
    full_invoice_data = clean_floats(full_invoice_data)
    
    return df_invoice_items, full_invoice_data

# =========================================
# Function to extract data from invoices(.png files) in a batch using LightOnOCR Model
# =========================================
@traceable(run_type="chain",name="Extract Invoice Batch")
def extract_invoice_batch(model, processor, image_file_location, batch_size, device, dtype, chat_history):
    """
    Extracts data from a batch of PNG invoice images using a LightOnOCR model.

    Args:
        model: The pre-trained LightOnOCRForConditionalGeneration model.
        processor: The LightOnOcrProcessor for the model.
        image_file_location (str): The path to the folder containing PNG invoice images.
        batch_size (int): The number of invoices to process in this batch.
        device (str): The device to run the model on (e.g., "cuda", "mps", "cpu").
        dtype: The data type for model tensors (e.g., torch.float32, torch.bfloat16).
        If image_file_location is a folder path, scans it for PNGs.
        If it's a list of paths, uses those directly.
    
    Returns:
        list: A list of dictionaries, where each dictionary contains the processed data
              for an invoice in JSON format.
    """
    # ── Resolve input: list of paths or directory ──
    if isinstance(image_file_location, list):
        # Already have full paths (e.g. from HF dataset download)
        all_png_files = sorted(image_file_location)
    else:
        # It's a directory path — scan for PNGs
        all_png_files = sorted([
            os.path.join(image_file_location, f)
            for f in os.listdir(image_file_location)
            if f.endswith('.png')
        ])
    
    files_to_process = all_png_files[:batch_size]    
    
    # List to store all processed invoice data
    global all_processed_invoices
    all_processed_invoices = []
    output_text = None  # <-- FIX: initialize for except block
    
    chat_history = []
    if not batch_size:
        chat_history.append({
                "role": "user",
                "content": batch_size
            })
        return chat_history if chat_history else []
    
    updated_chat_history = list(chat_history)
    
    try:
        if not all_png_files:
            logger.info(f"No PNG files found in '{image_file_location}' to process.")
            print(f"No PNG files found in '{image_file_location}' to process.")
            return []
        else:
            # Process only the first 'batch_size' files as requested by the user
            files_to_process = all_png_files[0:batch_size]
            if not files_to_process:
                logger.info("No files to process within the specified batch size.")
                print("No files to process within the specified batch size.")
            else:
                logger.info(f"Found {len(all_png_files)} PNG files. Processing the first {len(files_to_process)} invoices as requested.")
                print(f"Found {len(all_png_files)} PNG files. Processing the first {len(files_to_process)} invoices as requested.")

                # Call get_client_batch_id function
                client_invoice_batch_id = get_client_batch_id()
                logger.info(f"Processing batch with client_invoice_batch_id: {client_invoice_batch_id}")
                print(f"Processing batch with client_invoice_batch_id: {client_invoice_batch_id}")
                
                for idx, png_file_name in enumerate(files_to_process):
                    invoice_ord = idx + 1 # Assign order based on iteration
                    image_file_path_str = os.path.join(image_file_location, png_file_name)
                    image_file_path_str = image_file_path_str.replace('invoices/invoices','invoices/')
                    logger.info(f"\nProcessing: {image_file_path_str} (Invoice Order: {invoice_ord})")
                    print(f"\nProcessing: {image_file_path_str} (Invoice Order: {invoice_ord})")
                    
                    # Encode image for the OCR model, resizing to a max dimension of 1200 pixels
                    base64_image, mime_type = _encode_image_to_base64(image_file_path_str, max_dimension=1200)
                    data_url = f"data:{mime_type};base64,{base64_image}"
                    
                    conversation = [{"role": "user", "content": [{"type": "image", "url": data_url}]}] 
                    
                    inputs = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad(): # Disable gradient calculations to save memory
                        output_ids = model.generate(**inputs, max_new_tokens=1024)
                    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
                    output_text = processor.decode(generated_ids, skip_special_tokens=True)
                    print(output_text)
                    
                    # Process the OCR output using the defined function, passing the new fields
                    df_items, invoice_json_data = process_invoice_ocr_output(output_text, invoice_ord=invoice_ord, client_batch_id=client_invoice_batch_id)
                    all_processed_invoices.append(invoice_json_data)
                    
                    print("\n" + "=" * 70 + "\n") # Separator for processed JSON output

            updated_chat_history.append({
                "role": "user",
                "content": batch_size
            })
        
        return all_processed_invoices

    except Exception as e:
        error_message = f"An unexpected error occurred in extract_invoie_batch function call: {str(e)}"
        logger.error(error_message)
        print(error_message)
        updated_chat_history.append({
            "role": "user",
            "content": output_text
        })        
        updated_chat_history.append({
            "role": "user",
            "content": error_message
        })
        return updated_chat_history, none

# =========================================
# Function to get the LightOnOCR model, processor
# =========================================
def get_model_processor():
    return model, processor, device, dtype

# =========================================
# Call extract_invoices_batch function to process invoices
# =========================================
def extract_invoice_batch_generate_json_format(chat_history):
    # Define the folder where PNG images are stored (from PDF to PNG conversion)
    # image_file_location = "/tmp/hf_invoices"
    # image_file_location = "/home/user/app/invoices"
    image_file_location = 'invoices'
    
    # Get model & processor
    model, processor, device, dtype = get_model_processor()
    
    # Call the function to process the invoices
    all_processed_invoices = extract_invoice_batch(model, processor, image_file_location, invoice_batch_size, device, dtype,chat_history)
    
    # Convert all_processed_invoices into a json format
    # Define the output file name for all processed invoices
    output_all_invoices_filename = "all_processed_invoices.json"
    
    # Ensure all_processed_invoices is defined
    if 'all_processed_invoices' not in globals():
        logger.info("Warning: 'all_processed_invoices' not found in global scope. Initializing as empty list.")
        print("Warning: 'all_processed_invoices' not found in global scope. Initializing as empty list.")
        all_processed_invoices = []

    if all_processed_invoices:
        # Save the combined list of invoice data to a JSON file
        with open(output_all_invoices_filename, 'w') as f:
            json.dump(all_processed_invoices, f, indent=4, default=str)

        logger.info(f"Successfully saved all processed invoice data to '{output_all_invoices_filename}'")
        print(f"Successfully saved all processed invoice data to '{output_all_invoices_filename}'")
        print("Content of the saved JSON file:")
    
        # Optionally, print the content of the JSON file to verify
        with open(output_all_invoices_filename, 'r') as f:
            print(f.read())
    else:
        logger.info(f"Quitting the invoice batch process.")
        print(f"Quitting the invoice batch process.")

# =========================================
# Helper function to convert dict into human-readable text
# =========================================
def format_invoice_for_indexing(invoice_data: dict, invoice_id: str = None) -> str:
    """
    Convert structured invoice dict into human-readable text for vector store indexing.
    """
    id = invoice_id or invoice_data.get("invoice_details", {}).get("invoice_id", "unknown")
    inv = invoice_data.get("invoice_details", {})
    prov = invoice_data.get("provider_details", {})
    pat = invoice_data.get("patient_details", {})
    items = invoice_data.get("invoice_line_items", [])
    
    lines = [
        # f"Invoice ID: {id}",
        f"**Invoice #:** {inv.get('invoice_no', '')}",
        f"**Invoice Date:** {inv.get('invoice_date', '')}",
        f"**Provider:** {prov.get('provider', '')}",
        f"**Provider Address:** {prov.get('address', '')}, {prov.get('city', '')}, {prov.get('state', '')} {prov.get('zipcode', '')}".strip().rstrip(','),
        f"**Patient:** {pat.get('patient', '')}",
        f"**Patient Address:** {pat.get('address', '')}, {pat.get('city', '')}, {pat.get('state', '')} {pat.get('zipcode', '')}".strip().rstrip(','),
        f"**DOB:** {pat.get('dob', '')}",
        f"**Insurance Policy:** {pat.get('insurance_policy_no', '')}",
        "",
        "**Services:**",
    ]
    
    for item in items:
        if isinstance(item, dict):
            desc = item.get('Description', item.get('description', 'Unknown service'))
            amt = item.get('Amount', item.get('amount', ''))
            lines.append(f"- {desc}: ${amt}")
        else:
            lines.append(f"- {str(item)}")
    
    lines.extend([
        "",
        f"**Subtotal:** ${inv.get('subtotal', '')}",
        f"**Insurance Adjustment:** ${inv.get('insurance_adjustment', '')}",
        f"**Patient Responsibility:** ${inv.get('patient_responsibility', '')}",
        f"**Total Due:** ${inv.get('amount_due', '')}",
    ])
    
    return "\n".join(lines)        

# =========================================
# Function to handle calling the Postgress cloud database 'persist_invoice_batch' RPC function with error handling.
# =========================================
@traceable(run_type="chain", name="Medical Document Processing Platform")
def insert_invoice_batch(postgres_url: str, postgres_key: str, payload: list) -> list:
    """
    Handles calling the Postgres 'persist_invoice_batch' RPC function with error handling.

    Args:
        postgres_url (str): The URL of the Postgres project.
        postgres_key (str): The API key for the Postgres project.
        payload (list): The list of processed invoice data (JSON-serializable).

    Returns:
        list: A list of invoice_id (UUIDs) if the call is successful, otherwise an empty list.
    """
    try:
        postgres = create_client(postgres_url, postgres_key)

        # 🩹 Double-defensive: clean any rogue NaN/Inf before JSON serialization
        clean_payload = clean_floats(payload)

        # Call the RPC function with the corrected parameter name 'p_payload'
        result = (
            postgres
            .rpc("persist_invoice_batch_v3", {"p_batch": payload})
            .execute()
        )

        # Check if result.data is not None and is a list
        if result and result.data and isinstance(result.data, list):
            # Extract invoice_ids and sort them by invoice_ord
            rows = result.data
            invoice_ids_in_order = [r["out_invoice_id"] for r in sorted(rows, key=lambda x: x["out_invoice_ord"])]
            return invoice_ids_in_order
        else:
            logger.info("Postgres RPC call successful, but no data or unexpected data format returned.")
            print("Postgres RPC call successful, but no data or unexpected data format returned.")
            return []

    except Exception as e:
        error_message = f"An error occurred during Postgres RPC call(insert_invoice_batch): {str(e)}"
        logger.error(error_message)
        print(error_message)
        return error_message

# ===========================================
# Call insert_invoice_batch function to persist invoice data to PostgreSQL cloud database
# ===========================================
@tool
# @mcp.tool()
@traceable(run_type="chain", name="Process Invoice Batch Function")
# @spaces.GPU
def process_invoice_batch():
    """
    Handles Invoice Batch Processing:
    1. Extracts invoice data from PNG files
    2. Persists to PostgreSQL via Supabase
    3. Indexes human-readable text to Chroma vector store (with retry)
    """
    
    MAX_RETRIES = 2  # total attempts = 1 + MAX_RETRIES
    
    # ── Step 1: Extract invoices ─────────────────────────────
    try:
        chat_history = []
        extract_invoice_batch_generate_json_format(chat_history)
    except Exception as e:
        logger.error(f"❌ EXTRACTION FAILED: {e}")
        print(f"❌ EXTRACTION FAILED: {e}")
        return f"❌ Invoice extraction failed: {str(e)}"
    
    if not all_processed_invoices or len(all_processed_invoices) == 0:
        return "⚠️ No invoices were extracted. Check the invoice folder and batch size."
    
    extracted_count = len(all_processed_invoices)
    logger.info(f"✅ Extracted {extracted_count} invoices")
    print(f"✅ Extracted {extracted_count} invoices")
    
    # ── Step 2: Persist to PostgreSQL ──────────────────────
    returned_invoice_ids = None
    try:
        returned_invoice_ids = insert_invoice_batch(POSTGRES_URL, POSTGRES_KEY, all_processed_invoices)
        
        if not isinstance(returned_invoice_ids, list):
            raise ValueError(f"DB returned non-list: {returned_invoice_ids}")
        if len(returned_invoice_ids) != extracted_count:
            raise ValueError(
                f"DB ID count mismatch: expected {extracted_count}, got {len(returned_invoice_ids)}"
            )
        logger.info(f"✅ Database persisted: {returned_invoice_ids}")
        print(f"✅ Database persisted: {returned_invoice_ids}")
        
    except Exception as e:
        print(f"❌ DATABASE PERSIST FAILED: {e}")
        logger.error(f"❌ DATABASE PERSIST FAILED: {e}")
        return (
            f"⚠️ Partial failure — extracted {extracted_count} invoices, "
            f"but database save failed: {str(e)}"
        )
    
    # ── Step 3: Index to vector store (with retry) ──────────
    docs_to_index = []
    for invoice_data, invoice_id in zip(all_processed_invoices, returned_invoice_ids):
        readable_text = format_invoice_for_indexing(invoice_data, invoice_id)
        docs_to_index.append(
            Document(
                page_content=readable_text,
                metadata={
                    "invoice_id": invoice_id,
                    "source_type": "batch_process",
                    "batch_processed": True
                }
            )
        )
    
    last_exception = None
    # Retry if it fails to index
    for attempt in range(1 + MAX_RETRIES):
        try:
            if attempt > 0:
                wait_time = attempt * 2  # 2s, 4s backoff
                logger.info(f"🔄 Retry attempt {attempt}/{MAX_RETRIES} after {wait_time}s...")
                print(f"🔄 Retry attempt {attempt}/{MAX_RETRIES} after {wait_time}s...")
                time.sleep(wait_time)
            
            vectorstore.add_documents(docs_to_index)
            vectorstore.persist()

            logger.info(f"✅ Vector store indexed: {len(docs_to_index)} invoices")
            print(f"✅ Vector store indexed: {len(docs_to_index)} invoices")
            break  # success — exit retry loop
            
        except Exception as e:
            last_exception = e
            logger.error(f"⚠️ Vector store index attempt {attempt + 1} failed: {e}")
            print(f"⚠️ Vector store index attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES:
                # Exhausted retries
                print(f"❌ VECTOR STORE INDEX FAILED after {1 + MAX_RETRIES} attempts")
                return (
                    f"⚠️ Partial failure — extracted {extracted_count} invoices and "
                    f"saved to database (IDs: {returned_invoice_ids}), "
                    f"but vector store indexing failed after retries: {str(e)}"
                )
    
    # ── All steps succeeded ─────────────────────────────────
    return (
        f"✅ Fully processed {extracted_count} invoices.\n"
        f"💾 Database IDs: {returned_invoice_ids}\n"
        f"📚 Indexed to vector store: {len(docs_to_index)} records"
    )

# ===========================================
# Warpper function to process_invoice_batch function so that it can be called directly from the process batch button click event
# ===========================================
def run_invoice_batch(chat_history):
    """
    Triggered by the 'Process Batch' button in Gradio UI.
    """
    image_folder = 'invoices'  # or wherever your batch folder is
    batch_size = invoice_batch_size  # from env var
    
    return batch_process_invoices(image_folder, batch_size, chat_history)
    
# ===========================================
# Function to process images
# ===========================================
@traceable(run_type="chain", name="MedInvoice Image Analysis")
# @spaces.GPU
def process_invoice_image_for_chat(image_file_path_str, image_question, chat_history):
    print(f"DEBUG: image={image_file_path_str}, question={image_question}")

    if chat_history is None:
        chat_history = []

    updated_chat_history = list(chat_history)

    # Validation
    if not image_file_path_str or not isinstance(image_file_path_str, str) or not os.path.isfile(image_file_path_str):
        error_message = "❌ Error: Please upload a valid medical invoice image."
        return chat_history + [{"role": "assistant", "content": error_message}], None

    if not image_question or not image_question.strip():
        image_question = "What is the total amount due on this invoice?"

    # Load image with PIL
    image = Image.open(image_file_path_str)

    # Get model & processor (loaded at module level)
    model, processor, device, dtype = get_model_processor()
    
    try:
        # Use plain dicts, NOT LangChain objects
        messages = [
            {"role": "system", "content": VISION_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": image_question},
                {"type": "image"}
            ]}
        ]

        # Get the formatted prompt
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,  # get text first, then process with image
        )

        # Process text + image together
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Move to device / dtype
        inputs = {
            k: v.to(device=device, dtype=dtype) if v.dtype in (torch.float16, torch.float32, torch.bfloat16) else v.to(device)
            for k, v in inputs.items()
        }

        output_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)
        logger.info(f" Generated: {output_text}")
        print("Generated:", output_text)

        updated_chat_history.append({
            "role": "user",
            "content": f"📎 Image: {image_question}"
        })
        updated_chat_history.append({
            "role": "assistant",
            "content": output_text
        })

    except Exception as e:
        import traceback
        error_message = f"⚠️ Error analyzing invoice image: {str(e)}"
        logger.error(error_message)        
        print(error_message)
        print(traceback.format_exc())  # <-- print full traceback for debugging
        updated_chat_history.append({
            "role": "user",
            "content": f"📎 Image: {image_question}"
        })
        updated_chat_history.append({
            "role": "assistant",
            "content": error_message
        })

    return updated_chat_history, None

# ===========================================
# Functions for Review Queue
# ===========================================
def load_pending_reviews():
    try:
        postgres = create_client(POSTGRES_URL, POSTGRES_KEY)
        res = (
            postgres
            .rpc("load_pending_review", {} )
            .execute()
        )
        
        if not res.data:
            return [["-", "-", "-", "-", "No pending reviews", "-"]]
            
        rows = []
        for r in res.data:
            extracted = r.get("extracted_json", {}) or {}
            inv = extracted.get("invoice_details", {})
            prov = extracted.get("provider_details", {})
            pat = extracted.get("patient_details", {})
            
            rows.append([
                str(r["id"]),
                inv.get("invoice_no", "N/A"),
                prov.get("provider", "N/A"),
                pat.get("patient", "N/A"),
                "; ".join(r.get("failure_reasons", [])),
                r.get("created_at", "N/A")
            ])
        return rows
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return [["-", "-", "-", "-", f"Error: {str(e)}", "-"]]

def approve_review(review_id: str):
    if not review_id or not review_id.strip():
        return "⚠️ Please enter a Review ID"
    
    review_id = review_id.strip()
    try:
        # 1. Fetch pending review
        postgres = create_client(POSTGRES_URL, POSTGRES_KEY)
        res = (
            postgres
            .rpc("load_pending_review",
                 {
                     "p_review_id": review_id
                 }
                )
            .execute()
        )
        
        if not res.data or len(res.data) == 0:
            return f"❌ Review {review_id} not found or already processed"
            
        record = res.data[0]
        extracted = record.get("extracted_json")
        
        if not extracted:
            return f"❌ No extracted data in review {review_id}"
        
        # 2. Persist to PostgreSQL
        returned_ids = insert_invoice_batch(POSTGRES_URL, POSTGRES_KEY, [extracted])
        
        if not isinstance(returned_ids, list) or len(returned_ids) == 0:
            return f"❌ DB persist failed. Response: {returned_ids}"
            
        invoice_id = returned_ids[0]
        
        # 3. Index to vector store
        text_to_index = format_invoice_for_indexing(extracted, invoice_id)
        doc = Document(
            page_content=text_to_index,
            metadata={
                "invoice_id": invoice_id,
                "source": record.get("image_path", "review_queue"),
                "type": "invoice_image",
                "invoice_no": extracted.get("invoice_details", {}).get("invoice_no"),
                "approved_from_review": True
            }
        )
        vectorstore.add_documents([doc])
        vectorstore.persist()
        
        # 4. Mark review as approved
        res = (
            postgres
            .rpc("approve_review",
                 {
                     "p_review_id": review_id,
                     "p_reviewer_comment": "Approved as the invoice contains sufficient details."
                 }
                )
            .execute()
        )

        if not res.data:
            return f"❌ Review approval failed. Response: {review_id}"

        logger.info(f"✅ Approved {review_id}. DB Invoice ID: {invoice_id}. Indexed to vector store.")
        return f"✅ Approved {review_id}. DB Invoice ID: {invoice_id}. Indexed to vector store."
        
    except Exception as e:
        logger.error(f"❌ Approval error: {str(e)}")
        return f"❌ Approval error: {str(e)}"

def reject_review(review_id: str):
    if not review_id or not review_id.strip():
        return "⚠️ Please enter a Review ID"
    
    review_id = review_id.strip()
    try:
        postgres = create_client(POSTGRES_URL, POSTGRES_KEY)
        res = (
            postgres
            .rpc("reject_review",
                 {
                     "p_review_id": review_id,
                     "p_reviewer_comment": "Rejected due to Insufficient Invoice Details."
                 }
                )
            .execute()
        )          

        logger.info(f"❌ Rejected review {review_id}")
        return f"❌ Rejected review {review_id}"
    except Exception as e:
        logger.error(f"❌ Rejection error: {str(e)}")
        return f"❌ Rejection error: {str(e)}"

def save_pending_review(image_path: str, raw_text: str, extracted: dict, reasons: list):
    try:
        postgres = create_client(POSTGRES_URL, POSTGRES_KEY)
        result = postgres.table("invoice_reviews").insert({
            "image_path": image_path,
            "raw_ocr_text": raw_text,
            "extracted_json": extracted,
            "failure_reasons": reasons,
            "status": "pending"
        }).execute()
        review_id = result.data[0]['id']
        logger.info(f"✅ Queued review record: {review_id}")
        print(f"✅ Queued review record: {review_id}")
        return review_id
    except Exception as e:
        logger.error(f"❌ Failed to queue review: {e}")
        print(f"❌ Failed to queue review: {e}")
        return None      

# LangGraph Implementation for Deterministic Flow

# ===========================================
# 1. LangGraph State
# ===========================================
class InvoiceState(TypedDict):
    messages: list
    image_path: Optional[str]       # For single: image file path. For batch: folder path
    question: str
    rag_answer: Optional[str]
    extracted_data: Optional[dict]
    vision_answer: Optional[str]
    invoice_id: Optional[str]
    error: Optional[str]
    final_answer: str
    review_status: Optional[str]    # "approved" | "pending_review" | "rejected"
    review_reason: Optional[str]
    review_id: Optional[str]       # NEW: for tracking queued reviews
    batch_size: Optional[int]        # NEW: for batch processing
    batch_results: Optional[list]   # NEW: list of {invoice, status, review_id}

# ===========================================
# 2. Graph Nodes
# ===========================================

def rag_node(state: InvoiceState):
    """Node 1: Search vector store for existing invoice info."""
    question = state["question"]
    rag_raw = retrieve_context.invoke({"query": question})
    
    has_answer = (
        rag_raw
        and not rag_raw.lower().startswith("no relevant")
        and not rag_raw.lower().startswith("error")
    )
    return {"rag_answer": rag_raw if has_answer else None}

def router(state: InvoiceState):
    """Route to vision only if RAG found nothing."""
    # If RAG itself errored, skip to vision rather than surfacing a broken error string
    return "found" if state.get("rag_answer") else "vision"

def vision_node(state: InvoiceState):
    """Node 2: Run vision OCR and extract structured data."""
    image_path = state["image_path"]
    question = state["question"]
    
    result_history, _ = process_invoice_image_for_chat(image_path, question, [])
    
    raw_text = "No output extracted."
    for msg in reversed(result_history):
        if msg.get("role") == "assistant":
            raw_text = msg.get("content", "")
            break

    # Parse structured data
    extracted_data = None
    
    try:
        client_invoice_batch_id = get_client_batch_id()
        print(f"✓ client_batch_id is: {client_invoice_batch_id}")
        if not client_invoice_batch_id:
            return {"error": "No client_batch_id generated."}
            
        _, extracted_data = process_invoice_ocr_output(raw_text, client_batch_id=client_invoice_batch_id)
        print(f"✓ Parsed invoice data: {json.dumps(extracted_data, indent=2, default=str)}")
    except Exception as e:
        logger.error(f"⚠️ OCR parsing warning: {e}")
        print(f"⚠️ OCR parsing warning: {e}")
        extracted_data = {"raw_text": raw_text, "parse_error": str(e)}
    
    return {
        "vision_answer": raw_text,
        "extracted_data": extracted_data
    }

def persist_postgres_node(state: InvoiceState):
    """Node 3: Persist invoice to PostgreSQL with retry. Return invoice_id."""
    extracted = state.get("extracted_data")
    if not extracted:
        return {"error": "No extracted data available to persist"}

    # Max no. of retry attempts after a failure
    MAX_RETRIES = 2
    
    invoice_id = None
    last_error = None
    
    for attempt in range(1 + MAX_RETRIES):
        try:
            if attempt > 0:
                wait_time = attempt * 2
                logger.info(f"🔄 DB retry attempt {attempt}/{MAX_RETRIES} after {wait_time}s...")
                print(f"🔄 DB retry attempt {attempt}/{MAX_RETRIES} after {wait_time}s...")
                time.sleep(wait_time)
            
            result = insert_invoice_batch(POSTGRES_URL, POSTGRES_KEY, [extracted])
            
            if isinstance(result, list) and len(result) > 0:
                invoice_id = result[0]
                logger.info(f"✅ Persisted invoice to DB with ID: {invoice_id}")
                print(f"✅ Persisted invoice to DB with ID: {invoice_id}")
                break
            else:
                raise ValueError(f"Unexpected DB response: {result}")
                
        except Exception as e:
            last_error = str(e)
            logger.error(f"⚠️ DB persist attempt {attempt + 1} failed: {e}")
            print(f"⚠️ DB persist attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES:
                logger.error(f"❌ DB persist failed after {1 + MAX_RETRIES} attempts")
                print(f"❌ DB persist failed after {1 + MAX_RETRIES} attempts")
                return {"error": f"Database persist failed: {last_error}"}
    
    return {"invoice_id": invoice_id}

def index_vectorstore_node(state: InvoiceState):
    """Add invoice to vector store with invoice number in metadata for exact lookup."""
    extracted = state.get("extracted_data")
    invoice_id = state.get("invoice_id")

    MAX_RETRIES = 2 # Total attempts = 1 + MAX_RETRIES
    
    if not extracted:
        return {}
    
    invoice_no = extracted.get("invoice_details", {}).get("invoice_no")
    
    text_to_index = format_invoice_for_indexing(extracted, invoice_id)
    
    doc = Document(
        page_content=text_to_index,
        metadata={
            "source": state.get("image_path", "unknown"),
            "type": "invoice_image",
            "invoice_id": invoice_id,      # DB UUID
            "invoice_no": invoice_no,      # ← NEW: the actual invoice number like "859269226"
        }
    )

    for attempt in range(1 + MAX_RETRIES):
        try:
            if attempt > 0:
                wait_time = attempt * 2  # 2s, 4s backoff
                logger.info(f"🔄 Retry attempt {attempt}/{MAX_RETRIES} after {wait_time}s...")
                print(f"🔄 Retry attempt {attempt}/{MAX_RETRIES} after {wait_time}s...")
                time.sleep(wait_time)

            vectorstore.add_documents([doc])
            vectorstore.persist()

            logger.info(f"✅ Indexed invoice {invoice_no} (DB ID: {invoice_id})")
            print(f"✅ Indexed invoice {invoice_no} (DB ID: {invoice_id})")
            return {}

        except Exception as e:
            last_exception = e
            logger.error(f"⚠️ Vector store index attempt {attempt + 1} failed: {e}")
            print(f"⚠️ Vector store index attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES:
                logger.error(f"❌ Vector index failed after {1 + MAX_RETRIES} attempts")
                print(f"❌ Vector index failed after {1 + MAX_RETRIES} attempts")
                return {"error": f"Vector index failed: {last_error}"}

def respond_node(state: InvoiceState):
    error = state.get("error")
    rag_answer = state.get("rag_answer")
    vision_answer = state.get("vision_answer")
    invoice_id = state.get("invoice_id")
    
    # ── PRESERVE HITL / review messages already composed ──
    if state.get("final_answer"):
        answer = state["final_answer"]
    elif rag_answer:
        answer = rag_answer
    elif vision_answer:
        answer = vision_answer
    else:
        answer = "No answer available."
    
    if error:
        answer += f"\n\n⚠️ Processing error: {error}"
    elif invoice_id and not rag_answer:
        answer += f"\n\n💾 Invoice saved to database with ID: `{invoice_id}`"
    
    # Add review ID reference if applicable
    if state.get("review_status") == "pending_review" and state.get("review_id"):
        answer += f"\n\n🆔 Review ID: `{state['review_id']}`"
    
    return {"final_answer": answer}

def validate_invoice_node(state: InvoiceState):
    extracted = state.get("extracted_data", {})
    image_path = state.get("image_path")
    vision_answer = state.get("vision_answer", "")
    reasons = []

    # ── 1. IMAGE QUALITY ──
    if image_path and os.path.isfile(image_path):
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if lap_var < 80:
                    reasons.append(f"Image blurry (lap_var={lap_var:.1f})")
        except Exception as e:
            logger.error(f"Blur check skipped: {e}")
            print(f"Blur check skipped: {e}")

    # ── 2. HEADER FIELD COMPLETENESS ──
    inv = extracted.get("invoice_details", {})
    prov = extracted.get("provider_details", {})
    pat = extracted.get("patient_details", {})

    required = {
        "invoice_no": inv.get("invoice_no"),
        "invoice_date": inv.get("invoice_date"),
        "provider": prov.get("provider"),
        "patient": pat.get("patient"),
        "amount_due": inv.get("amount_due"),
    }
    for field, val in required.items():
        if not val or str(val).strip() in ("", "None", "null"):
            reasons.append(f"Missing or empty: {field}")

    # ── 3. LINE ITEM DATA QUALITY ──
    items = extracted.get("invoice_line_items", [])
    
    def is_real_item(item):
        code = str(item.get("line_item_code") or item.get("Code") or "").strip()
        desc = str(item.get("line_item_description") or item.get("Description") or "").strip()
        amt = item.get("line_item_amount") or item.get("Amount")
        
        if str(amt).lower() in ("nan", "nat", "none", "null", ""):
            amt = None
            
        if code.lower() in ("code:", "code", "item code", "service code"):
            return False
        if desc.lower() in ("description", "desc", "service"):
            return False
            
        if not code and not desc:
            return False
        try:
            float(str(amt).replace('$', '').replace(',', ''))
        except (ValueError, TypeError):
            return False
        return True

    real_items = [it for it in items if is_real_item(it)]
    
    if not real_items:
        reasons.append("Invoice line items are unreadable, empty, or could not be parsed")
    elif len(real_items) < len(items):
        reasons.append(f"{len(items) - len(real_items)}/{len(items)} line item rows are junk/headers")

    # ── 4. DB DUPLICATE CHECK ──
    invoice_no = inv.get("invoice_no")
    if invoice_no:
        existing = lookup_invoice_in_database(invoice_no)
        if existing:
            reasons.append(f"Invoice {invoice_no} already exists")

    # ── 5. ROUTE TO HITL IF ANY ISSUES ──
    if reasons:
        review_id = save_pending_review(image_path, vision_answer, extracted, reasons)
        return {
            "review_status": "pending_review",
            "review_reason": "; ".join(reasons),
            "review_id": review_id,
            "final_answer": (
                "⚠️ **This invoice requires manual review before it can be saved.**\n\n"
                "**Reasons detected:**\n" + "\n".join(f"- {r}" for r in reasons)
            )
        }

    return {"review_status": "approved", "review_reason": None, "review_id": None}

def human_review_node(state: InvoiceState):
    """Return the already-composed warning message to the user."""
    return {"final_answer": state.get("final_answer", "Manual review required.")}

def batch_process_invoices(image_folder: str, batch_size: int, chat_history):
    """
    Process multiple invoices by invoking the existing invoice_agent per image.
    Each invoice goes through the full pipeline: vision → validate → persist/index OR HITL queue.
    """
    # Resolve files
    if isinstance(image_folder, list):
        all_png_files = sorted(image_folder)
    else:
        all_png_files = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.endswith('.png')
        ])
    
    files_to_process = all_png_files[:batch_size]
    
    if not files_to_process:
        chat_history = list(chat_history) if chat_history else []
        chat_history.append({
            "role": "assistant",
            "content": "⚠️ No PNG invoices found to process."
        })
        return chat_history
    
    # Results tracking
    approved_count = 0
    review_count = 0
    failed_count = 0
    review_ids = []
    persisted_ids = []
    failed_files = []
    
    client_batch_id = get_client_batch_id()
    print(f"📦 Starting batch {client_batch_id} with {len(files_to_process)} invoices")
    
    for idx, png_path in enumerate(files_to_process):
        invoice_ord = idx + 1
        print(f"\n{'='*60}")
        print(f"Processing {invoice_ord}/{len(files_to_process)}: {png_path}")
        print(f"{'='*60}")
        
        try:
            # Invoke the EXISTING single-invoice graph
            result = invoice_agent.invoke({
                "messages": [],
                "image_path": png_path,
                "question": "Extract all invoice details and line items from this image.",
            })
            
            status = result.get("review_status", "unknown")
            
            if status == "pending_review":
                review_count += 1
                rid = result.get("review_id", "N/A")
                review_ids.append(rid)
                print(f"⏸️ Queued for review: {rid}")
                
            elif status == "approved":
                approved_count += 1
                pid = result.get("invoice_id", "N/A")
                persisted_ids.append(pid)
                print(f"✅ Approved and persisted: {pid}")
                
            else:
                failed_count += 1
                failed_files.append(os.path.basename(png_path))
                print(f"❌ Unknown status: {status}")
                
        except Exception as e:
            failed_count += 1
            failed_files.append(os.path.basename(png_path))
            logger.error(f"❌ Exception processing {png_path}: {e}")
            print(f"❌ Exception processing {png_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Build summary message
    summary_lines = [
        f"📦 **Batch Processing Complete** — Batch ID: `{client_batch_id}`",
        f"",
        f"📁 Total invoices scanned: **{len(files_to_process)}**",
        f"✅ Auto-approved & persisted: **{approved_count}**",
        f"⏸️ Sent to review queue: **{review_count}**",
    ]
    
    if failed_count > 0:
        summary_lines.append(f"❌ Failed / errors: **{failed_count}**")
    
    if persisted_ids:
        summary_lines.append(f"\n💾 Persisted Invoice IDs: {', '.join(str(x) for x in persisted_ids if x)}")
    
    if review_ids:
        summary_lines.append(f"\n🆔 Review Queue IDs: {', '.join(str(x) for x in review_ids if x)}")
    
    if failed_files:
        summary_lines.append(f"\n⚠️ Failed files: {', '.join(failed_files)}")
    
    summary_lines.append(f"\n_Open the **🔍 Review Queue** tab to approve pending invoices._")
    
    chat_history = list(chat_history) if chat_history else []
    chat_history.append({
        "role": "assistant",
        "content": "\n".join(summary_lines)
    })
    
    return chat_history    

# ===========================================
# 3. Build & Compile the Graph
# ===========================================
builder = StateGraph(InvoiceState)

builder.add_node("rag", rag_node)
builder.add_node("vision", vision_node)
builder.add_node("validate", validate_invoice_node)
builder.add_node("persist_postgres", persist_postgres_node)
builder.add_node("index_vectorstore", index_vectorstore_node)
builder.add_node("human_review", human_review_node)
builder.add_node("respond", respond_node)

builder.set_entry_point("rag")
builder.add_conditional_edges(
    "rag",
    router,
    {"found": "respond", "vision": "vision"}
)

# After vision, run validation
builder.add_edge("vision", "validate")

# Validation routes to either persist or human review
builder.add_conditional_edges(
    "validate",
    lambda s: "human_review" if s.get("review_status") == "pending_review" else "persist_postgres",
    {"human_review": "human_review", "persist_postgres": "persist_postgres"}
)

# Approved path
builder.add_edge("persist_postgres", "index_vectorstore")
builder.add_edge("index_vectorstore", "respond")

# Pending review path
builder.add_edge("human_review", "respond")

builder.add_edge("respond", END)

invoice_agent = builder.compile()

# ===========================================
# 4. Gradio Event Handler
# ===========================================
@traceable(run_type="chain", name="Invoice Submit")
def handle_invoice_submit(image_path, question, chat_history):
    chat_history = list(chat_history) if chat_history else []
    current_run_id = None  # <-- For LangSmith feedback

    # Capture LangSmith run ID from the agent execution
    try:
        run_tree = get_current_run_tree()
        current_run_id = run_tree.id if run_tree else None
    except Exception:
        current_run_id = None    
    
    if not question or not question.strip():
        return chat_history + [{"role": "assistant", "content": "Please enter a question about the invoice."}], None, current_run_id
    
    # ── SCENARIO 1: Image uploaded → Full LangGraph pipeline ──
    if image_path and isinstance(image_path, str) and os.path.isfile(image_path):
        result = invoice_agent.invoke({
            "messages": chat_history,
            "image_path": image_path,
            "question": question,
        })
        
        chat_history.append({"role": "user", "content": f"📎 {question}"})
        chat_history.append({"role": "assistant", "content": result["final_answer"]})
        return chat_history, None, current_run_id
    
    # ── SCENARIO 2: Text-only question (no image) ──
    else:
        # 2a: Try vector store RAG
        rag_raw = retrieve_context.invoke({"query": question})
        has_answer = (
            rag_raw 
            and not rag_raw.lower().startswith("no relevant")
            and not rag_raw.lower().startswith("error")
        )
        
        if has_answer:
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": rag_raw})
            return chat_history, None, current_run_id
        
        # 2b: RAG missed → Query PostgreSQL by invoice number
        invoice_no = extract_invoice_number(question)
        
        if invoice_no:
            print(f"🔍 RAG missed. Querying PostgreSQL for invoice: {invoice_no}")
            db_result = lookup_invoice_in_database(invoice_no)
            
            chat_history.append({"role": "user", "content": question})
            
            if db_result:
                # Cache DB result to vector store for future RAG
                vectorstore.add_documents([Document(
                     page_content=db_result,
                     metadata={"invoice_no": invoice_no, "source": "db_cache"}
                 )])
                vectorstore.persist()
                
                chat_history.append({"role": "assistant", "content": db_result})
            else:
                chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Invoice **{invoice_no}** does not exist in the system. Please verify the invoice number or upload the invoice image."
                })
            return chat_history, None, current_run_id
        
        # 2c: No invoice number and RAG missed
        chat_history.append({"role": "user", "content": question})
        chat_history.append({
            "role": "assistant",
            "content": "I couldn't find that invoice in the system. Please provide the invoice number or upload the invoice image."
        })
        return chat_history, None, current_run_id     

# =======================================
# Run the MCP server in a thread to keep the UI responsive
# =======================================
# def run_mcp_server():
#     # Use streamable-http for compatibility
#     mcp.run(transport="streamable-http")

# server_thread = threading.Thread(target=run_mcp_server, daemon=True)
# server_thread.start()

# async def run_agent():
#     # =======================================
#     # Connect the client to the MCP server running in the thread
#     # Note: FastMCP default streamable-http port is often 8000
#     # =======================================
#     client = MultiServerMCPClient({
#         "math": {
#             "transport": "http",
#             "url": "http://localhost:8000/mcp",
#         }
#     })
    
#     # Discover and use tools
#     tools = await client.get_tools()
    
#     # Tools for LangChain Agent
#     #tools = [vectorstore, process_invoice_batch]    
    
    # =======================================
    # LangChain - Global Agent Setup
    # =======================================
    
#     agent = create_agent(
#         model,
#         tools=tools,
#         system_prompt=SYSTEM_PROMPT,
#     )
    
#     # Test the MCP Tool
#     math_addition_response = await agent.ainvoke({"messages": [{"role": "user", "content": "What is 10 + 5?"}]})
#     print(math_addition_response["messages"][-1].content)

# run_agent()

# Tools for LangChain Agent
tools = [retrieve_context, process_invoice_batch]    

# =======================================
# LangChain - Global Agent Setup
# =======================================

agent = create_agent(
    model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)

# ===========================================
# Create the Gradio app - UI
# ===========================================
with gr.Blocks() as demo:  
    
    with gr.Tab("🏥 Medical Invoice Assistant"):
                
        gr.Markdown(
            """
            <h2 style='text-align: center;'>
                Medical Invoice Assistant
            </h2>
            <p style='text-align: center; color: #555;'>
                Process medical invoices in a batch automatically, upload an invoice image and ask questions, see pending invoices in a review queue for human approval.
            </p>
            """
        )    

        run_id_state = gr.State(value=None)  

        with gr.Row():

            with gr.Column(scale=3):
                
                with gr.Column(elem_classes="input-panel"):
                    chatbot = gr.Chatbot(
                        label="Conversation History",
                        height=360,
                        allow_tags=False,
                    )

                # Feedback UI
                with gr.Row():
                    with gr.Column(scale=0, min_width=180):
                        gr.Textbox(
                            value="Rate this response:", 
                            show_label=False, 
                            container=False, 
                            interactive=False
                        )
                    with gr.Column(scale=0, min_width=50):
                        btn_up = gr.Button("👍", min_width=50, variant="secondary")
                    with gr.Column(scale=0, min_width=50):
                        btn_down = gr.Button("👎", min_width=50, variant="secondary")
                    with gr.Column(scale=1):
                        pass  # spacer
                    with gr.Column(scale=0, min_width=170):
                        clear_btn = gr.ClearButton(value="🗑️ Clear Chat")                

                # Feedback Input/Text input panel
                with gr.Column(elem_classes="input-panel"):
                    fb_comment = gr.Textbox(
                        placeholder="💬 Leave feedback",
                        show_label=False,
                        lines=1,
                        elem_classes=["prompt-box"]
                    )
                    fb_status = gr.Textbox(
                        value="", 
                        show_label=False, 
                        interactive=False,
                        container=False,
                        visible=True
                    )                    
                    image_question = gr.Textbox(
                        show_label=False, 
                        placeholder="What is the total amount due on this invoice?",
                        lines=2,
                        elem_classes=["prompt-box"]
                    )

                with gr.Row():
                    submit_btn = gr.Button("💬 Submit Question", variant="primary", scale=1)

                with gr.Column():

                    with gr.Row(elem_classes="equal-height"):
                        # Left: Upload Invoice Image
                        with gr.Column(scale=1, elem_classes=["input-panel-mod", "multimodal-card"]):
                            with gr.Row():
                                gr.HTML("<div class='panel-badge-mod'>Upload Invoice Image</div>")
                            with gr.Row():
                                upload_file = gr.File(
                                    file_types=["image"],
                                    label="",
                                    min_width=130,
                                    scale=1
                                )                              
                            preview = gr.Image(
                                label="Preview", 
                                visible=False, 
                                height=180
                            )
                        
                        # Middle: Extract Invoice(s)
                        with gr.Column(scale=1, min_width=100, elem_classes=["input-panel-mod", "multimodal-card"]):
                            with gr.Row():
                                gr.HTML(f"<div class='panel-badge-mod'>Admin</div>")
                            with gr.Row():
                                gr.HTML(f"<div class='panel-badge-mod'>Invoice Batch (PG|KB) / Batch Size: {invoice_batch_size}</div>")
                                process_invoice_batch_btn = gr.Button("Process Batch",
                                                                 variant="primary",
                                                                 min_width=200
                                                             )
                            with gr.Row():
                                gr.HTML("<div class='panel-badge'>Knowledge Base / Save Vector Store</div>")
                            with gr.Row():
                                # gr.HTML("<div class='panel-badge-mod'>Save Vector Store:</div>")
                                save_vectordb_btn = gr.Button("Index Invoice",
                                                                 variant="primary",
                                                                 min_width=200
                                                                )
                        # Right: Reporting Dashboard
                        with gr.Column(scale=3, min_width=130, elem_classes=["input-panel-mod", "multimodal-card"]):
                            with gr.Row():
                                gr.HTML("<div class='panel-badge-mod'>Reporting Dashboard</div><div class='panel-badge-mod2'>(Open report in new tab)</div>")
                            with gr.Row():
                                with gr.Column(scale=3, min_width=65, elem_classes=["input-panel-mod", "multimodal-card"]):
                                    # with gr.Row():
                                    #     gr.Button("",
                                    #               variant="secondary",
                                    #               min_width=25
                                    #              )
                                    with gr.Row():
                                        gr.Button("📊 Invoices by Date",
                                                  link="https://app.powerbi.com/groups/me/reports/088dede8-660a-4f72-8be6-ba61523a0403/be5c721176c1733a28ec?experience=power-bi",
                                                  variant="primary",
                                                  min_width=200
                                                 )
                                    # 📊 Power BI button — opens in new tab
                                    with gr.Row():
                                        gr.Button(
                                            "📊 Invoice by Total Amt Due",
                                            link="https://app.powerbi.com/groups/me/reports/3978942e-0d1b-4d4a-a871-5cb255b14318/dfb504468e2c29b950a5?experience=power-bi",
                                            variant="primary",
                                            min_width=200
                                            )                                

                                with gr.Column(scale=3, min_width=65, elem_classes=["input-panel-mod", "multimodal-card"]):
                                    with gr.Row():
                                        gr.Button("📊 Invoices by Line Items",
                                                  link="https://app.powerbi.com/groups/me/reports/586c08c9-5c15-4f38-878a-a86b25b131a8/08faf32e068341d017cb?experience=power-bi",
                                                  variant="primary",
                                                  min_width=200
                                                 )

    with gr.Tab("🔍 Review Queue"):
        gr.Markdown("## Pending Invoice Reviews")
        
        review_table = gr.Dataframe(
            headers=["Review ID", "Invoice #", "Provider", "Patient", "Failure Reasons", "Created"],
            label="Pending Reviews"
        )
        
        # Auto-load on startup
        demo.load(load_pending_reviews, outputs=review_table)
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Queue")
        
        with gr.Row():
            review_id_input = gr.Textbox(label="Review ID", placeholder="Paste review UUID here")
        
        with gr.Row():
            approve_btn = gr.Button("✅ Approve & Save to DB", variant="primary")
            reject_btn = gr.Button("❌ Reject", variant="secondary")
        
        review_action_output = gr.Textbox(
            label="Result",
            interactive=False,
            visible=True
        )
        
        # Bind events
        refresh_btn.click(load_pending_reviews, outputs=review_table)
        approve_btn.click(approve_review, inputs=review_id_input, outputs=review_action_output)
        reject_btn.click(reject_review, inputs=review_id_input, outputs=review_action_output)                                
    
    # ===========================================
    # Function enables LLM to respond to user prompts/actions
    # ===========================================
    @traceable(run_type="chain", name="MedIntel Respond")
    def respond(user_input, chat_history):
        print(f"DEBUG user_input type: {type(user_input)}, value: {repr(user_input)}")
        
        # CAPTURE LANGSMITH RUN ID
        try:
            run_tree = get_current_run_tree()
            current_run_id = run_tree.id if run_tree else None
        except Exception:
            current_run_id = None
        
        try:
            # 1. Clean Gradio multipart format
            user_input = extract_gradio_text(user_input)
            if not user_input:
                return "", chat_history, current_run_id
        
            if chat_history is None:
                chat_history = []
        
            # 2. Sanitize history before passing downstream
            clean_history = []
            for msg in chat_history:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = extract_gradio_text(content)
                    clean_history.append({
                        "role": msg.get("role", "user"),
                        "content": str(content)
                    })
        
            # 3. Run the deterministic pipeline
            new_chat_history = generate_chat(user_input, clean_history)
            return "", new_chat_history, current_run_id
        
        except Exception as e:
            error_message = f"Error: {e}"
            logger.error(f"Error in respond: {e}")
            print(f"Error in respond: {e}")
        
            if chat_history is None:
                chat_history = []
        
            clean_input = extract_gradio_text(user_input) if isinstance(user_input, (list, dict)) else str(user_input)
            updated = _append_turn(chat_history, clean_input, error_message)
            return "", updated, current_run_id

    # ============================================
    # Feedback button event handler
    # ============================================        
    def submit_feedback(run_id, rating, comment=""):
        if not run_id:
            return gr.Textbox(value="⚠️ Send a message first to enable rating.", visible=True)
        
        score = 1.0 if rating == "up" else 0.0
        
        try:
            ls_client.create_feedback(
                run_id=run_id,
                key="user_rating",
                score=score,
                comment=comment or None,
            )
            return gr.Textbox(value="✅ Thanks for your Feedback. Saved to LangSmith.", visible=True)
        except Exception as e:
            logger.error(f"❌ Feedback error: {str(e)}")
            return gr.Textbox(value=f"❌ Feedback error: {str(e)}", visible=True)             
                                
    #===============================================
    # Bind Events
    #===============================================    
    #Image preview on upload
    upload_file.change(
        lambda f: gr.Image(value=f, visible=bool(f)) if f else gr.Image(visible=False),
        inputs=upload_file,
        outputs=preview
    )
    
    # WORKFLOW 1: Image + Question → LangGraph agent    
    submit_btn.click(
        handle_invoice_submit,
        inputs=[upload_file, image_question, chatbot],
        outputs=[chatbot, upload_file, run_id_state],
    )
    
    # WORKFLOW 2: Text-only fallback (press Enter in textbox)
    image_question.submit(
        handle_invoice_submit,        
        inputs=[upload_file, image_question, chatbot],
        outputs=[chatbot, image_question, run_id_state],
    )    

    process_invoice_batch_btn.click(
        run_invoice_batch,
        inputs=[chatbot],
        outputs=[chatbot]
    )
    
    # Feedback buttons
    btn_up.click(
        lambda rid, c: submit_feedback(rid, "up", c),
        inputs=[run_id_state, fb_comment],
        outputs=fb_status
    )
    
    btn_down.click(
        lambda rid, c: submit_feedback(rid, "down", c),
        inputs=[run_id_state, fb_comment],
        outputs=fb_status
    )    
    
    # save_vectordb_btn.click(
    #     add_documents_to_vectorstore(vectorstore, pdf_local_file_path, output_vectorstore_path),
    #     inputs=[chatbot],
    #     outputs=[chatbot]
    # )
    
    clear_btn.click(lambda: [], None, chatbot, queue=False)
        
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch(theme=medical_theme,
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,  # Disable SSR to avoid the experimental warning
        css=custom_css,
        share=True
    )      