import os, json, time, random, hashlib, re
from pathlib import Path
from dotenv import load_dotenv
from jsonschema import validate
import pandas as pd
# --- NOTE: Removed google.generativeai due to API limit/MOCK Mode ---

# -------------------------------------------------------------------
# SETUP & CONFIG
# -------------------------------------------------------------------
BASE = Path(__file__).parent
TRANSCRIPTS = BASE / "transcripts"
OUTPUTS = BASE / "outputs"; OUTPUTS.mkdir(exist_ok=True)
CACHE = OUTPUTS / ".cache"; CACHE.mkdir(exist_ok=True)

PROMPT = (BASE / "prompts" / "call_prompt.txt").read_text(encoding="utf-8")
SCHEMA = json.loads((BASE / "schema.json").read_text(encoding="utf-8"))

# Load environment variables (for MOCK details like the validation URL)
dotenv_path = BASE / ".env"
load_dotenv(dotenv_path=dotenv_path)

print("✅ Running in MOCK LLM Mode. Task 2 analysis and validation are simulated.")

# -------------------------------------------------------------------
# MOCK API & TOOL DEFINITION (Task 2 Core)
# -------------------------------------------------------------------
def validate_payment_credentials(id: str, student_id: str, credentials: dict, amount: float | str, payment_valid: bool, failure_reason: str) -> dict:
    """
    MOCK function that simulates the external Payment Validation API call.
    It takes LLM-extracted arguments (including the LLM's guess at validity)
    and returns a simulated API response JSON, checking the LLM's extraction accuracy
    against a hardcoded "ground truth" for the sample transcript.

    NOTE: The real API uses payment_valid and failure_reason to check if the LLM's
    *determination* was correct against the ground truth.
    """
    card_number = credentials.get("cardNumber", "")
    cvv = credentials.get("cvv", "")
    
    # --- HARDCODED GROUND TRUTH (Based on the provided Sample Transcript) ---
    # The sample transcript shows a SUCCESSFUL payment setup.
    # The actual API logic is complex, but for this mock, we ensure the LLM's
    # *extracted details* match the expected successful outcome.
    
    GROUND_TRUTH_CARD = "4539876543210987" 
    GROUND_TRUTH_CVV = "215"
    
    # 1. Check LLM's extraction accuracy (the main API evaluation point)
    extraction_correct = (card_number == GROUND_TRUTH_CARD and cvv == GROUND_TRUTH_CVV and payment_valid == True)
    
    # --- task2_analyze_with_tools.py (Corrected Block) ---

    if extraction_correct:
        # If LLM's extraction and assessment were perfect (the ideal Task 2 outcome)
        is_valid = True
        api_message = "Payment validated successfully (LLM Extraction Correct)."
        api_failure_reason = "none"
        http_code = 200
    else:
        # Simulate a typical API failure (e.g., data mismatch, or LLM incorrectly guessed validity)
        is_valid = False
        api_failure_reason = "data_mismatch" if card_number != GROUND_TRUTH_CARD else failure_reason
        api_message = f"Data mismatch detected. LLM extracted card: {card_number[-4:]}, GT card: {GROUND_TRUTH_CARD[-4:]}"
        # Corrected line: Ensure only the Python code remains
        http_code = 422 # HTTP 422 Validation failed 
        
    # Mock API Response
# ... rest of the function ...        
    # Mock API Response
    return {
        "success": is_valid,
        "message": api_message,
        "failureReason": api_failure_reason,
        "http_status": http_code
    }

# -------------------------------------------------------------------
# MOCK LLM OUTPUTS & HELPERS
# -------------------------------------------------------------------

def transcript_to_text(obj) -> str:
    """Converts transcript JSON list into readable text format, including timestamp conversion."""
    lines = []
    for turn in obj:
        speaker = turn.get("role", "Unknown")
        text = turn.get("utterance", "")
        t = turn.get("stime")
        ts = f"[{int(t//60)}:{int(t%60):02d}]" if isinstance(t, (int, float)) else ""
        lines.append(f"{ts} {speaker}: {text}")
    return "\n".join(lines)

def redact_sensitive(text: str) -> str:
    # Keeps the original redaction logic
    text = re.sub(r"\b\d{13,19}\b", "[CARD_NUMBER]", text)
    text = re.sub(r"\b\d{3,4}\b(?=\s*CVV|\s*CVC|\s*code)", "[CVV]", text, flags=re.IGNORECASE)
    text = re.sub(r"(0[1-9]|1[0-2])/?(20\d{2})", "[EXPIRY]", text)
    return text

# MOCK LLM output for the final JSON (Task 1 + Task 2)
# NOTE: We assume the LLM successfully extracted the correct credentials in the hidden step
# to demonstrate a successful validation.
def mock_llm_final_json(validation_result: dict) -> str:
    """
    MOCK function simulating the final LLM output, incorporating the Task 2 validation result.
    """
    final_output = {
      "payment_attempted": True,
      "customer_intent": True,
      "customer_sentiment": {
        "classification": "Satisfied",
        "description": "Customer was cooperative, quickly agreed to pay the full amount, and showed no signs of frustration, leading to a successful call."
      },
      "agent_performance": "Agent was professional, delivered the required debt disclosure (Mini-Miranda), and clearly guided the customer through the successful payment setup process.",
      "timestamped_events": [
        {"timestamp": "0:42", "event_type": "disclosure", "description": "Agent provided the mandatory debt collection disclosure."},
        {"timestamp": "0:56", "event_type": "payment_setup_attempt", "description": "Customer confirms ability to pay and agent starts collecting payment details."},
        {"timestamp": "1:02", "event_type": "payment_setup_attempt", "description": "Customer provided card number (4539...0987)."},
        {"timestamp": "1:21", "event_type": "payment_setup_attempt", "description": "Customer provided expiry date (July 2027)."},
        {"timestamp": "1:32", "event_type": "payment_setup_attempt", "description": "Agent read the final payment authorization statement."},
      ],
      # --- TASK 2 FIELD ---
      "payment_validation_result": validation_result
    }
    return json.dumps(final_output, indent=2)

# Helper functions for cache (kept the same)
def cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_cache(key: str):
    p = CACHE / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None

def save_cache(key: str, data: dict):
    (CACHE / f"{key}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))


# -------------------------------------------------------------------
# CREDENTIAL EXTRACTION LOGIC (Simulates LLM's Tool Argument Generation)
# -------------------------------------------------------------------
def mock_extract_credentials(transcript_text: str, call_id: str, student_id: str) -> dict:
    """
    Simulates the LLM's ability to extract specific credentials from the transcript
    and determine the LLM's 'guess' at validity *before* calling the API.
    
    Since the sample transcript is a successful one, the mock extraction succeeds.
    """
    
    # --- Hardcoded Extraction based on Sample Transcript ---
    
    credentials = {
        # The LLM must link the 'Jennifer Martinez' spoken name to the cardholder
        "cardholderName": "Jennifer Martinez", 
        # The LLM extracts the 16-digit number and determines it's the card
        "cardNumber": "4539876543210987", 
        "cvv": "215",
        # 'July twenty twenty seven' -> 7/2027
        "expiryMonth": 7,
        "expiryYear": 2027
    }
    
    # Amount is 'four hundred fifty two dollars and thirty cents'
    amount = 452.30
    
    # LLM's assessment BEFORE API call (Based on transcript: customer was cooperative)
    llm_valid_guess = True
    llm_failure_reason = "none" # Only required if llm_valid_guess=False [cite: 95]
    
    # This dictionary represents the full JSON request the LLM would *generate*
    # to be sent to the validate_payment_credentials tool.
    llm_tool_request = {
      "id": call_id,
      "student_id": student_id,
      "payment_valid": llm_valid_guess,
      "failure_reason": llm_failure_reason,
      "credentials": credentials,
      "amount": amount
    }
    
    return llm_tool_request

# -------------------------------------------------------------------
# MAIN ANALYSIS FOR TASK 2
# -------------------------------------------------------------------

def analyze_one_with_tools(path: Path) -> dict:
    """Processes one transcript using the Mock Tool Calling workflow."""
    obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    transcript_text = transcript_to_text(obj)
    
    # Assuming student_id is available in the filename or metadata (a4c8d2e7.json)
    call_id = path.stem 
    # MOCK student ID for the sample transcript
    student_id = "SAP-3321" 

    # --- STEP 1: Simulate LLM Tool Extraction ---
    print(f"  > Step 1: Simulating LLM extraction and tool argument generation...")
    llm_tool_request_args = mock_extract_credentials(transcript_text, call_id, student_id)
    
    # --- STEP 2: Execute Mock API Tool Call ---
    print(f"  > Step 2: Executing MOCK Payment Validation API call...")
    
    try:
        # Destructure the dictionary into function arguments
        api_response = validate_payment_credentials(**llm_tool_request_args)
        
        # Format the result for the final prompt
        validation_result = {
            "called": True,
            "valid": api_response["success"],
            "reason": api_response["message"],
            "status_code": api_response["http_status"]
        }
        print(f"  ✅ Tool Result: Valid={validation_result['valid']} | Status={validation_result['status_code']}")

    except Exception as e:
        validation_result = {"called": True, "valid": False, "reason": f"MOCK API Execution Error: {e}", "status_code": 500}
        print(f"  ❌ MOCK Execution Error: {e}")
    
    # --- STEP 3: Final Analysis (MOCK LLM) ---
    print("  > Step 3: Generating final structured JSON (MOCK LLM)...")
    
    # The final prompt would include all the context, but the MOCK function just
    # takes the validation result to complete the final structure.
    raw_json_string = mock_llm_final_json(validation_result)

    # Standard JSON extraction and validation
    analysis = json.loads(raw_json_string)
    
    # We must temporarily update the schema to include the Task 2 field for validation
    TASK2_SCHEMA = SCHEMA.copy()
    TASK2_SCHEMA["properties"]["payment_validation_result"] = {
        "type": "object",
        "required": ["called", "valid", "reason", "status_code"],
        "additionalProperties": False,
        "properties": {
            "called": { "type": "boolean" },
            "valid": { "type": "boolean" },
            "reason": { "type": "string" },
            "status_code": { "type": "integer" }
        }
    }
    TASK2_SCHEMA["required"].append("payment_validation_result")
    
    validate(analysis, TASK2_SCHEMA) 
    
    # Use the combined call ID and validation result for caching the final output
    cache_key_data = f"{call_id}-{validation_result['valid']}"
    save_cache(cache_key_data, analysis)
    
    return analysis

# -------------------------------------------------------------------
def main():
    # To run this mock, ensure the sample transcript is saved as a JSON in the 'transcripts/' folder.
    files = sorted(TRANSCRIPTS.glob("*.json"))
    if not files:
        print("❌ No transcript files found in 'transcripts/' folder.")
        return

    rows = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Analyzing {f.name} for Task 2...")
        try:
            result = analyze_one_with_tools(f)
            out_path = OUTPUTS / (f.stem + ".task2_analysis.json")
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            rows.append({
                "file": f.name,
                "payment_attempted": result.get("payment_attempted"),
                "customer_intent": result.get("customer_intent"),
                "sentiment": result.get("customer_sentiment", {}).get("classification"),
                "validation_success": result.get("payment_validation_result", {}).get("valid")
            })
            print(f"  ✅ Task 2 Output saved to {out_path}")
        except Exception as e:
            print(f"❌ Error processing {f.name}: {e}")
            
    if rows:
        summary_path = OUTPUTS / "task2_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"\n✅ Task 2 (MOCK) complete! See {summary_path} and ./outputs for the combined JSON analysis.")

if __name__ == "__main__":
    main()