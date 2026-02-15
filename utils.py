# utils.py
"""
Hỗ trợ cho Medical FAQ Assistant

Chứa:
- PROMPTS (SYSTEM_INSTRUCTION_MEDICAL, SYSTEM_INSTRUCTION_CHAT)
- is_medical_query(text): rule-based classifier nhẹ
- detect_emergency(text): phát hiện từ khóa khẩn cấp
- extract_json_from_text(text): cố gắng tách object JSON cuối cùng từ output LLM
- normalize_text_for_model(text): làm sạch input/output ngắn
"""

from typing import Optional, Any, List
import re
import json

# ====== Configuration / Constants ======

EMERGENCY_KEYWORDS = [
    "khó thở", "mất ý thức", "ngất", "chảy máu", "đau ngực",
    "severe bleeding", "difficulty breathing", "unconscious",
    "call emergency", "emergency", "không thở được", "ngạt thở"
]

# Từ khoá y tế cơ bản để phân loại câu hỏi có liên quan y tế hay không.
MEDICAL_KEYWORDS = [
    "triệu chứng", "bệnh", "điều trị", "thuốc", "viêm", "nhiễm", "sốt",
    "ho", "đau", "viêm", "khám", "chẩn đoán", "phẫu thuật", "tiêm", "vaccine",
    "thuốc", "liều", "triệu chứng", "hồi phục", "khẩn cấp", "cấp cứu", "xét nghiệm",
    "gan", "tim", "thận", "tiêu hoá", "hô hấp", "TMH", "nhi khoa"
]

# Nếu số từ khóa y tế xuất hiện >= THRESHOLD -> coi là truy vấn y tế
MEDICAL_KEYWORD_THRESHOLD = 1


SYSTEM_INSTRUCTION_MEDICAL = """Bạn là một trợ lý thông tin y tế dựa trên tài liệu. CHỈ SỬ DỤNG những đoạn có trong CONTEXT.
Nếu câu hỏi là y tế, TRẢ VỀ DUY NHẤT một JSON hợp lệ với các trường:
- answer: string (ngắn, <=300 words)
- citations: [{source, url (if có, else empty), chunk_id, snippet}]
- confidence: low|medium|high
- action: informational|see_physician|emergency
- extracted_facts: [string]

Quy tắc bắt buộc:
1) Nếu không đủ thông tin trong CONTEXT => trả {"answer":"INSUFFICIENT_DATA", "citations":[], "confidence":"low", "action":"see_physician", "extracted_facts":[]}.
2) Không chẩn đoán và không kê toa. Nếu mô tả triệu chứng khẩn cấp => action="emergency" và trả chỉ dẫn gọi cấp cứu.
3) Mỗi citation phải dựa trên CONTEXT; bao gồm chunk_id và một snippet ngắn.
4) Output phải là JSON duy nhất (KHÔNG có text tự do khác).
"""

SYSTEM_INSTRUCTION_CHAT = """Bạn là một trợ lý thông thường: trả lời câu hỏi tóm tắt, thân thiện và ngắn gọn.
Nếu người hỏi cần thông tin y tế chuyên sâu, chuyển hướng họ sang trợ lý y tế (ví dụ: "Nếu bạn cần tư vấn chuyên sâu về triệu chứng, hãy hỏi trợ lý y tế.")."""

# ====== Helpers ======

def normalize_text_for_model(text: str) -> str:
    """Loại bỏ control tokens, nhiều dòng thừa, trim."""
    if text is None:
        return ""
    # Remove repeated whitespace including non-breaking spaces
    s = re.sub(r"\s+", " ", text).strip()
    return s

def safe_json_load(s: str) -> Optional[Any]:
    """Cố gắng load JSON, trả None nếu fail."""
    try:
        return json.loads(s)
    except Exception:
        return None

# ====== Query classification & emergency detection ======

def is_medical_query(text: str, threshold: int = MEDICAL_KEYWORD_THRESHOLD) -> bool:
    """
    Quyết định xem một câu có mang ý nghĩa y tế hay không bằng rule-based keyword matching.
    Trả True nếu số keywords >= threshold.
    """
    if not text:
        return False
    t = text.lower()
    count = 0
    for kw in MEDICAL_KEYWORDS:
        if kw in t:
            count += 1
            if count >= threshold:
                return True
    # Nếu có dạng câu hỏi và chứa các từ gợi ý triệu chứng => coi là y tế
    question_words = ["?", "thế nào", "là gì", "cách", "khi nào", "nên"]
    if any(w in t for w in question_words) and any(k in t for k in ["triệu", "triệu chứng", "bệnh", "điều trị"]):
        return True
    return False

def detect_emergency(text: str) -> bool:
    """Phát hiện từ khóa khẩn cấp (case-insensitive)."""
    if not text:
        return False
    t = text.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in t:
            return True
    return False

# ====== JSON extraction from LLM output ======

_RE_THINK = re.compile(r"<think\b[^>]*>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def _strip_thinking_blocks(s: str) -> str:
    """Loại bỏ block <think>...</think> nếu model in ra."""
    return _RE_THINK.sub("", s)

def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Cố gắng tách object JSON cuối cùng hợp lệ từ output của model.
    - Loại bỏ các block <think>...</think>.
    - Tìm mọi substring có dạng {...} (cân bằng ngoặc) và thử json.loads.
    - Trả object JSON cuối cùng hợp lệ (ưu tiên object dài nhất/ở cuối).
    """
    if not text:
        return None
    s = _strip_thinking_blocks(text).strip()

    # quick path: if whole text is JSON
    whole = safe_json_load(s)
    if isinstance(whole, dict):
        return whole

    results: List[dict] = []

    # Find all '{' positions and try to parse balanced braces
    open_positions = [m.start() for m in re.finditer(r"\{", s)]
    for start in open_positions:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i+1]
                    parsed = safe_json_load(candidate)
                    if isinstance(parsed, dict):
                        results.append((start, i+1, parsed))
                    break

    if not results:
        # fallback: try to find a JSON-like substring with regex (less strict)
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            parsed = safe_json_load(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        return None

    # Choose the JSON object that appears last (largest start index), or largest span
    results.sort(key=lambda x: (x[0], x[1]))  # sort by start then end
    # prefer the one with largest start (i.e., nearest to end)
    chosen = results[-1][2]
    return chosen

# ====== Small utility to build RAG prompt or chat prompt ======

def build_rag_prompt(context_snippets: List[dict], question: str) -> str:
    """
    Tạo string prompt đơn giản cho model (bổ sung SYSTEM_INSTRUCTION_MEDICAL khi cần).
    context_snippets: list of {"chunk_id","source","text"}
    """
    ctx_parts = []
    for sn in context_snippets:
        chunk_id = sn.get("chunk_id", "")
        source = sn.get("source", "")
        text = sn.get("text", "")
        ctx_parts.append(f"[{chunk_id} | {source}]\n{text}")
    ctx = "\n---\n".join(ctx_parts)
    prompt = f"{SYSTEM_INSTRUCTION_MEDICAL}\n\nCONTEXT:\n{ctx}\n\nQUESTION:\n{question}\n\nReturn JSON only."
    return prompt

def build_chat_prompt(history: List[dict], user_input: str) -> str:
    """
    Tạo prompt cho mode chat thông thường. history là list các dict {'role','content'}.
    """
    lines = []
    for item in history or []:
        role = item.get("role", "user")
        content = item.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append(f"user: {user_input}")
    prompt = f"{SYSTEM_INSTRUCTION_CHAT}\n\n" + "\n".join(lines)
    return prompt
