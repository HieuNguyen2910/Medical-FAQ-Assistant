import os
import re
import threading
import logging
import uvicorn
import torch
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from model_wrapper import QwenLocal
from utils import detect_emergency, is_medical_query

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
DEFAULT_QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-1.7B")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("medical-rag")

app = FastAPI(title="Medical FAQ Assistant — RAG (Qwen Local)", version="1.2")

emb: Optional[HuggingFaceEmbeddings] = None
vectordb: Optional[Chroma] = None
retriever = None

qwen: Optional[QwenLocal] = None
model_lock = threading.Lock()
model_loaded = False
model_name_loaded: Optional[str] = None

class QueryIn(BaseModel):
    question: str
    history: Optional[List[Dict[str, Any]]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global emb, vectordb, retriever, qwen, model_loaded, model_name_loaded
    hf_token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_HUB_TOKEN"] = hf_token
    try:
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        if not os.path.exists(CHROMA_DIR):
            vectordb = None
            retriever = None
            logger.warning("Chroma DB not found at '%s' — retrieval disabled.", CHROMA_DIR)
        else:
            vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            logger.info("Vector DB loaded from %s (k=3).", CHROMA_DIR)
    except Exception as e:
        logger.exception("Failed to initialize embeddings or Chroma: %s", e)
        emb = None
        vectordb = None
        retriever = None
    with model_lock:
        try:
            qwen = QwenLocal(model_name=DEFAULT_QWEN_MODEL)
            model_loaded = True
            model_name_loaded = DEFAULT_QWEN_MODEL
            logger.info("Model loaded successfully: %s", DEFAULT_QWEN_MODEL)
        except Exception as e:
            model_loaded = False
            model_name_loaded = None
            qwen = None
            msg = str(e)
            logger.exception("Failed to load Qwen model at startup: %s", e)
            if "RepositoryNotFoundError" in msg or "not a valid model identifier" in msg or "401" in msg:
                logger.error("Model could not be found or is gated/private. Set HF_HUB_TOKEN if needed.")
            elif "unknown model" in msg.lower() or "architectur" in msg.lower():
                logger.error("Transformers may be too old to support this model architecture.")
    yield
    try:
        if qwen and hasattr(qwen, "close"):
            try:
                qwen.close()
                logger.info("QwenLocal closed.")
            except Exception:
                logger.exception("Exception while closing QwenLocal.")
    except Exception:
        logger.exception("Exception during shutdown cleanup.")

app.router.lifespan_context = lifespan

def format_context_snippets(docs: List[Any]) -> List[Dict[str, str]]:
    snippets = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        text = getattr(d, "page_content", "") or ""
        snippets.append({
            "chunk_id": md.get("chunk_id", ""),
            "source": md.get("source", ""),
            "text": text
        })
    return snippets

def retrieve_documents_for_query(question: str) -> List[Any]:
    if retriever is None:
        return []
    try:
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(question)
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(question)
            if isinstance(docs, dict) and "documents" in docs:
                return docs["documents"]
            return docs
        if hasattr(retriever, "get_documents"):
            return retriever.get_documents(question)
        return []
    except Exception as e:
        logger.exception("Retriever call failed: %s", e)
        raise

def chat_with_qwen(question: str, history: Optional[List[Dict[str, Any]]] = None, max_new_tokens: int = 256) -> str:
    if qwen is None:
        return ""
    messages = []
    if history:
        for item in history:
            role = item.get("role", "user")
            content = item.get("content", "")
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    text = qwen.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = qwen.tokenizer([text], return_tensors="pt").to(qwen.model.device)
    with torch.no_grad():
        outputs = qwen.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True
        )
    generated_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    out_text = qwen.tokenizer.decode(generated_ids, skip_special_tokens=True)
    out_text = re.sub(r"<think\b[^>]*>.*?</think>", "", out_text, flags=re.DOTALL | re.IGNORECASE).strip()
    return out_text

@app.get("/status")
def status():
    return {
        "model_loaded": model_loaded,
        "model_name": model_name_loaded,
        "retriever_ready": retriever is not None
    }

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Medical FAQ Assistant</title>
  <style>
    :root{
      --bg-1: #071127;
      --bg-2: #081029;
      --card: #061021;
      --muted: #9aa4b2;
      --accent: #6c5ce7;
      --accent-2: #06b6d4;
      --glass: rgba(255,255,255,0.03);
      --bubble-user: linear-gradient(90deg,var(--accent),var(--accent-2));
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial;
      color-scheme: dark;
    }

    /* Page */
    html, body { height:100%; margin:0; background: linear-gradient(180deg,var(--bg-1) 0%, var(--bg-2) 100%); }
    body { display:flex; align-items:flex-start; justify-content:center; padding:20px; color:#e6eef8; }

    /* Container sizing: wider and full-height aware */
    .container { width:100%; max-width:1200px; padding: 8px; box-sizing:border-box; }

    /* Card now fills most of viewport height; stable from start */
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border-radius:18px;
      padding:18px;
      box-shadow: 0 12px 40px rgba(3,7,18,0.6);
      border:1px solid rgba(255,255,255,0.03);
      display:flex;
      flex-direction:column;
      gap:12px;
      height: calc(100vh - 60px); /* occupy nearly full viewport height */
      max-height: calc(100vh - 60px);
      box-sizing: border-box;
    }

    header { display:flex; align-items:center; gap:12px; }
    .logo { width:56px; height:56px; border-radius:12px; background:linear-gradient(135deg,var(--accent),var(--accent-2)); display:flex; align-items:center; justify-content:center; font-weight:700; font-size:18px; box-shadow: 0 6px 18px rgba(12,10,20,0.6); }
    h1 { margin:0; font-size:18px; }
    p.lead { margin:2px 0 0 0; color:var(--muted); font-size:13px; }

    /* Top row: header + status */
    .top-row { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
    #status-area { text-align:right; color:var(--muted); font-size:13px; min-width:200px; }

    /* Content area: make a flexible container so chat can grow */
    .content { display:flex; flex-direction:column; gap:12px; flex:1 1 auto; min-height:0; }

    /* Chat area fills available vertical space */
    #chat {
      flex:1 1 auto;      /* allow to grow and shrink */
      min-height:0;       /* important for flex children to scroll properly */
      overflow:auto;
      padding:20px;
      border-radius:12px;
      background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005));
      border:1px solid rgba(255,255,255,0.02);
      display:flex;
      flex-direction:column;
      gap:12px;
      box-sizing:border-box;
    }

    /* Messages */
    .msg { display:flex; gap:12px; align-items:flex-start; }
    .msg.user { justify-content:flex-end; }
    .bubble {
      max-width:78%;
      padding:12px 14px;
      border-radius:12px;
      font-size:14px;
      line-height:1.45;
      white-space:pre-wrap;
      box-shadow: 0 6px 18px rgba(3,7,18,0.35);
      border:1px solid rgba(255,255,255,0.02);
      background: rgba(255,255,255,0.02);
    }
    .msg.user .bubble {
      background: var(--bubble-user);
      color: white;
      border-bottom-right-radius:6px;
      border-bottom-left-radius:12px;
    }
    .msg.bot .bubble {
      background: rgba(255,255,255,0.03);
      color: inherit;
      border-bottom-left-radius:6px;
      border-bottom-right-radius:12px;
    }

    .citation { margin-top:8px; font-size:13px; color:var(--muted); background:rgba(255,255,255,0.01); padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.01); }

    /* Placeholder centered vertically within chat */
    .placeholder {
      margin:auto;
      text-align:center;
      color:var(--muted);
      font-size:14px;
      max-width:720px;
      line-height:1.6;
    }
    .placeholder .kicker { font-weight:600; color: #dfe8ff; margin-bottom:8px; font-size:15px; }

    /* Input: sticky-bottom-like inside card (but part of flow) */
    .input-row { display:flex; gap:12px; align-items:flex-end; }
    textarea { flex:1; min-height:96px; max-height:240px; resize:vertical; padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.06); background:var(--glass); color:inherit; font-size:14px; outline:none; box-sizing:border-box; }
    .btn { width:120px; background:linear-gradient(90deg,var(--accent),var(--accent-2)); padding:12px 18px; border-radius:12px; border:none; color:white; font-weight:700; cursor:pointer; transition: transform .12s ease, box-shadow .12s ease; box-shadow: 0 8px 24px rgba(6,11,30,0.45); }
    .btn:hover { transform:translateY(-3px); }
    .btn[disabled] { opacity:0.6; cursor:not-allowed; transform:none; box-shadow:none; }

    footer { margin-top:6px; color:var(--muted); font-size:13px; text-align:center; opacity:0.95; }

    /* Responsive */
    @media (max-width:900px) {
      .container { padding:6px; }
      .card { height: calc(100vh - 28px); padding:12px; }
      .logo { width:48px; height:48px; font-size:16px; }
      textarea { min-height:80px; }
    }

    /* spinner animation appended by JS also uses @keyframes spin */
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <div class="top-row">
        <header style="display:flex; align-items:center; gap:12px;">
          <div class="logo">MED</div>
          <div>
            <h1>Medical FAQ Assistant</h1>
            <p class="lead">RAG + Local Qwen3</p>
          </div>
        </header>
        <div id="status-area">
          <div id="status">Checking status...</div>
          <div id="conf-action" style="color:var(--muted); font-size:12px; margin-top:6px;">Confidence: — · Action: —</div>
        </div>
      </div>

      <div class="content">
        <div id="chat" aria-live="polite" role="log">
          <div class="placeholder" id="welcome-placeholder">
            <div class="kicker">Xin chào — tôi có thể giúp gì cho bạn?</div>
            Hỏi về triệu chứng, cách chăm sóc cơ bản, hoặc các câu hỏi y tế thông tin.
            <br/><br/>
            Ví dụ: <em>"Thường xuyên bị đau đầu, nên làm gì?"</em> — hoặc gõ câu hỏi của bạn vào ô dưới rồi nhấn <strong>Ask</strong>.
          </div>
        </div>

        <div class="input-row">
          <textarea id="question" placeholder="Ask a question... (Press Enter to send, Shift+Enter for newline)"></textarea>
          <button id="ask" class="btn">Ask</button>
        </div>
      </div>

      <footer>
        This assistant provides informational content only. In emergency, contact local medical services.
      </footer>
    </div>
  </div>

<script>
let history = [];

/* Render messages; if no messages show placeholder (keeps chat box size stable) */
function renderHistory(){
  const chat = document.getElementById('chat');
  chat.innerHTML = '';
  if(!history || history.length === 0){
    const placeholder = document.getElementById('welcome-placeholder');
    chat.appendChild(placeholder);
    // Ensure placeholder is centered by forcing layout then scroll to center (auto)
    chat.scrollTop = chat.scrollHeight / 2;
    return;
  }

  for(const item of history){
    const row = document.createElement('div');
    row.className = 'msg ' + (item.role === 'user' ? 'user' : 'bot');

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = item.content || '';

    row.appendChild(bubble);

    if(item.citations && item.citations.length){
      const cit = document.createElement('div');
      cit.className = 'citation';
      cit.innerHTML = item.citations.map(c => `<div><strong>${c.source || ''}</strong>${c.snippet ? ' — ' + c.snippet : ''}</div>`).join('');
      row.appendChild(cit);
    }
    chat.appendChild(row);
  }

  // auto-scroll to bottom
  requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
}

/* Request status from backend */
async function refreshStatus(){
  try{
    const r = await fetch('/status');
    const j = await r.json();
    const st = document.getElementById('status');
    st.textContent = j.model_loaded ? `Model: ${j.model_name || 'ready'}` : 'Model: not ready';
  }catch(e){
    document.getElementById('status').textContent = 'Status unavailable';
  }
}

/* Send question and update UI */
async function sendQuestion(){
  const ta = document.getElementById('question');
  const question = ta.value.trim();
  if(!question) return;
  history.push({role:'user', content: question});
  renderHistory();
  ta.value = '';
  const btn = document.getElementById('ask');
  btn.disabled = true;
  btn.innerHTML = '<span style="display:inline-block;width:18px;height:18px;border-radius:50%;border:3px solid rgba(255,255,255,0.2);border-top-color:white;animation:spin 1s linear infinite;"></span> Thinking...';
  try{
    const r = await fetch('/api/query',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question, history})
    });
    const data = await r.json();
    if(r.status !== 200){
      alert(data.error || "Server error");
      document.getElementById('conf-action').textContent = "Confidence: — · Action: —";
    } else {
      const ansText = data.answer || "(no answer)";
      history.push({role:'assistant', content: ansText, citations: data.citations || []});
      renderHistory();
      document.getElementById('conf-action').textContent = 'Confidence: ' + (data.confidence || '—') + ' · Action: ' + (data.action || '—');
    }
  }catch(e){
    alert("Request failed: " + e.message);
  }
  btn.disabled = false;
  btn.innerHTML = "Ask";
}

/* Events */
document.getElementById('ask').addEventListener('click', sendQuestion);
document.getElementById('question').addEventListener('keydown', function(e){
  if(e.key === 'Enter' && !e.shiftKey){
    e.preventDefault();
    sendQuestion();
  }
});

/* init */
renderHistory();
refreshStatus();
setInterval(refreshStatus, 5000);
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

@app.post("/query")
async def query_form(question: str = Form(...)):
    return await process_query({"question": question})

@app.post("/api/query")
async def api_query(qin: QueryIn):
    return await process_query(qin.dict())

async def process_query(payload: dict):
    question: str = (payload.get("question") or "").strip()
    history: Optional[List[Dict[str, Any]]] = payload.get("history")
    if not question:
        return JSONResponse({"error":"Question cannot be empty."}, status_code=400)

    if detect_emergency(question):
        return JSONResponse({
            "answer": "Detected potential emergency signs in your query. If you or someone is in immediate danger, call local emergency services right away.",
            "citations": [],
            "confidence": "high",
            "action": "emergency"
        })

    medical = is_medical_query(question)

    if not medical:
        fallback_medical_intro = (
            "Mình là một trợ lý y tế thông tin. Mình chuyên hỗ trợ về triệu chứng, chẩn đoán tham khảo và cách chăm sóc cơ bản.\n\n"
            "Câu hỏi bạn vừa gửi không nằm trong phạm vi y tế. Nếu bạn cần trợ giúp y tế, hãy mô tả triệu chứng (ví dụ: triệu chứng, khi bắt đầu, mức độ, có kèm sốt hay không, tiền sử), "
            "mình sẽ cố gắng hỗ trợ và đưa thông tin tham khảo. Nếu bạn vẫn muốn hỏi về chủ đề khác, mình có thể cố trả lời nhưng ưu tiên là các vấn đề y tế."
        )
        return JSONResponse({
            "answer": fallback_medical_intro,
            "citations": [],
            "confidence": "high",
            "action": "informational"
        })

    context_snippets = []
    if retriever is None:
        logger.warning("Retriever not available; returning INSUFFICIENT_DATA.")
    else:
        try:
            docs = retrieve_documents_for_query(question)
            if isinstance(docs, dict) and "documents" in docs:
                docs_list = docs["documents"]
            else:
                docs_list = docs
            context_snippets = format_context_snippets(docs_list or [])
        except Exception as e:
            logger.exception("Retrieval error: %s", e)
            return JSONResponse({"error": f"Retrieval error: {str(e)}"}, status_code=500)

    if not model_loaded or qwen is None:
        return JSONResponse({"error":"Local model not available. Check server logs."}, status_code=503)

    try:
        parsed = qwen.generate_json(context_snippets, question)
    except Exception as e:
        logger.exception("Model inference failed: %s", e)
        return JSONResponse({"error": f"Model inference error: {str(e)}"}, status_code=500)

    if isinstance(parsed, dict):
        if parsed.get("answer","").startswith("INSUFFICIENT_DATA") or not parsed.get("citations"):
            parsed.setdefault("confidence", "low")
            parsed.setdefault("action", "see_physician")
    else:
        parsed = {
            "answer": str(parsed),
            "citations": context_snippets,
            "confidence": "low",
            "action": "informational"
        }

    parsed.setdefault("answer", "")
    parsed.setdefault("citations", [])
    parsed.setdefault("confidence", "low")
    parsed.setdefault("action", "informational")

    return JSONResponse(parsed)

if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
