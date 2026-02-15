# model_wrapper.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import extract_json_from_text, SYSTEM_INSTRUCTION_MEDICAL



class QwenLocal:
    def __init__(
        self,
        model_name="Qwen/Qwen3-1.7B",   # 🔥 đổi sang 1.7B
        local_dir="./models",
        device=None,
        enable_thinking=False
    ):
        self.model_name = model_name
        self.local_dir = local_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_thinking = enable_thinking

        os.makedirs(self.local_dir, exist_ok=True)

        print(f"Downloading/loading model: {self.model_name}")
        print(f"Model cache directory: {self.local_dir}")

        # Load tokenizer (tự động cache về ./models)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.local_dir,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=self.local_dir,
            trust_remote_code=True
        )

        if not torch.cuda.is_available():
            self.model.to(self.device)

        print("✅ Model loaded successfully.")

    def build_prompt(self, context_snippets, question):
        ctx = ""
        for sn in context_snippets:
            ctx += f"[{sn.get('chunk_id')} | {sn.get('source')}]\n{sn.get('text')}\n---\n"

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION_MEDICAL},
            {
                "role": "user",
                "content": f"CONTEXT:\n{ctx}\nQUESTION:\n{question}\nReturn JSON only."
            }
        ]
        return messages

    def generate_json(
        self,
        context_snippets,
        question,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        top_k=20
    ):
        messages = self.build_prompt(context_snippets, question)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )

        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        out_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        parsed = extract_json_from_text(out_text)

        if parsed is None:
            return {
                "answer": out_text,
                "citations": [],
                "confidence": "low",
                "action": "informational",
                "raw": out_text
            }

        return parsed
