import time, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="Kaushik/grporlhf-demo-adapter")
    args = a.parse_args()

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").cuda()
    model = PeftModel.from_pretrained(base, args.model).eval()

    t0 = time.time()
    for _ in range(100):
        ids = tok("Hello", return_tensors="pt").input_ids.cuda()
        _ = model.generate(ids, max_new_tokens=5)
    print(f"100 gens in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
