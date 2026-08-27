"""DEFER - Phase 0.1, the access check.

The cheapest possible test of the whole chain: Kaggle secret -> Hugging Face
token -> gated Llama weights. It downloads config.json (a couple of kilobytes),
never the 6 GB of weights, so it runs on CPU in seconds and spends no GPU quota.

Why this exists as its own run. Llama-3.2-3B-Instruct is *gated*: Hugging Face
refuses to serve it until Meta's licence has been accepted by the account behind
the token. Discovering that nine hours into a GPU session is the single most
expensive way to learn it. This is the assert placed in front of the expensive
thing.

Every failure below prints what it means and what to do about it. A run that
ends without PASS has told you exactly which link in the chain is broken.
"""
import json
import sys
import traceback

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
EXPECTED_PARAMS_B = 3.2   # billions, roughly, from the model card
SECRET_NAME = "HF_TOKEN"


def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def fail(what, why, fix):
    print(f"\n  FAILED: {what}")
    print(f"  what it means: {why}")
    print(f"  what to do:    {fix}")
    sys.exit(1)


# --------------------------------------------------------------- 1. the token
line("1. reading the Hugging Face token from Kaggle Secrets")

token = None
try:
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret(SECRET_NAME)
    print(f"  secret {SECRET_NAME!r} found, {len(token)} characters")
except Exception as exc:
    fail(
        f"could not read the secret {SECRET_NAME!r}",
        f"{type(exc).__name__}: {exc}\n"
        "                 The secret is either not created, or not attached to "
        "this notebook.\n"
        "                 Creating it once is not enough - each notebook must "
        "have it switched on.",
        "In the Kaggle editor: Add-ons -> Secrets -> create HF_TOKEN, then "
        "tick its\n"
        "                 checkbox for THIS notebook. Then re-run.",
    )

if not token or not token.startswith("hf_"):
    fail(
        "the secret does not look like a Hugging Face token",
        "HF read tokens start with 'hf_'. This one does not, so it is probably "
        "the\n                 wrong value pasted into the right box.",
        "Make a read token at huggingface.co/settings/tokens and paste it in.",
    )

# ------------------------------------------------------------ 2. the identity
line("2. checking who that token belongs to")

try:
    from huggingface_hub import HfApi

    me = HfApi().whoami(token=token)
    print(f"  authenticated as: {me.get('name')}  ({me.get('type')})")
except Exception as exc:
    fail(
        "the token was rejected by Hugging Face",
        f"{type(exc).__name__}: {exc}\n"
        "                 Either the token is expired/revoked, or this notebook "
        "has no internet.",
        "Check Notebook options -> Internet is ON (it needs a phone-verified "
        "account),\n"
        "                 then regenerate the token if it still fails.",
    )

# --------------------------------------------------------- 3. the gated model
line(f"3. requesting the gate-protected config for {MODEL}")

try:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=MODEL, filename="config.json", token=token)
    cfg = json.load(open(path))
    print(f"  config.json downloaded: {path}")
except Exception as exc:
    detail = f"{type(exc).__name__}: {exc}"
    is_gate = "403" in detail or "gated" in detail.lower() or "awaiting" in detail.lower()
    fail(
        "could not download the model config",
        detail + (
            "\n                 A 403 here means the licence has not been "
            "accepted by THIS account,\n"
            "                 or the request is still pending Meta's approval."
            if is_gate else ""
        ),
        f"Open huggingface.co/{MODEL} while signed in as the account above, "
        "accept the\n"
        "                 licence, wait for it to say 'granted', then re-run.",
    )

# ------------------------------------------------- 4. is it the right model?
line("4. confirming it is the model we think it is")

hidden = cfg.get("hidden_size")
layers = cfg.get("num_hidden_layers")
vocab = cfg.get("vocab_size")
print(f"  architecture:      {cfg.get('architectures')}")
print(f"  hidden size:       {hidden}")
print(f"  layers:            {layers}")
print(f"  vocab size:        {vocab}")
print(f"  max positions:     {cfg.get('max_position_embeddings')}")

if not (hidden and layers):
    fail("the config is missing basic fields",
         "The download succeeded but the file is not a model config.",
         "Check the MODEL id at the top of this script.")

# Rough transformer parameter count, enough to catch a wrong-size model:
# attention+mlp blocks plus the embedding and output matrices.
approx = (12 * layers * hidden ** 2 + 2 * vocab * hidden) / 1e9
print(f"  approx params:     {approx:.1f}B  (expected around {EXPECTED_PARAMS_B}B)")
if abs(approx - EXPECTED_PARAMS_B) > 1.5:
    fail(
        "this is not the size of model the study is designed around",
        f"Estimated {approx:.1f}B against an expected {EXPECTED_PARAMS_B}B. "
        "The previous study\n"
        "                 found 1.5B too small to learn the behaviour, so size "
        "is not cosmetic.",
        "Fix MODEL at the top of this script.",
    )

# ----------------------------------------------------- 5. the chat template
line("5. loading the tokenizer and its chat template")

try:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, token=token)
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": "Answer only from the passage."},
            {"role": "user", "content": "Passage: The capital, Lyon, sits on the "
                                        "Rhone.\n\nQuestion: What is the capital?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"  tokenizer loaded, vocab {len(tok)}")
    print("  --- rendered prompt, exactly as the model will see it ---")
    print(prompt)
    print("  --- end ---")
    print(f"  that prompt is {len(tok(prompt)['input_ids'])} tokens")
except Exception as exc:
    traceback.print_exc()
    fail(
        "the tokenizer would not load",
        f"{type(exc).__name__}: {exc}\n"
        "                 The gate is open (step 3 passed) but the tokenizer "
        "files did not load.",
        "Usually a transient network error - re-run once before digging.",
    )

line("PASS - the whole chain works")
print("""
  Kaggle secret -> HF token -> accepted licence -> gated weights -> tokenizer.

  Next: the closed-book probe, which is the first run that actually needs a GPU.
  Nothing before this point spends quota, which is the entire point of running
  this separately.
""")
