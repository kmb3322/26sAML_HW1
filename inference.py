"""Load a trained checkpoint and run inference.

Examples:
    # Modular arithmetic prediction:
    python inference.py --checkpoint_dir runs/add_p97_l1_seed0/best \
        --a 12 --b 34 --op + --p 97

    # Sanity-check generation from BOS:
    python inference.py --checkpoint_dir runs/sanity/final

    # Sanity-check generation with a partial prompt (BOS prepended automatically):
    python inference.py --checkpoint_dir runs/sanity_masked/final \
        --sanity_prompt "I love"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from model import GPT, GPTConfig
from tokenizer_data import ResidueTokenizer, SanityTokenizer, load_tokenizer_from_dict


def load_checkpoint(checkpoint_dir: str) -> Tuple[GPT, object, torch.device, dict]:
    """Load a checkpoint directory containing ``ckpt.pt``.

    Returns ``(model, tokenizer, device, ckpt_dict)``.  The model is in eval
    mode and on the best available device.
    """
    ckpt_path = Path(checkpoint_dir) / "ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    tokenizer = load_tokenizer_from_dict(ckpt["tokenizer"])
    model_cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device, ckpt


@torch.no_grad()
def predict_modular_answer(
    model: GPT,
    tokenizer: ResidueTokenizer,
    a: int,
    b: int,
    op: str,
    p: int,
    device: Optional[torch.device] = None,
) -> int:
    """Predict ``c`` such that ``a op b ≡ c (mod p)``.

    The argmax is taken over the ``p`` numeric tokens ``<n:0>, ..., <n:p-1>``
    only -- so the model can never return a punctuation/special token even
    when undertrained.
    """
    if device is None:
        device = next(model.parameters()).device

    prompt = tokenizer.encode_problem(a, b, op, c=None)
    x = torch.tensor([prompt], dtype=torch.long, device=device)
    logits = model(x)[:, -1, :]

    candidate_ids = torch.tensor([tokenizer.num_id(i) for i in range(p)], device=device)
    restricted = logits.index_select(1, candidate_ids)
    pred_offset = int(restricted.argmax(dim=-1).item())
    pred_token_id = int(candidate_ids[pred_offset].item())
    return tokenizer.id_to_num(pred_token_id)


@torch.no_grad()
def greedy_generate(
    model: GPT,
    input_ids: List[int],
    eos_id: Optional[int] = None,
    max_new_tokens: int = 32,
    device: Optional[torch.device] = None,
) -> List[int]:
    """Greedy decode starting from ``input_ids``.  Returns the full sequence."""
    if device is None:
        device = next(model.parameters()).device

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]
        next_id = int(logits.argmax(dim=-1).item())
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if eos_id is not None and next_id == eos_id:
            break
    return ids[0].detach().cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--a", type=int, default=None)
    parser.add_argument("--b", type=int, default=None)
    parser.add_argument("--op", type=str, default=None, choices=["+", "-", "/"])
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument(
        "--sanity_prompt",
        type=str,
        default=None,
        help="Words separated by spaces; BOS is prepended automatically.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    model, tokenizer, device, ckpt = load_checkpoint(args.checkpoint_dir)
    print(f"Loaded checkpoint step={ckpt.get('step')}, tokenizer={ckpt['tokenizer']}")

    if isinstance(tokenizer, ResidueTokenizer):
        if args.a is None or args.b is None or args.op is None or args.p is None:
            raise SystemExit(
                "Modular checkpoint requires --a --b --op --p for inference."
            )
        ans = predict_modular_answer(
            model, tokenizer, args.a, args.b, args.op, args.p, device=device
        )
        print(f"{args.a} {args.op} {args.b} mod {args.p} = {ans}")

    elif isinstance(tokenizer, SanityTokenizer):
        words = args.sanity_prompt.split() if args.sanity_prompt else []
        unknown = [w for w in words if w not in tokenizer.stoi]
        if unknown:
            raise SystemExit(f"Unknown words for sanity tokenizer: {unknown}")
        prompt_ids = [tokenizer.bos_id] + [tokenizer.stoi[w] for w in words]
        out_ids = greedy_generate(
            model,
            prompt_ids,
            eos_id=tokenizer.eos_id,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        print("Generated:", tokenizer.decode(out_ids))

    else:
        raise SystemExit(f"Unsupported tokenizer kind: {type(tokenizer).__name__}")


if __name__ == "__main__":
    main()
