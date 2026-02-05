#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
from typing import Any, Dict, List, Optional

GENDERED_RE = re.compile(r"\[gendered_response:\s*([^,\]]+)\s*,\s*([^\]]+)\]")


def capitalize_first_cased_char(s: str) -> str:
    """
    Uppercase the first cased character (letter) in the string, preserving
    leading whitespace/punctuation and leaving the rest unchanged.
    """
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1 :]
    return s


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_label_location(template: str, sentence_id: str) -> None:
    if "[label_location]" not in template:
        raise ValueError(f"Sentence {sentence_id} is missing required placeholder [label_location].")


def sentence_has_gendered_placeholder(template: str) -> bool:
    return GENDERED_RE.search(template) is not None


def render_template(template: str, label_text: str, gender: Optional[str]) -> str:
    """
    gender: "female" or "male" if template contains gendered placeholders,
            otherwise None for non-gendered templates.
    """
    filled = template.replace("[label_location]", label_text)

    has_gendered = sentence_has_gendered_placeholder(template)
    if has_gendered and gender not in ("female", "male"):
        raise ValueError("Template contains [gendered_response: ...] but no gender was provided.")
    if (not has_gendered) and gender is not None:
        raise ValueError("Gender provided for template that has no [gendered_response: ...].")

    if has_gendered:
        idx = 0 if gender == "female" else 1

        def repl(m: re.Match) -> str:
            return (m.group(1).strip() if idx == 0 else m.group(2).strip())

        filled = GENDERED_RE.sub(repl, filled)

    return filled


def label_options_from_label(label: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of options, each with:
      - label_text
      - label_gender: "female" | "male" | "neutral"
      - label_type: "gendered" | "neutral"
      - label_id
    """
    ltype = label.get("type")
    lid = label.get("id", "unknown")

    if ltype == "gendered":
        if "female" not in label or "male" not in label:
            raise ValueError(f"Gendered label {lid} must include 'female' and 'male'.")
        return [
            {"label_id": lid, "label_type": "gendered", "label_gender": "female", "label_text": str(label["female"])},
            {"label_id": lid, "label_type": "gendered", "label_gender": "male", "label_text": str(label["male"])},
        ]

    if ltype == "neutral":
        variants = label.get("variants")
        if not variants or not isinstance(variants, list):
            raise ValueError(f"Neutral label {lid} must include a non-empty 'variants' list.")
        return [
            {"label_id": lid, "label_type": "neutral", "label_gender": "neutral", "label_text": str(v)}
            for v in variants
        ]

    raise ValueError(f"Unknown label type for {lid}: {ltype}")


def stratified_sample_by_image(
    rows: List[Dict[str, Any]], images: List[str], max_items: int, seed: int
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_image: Dict[str, List[Dict[str, Any]]] = {img: [] for img in images}
    for r in rows:
        by_image[r["poster_image"]].append(r)

    n_images = len(images)
    base = max_items // n_images
    remainder = max_items % n_images

    sampled: List[Dict[str, Any]] = []
    for i, img in enumerate(images):
        k = base + (1 if i < remainder else 0)
        pool = by_image[img]
        if k > len(pool):
            raise ValueError(f"Not enough rows to sample {k} items for image={img}. Have {len(pool)}.")
        sampled.extend(rng.sample(pool, k))

    rng.shuffle(sampled)
    return sampled


def write_tsv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # gender is inserted between label and image
    fieldnames = ["trial", "post_text", "label", "gender", "poster_image", "gendered_pronouns", "filled_post"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def safe_image_stem(image_name: str) -> str:
    base = os.path.basename(image_name)
    stem, _ext = os.path.splitext(base)
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", stem)


def main():
    ap = argparse.ArgumentParser(description="Generate TSV stimuli from sentences/labels/images JSON.")
    ap.add_argument("--sentences", required=True, help="Path to sentences.json")
    ap.add_argument("--labels", required=True, help="Path to labels.json")
    ap.add_argument("--images", required=True, help="Path to images.json")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    ap.add_argument("--out_full", default="stimuli_full.tsv", help="Full TSV filename")
    ap.add_argument("--max_items", type=int, default=None, help="Cap final TSV to this many rows (e.g., 80).")
    ap.add_argument("--min_items", type=int, default=30, help="Minimum required rows (default 30).")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for sampling (default 123).")
    args = ap.parse_args()

    sjson = load_json(args.sentences)
    ljson = load_json(args.labels)
    ijson = load_json(args.images)

    sentences = sjson.get("sentences", [])
    labels = ljson.get("labels", [])
    images = ijson.get("images", [])

    if not (isinstance(sentences, list) and len(sentences) > 0):
        raise ValueError("sentences.json must contain a non-empty 'sentences' list.")
    if not (isinstance(labels, list) and len(labels) > 0):
        raise ValueError("labels.json must contain a non-empty 'labels' list.")
    if not (isinstance(images, list) and len(images) > 0):
        raise ValueError("images.json must contain a non-empty 'images' list.")

    # Expand label options
    label_options: List[Dict[str, Any]] = []
    for lab in labels:
        label_options.extend(label_options_from_label(lab))

    # Build exhaustive rows
    all_rows: List[Dict[str, Any]] = []
    for s in sentences:
        sid = s.get("id", "unknown_sentence")
        template = s.get("template", "")
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"Sentence {sid} template must be a non-empty string.")
        ensure_label_location(template, sid)

        sent_gendered = sentence_has_gendered_placeholder(template)

        for lo in label_options:
            # Determine which gender renders are valid
            if sent_gendered:
                if lo["label_gender"] == "female":
                    genders = ["female"]
                elif lo["label_gender"] == "male":
                    genders = ["male"]
                else:
                    # neutral label -> both genders are possible for gendered sentences
                    genders = ["female", "male"]
            else:
                genders = [None]  # no gender substitution needed

            for g in genders:
                filled = render_template(template, lo["label_text"], g)
                filled = capitalize_first_cased_char(filled)

                # gender column reflects LABEL gender only (NOT rendered pronouns)
                # values must be EXACT: NONE, female, male
                if lo["label_gender"] in ("female", "male"):
                    gender_value = lo["label_gender"]
                else:
                    gender_value = "NONE"

                # gendered_pronouns is TRUE if either the sentence itself uses gendered placeholders,
                # or the label is gendered (sis/bro etc.)
                gendered_pronouns = sent_gendered or (lo["label_type"] == "gendered")

                for img in images:
                    all_rows.append(
                        {
                            "trial": 0,  # assigned later
                            "post_text": template,
                            "label": lo["label_text"],
                            "gender": gender_value,
                            "poster_image": img,
                            "gendered_pronouns": str(bool(gendered_pronouns)),
                            "filled_post": filled,
                        }
                    )

    # Optionally cap size (stratified by image)
    final_rows = all_rows
    if args.max_items is not None and len(final_rows) > args.max_items:
        final_rows = stratified_sample_by_image(final_rows, images, args.max_items, args.seed)

    if len(final_rows) < args.min_items:
        raise ValueError(f"Only generated {len(final_rows)} rows, which is below min_items={args.min_items}.")

    # Assign trial indices
    for i, r in enumerate(final_rows, start=1):
        r["trial"] = i

    # Write full TSV
    out_full_path = os.path.join(args.out_dir, args.out_full)
    write_tsv(out_full_path, final_rows)

    # Write per-image TSVs (still keep image column)
    for img in images:
        img_rows = [r for r in final_rows if r["poster_image"] == img]
        stem = safe_image_stem(img)
        out_path = os.path.join(args.out_dir, f"stimuli_{stem}.tsv")
        write_tsv(out_path, img_rows)

    print(f"Wrote full TSV: {out_full_path} ({len(final_rows)} rows)")
    for img in images:
        stem = safe_image_stem(img)
        print(f"Wrote per-image TSV: {os.path.join(args.out_dir, f'stimuli_{stem}.tsv')}")


if __name__ == "__main__":
    main()
