"""
Meteorite Identification - LoRA + LLM Fusion Classification
Combines DINOv2 LoRA probabilities with GPT-4o-mini for uncertain samples.

Strategy:
- LoRA prob < 0.1 or > 0.9: trust LoRA directly
- Middle range (0.15-0.85): use LLM confidence >= 0.5 as positive
- NO Top-N selection - let LLM decide naturally
"""

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ============ Configuration ============
ROOT = Path(__file__).resolve().parent
TEST_IMG = ROOT / "test_images"
LORA_DETAIL = ROOT / "dinov2算法/dinov2_lora_v2_detail.csv"
TEMPLATE_CSV = ROOT / "sample_submission.csv"
OUT = ROOT / "outputs"

# Read API configuration from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")  # Default to OpenAI

# Validate API key is set
if OPENAI_API_KEY is None:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set.\n"
        "Please set it using:\n"
        "  Linux/Mac: export OPENAI_API_KEY='your-key-here'\n"
        "  Windows: set OPENAI_API_KEY=your-key-here\n"
        "  Or create a .env file with: OPENAI_API_KEY=your-key-here"
    )

MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # Allow model override
MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", "8"))

TRUST_THRESHOLD_LOW = 0.1
TRUST_THRESHOLD_HIGH = 0.9
LLM_QUERY_RANGE_LOW = 0.20
LLM_QUERY_RANGE_HIGH = 0.85
LLM_DECISION_THRESHOLD = 0.5  # LLM confidence >= 0.5 means positive


# ============ Improved LLM Prompt ============
METEORITE_PROMPT = """You are an expert in meteorite identification. Analyze this image and rate your confidence that it is a REAL meteorite.

## Decision Criteria (MUST follow these exact standards)

### MUST-HAVE Features (any ONE present suggests meteorite)
1. **Fusion crust** - dark glassy coating with visible flow texture or weathering patterns
2. **Regmaglypts** - distinct thumbprint-like depressions on surface
3. **Metal grains** - bright metallic specks visible on surface (iron meteorites)
4. **Chondrules** - small round mineral structures visible (stony meteorites)

### Strong Negative Indicators (any ONE present suggests NOT meteorite)
1. **Perfectly smooth/polished surface** - appears worn by water or artificial
2. **Geometric shape** - spheres, cubes, or perfect cylinders
3. **Uniform color throughout** - no weathering crust or fusion crust
4. **Conchoidal fractures** - curved fracture patterns typical of volcanic glass
5. **Crystal inclusions** - visible internal crystal structures (geodes)
6. **Green/copper patina** - indicates copper oxide, not meteorite

### Shape Analysis
- Meteorites: Naturally irregular, often withunes surfaces
- NOT meteorites: Water-worn smooth stones, manufactured objects

## Confidence Scale (MUST output value in this exact range)
Rate your confidence on 0.0 to 1.0 scale:
- **1.0**: ALL must-have features present (fusion crust + metal grains + regmaglypts)
- **0.9**: 3 must-have features present
- **0.8**: 2 must-have features, no strong negatives
- **0.7**: 1 must-have feature, no strong negatives
- **0.6**: 1 must-have feature, minor doubts
- **0.5**: Mixed signals - some positive indicators but also some concerns
- **0.4**: Some negative indicators, uncertain
- **0.3**: 1 strong negative indicator, no must-haves
- **0.2**: 2+ strong negative indicators
- **0.1**: Strong negatives dominant, almost certainly not meteorite
- **0.0**: Clear non-meteorite features (manufactured, water-worn, volcanic)

## Output Format
Respond with JSON only, no other text:
{"confidence": 0.0 to 1.0, "reason": "brief explanation of your decision"}

## Examples

**Example 1:**
{"confidence": 0.95, "reason": "Clear fusion crust with flow texture, visible metal grains, and regmaglypts present on irregular dark surface"}

**Example 2:**
{"confidence": 0.15, "reason": "Perfectly smooth river stone with uniform gray color, no fusion crust or metal grains, water-worn appearance"}

**Example 3:**
{"confidence": 0.55, "reason": "Irregular dark stone but surface too smooth, no clear regmaglypts, metal content uncertain"}"""


# ============ Image Preprocessing ============
def compute_fg_mask(img: np.ndarray) -> np.ndarray:
    """Detect foreground region."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    import scipy.ndimage as ndimage
    labeled, num = ndimage.label(closed)
    if num > 0:
        sizes = ndimage.sum(closed, labeled, range(1, num + 1))
        largest = sizes.argmax() + 1
        return (labeled == largest).astype(np.uint8)
    return np.zeros_like(gray, dtype=np.uint8)


def crop_to_fg(img: np.ndarray, mask: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """Crop to foreground region."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = img.shape[:2]
    margin_h = int((rmax - rmin) * margin)
    margin_w = int((cmax - cmin) * margin)

    rmin = max(0, rmin - margin_h)
    rmax = min(h, rmax + margin_h)
    cmin = max(0, cmin - margin_w)
    cmax = min(w, cmax + margin_w)

    cropped = img[rmin:rmax, cmin:cmax]
    return cv2.resize(cropped, (512, 512))


def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess image: detect and crop to foreground."""
    img = np.array(Image.open(image_path).convert("RGB"))
    mask = compute_fg_mask(img)

    fg_ratio = mask.sum() / (img.shape[0] * img.shape[1])
    if fg_ratio > 0.05:
        cropped = crop_to_fg(img, mask)
        return cropped
    return cv2.resize(img, (512, 512))


# ============ LLM Classification ============
def encode_image(image_path: str) -> str:
    """Encode image as base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def classify_gpt(image_path: str, prompt: str = METEORITE_PROMPT) -> tuple[float, str]:
    """Use GPT-4o-mini to classify image, return confidence and reason."""
    import openai
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}},
                {"type": "text", "text": prompt}
            ]
        }],
        max_tokens=500,
        temperature=0.1
    )

    result_text = response.choices[0].message.content

    try:
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        else:
            result_text = result_text.strip()

        result = json.loads(result_text)
        confidence = float(result.get("confidence", 0.5))
        reason = result.get("reason", "")
        return confidence, reason
    except json.JSONDecodeError:
        return 0.5, "Failed to parse LLM response"


# ============ Fusion Logic ============
def make_decision(lora_prob: float, llm_confidence: float) -> int:
    """
    Make final decision based on LoRA and LLM.

    - LoRA < 0.1: negative (trust LoRA)
    - LoRA > 0.9: positive (trust LoRA)
    - Middle range: LLM confidence >= 0.5 means positive
    """
    if lora_prob < TRUST_THRESHOLD_LOW:
        return 0
    if lora_prob > TRUST_THRESHOLD_HIGH:
        return 1
    # Middle range: LLM decides
    return 1 if llm_confidence >= LLM_DECISION_THRESHOLD else 0


def make_final_submission(all_results):
    """
    Create final submission by fusing LoRA and LLM predictions.
    all_results already contains lora_prob and llm_confidence columns.
    """
    # Apply fusion for each sample
    final_labels = []

    for _, row in all_results.iterrows():
        lora_prob = row["lora_prob"]
        llm_conf = row.get("llm_confidence", 0.5) if pd.notna(row.get("llm_confidence")) else 0.5
        label = make_decision(lora_prob, llm_conf)
        final_labels.append(label)

    all_results = all_results.copy()
    all_results["final_label"] = final_labels

    # Create submission matching sample_submission.csv format
    template = pd.read_csv(TEMPLATE_CSV)
    submission = template.copy()
    submission = submission.merge(all_results[["id", "final_label"]], on="id", how="left")
    submission["label"] = submission["final_label"].astype(int)
    submission = submission[["id", "label"]]

    return submission, all_results


# ============ Main Pipeline ============
def main():
    OUT.mkdir(exist_ok=True)

    # Load LoRA probabilities
    lora_df = pd.read_csv(LORA_DETAIL)
    print(f"Loaded {len(lora_df)} LoRA predictions")
    print(f"LoRA prob range: {lora_df['lora_prob'].min():.4f} - {lora_df['lora_prob'].max():.4f}")

    # Identify samples needing LLM classification
    middle_range = lora_df[(lora_df["lora_prob"] >= LLM_QUERY_RANGE_LOW) &
                           (lora_df["lora_prob"] <= LLM_QUERY_RANGE_HIGH)].copy()
    print(f"Samples in middle range ({LLM_QUERY_RANGE_LOW}-{LLM_QUERY_RANGE_HIGH}): {len(middle_range)}")

    # Show distribution
    low_trust = (lora_df["lora_prob"] < TRUST_THRESHOLD_LOW).sum()
    high_trust = (lora_df["lora_prob"] > TRUST_THRESHOLD_HIGH).sum()
    print(f"Trust zones: LoRA<{TRUST_THRESHOLD_LOW}={low_trust}, LoRA>{TRUST_THRESHOLD_HIGH}={high_trust}")

    # Prepare LLM classification tasks
    tasks = []
    for _, row in middle_range.iterrows():
        img_path = TEST_IMG / row["id"]
        if img_path.exists():
            tasks.append((row["id"], str(img_path)))
        else:
            print(f"Warning: Image not found: {img_path}")

    print(f"LLM queries needed: {len(tasks)}")

    # Process with LLM (concurrent)
    llm_results = []
    if tasks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(classify_gpt, img_path): img_id
                       for img_id, img_path in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Classification"):
                img_id = futures[future]
                try:
                    confidence, reason = future.result()
                    llm_results.append({
                        "id": img_id,
                        "llm_confidence": confidence,
                        "llm_reason": reason
                    })
                except Exception as e:
                    print(f"Error classifying {img_id}: {e}")
                    llm_results.append({
                        "id": img_id,
                        "llm_confidence": 0.5,
                        "llm_reason": f"Error: {str(e)}"
                    })
    else:
        print("No samples need LLM classification")

    llm_df = pd.DataFrame(llm_results)
    if len(llm_df) > 0:
        llm_df.to_csv(OUT / "llm_fusion_results.csv", index=False)
        print(f"Saved LLM results: {OUT / 'llm_fusion_results.csv'}")

    # For all samples, initialize with LoRA values
    all_results = lora_df.copy()
    all_results["llm_confidence"] = all_results["lora_prob"]
    all_results["llm_reason"] = "From LoRA (trust zone)"

    # Update with actual LLM results where available
    if len(llm_df) > 0:
        for _, row in llm_df.iterrows():
            mask = all_results["id"] == row["id"]
            all_results.loc[mask, "llm_confidence"] = row["llm_confidence"]
            all_results.loc[mask, "llm_reason"] = row["llm_reason"]

    # Create final submission (LLM decides naturally)
    submission, detail_df = make_final_submission(all_results)

    # Save outputs
    submission.to_csv(OUT / "submission_fusion.csv", index=False)
    detail_df.to_csv(OUT / "submission_fusion_detail.csv", index=False)

    # Summary statistics
    positive_count = submission["label"].sum()
    print(f"\n=== Fusion Summary ===")
    print(f"Total samples: {len(submission)}")
    print(f"Positive predictions: {positive_count} (natural LLM decision)")
    print(f"Decision threshold: LLM confidence >= {LLM_DECISION_THRESHOLD}")

    # Count by source
    llm_used = len(tasks)
    lora_trusted = len(lora_df) - llm_used
    print(f"From LLM: {llm_used}, From LoRA (trust): {lora_trusted}")

    # Show positive predictions
    positive_samples = detail_df[detail_df["final_label"] == 1].sort_values("llm_confidence", ascending=False)
    print(f"\n=== Positive predictions ({len(positive_samples)}) ===")
    for _, row in positive_samples.head(15).iterrows():
        source = "LoRA" if row["lora_prob"] < TRUST_THRESHOLD_LOW or row["lora_prob"] > TRUST_THRESHOLD_HIGH else "LLM"
        print(f"  {row['id']}: {source} conf={row['llm_confidence']:.3f} - {row['llm_reason'][:60]}")

    print(f"\nSubmission: {OUT / 'submission_fusion.csv'}")
    print(f"Detail: {OUT / 'submission_fusion_detail.csv'}")


if __name__ == "__main__":
    main()