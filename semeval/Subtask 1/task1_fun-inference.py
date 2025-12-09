import json
import sys
import numpy as np
import os
import glob
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification,
    TrainingArguments,
)
from collections import defaultdict
from scipy.special import softmax

MODEL_PATH_BASE = "distilbert-single-type-simplified"
MARKER_TYPES_TO_INFER = ["Action", "Actor", "Effect", "Evidence", "Victim"]
TEST_FILE = r"D:\Pycharm_2025.2.4\test\dev_rehydrated.jsonl"
SUBMISSION_FILE = r"D:\Pycharm_2025.2.4\test\submission.jsonl"
MODEL_NAME = r"D:\models\distilbert-base-uncased"
BATCH_SIZE = 64
CONFIDENCE_THRESHOLD = 0.30
MAX_SPAN_GAP = 10
MAX_SEQ_LEN = 256


def find_latest_checkpoint(base_path, marker_type):
    full_path = f"{base_path}-{marker_type}"
    checkpoint_dirs = glob.glob(os.path.join(full_path, "checkpoint-*"))

    if not checkpoint_dirs:
        print(f"Warning: No 'checkpoint-*' folder found. Assuming model files are in: {full_path}")
        return full_path

    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))
    latest_checkpoint = checkpoint_dirs[-1]
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                item["_id"] = item.get("_id", f"sample_{i}")
                item["text"] = item.get("text", "")
                item["markers"] = item.get("markers", [])
                item["conspiracy"] = item.get("conspiracy", "No")
                data.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at index {i} in {file_path}: {line.strip()}")
    print(f"Loaded {len(data)} samples for inference.")
    return data


def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        return_offsets_mapping=True
    )
    tokenized_inputs["labels"] = [[-100] * len(offset_map) for offset_map in tokenized_inputs["offset_mapping"]]
    return tokenized_inputs


def smooth_predictions(pred_ids, window_size=3):
    pred_ids = np.array(pred_ids)
    smoothed = pred_ids.copy()
    half = window_size // 2
    for i in range(len(pred_ids)):
        start = max(0, i - half)
        end = min(len(pred_ids), i + half + 1)
        window = pred_ids[start:end]
        if (window == 1).sum() >= 2:
            smoothed[i] = 1
    return smoothed.tolist()


def confidence_based_filtering(pred_ids, softmax_scores=None, threshold=CONFIDENCE_THRESHOLD):
    if not isinstance(pred_ids, (list, np.ndarray)):
        pred_ids = [int(pred_ids)]
    pred_ids = list(pred_ids)
    if softmax_scores is None:
        return pred_ids
    if not isinstance(softmax_scores, (list, np.ndarray)):
        softmax_scores = [float(softmax_scores)]
    return [
        pid if softmax_scores[i] > threshold else 0
        for i, pid in enumerate(pred_ids)
    ]


def merge_close_spans(spans, max_gap=MAX_SPAN_GAP):
    if not spans:
        return spans
    merged = [spans[0]]
    for s in spans[1:]:
        if s["startIndex"] - merged[-1]["endIndex"] <= max_gap:
            merged[-1]["endIndex"] = s["endIndex"]
            merged[-1]["text"] = merged[-1]["text"] + " " + s["text"]
        else:
            merged.append(s)
    return merged


def filter_short_spans(spans, min_len=1):
    return [s for s in spans if (s["endIndex"] - s["startIndex"]) >= min_len]


def reconstruct_spans(predictions, tokenized_dataset, id_to_label, softmax_scores_list=None):
    reconstructed_markers = defaultdict(list)
    positive_label_type = id_to_label.get(1)

    if not positive_label_type or positive_label_type == "O":
        print("Error: Model configuration does not match simplified binary training (ID 1 is not the marker type).")
        return reconstructed_markers

    for i, pred_ids in enumerate(predictions):
        smoothed_pred_ids = smooth_predictions(pred_ids)
        token_softmax = None
        if softmax_scores_list is not None:
            token_softmax = softmax_scores_list[i]
        filtered_pred_ids = confidence_based_filtering(smoothed_pred_ids, softmax_scores=token_softmax)

        offsets = tokenized_dataset[i]['offset_mapping']
        original_text = tokenized_dataset[i]['text']

        current_span_start_char = None

        for token_idx, label_id in enumerate(filtered_pred_ids):
            offset_tuple = offsets[token_idx]
            if offset_tuple is None or offset_tuple[0] is None or offset_tuple[1] is None or (
                    offset_tuple[0] == 0 and offset_tuple[1] == 0):
                if current_span_start_char is not None:
                    prev_end_char = offsets[token_idx - 1][1] if token_idx > 0 and offsets[token_idx - 1][1] else None
                    if prev_end_char is not None:
                        span_text = original_text[current_span_start_char:prev_end_char]
                        reconstructed_markers[i].append({
                            "startIndex": current_span_start_char,
                            "endIndex": prev_end_char,
                            "type": positive_label_type,
                            "text": span_text
                        })
                    current_span_start_char = None
                continue

            label = id_to_label[label_id]
            start_char = offset_tuple[0]

            if label == positive_label_type:
                if current_span_start_char is None:
                    current_span_start_char = start_char
            else:
                if current_span_start_char is not None:
                    prev_end_char = offsets[token_idx - 1][1] if token_idx > 0 and offsets[token_idx - 1][1] else start_char
                    span_text = original_text[current_span_start_char:prev_end_char]
                    reconstructed_markers[i].append({
                        "startIndex": current_span_start_char,
                        "endIndex": prev_end_char,
                        "type": positive_label_type,
                        "text": span_text
                    })
                    current_span_start_char = None

        if current_span_start_char is not None:
            last_valid_end = None
            last_token_idx = len(filtered_pred_ids) - 1
            while last_token_idx >= 0:
                offset_tuple_end = offsets[last_token_idx]
                if offset_tuple_end is not None and offset_tuple_end[1] is not None and offset_tuple_end[1] != 0:
                    last_valid_end = offset_tuple_end[1]
                    break
                last_token_idx -= 1
            if last_valid_end is not None:
                span_text = original_text[current_span_start_char:last_valid_end]
                reconstructed_markers[i].append({
                    "startIndex": current_span_start_char,
                    "endIndex": last_valid_end,
                    "type": positive_label_type,
                    "text": span_text
                })

    # merge spans across tokens
    for i in reconstructed_markers:
        reconstructed_markers[i] = merge_close_spans(reconstructed_markers[i], max_gap=MAX_SPAN_GAP)
        reconstructed_markers[i] = filter_short_spans(reconstructed_markers[i], min_len=1)

    return reconstructed_markers


if __name__ == '__main__':
    raw_data = load_data(TEST_FILE)
    if not raw_data:
        print("Error: No data loaded. Cannot perform inference.")
        sys.exit(-1)

    unique_ids = [d["_id"] for d in raw_data]
    conspiracy_keys = [d["conspiracy"] for d in raw_data]
    test_dataset = Dataset.from_list(raw_data)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    dummy_label_to_id = {"O": 0}

    tokenized_test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=[col for col in test_dataset.column_names if col not in ['text', 'offset_mapping', '_id', 'conspiracy']],
        fn_kwargs={"tokenizer": tokenizer, "label_to_id": dummy_label_to_id}
    )

    all_predicted_markers = defaultdict(list)

    for marker_type in MARKER_TYPES_TO_INFER:
        model_directory = find_latest_checkpoint(MODEL_PATH_BASE, marker_type)
        print(f"\n--- Running inference for type: {marker_type} ---")
        print(f"Loading model from: {model_directory}")

        try:
            model = DistilBertForTokenClassification.from_pretrained(model_directory)
            id_to_label = {0: "O", 1: marker_type}
        except Exception as e:
            print(f"Error loading model for {marker_type} from '{model_directory}'. Details: {e}")
            continue

        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer_infer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"./tmp_inference_span_{marker_type}",
                per_device_eval_batch_size=BATCH_SIZE,
                report_to="none"
            ),
            data_collator=data_collator
        )

        predictions_output = trainer_infer.predict(tokenized_test_dataset)
        logits = predictions_output.predictions
        predicted_class_ids = np.argmax(logits, axis=2)
        token_softmax_scores = softmax(logits, axis=2).max(axis=2)

        current_marker_map = reconstruct_spans(predicted_class_ids, tokenized_test_dataset, id_to_label, softmax_scores_list=token_softmax_scores)

        for i, markers in current_marker_map.items():
            all_predicted_markers[i].extend(markers)

    print(f"\nSaving final aggregated predictions ({len(raw_data)} samples) to {SUBMISSION_FILE} (JSONL format)...")

    jsonl_lines = []
    for i in range(len(raw_data)):
        predicted_markers = all_predicted_markers.get(i, [])
        jsonl_obj = {
            "_id": unique_ids[i],
            "conspiracy": conspiracy_keys[i],
            "markers": predicted_markers
        }
        jsonl_lines.append(json.dumps(jsonl_obj))

    with open(SUBMISSION_FILE, 'w') as f:
        f.write('\n'.join(jsonl_lines) + '\n')

    print(f"Submission file '{SUBMISSION_FILE}' generated successfully.")
