import os
import torch
from transformers import AutoModel, AutoProcessor


class ClipModel:
    SUPPORTED_MODELS = {
        "kaveh/rclip",
        "openai/clip-vit-patch32",
        "stanfordmimi/RoentGen-v2/text_encoder_and_tokenizer",
    }

    def __init__(self, model_name="kaveh/rclip", device="cuda"):
        self.initialized = False
        self.model_name = model_name
        self.device = device
        self.clip_model = None
        self.clip_processor = None

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported CLIP model: {self.model_name}. "
                f"Supported: {sorted(self.SUPPORTED_MODELS)}"
            )

        self.classes = [
            "normal",
            "pneumonia",
            "covid",
            "enlarged cardiomediastinum",
            "cardiomegaly",
            "lung opacity",
            "lung lesion",
            "edema",
            "consolidation",
            "atelectasis",
            "pneumothorax",
            "pleural effusion",
            "pleural other",
            "fracture",
            "infiltration",
            "effusion",
            "nodule",
            "mass",
            "pleural_thickening",
            "emphysema",
            "fibrosis",
            "hernia",
        ]

        # Class prompts aligned with radiology-report phrasing.
        self.class_texts = [
            "no acute cardiopulmonary findings",  # normal
            "evidence of pneumonia",
            "consistent with COVID-19 pneumonia",
            "an enlarged cardiomediastinum",
            "cardiomegaly",
            "lung opacity",
            "a lung lesion",
            "pulmonary edema",
            "consolidation",
            "atelectasis",
            "pneumothorax",
            "pleural effusion",
            "other pleural abnormalities",
            "rib fracture",
            "pulmonary infiltrates",
            "effusion",
            "a nodule",
            "a mass",
            "pleural thickening",
            "emphysema",
            "pulmonary fibrosis",
            "hernia",
        ]

    def _init_if_needed(self):
        if not self.initialized:
            model_path = os.path.join("../HuggingFace_Model", self.model_name)
            self.clip_model = AutoModel.from_pretrained(
                model_path, local_files_only=True
            )
            self.clip_processor = AutoProcessor.from_pretrained(
                model_path, local_files_only=True
            )
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            self.initialized = True

    def encode_texts(self, texts):
        self._init_if_needed()
        if isinstance(texts, str):
            texts = [texts]

        text_inputs = self.clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            try:
                text_features = self.clip_model.get_text_features(**text_inputs)
            except AttributeError:
                outputs = self.clip_model(**text_inputs)
                text_features = (
                    outputs.pooler_output
                )  # This is the equivalent embedding
        return text_features

    def encode_images(self, images):
        self._init_if_needed()
        rgb_images = [img.convert("RGB") for img in images]

        image_inputs = self.clip_processor(images=rgb_images, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(
                pixel_values=image_inputs["pixel_values"]
            )
        return image_features

    def multilabel_to_indices(self, labels):
        """
        Convert a (batch, classes) multi-label tensor into active-class index lists.

        Args:
            labels: Tensor of shape (batch, classes) with binary values

        Returns:
            list of lists: active class indices for each sample.
        """
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        result = []
        for i in range(labels.size(0)):
            nonzero_indices = torch.nonzero(labels[i], as_tuple=True)[0].tolist()
            result.append(nonzero_indices)

        return result

    def combine_texts(self, label_indices):
        """Combine multiple class descriptions into a concise natural-language report."""
        self._init_if_needed()
        findings = [self.class_texts[i] for i in label_indices]

        if not findings:
            return "a chest X-ray with no acute cardiopulmonary findings"

        prefix = "a chest X-ray showing"
        if len(findings) == 1:
            return f"{prefix} {findings[0]}"

        # Join multiple findings with commas and “and”.
        return f"{prefix} {', '.join(findings[:-1])}, and {findings[-1]}"

    def combine_texts_RoentGen(self, label_indice):
        index_to_text = {
            0: "no acute cardiopulmonary findings",
            1: "evidence of pneumonia",
            2: "findings consistent with COVID-19 pneumonia",
            3: "an enlarged cardiomediastinum",
            4: "cardiomegaly",
            5: "lung opacity",
            6: "a lung lesion",
            7: "pulmonary edema",
            8: "consolidation",
            9: "atelectasis",
            10: "pneumothorax",
            11: "pleural effusion",
            12: "other pleural abnormalities",
            13: "rib fracture",
            14: "pulmonary infiltrates",
            15: "effusion",
            16: "a pulmonary nodule",
            17: "a pulmonary mass",
            18: "pleural thickening",
            19: "emphysema",
            20: "pulmonary fibrosis",
            21: "hernia",
        }

        # Use “normal” when no positive label is present.
        if not label_indice:
            findings = [index_to_text[0]]
        else:
            # Ignore “normal” when another label is positive.
            if 0 in label_indice and len(label_indice) > 1:
                label_indice = [i for i in label_indice if i != 0]
            findings = [index_to_text[i] for i in label_indice]

        # Build the report-style subject sentence.
        # Concise report style: "50 year old female. Normal chest radiograph."
        if findings == [index_to_text[0]]:
            main = "Normal chest radiograph."
        else:
            # Separate findings with semicolons to keep the prompt short.
            main = "Chest radiograph: " + "; ".join(findings) + "."
        # else:
        #     # More natural report-style phrasing.
        #     if findings == [index_to_text[0]]:
        #         main = "Normal chest radiograph."
        #     elif len(findings) == 1:
        #         main = f"Chest radiograph shows {findings[0]}."
        #     else:
        #         main = "Chest radiograph shows " + ", ".join(findings[:-1]) + ", and " + findings[-1] + "."
        return main

    def build_prompt(self, labels):
        label_indices = self.multilabel_to_indices(labels)
        text_descriptions = [
            self.combine_texts_RoentGen(indices) for indices in label_indices
        ]
        return text_descriptions

    def caculate_embdding(self, labels):
        label_indices = self.multilabel_to_indices(labels)
        text_descriptions = [self.combine_texts(indices) for indices in label_indices]
        return self.encode_texts(text_descriptions)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip = ClipModel(
        model_name="stanfordmimi/RoentGen-v2/text_encoder_and_tokenizer", device=device
    )

    # Test a single-label prompt.
    single_label = torch.tensor(
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    print("Single label description:", clip.build_prompt(single_label))
