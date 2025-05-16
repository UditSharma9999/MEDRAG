# # # from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# # # tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
# # # model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# # # dataset = TextDataset(
# # #     tokenizer=tokenizer,
# # #     file_path="train.txt",  # your processed data
# # #     block_size=128,
# # # )

# # # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # # training_args = TrainingArguments(
# # #     output_dir="./slm_model",
# # #     overwrite_output_dir=True,
# # #     per_device_train_batch_size=4,
# # #     num_train_epochs=3,
# # #     save_steps=500,
# # #     save_total_limit=2,
# # # )

# # # trainer = Trainer(
# # #     model=model,
# # #     args=training_args,
# # #     data_collator=data_collator,
# # #     train_dataset=dataset,
# # # )

# # # trainer.train()


# # # train_slm_from_pdfs.py
# # """
# # This script extracts text from medical PDFs, preprocesses it into a training file,
# # then fine-tunes a DistilGPT2 model on the resulting text.
# # """
# # import os
# # import re
# # from pathlib import Path
# # import fitz  # PyMuPDF
# # from transformers import (
# #     AutoTokenizer,
# #     AutoModelForCausalLM,
# #     Trainer,
# #     TrainingArguments,
# #     TextDataset,
# #     DataCollatorForLanguageModeling
# # )

# # # Phase 1: PDF Extraction and Preprocessing
# # def extract_text_from_pdfs(pdf_folder):
# #     """
# #     Reads all PDF files in the given folder, extracts text, cleans whitespace,
# #     and returns a list of text chunks.
# #     """
# #     pdf_folder = Path(pdf_folder)
# #     all_texts = []
# #     for pdf_path in pdf_folder.glob("*.pdf"):
# #         print(f"Processing {pdf_path.name}...")
# #         doc = fitz.open(pdf_path)
# #         for page in doc:
# #             text = page.get_text()
# #             # Clean multiple spaces and line breaks
# #             cleaned = re.sub(r"\s+", " ", text).strip()
# #             if len(cleaned) > 100:
# #                 all_texts.append(cleaned)
# #     return all_texts


# # def save_as_training_file(text_chunks, output_path):
# #     """
# #     Saves a list of text chunks to a single text file, separated by blank lines.
# #     """
# #     with open(output_path, "w", encoding="utf-8") as f:
# #         for chunk in text_chunks:
# #             f.write(chunk + "\n\n")
# #     print(f"Saved {len(text_chunks)} text chunks to {output_path}")


# # # Phase 2: Dataset Preparation
# # TRAIN_FILE = "train_medical.txt"
# # PDF_FOLDER = r"./data"

# # # Extract and save
# # text_chunks = extract_text_from_pdfs(PDF_FOLDER)
# # save_as_training_file(text_chunks, TRAIN_FILE)

# # # Phase 3: Fine-tuning the SLM
# # MODEL_NAME = "distilgpt2"  # Base model
# # OUTPUT_DIR = "./slm_medical_finetuned"

# # # Load tokenizer and model
# # print("Loading tokenizer and model...")
# # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# # # Prepare dataset
# # print("Preparing training dataset...")
# # train_dataset = TextDataset(
# #     tokenizer=tokenizer,
# #     file_path=TRAIN_FILE,
# #     block_size=512
# # )

# # # Data collator
# # data_collator = DataCollatorForLanguageModeling(
# #     tokenizer=tokenizer,
# #     mlm=False
# # )

# # # Training arguments
# # training_args = TrainingArguments(
# #     output_dir=OUTPUT_DIR,
# #     overwrite_output_dir=True,
# #     num_train_epochs=3,
# #     per_device_train_batch_size=2,
# #     save_steps=500,
# #     save_total_limit=2,
# #     logging_steps=100,
# #     logging_dir='./logs'
# # )

# # # Initialize Trainer
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     data_collator=data_collator,
# #     train_dataset=train_dataset
# # )

# # # Start training
# # print("Starting training...")
# # trainer.train()

# # # Save the fine-tuned model and tokenizer
# # print("Saving fine-tuned model...")
# # model.save_pretrained(OUTPUT_DIR)
# # tokenizer.save_pretrained(OUTPUT_DIR)
# # print(f"Model saved to {OUTPUT_DIR}")












import os
import re
import fitz  # PyMuPDF
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling
)

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        text = page.get_text()
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) > 100:
            texts.append(cleaned)
    return texts

# Step 2: Save extracted text as training file
def save_text_to_file(texts, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n\n")

# Step 3: Fine-tune the model
def fine_tune_model(train_file, model_name="distilgpt2", output_dir="./slm_disease_finetuned"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=512
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        logging_dir='./logs',
        fp16=torch.cuda.is_available(), # Enable mixed precision for faster training
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to: {output_dir}")

# Run the full pipeline
if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0))
    
    pdf_path = "./diseases.pdf"  # Your uploaded PDF
    train_txt = "train_disease.txt"
    output_dir = "./slm_disease_finetuned"

    print("🔍 Extracting PDF content...")
    text_data = extract_text_from_pdf(pdf_path)

    print("💾 Saving text to file...")
    save_text_to_file(text_data, train_txt)

    print("🚀 Fine-tuning the model...")
    fine_tune_model(train_txt, output_dir=output_dir)


