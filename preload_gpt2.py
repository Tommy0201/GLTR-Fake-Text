from transformers import (BertForMaskedLM, BertTokenizer, ElectraForMaskedLM,
                          ElectraTokenizer, GPT2LMHeadModel, GPT2Tokenizer,
                          RobertaForMaskedLM, RobertaTokenizer)


def run(model_name_or_path):
    if model_name_or_path == "gpt2":
        GPT2Tokenizer.from_pretrained(model_name_or_path)
        GPT2LMHeadModel.from_pretrained(model_name_or_path)
        print("Loaded GPT-2 model!")
    elif model_name_or_path == "bert-base-cased":
        BertTokenizer.from_pretrained(model_name_or_path)
        BertForMaskedLM.from_pretrained(model_name_or_path)
        print("Loaded BERT model!")
    elif model_name_or_path == "roberta-base":
        RobertaTokenizer.from_pretrained(model_name_or_path)
        RobertaForMaskedLM.from_pretrained(model_name_or_path)
        print("Loaded RoBERTa model!")
    elif model_name_or_path == "google/electra-large-discriminator":
        ElectraTokenizer.from_pretrained(model_name_or_path)
        ElectraForMaskedLM.from_pretrained(model_name_or_path)
        print("Loaded ELECTRA model!")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        # Preload all models
        models = ["gpt2", "bert-base-cased", "roberta-base", "google/electra-large-discriminator"]
        for model in models:
            print(f"\\nPreloading {model}...")
            try:
                run(model)
            except Exception as e:
                print(f"Error loading {model}: {e}")

