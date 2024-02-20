from transformers import AutoTokenizer, OpenAIGPTModel
import torch
import sys

if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
    model = OpenAIGPTModel.from_pretrained("openai-gpt")

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in all_sents:
        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        # sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()  # .lower()
            inputs = tokenizer(sentences, return_tensors="pt")
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            last_hidden_states = last_hidden_states.mean(axis=1).detach().numpy()
            print(type(last_hidden_states))
            print(last_hidden_states.shape)