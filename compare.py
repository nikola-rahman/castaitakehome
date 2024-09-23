import pickle
from datasets import load_dataset
from evaluate import load as load_metric
import matplotlib.pyplot as plt
import numpy as np
import tiktoken

# Function to load predictions from a pickle file
def load_predictions_from_file(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    predictions = [{"id": id, "prediction_text": pred} for id, pred in zip(data["id"], data["predictions"])]
    return predictions, data

# Function to evaluate predictions using the SQuAD metric
def evaluate_predictions(predictions, dataset):
    squad_metric = load_metric("squad")
    references = [{"id": example["id"], "answers": {"text": example['answers']['text'], "answer_start": example['answers']['answer_start']}} for example in dataset]
    results = squad_metric.compute(predictions=predictions, references=references)
    return results

def llama_cost(price_instance_per_hour, tokens_per_second):
    # calculate $ per million tokens based on the price per instance and tokens per second
    price_per_million_tokens = price_instance_per_hour / (tokens_per_second * 3600) * 1e6

    return price_per_million_tokens

def gpt_cost(price_input_tokens_per_million, price_output_tokens_per_million, input_tokens_percent):
    # calculate cost per million tokens based on input and output token prices
    price_per_million_tokens = price_input_tokens_per_million * input_tokens_percent + price_output_tokens_per_million * (1 - input_tokens_percent)

    return price_per_million_tokens

# Load the validation dataset
val_dataset = load_dataset("rajpurkar/squad", split="validation")#, split=[f"validation[{k}%:{k+1}%]" for k in range(0, 100, 100)])
val_dataset = val_dataset.select(range(0, len(val_dataset), 100))

# List of pickle files with results
models = ["baseline_zero_shot", "openai_gpt35", "openai_gpt4o", "finetuned"]
pickle_files = [f"predictions_{model}.pickle" for model in models]
pickle_files = [f"data/{x}" for x in pickle_files]

# Store results for visualization
exact_match_scores = []
f1_scores = []

# Compare models
for pickle_file in pickle_files:
    predictions, data = load_predictions_from_file(pickle_file)
    results = evaluate_predictions(predictions, val_dataset)
    exact_match_scores.append(results['exact_match'])
    f1_scores.append(results['f1'])
    print(f"Results for {pickle_file}:")
    print(results)
    print("\n")

# Visualization
x = range(len(models))

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

model_names = [
    "Llama 3.1 8B (baseline)",
    "GPT-3.5 Instruct (few-shot)",
    "GPT-4o (few-shot)",
    "Llama 3.1 8B (fine-tuned)"
]

# Exact Match Scores
ax[0].bar(x, exact_match_scores, color='skyblue')
ax[0].set_xticks(x)
# ax[0].set_xticklabels(model_names)
ax[0].set_title('Exact Match Scores')
ax[0].set_ylabel('Exact Match (%)')
# set text annotations for model names instead of tick labels
for i, model_name in enumerate(model_names):
    ax[0].text(i, exact_match_scores[i], model_name, ha='center', va='bottom')
# remove all xtick labels
ax[0].set_xticklabels([])

# F1 Scores
ax[1].bar(x, f1_scores, color='lightgreen')
ax[1].set_xticks(x)
# ax[1].set_xticklabels(model_names)
ax[1].set_title('F1 Scores')
ax[1].set_ylabel('F1 Score (%)')
# set text annotations for model names instead of tick labels
for i, model_name in enumerate(model_names):
    ax[1].text(i, f1_scores[i], model_name, ha='center', va='bottom')
# remove all xtick labels
ax[1].set_xticklabels([])

plt.tight_layout()
# save the plot
plt.savefig("plots/comparison_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Function to calculate the length of question and context in tokens
def calculate_length_in_tokens(example, enc):
    question_tokens = enc.encode(example['question'])
    context_tokens = enc.encode(example['context'])
    answer_tokens = enc.encode(example['answers']['text'][0])
    return len(question_tokens) + len(context_tokens) + len(answer_tokens)

# Function to calculate accuracy for each example
def calculate_accuracy(predictions, dataset):
    correct = 0
    total = len(predictions)
    for pred, example in zip(predictions, dataset):
        if pred['prediction_text'] in example['answers']['text']:
            correct += 1
    return correct / total if total > 0 else 0

# Initialize the tokenizer
enc = tiktoken.encoding_for_model("gpt-4o")

# plot distribution of question + context lengths in tokens
lengths_in_tokens = [calculate_length_in_tokens(example, enc) for example in val_dataset]
plt.figure(figsize=(14, 6))
plt.hist(lengths_in_tokens, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Question + Context Length (Tokens)')
plt.ylabel('Count')
plt.title('Distribution of Question + Context Lengths (Tokens)')
plt.savefig("plots/length_distribution_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Define bins for lengths in tokens
bins = [0, 100, 150, 1000]
bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

# Store lengths and accuracies for visualization
lengths_in_tokens = []
accuracies = {model: [] for model in models}

# Calculate lengths in tokens
for example in val_dataset:
    length_in_tokens = calculate_length_in_tokens(example, enc)
    lengths_in_tokens.append(length_in_tokens)

# Bin the lengths in tokens
binned_lengths_in_tokens = np.digitize(lengths_in_tokens, bins)

# Calculate accuracies for each bin
for model, pickle_file in zip(models, pickle_files):
    predictions, _ = load_predictions_from_file(pickle_file)
    bin_accuracies = []
    for i in range(1, len(bins)):
        bin_indices = [index for index, bin_num in enumerate(binned_lengths_in_tokens) if bin_num == i]
        bin_predictions = [predictions[index] for index in bin_indices]
        bin_dataset = [val_dataset[index] for index in bin_indices]
        accuracy = calculate_accuracy(bin_predictions, bin_dataset)
        bin_accuracies.append(accuracy)
    accuracies[model] = bin_accuracies

# Plotting
x = range(len(bin_labels))

fig, ax = plt.subplots(figsize=(14, 6))

for model in models:
    ax.plot(x, accuracies[model], label=model, marker='o')

ax.set_xticks(x)
ax.set_xticklabels(bin_labels, rotation=45)
ax.set_xlabel('Question + Context Length Bins (Tokens)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Question + Context Length Bins (Tokens)')
ax.legend()

plt.tight_layout()
plt.savefig("plots/accuracy_vs_length_bins_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# print finetuned and gpt4o predictions for examples where length is less than 50 tokens
for example, length_in_tokens in zip(val_dataset, lengths_in_tokens):
    if length_in_tokens < 100:
        print(100*"-")
        print(f"Question: {example['question']}")
        print(f"Context: {example['context']}")
        print(f"Answer: {example['answers']['text']}")
        for model, pickle_file in zip(models, pickle_files):
            predictions, _ = load_predictions_from_file(pickle_file)
            prediction = [pred['prediction_text'] for pred in predictions if pred['id'] == example['id']]
            print(f"{model} Prediction: {prediction[0]}")
        print("\n")

# plot accuracy vs cost for different models
price_instance_per_hour = 0.0672
tokens_per_second = 10
baseline_cost = llama_cost(price_instance_per_hour, tokens_per_second)
fine_tuned_cost = llama_cost(price_instance_per_hour, tokens_per_second)
gpt35_cost = gpt_cost(1.5, 2.0, 0.9)
gpt4o_cost = gpt_cost(5, 15, 0.9)

costs = [baseline_cost, gpt35_cost, gpt4o_cost, fine_tuned_cost]

# plot accuracy vs cost
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(costs, f1_scores, color='skyblue')
plt.xlabel('Cost per Million Tokens ($)')
plt.ylabel('F1 Score (%)')
plt.title('F1 Score vs Cost per Million Tokens')
for i, model in enumerate(model_names):
    print(model)
    ax.annotate(model, (costs[i], f1_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.savefig("plots/accuracy_vs_cost_plot.png", dpi=300, bbox_inches='tight')
plt.show()
