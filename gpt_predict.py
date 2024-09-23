import pickle

from datasets import load_dataset, concatenate_datasets
from evaluate import load as load_metric
from tqdm import tqdm

from openai import OpenAI

client = OpenAI()

test_flag = False

# Load the validation dataset slices
if test_flag:
    val_dataset = load_dataset("rajpurkar/squad", split=f"validation[:10]")
else:
    val_dataset = load_dataset("rajpurkar/squad", split="validation")#, split=[f"validation[{k}%:{k+1}%]" for k in range(0, 100, 100)])
    val_dataset = val_dataset.select(range(0, len(val_dataset), 100))

# val_dataset = val_dataset.map(formatting_prompts_func, batched = True,)

print(val_dataset)

# exit()

# Define the prompt template
squad_prompt_template = """Below is a question, paired with a context. Write a response that extracts the answer from the context. Only include the extracted text from the context in the answer without any extra words.

### Question:
Which individual worked on projects at Notre Dame that eventually created neoprene?

### Context:
In 1882, Albert Zahm (John Zahm's brother) built an early wind tunnel used to compare lift to drag of aeronautical models. Around 1899, Professor Jerome Green became the first American to send a wireless message. In 1931, Father Julius Nieuwland performed early work on basic reactions that was used to create neoprene. Study of nuclear physics at the university began with the building of a nuclear accelerator in 1936, and continues now partly through a partnership in the Joint Institute for Nuclear Astrophysics.

### Answer:
Father Julius Nieuwland

### Question:
What did Beyoncé announce in January 2010?

### Context:
Beyoncé announced a hiatus from her music career in January 2010, heeding her mother's advice, "to live life, to be inspired by things again". During the break she and her father parted ways as business partners. Beyoncé's musical break lasted nine months and saw her visit multiple European cities, the Great Wall of China, the Egyptian pyramids, Australia, English music festivals and various museums and ballet performances.

### Answer:
a hiatus

### Question:
{}

### Context:
{}

### Answer:
"""



# Function to get predictions from GPT-4
def get_gpt4_predictions(dataset):
    predictions = []
    for example in tqdm(dataset, total=len(dataset)):

        squad_prompt = squad_prompt_template.format(
            example['question'],  # question
            example['context'],   # context
            ""                    # answer - leave blank for generation
        )

        # gpt3
        # response = client.completions.create(
        #     model="gpt-3.5-turbo-instruct",
        #     prompt=squad_prompt,
        #     max_tokens=64,
        #     n=1,
        #     stop=None,
        #     temperature=0.0
        # )

        # gpt4
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You answer questions by extracting answer from context."},
                {"role": "user", "content": squad_prompt}
            ],
            max_tokens=64,
            n=1,
            stop=None,
            temperature=0.0
        )

        # gpt3
        # generated_answer = response.choices[0].text.strip()
        # generated_answer = generated_answer.split("Answer:")[1].strip() if "Answer:" in generated_answer else generated_answer

        # gpt4
        generated_answer = response.choices[0].message.content.strip()


        predictions.append({
            "id": example["id"],  # use the id from the dataset
            "prediction_text": generated_answer.strip()  # the generated text
        })

    return predictions

# Function to save predictions to a pickle file
def save_predictions_to_file(predictions, filename):
    data = val_dataset.to_dict()
    data["predictions"] = [p["prediction_text"] for p in predictions]

    with open(filename, "wb") as f:
        pickle.dump(data, f)

# Function to evaluate predictions using the SQuAD metric
def evaluate_predictions(predictions, dataset):
    squad_metric = load_metric("squad")
    references = [{"id": example["id"], "answers": {"text": example['answers']['text'], "answer_start": example['answers']['answer_start']}} for example in dataset]
    results = squad_metric.compute(predictions=predictions, references=references)
    return results

# Main function to run the prediction and evaluation pipeline
def main():
    # Get GPT-4 predictions
    predictions = get_gpt4_predictions(val_dataset)

    # Save predictions to a JSON file
    save_predictions_to_file(predictions, filename="predictions_openai_gpt4o.pickle")

    # Print the first 10 predictions for inspection
    for i in range(10):
        print(f"Prediction {i+1}: {predictions[i]['prediction_text']}")

    # Evaluate the predictions
    results = evaluate_predictions(predictions, val_dataset)

    # Print the evaluation results
    print(results)


if __name__ == "__main__":
    main()

