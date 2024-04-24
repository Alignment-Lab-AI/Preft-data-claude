import os
import json
import argparse
import requests
from pathlib import Path
from datasets import load_dataset

API_URL = "https://api.anthropic.com/v1/messages"
API_KEY = "YOUR_API_KEY"  # Replace with your actual API key

def process_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    if file_path.endswith(".py"):
        prompt = content + "\n\n<instruction>respond with a flat jsonl in the given format, simulating a 3-6 turn conversation between a user(human) and assistant(gpt), demonstrating a plausable situation requiring the assistant to articulate a strong understanding of the logical elements of the content in the example</instruction>\n<format>{\"conversations\": [{\"from\": \"human\", \"value\": \"...\"}, {\"from\": \"gpt\", \"value\": \"...\"}, {\"from\": \"human\", \"value\": \"...\"}, ...]}</format>"
        paragraphs = [prompt]
    else:
        paragraphs = content.split("\n\n")

    dataset = []
    ratings = []

    for paragraph in paragraphs:
        payload = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": paragraph}],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
        }

        response = requests.post(API_URL, json=payload, headers=headers)
        response_data = response.json()

        if "content" in response_data:
            model_response = response_data["content"][0]["text"]
            jsonl_row = model_response.strip()
            dataset.append({"input": paragraph, "output": jsonl_row})

            rating_prompt = paragraph + "\n\n<instruction>rate the quality of this piece of data on a scale of 1 to 10 where 1 is useless, nonsensical, or otherwise blatantly wrong, and 10 is perfect, verbose, reasoning-heavy, etc   the use case is for training data.</instruction>\n<condition>ONLY REPLY WITH THE INTEGER CORRESPONDING TO YOUR RATING. DO NOT RESPOND WITH ANYTHING BUT THE VALUE</condition>"
            rating_payload = {
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": rating_prompt}],
                "max_tokens": 1,
                "temperature": 0.0,
            }
            rating_response = requests.post(API_URL, json=rating_payload, headers=headers)
            rating_data = rating_response.json()

            if "content" in rating_data:
                rating = int(rating_data["content"][0]["text"].strip())
                ratings.append({"input": paragraph, "output": rating})

    return dataset, ratings

def process_folder(folder_path):
    dataset = []
    ratings = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and (file_path.endswith(".txt") or file_path.endswith(".py")):
            file_dataset, file_ratings = process_file(file_path)
            dataset.extend(file_dataset)
            ratings.extend(file_ratings)

    return dataset, ratings

def process_repo(repo_path):
    dataset = load_dataset(repo_path)
    selected_rows = []

    print("Dataset rows:")
    for i, row in enumerate(dataset["train"]):
        print(f"{i}: {row}")

    while True:
        row_numbers = input("Enter the row numbers to select (comma-separated) or press Enter to finish: ")
        if not row_numbers:
            break

        selected_row_numbers = [int(num.strip()) for num in row_numbers.split(",")]
        selected_rows.extend([dataset["train"][num] for num in selected_row_numbers])

    dataset = []
    ratings = []

    for row in selected_rows:
        row_content = "\n".join([f"{key}: {value}" for key, value in row.items()])
        payload = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": row_content}],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
        }

        response = requests.post(API_URL, json=payload, headers=headers)
        response_data = response.json()

        if "content" in response_data:
            model_response = response_data["content"][0]["text"]
            jsonl_row = model_response.strip()
            dataset.append({"input": row_content, "output": jsonl_row})

            rating_prompt = row_content + "\n\n<instruction>rate the quality of this piece of data on a scale of 1 to 10 where 1 is useless, nonsensical, or otherwise blatantly wrong, and 10 is perfect, verbose, reasoning-heavy, etc   the use case is for training data.</instruction>\n<condition>ONLY REPLY WITH THE INTEGER CORRESPONDING TO YOUR RATING. DO NOT RESPOND WITH ANYTHING BUT THE VALUE</condition>"
            rating_payload = {
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": rating_prompt}],
                "max_tokens": 1,
                "temperature": 0.0,
            }
            rating_response = requests.post(API_URL, json=rating_payload, headers=headers)
            rating_data = rating_response.json()

            if "content" in rating_data:
                rating = int(rating_data["content"][0]["text"].strip())
                ratings.append({"input": row_content, "output": rating})

    return dataset, ratings

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process files or a dataset repository using the Claude API")
parser.add_argument("--folder", type=str, help="Path to the folder containing the files")
parser.add_argument("--repo", type=str, help="Hugging Face dataset repository path")
parser.add_argument("--rate", action="store_true", help="Rate the quality of the data")
args = parser.parse_args()

if args.folder:
    dataset, ratings = process_folder(args.folder)
elif args.repo:
    dataset, ratings = process_repo(args.repo)
else:
    print("Please provide either a folder path or a dataset repository path.")
    exit(1)

# Save the dataset to a JSON file
output_file = "dataset.json"
with open(output_file, "w") as file:
    json.dump(dataset, file, indent=2)

print(f"Dataset saved to {output_file}")

if args.rate:
    # Save the ratings to a JSONL file
    ratings_file = "ratings.jsonl"
    with open(ratings_file, "w") as file:
        for rating in ratings:
            file.write(json.dumps(rating) + "\n")

    print(f"Ratings saved to {ratings_file}")
