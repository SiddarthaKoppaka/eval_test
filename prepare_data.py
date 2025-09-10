import csv
import json
from main import generate_structured_story

def create_qa_dataset():
    story_arcs = [
        "A lonely lighthouse keeper discovers a mysterious glowing pearl that washes ashore.",
        "A young witch accidentally turns her grumpy cat into a talking, flying broomstick.",
        "An astronaut gets stranded on a planet where the plants communicate through music.",
        "A detective in a steampunk city must solve the theft of a priceless clockwork heart.",
        "Two rival chefs must team up to win a magical cooking competition.",
        "A timid librarian finds a book that allows him to enter into the stories he reads.",
        "A group of kids builds a spaceship out of junk and actually travels to the moon.",
        "A knight who is afraid of dragons is tasked with saving a princess from one.",
        "An ancient robot wakes up in a post-apocalyptic world and tries to find its purpose.",
        "A musician discovers her guitar can control the weather when she plays certain chords."
    ]

    qa_data = []

    print("Starting dataset generation...")
    for arc in story_arcs:
        try:
            story_json = generate_structured_story(arc)
            qa_pair = {
                "question": arc,
                "answer": json.dumps(story_json)
            }
            qa_data.append(qa_pair)
        except Exception as e:
            print(f"Could not process story arc '{arc}'. Error: {e}")

    output_file = 'qa_dataset.csv'
    if qa_data:
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['question', 'answer'])
                writer.writeheader()
                writer.writerows(qa_data)
            print(f"\nSuccessfully generated {len(qa_data)} QA pairs and saved them to '{output_file}'")
        except IOError as e:
            print(f"\nError writing to file '{output_file}'. Error: {e}")
    else:
        print("\nNo data was generated, so the CSV file was not created.")

if __name__ == '__main__':
    create_qa_dataset()
