import nltk
import time
import spacy
import string
import torch
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.collections import Trie
from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer

# Download the word list from nltk if not already available
nltk.download('words')
nltk.download('punkt')  # For tokenization

# Load the spaCy language model
nlp = spacy.load("en_core_web_lg")

# Load a pre-trained language model and tokenizer
model_name = "distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Implementing the Levenshtein Distance Function with dynamic programming
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# Load a comprehensive dictionary
dictionary_words = set(words.words())
trie_dict = Trie()
for word in dictionary_words:
    trie_dict.insert(word)
def spell_check(word, sentence):
    """
    Spell check a word within the context of its sentence.
    """
    # Process the sentence with spaCy to get POS tags
    doc = nlp(sentence)

    # Find the POS of the target word
    target_pos = None
    for token in doc:
        if token.text.lower() == word.lower():
            target_pos = token.pos_
            break

    if word in dictionary_words:
        return word  # The word is correctly spelled

    # Generate suggestions based on Levenshtein distance and context
    suggestions = []
    for w in dictionary_words:
        if levenshtein_distance(word, w) <= 2:
            suggestions.append(w)

    # Prioritize suggestions based on matching POS tags and context
    if target_pos:
        suggestions = prioritize_corrections_based_on_pos(suggestions, target_pos)

    # Handle word forms
    suggestions = handle_word_forms(word, suggestions)

    # Preserve sentence structure by maintaining punctuation and capitalization
    original_tokens = nltk.word_tokenize(sentence)
    suggestions = preserve_sentence_structure(word, suggestions, original_tokens)

    # Return the best suggestion or the original word if no suggestions are found
    return min(suggestions, key=lambda w: levenshtein_distance(word, w), default=word)

def select_best_correction(original_word, suggestions, sentence):
    """
    Select the best correction for a word based on context using a language model.
    """
    if not suggestions:
        return original_word

    # Replace the original word with a mask token
    masked_sentence = sentence.replace(original_word, tokenizer.mask_token, 1)

    # Tokenize the masked sentence
    tokenized_sentence = tokenizer(masked_sentence, return_tensors='pt', add_special_tokens=True)

    # Get the input IDs and attention mask
    input_ids = tokenized_sentence["input_ids"]
    attention_mask = tokenized_sentence["attention_mask"]

    # Find the index of the mask token in the input IDs
    mask_token_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)
    if mask_token_indices.numel() == 0:
        # Mask token not found, skip prediction for this sentence
        return original_word

    for mask_token_index in mask_token_indices:
        mask_token_index = mask_token_index[0].item()  # Get the first item of the tensor

        # Get predictions from the language model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get the logits for the mask token
        mask_token_logits = logits[0, mask_token_index]

        # Get the top-k predictions for the mask token
        top_k = 5
        top_k_tokens = torch.topk(mask_token_logits, k=top_k, dim=-1).indices.tolist()

        # Convert the tokens back to words
        top_k_words = tokenizer.convert_ids_to_tokens(top_k_tokens)

        # Filter out the [CLS], '[SEP]', and '[PAD]' tokens
        filtered_words = [word for word in top_k_words if word not in ['[CLS]', '[SEP]', '[PAD]']]

        # Update the best suggestion if a suitable suggestion is found
        for suggestion in filtered_words:
            if len(suggestion) > 1:
                return suggestion

    return original_word

def prioritize_corrections_based_on_pos(suggestions, target_pos):
    """
    Re-rank suggestions based on their match with the target POS tag.
    """
    ranked_suggestions = []
    for suggestion in suggestions:
        doc = nlp(suggestion)
        # Assuming each suggestion is a single word for simplicity
        if doc[0].pos_ == target_pos:
            ranked_suggestions.insert(0, suggestion)  # Prioritize this suggestion
        else:
            ranked_suggestions.append(suggestion)

    return ranked_suggestions

def handle_word_forms(word, suggestions):
    """
    Handle word forms (e.g., singular/plural, tense) in suggestions.
    """
    # Add rules or heuristics to handle word forms
    # For example, convert all suggestions and the target word to lowercase for comparison
    word_lower = word.lower()
    suggestions_lower = [s.lower() for s in suggestions]
    return suggestions[0] if suggestions else word

def preserve_sentence_structure(word, suggestions):
    """
    Preserve sentence structure by maintaining punctuation and capitalization.
    """
    preserved_suggestions = []
    for suggestion in suggestions:
        # Check if the original word is capitalized
        if word[0].isupper():
            # Capitalize the suggestion
            suggestion = suggestion.capitalize()
        else:
            # Ensure the suggestion is lowercase
            suggestion = suggestion.lower()

        # Check if the original word has punctuation at the end
        if word[-1] in string.punctuation:
            # Append the punctuation to the suggestion
            suggestion += word[-1]

        return suggestion

# Function to tokenize user input into words
def tokenize_user_input(text):
    """Tokenizes user input into words."""
    return word_tokenize(text)

# Function to get user input
def get_user_input():
    """Handles user input for direct text entry or file upload."""
    choice = input("Type 'text' to enter text directly or 'file' to upload a file: ").strip().lower()
    if choice == 'text':
        return input("Enter your text here: ")
    elif choice == 'file':
        file_path = input("Enter the full path to your file: ").strip().replace('"', '')  # Remove double quotes
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                return file.read()
        except FileNotFoundError:
            print("Error: File not found. Please check the path and try again.")
            return None

def process_sentences(sentences):
    corrections = {}
    for sentence in sentences:
        doc = nlp(sentence)
        for token in doc:
            word = token.text  # Use the original token without converting to lowercase
            if not word.isalpha():  # Skip non-alphabetic tokens
                continue
            corrected_word = spell_check(word, sentence)
            if corrected_word != word:
                corrections[word] = corrected_word  # Store corrections
    return corrections

def main():
    print("Spell checker program is starting...")

    user_input = get_user_input()
    if user_input is None:
        return  # Exit if no input was provided

    print("Processing input...")

    # Tokenize input into sentences
    sentences = nltk.sent_tokenize(user_input)
    total_sentences = len(sentences)

    corrections = {}  # Dictionary to hold word corrections
    processed_words = set()  # Keep track of words that have been processed

    for i, sentence in enumerate(sentences, start=1):
        start_time = time.time()  # Start timing
        print(f"Processing sentence {i} of {total_sentences}...")

        doc = nlp(sentence)
        for token in doc:
            word = token.text.lower()
            if word not in processed_words and word.isalpha():  # Skip non-alphabetic tokens and already processed words
                corrected_word = spell_check(word, sentence)
                if corrected_word != word:
                    corrections[word] = corrected_word  # Store corrections
                processed_words.add(word)

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        print(f"Processing sentence {i} took {elapsed_time:.2f} seconds.")

    corrected_text = ""
    for sentence in sentences:
        corrected_sentence = " ".join([corrections.get(token.lower(), token) for token in nltk.word_tokenize(sentence)])
        corrected_text += corrected_sentence + " "

    corrected_text = corrected_text.strip()  # Remove trailing spaces

    # Write corrections to file
    with open('SpellCheckCorrections.txt', 'w', encoding='utf-8') as output_file:
        output_file.write("Original Text:\n")
        output_file.write(user_input + "\n\n")
        output_file.write("Corrected Text:\n")
        output_file.write(corrected_text + "\n\n")
        output_file.write("List of Corrections:\n")
        for misspelled, correction in corrections.items():
            output_file.write(f"{misspelled} -> {correction}\n")

    print("\nCorrected Text:")
    print(corrected_text)
    print("\nList of Corrections:")
    for misspelled, correction in corrections.items():
        print(f"{misspelled} -> {correction}")

    print("Spell checker program has finished.")

if __name__ == "__main__":
    main()