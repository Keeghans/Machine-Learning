import nltk
from nltk.tag import hmm
from nltk.corpus import treebank
from collections import Counter
import math
from collections import defaultdict
from transformers import MarianMTModel, MarianTokenizer
from CompletedParser import parse_text
from CompletedParser import main as get_parsed_text


def display_menu():
    print("\nSelect the type of NLP analysis:")
    print("1. Rule-based Systems")
    print("2. Statistical Models")
    print("3. Hidden Markov Models")
    print("4. Bayesian Methods")
    print("5. Machine Translation Techniques")
    print("6. Information Retrieval Techniques")
    print("7. Exit")
    return input("Enter your choice (1-7): ")

def analyze_rule_based_systems(chunks):
    print("Analyzing with Rule-based Systems approach...")
    for chunk in chunks:
        if isinstance(chunk, list):
            for sentence in chunk:
                if "hello" in sentence.lower():
                    print(f"Found 'hello' in: {sentence}")
        else:
            if "hello" in chunk.lower():
                print(f"Found 'hello' in: {chunk}")

def analyze_statistical_models(chunks):
    print("Analyzing with Statistical Models approach...")
    word_freq = {}
    for chunk in chunks:
        if isinstance(chunk, list):
            for sentence in chunk:
                words = sentence.split()
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
        else:
            words = chunk.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

    for word, freq in word_freq.items():
        print(f"'{word}': {freq}")
                
def analyze_hidden_markov_models(chunks):
    # Download necessary NLTK datasets
    nltk.download('treebank')
    nltk.download('universal_tagset')
    
    # Use a subset of the Penn Treebank corpus for training
    train_data = treebank.tagged_sents(tagset='universal')[:3000]
    
    # Initialize and train an HMM model
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(train_data)
    
    print("HMM trained on the Penn Treebank corpus.")
    print("Here's an analysis of the text chunks using the trained HMM for POS tagging:")
    
# Analyze each text chunk
    for chunk in chunks:
        if isinstance(chunk, list):
            for sentence in chunk:
                tokens = nltk.word_tokenize(sentence)
                tagged = tagger.tag(tokens)
                print(f"Text chunk: {sentence}")
                print(f"POS Tags: {tagged}\n")
        else:
            tokens = nltk.word_tokenize(chunk)
            tagged = tagger.tag(tokens)
            print(f"Text chunk: {chunk}")
            print(f"POS Tags: {tagged}\n")

class BayesianFilter:
    def __init__(self):
        self.num_messages = {'spam': 0, 'ham': 0}
        self.log_class_priors = {'spam': 0, 'ham': 0}
        self.word_counts = {'spam': defaultdict(int), 'ham': defaultdict(int)}
        self.vocab = set()
    
    def train(self, data):
        for text, label in data:
            counts = Counter(parse_text(text, ' ')[0])  # Use space as the delimiter
            for word in counts:
                if word not in self.vocab:
                    self.vocab.add(word)
                self.word_counts[label][word] += counts[word]
            self.num_messages[label] += 1
        
        total_messages = sum(self.num_messages.values())
        self.log_class_priors['spam'] = math.log(self.num_messages['spam'] / total_messages)
        self.log_class_priors['ham'] = math.log(self.num_messages['ham'] / total_messages)
    
    def predict(self, text):
        message_words = set(parse_text(text, ' ')[0])  # Use space as the delimiter
        spam_log_prob = self.log_class_priors['spam']
        ham_log_prob = self.log_class_priors['ham']
        for word in message_words:
            if word in self.vocab:
                # Laplace smoothing
                spam_log_prob += math.log((self.word_counts['spam'].get(word, 0) + 1) / 
                                     (self.num_messages['spam'] + len(self.vocab)))
                ham_log_prob += math.log((self.word_counts['ham'].get(word, 0) + 1) / 
                                    (self.num_messages['ham'] + len(self.vocab)))

        return 'spam' if spam_log_prob > ham_log_prob else 'ham'

def analyze_bayesian_methods(chunks):
    bf = BayesianFilter()
    training_data = [
        ("Act now and get a discount!", "spam"),
        ("Reminder: Your dentist appointment is tomorrow", "ham"),
        ("Get a free trial today", "spam"),
        ("Don't miss out on this opportunity", "spam"),
        ("Meeting rescheduled to next week", "ham"),
        ("Limited time offer: Buy one, get one free", "spam"),
        ("Please review and approve the document", "ham"),
        ("Congratulations! You've been selected as a winner", "spam"),
        ("Your package has been delivered", "ham"),
        ("Claim your prize now", "spam"),
        ("Don't forget to RSVP for the party", "ham"),
        ("Get rich quick scheme", "spam"),
        ("Meeting agenda for next week", "ham"),
        ("Exclusive offer for loyal customers", "spam"),
        ("Reminder: Pay your utility bill by Friday", "ham"),
        ("Special promotion: Buy two, get one free", "spam"),
        ("Vacation rental available for booking", "ham"),
        ("Limited time offer: 50% off on all items", "spam"),
        ("Review the attached report", "ham"),
        ("Congratulations! You've won a trip", "spam"),
        ("Meeting minutes from last week", "ham"),
        ("Act fast to secure your spot", "spam"),
        ("Reminder: Parent-teacher conference tomorrow", "ham"),
        ("Don't miss this amazing deal", "spam"),
        ("Lunch menu for the week", "ham"),
        ("Exclusive deal: 24-hour sale", "spam"),
        ("Update on project status", "ham"),
        ("Get a quote for car insurance", "spam"),
        ("Reminder: Submit your expenses by end of day", "ham"),
        ("Limited time offer: Free shipping on all orders", "spam"),
        ("Confirm your attendance for the event", "ham"),
    ]
    bf.train(training_data)

    for chunk in chunks:
        # Ensure chunk is a string
        if isinstance(chunk, list):
            chunk = ' '.join(chunk)

        prediction = bf.predict(chunk)
        print(f"Text: {chunk}")
        print(f"Classification: {prediction}\n")

class MachineTranslationAnalyzer:
    def __init__(self):
        # Load the tokenizer and model for English to French translation
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

    def translate_chunks(self, chunks):
        translated_chunks = []
        for chunk in chunks:
            sentence = ' '.join(chunk)
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True)
            translated_tokens = self.model.generate(**inputs)
            translated_sentence = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            translated_chunk = translated_sentence.split()
            translated_chunks.append(translated_chunk)
        return translated_chunks

    def analyze_machine_translation_techniques(self, chunks):
        print("Analyzing using Machine Translation Techniques...")
        translated_chunks = self.translate_chunks(chunks)
        for original, translation in zip(chunks, translated_chunks):
            print("Original:", ' '.join(original))
            print("Translated:", ' '.join(translation))
            print()



def compute_tf(word, token_list):
    """Calculate term frequency in a document"""
    return token_list.count(word) / len(token_list)

def compute_idf(word, docs_containing_word, num_docs):
    """Calculate inverse document frequency across documents"""
    return math.log(num_docs / (1 + docs_containing_word[word]))

def compute_tf_idf_for_doc(token_list, idf_scores):
    """Compute TF-IDF scores for all unique words in a document"""
    words = set(token_list)
    tf_idf_scores = {word: compute_tf(word, token_list) * idf_scores[word] for word in words}
    return tf_idf_scores

def analyze_information_retrieval_techniques(chunks):
    print("Analyzing using Information Retrieval Techniques (TF-IDF)...")

    # Assuming chunks are already parsed and tokenized as per previous discussions
    list_of_token_lists = chunks
    
    # Precompute IDF scores for all words
    num_docs = len(list_of_token_lists)
    docs_containing_word = {}
    for token_list in list_of_token_lists:
        for word in set(token_list):
            docs_containing_word[word] = docs_containing_word.get(word, 0) + 1
            
    idf_scores = {word: compute_idf(word, docs_containing_word, num_docs) for word in docs_containing_word}

    # Calculate and display TF-IDF scores for each document
    for i, token_list in enumerate(list_of_token_lists):
        tf_idf_scores = compute_tf_idf_for_doc(token_list, idf_scores)
        print(f"\nchunks {i+1}: {' '.join(token_list)}")
        for word, score in tf_idf_scores.items():
            print(f"Word: '{word}', TF-IDF Score: {score:.5f}")
            
def main():
    chunks = get_parsed_text()
    while True:
        choice = display_menu()
        if choice == '1':
            analyze_rule_based_systems(chunks)
        elif choice == '2':
            analyze_statistical_models(chunks)
        elif choice == '3':
            analyze_hidden_markov_models(chunks)
        elif choice == '4':
            analyze_bayesian_methods(chunks)
        elif choice == '5':
            analyzer = MachineTranslationAnalyzer()
            analyzer.analyze_machine_translation_techniques(chunks)
        elif choice == '6':
            analyze_information_retrieval_techniques(chunks)
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")
        input("\nPress the spacebar then enter to continue...")

if __name__ == "__main__":
    main()