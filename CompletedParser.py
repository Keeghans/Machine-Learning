import string

def parse_text(text, delimiter="\n\n"):
    """
    Splits text into a list of 'pages' based on a specified delimiter.
    Each page is further split into words, stripped of punctuation, and lowercased.
    """
    pages = text.split(delimiter)
    word_jagged_array = []
    for page in pages:
        # Replace non-ASCII characters with their closest ASCII equivalents
        page = page.replace('\u2019', "'").replace('\u2018', "'").replace('\u201c', '"').replace('\u201d', '"')
        # Additional step: Remove any characters outside the ASCII range
        page = ''.join([char for char in page if ord(char) < 128])
        # Remove punctuation and convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        cleaned_page = page.translate(translator).lower()
        words = cleaned_page.split()
        word_jagged_array.append(words)
    return word_jagged_array

def display_chunks(chunks): 
    """Displays the processed text to the user in a chunked format."""
    print("\nChopped-up Text:")
    for page_number, page_words in enumerate(chunks, start=1):
        print(f"Page {page_number}: {' '.join(page_words)}\n")

def get_user_input():
    """Handles user input for direct text entry or file upload."""
    while True:
        choice = input("Type 'text' to enter text directly or 'file' to upload a file: ").strip().lower()
        if choice == 'text':
            return input("Enter your text here: ")
        elif choice == 'file':
            file_path = input("Enter the full path to your file: ")
            file_path = file_path.replace('"', '')  # Remove double quotes from the file path
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    return file.read()
            except FileNotFoundError:
                print("Error: File not found. Please check the path and try again.")
        else:
            print("Invalid option selected. Please try again.")
def main():
    text = get_user_input()
    word_jagged_array = parse_text(text)
    if word_jagged_array:
        display_choice = input("Do you want to display the parsed text? (yes/no): ")
        if display_choice.lower() == 'yes':
            display_chunks(word_jagged_array)
    else:
        print("No content found. Please ensure the text is not empty and try again.")
    # Return the parsed text
    return word_jagged_array

    # Wait for user action to exit
    input("Press the spacebar then enter to exit...")

if __name__ == "__main__":
    main()