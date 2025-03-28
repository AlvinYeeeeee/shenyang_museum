
import re
import torch

class Explanation:
    title: str
    serial_number: int
    text: str
    audio_path: str
    audio_duration: float
    audio_tensor: torch.Tensor

    def __init__(self,
                 title,
                 serial_number,
                 text,
                 audio_path="",
                 audio_duration=0.0):
        self.title = title
        self.serial_number = int(serial_number)
        self.text = text
        self.audio_path = audio_path
        self.audio_duration = audio_duration

    def __str__(self):
        return f"Title: {self.title}\nSerial Number: {self.serial_number}\nText: {self.text}\nAudio Path: {self.audio_path}\nAudio Duration: {self.audio_duration}"


def extract_numbers(text):
    """
    Extract digital numbers from a string and return both the cleaned string and the extracted numbers.
    
    Args:
        text (str): Input string containing numbers to extract
        
    Returns:
        tuple: (cleaned_string, extracted_numbers)
    """
    # Find all consecutive digits
    numbers = re.findall(r'\d+', text)

    # Join multiple number sequences if found
    extracted_numbers = ''.join(numbers)

    # Remove all digits from the original string
    cleaned_string = re.sub(r'\d+', '', text)

    return (cleaned_string, extracted_numbers)


def process_one_line(line: str) -> Explanation:
    title, text = line.split("\t")
    cleaned_title, serial_number = extract_numbers(title)

    return Explanation(title=cleaned_title,
                       serial_number=serial_number,
                       text=text,
                       audio_path="",
                       audio_duration=0.0)


def read_file_line_by_line(file_path) -> list[Explanation]:
    list_of_explanations = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) < 2:
                continue
            list_of_explanations.append(process_one_line(line))
    return list_of_explanations
