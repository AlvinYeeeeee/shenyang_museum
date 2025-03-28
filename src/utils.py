import os
import pathlib


def is_path_under_directory(file_path, directory):
    """
    Check if file_path is under the specified directory.
    
    Args:
    file_path (str): The path to check
    directory (str): The directory to compare against
    
    Returns:
    bool: True if file_path is under directory, False otherwise
    """
    # Normalize and resolve both paths to absolute paths
    # This resolves symlinks, '..' and '.' components
    try:
        normalized_file_path = os.path.realpath(os.path.abspath(file_path))
        normalized_directory = os.path.realpath(os.path.abspath(directory))
        
        # Ensure the paths end with a separator to prevent partial matches
        normalized_directory = os.path.normpath(normalized_directory + os.sep)
        
        # Check if the normalized file path starts with the normalized directory path
        return os.path.commonpath([normalized_file_path, normalized_directory]) == normalized_directory
    
    except Exception as e:
        print(f"Error checking path: {e}")
        return False


def check_request_audio_file_path(audio_file_path: str, response_dir) -> bool:
    if audio_file_path is None or \
        audio_file_path == "" or \
            is_path_under_directory(audio_file_path, response_dir) is False:
        return False
    return True
