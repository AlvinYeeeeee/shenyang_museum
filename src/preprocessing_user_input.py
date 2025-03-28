

import re
from typing import Union


def extract_int_from_string(input_string: str)->Union[int, None]:
    """
    Extract the integer from the input string
    """
    out = re.search(r'\d+', input_string)
    # Valid user input
    if out is None or len(out.group()) == 0:
        return None
    num = out.group(0)
    return int(num)


def preprocessing_user_input(user_input: str)->str:
    """
    Preprocess the user input to remove any special characters
    """
    user_input = user_input.strip()
    
    return user_input


def test_extract_int_from_string():
    assert extract_int_from_string("123abc123") == 123
    assert extract_int_from_string("abc123") == 123
    assert extract_int_from_string("abc") == None
    assert extract_int_from_string("123") == 123
    assert extract_int_from_string("abc00123abc") == 123
    assert extract_int_from_string("abc00123abc中文") == 123


if __name__ == "__main__":
    test_extract_int_from_string()
    print("All tests passed!")