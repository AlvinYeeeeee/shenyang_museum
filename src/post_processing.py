

DEFAULT_USER_PROMPT = '请输入讲解词的编号，例如"12"；或者直接输入想了解的问题，例如"沈阳旧石器时代的遗址"'

def text_post_processing(response: str) -> str:
    if response.find("I don't know") != -1:
        return DEFAULT_USER_PROMPT
    else:
        return response