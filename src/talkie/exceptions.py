class TalkieError(Exception):
    pass

class ASRError(TalkieError):
    pass

class TTSError(TalkieError):
    pass

class LLMError(TalkieError):
    pass
