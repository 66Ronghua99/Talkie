class BaseVAD:
    pass

class SileroVAD(BaseVAD):
    pass

class VADResult:
    pass

async def create_vad(*args, **kwargs):
    return SileroVAD()
