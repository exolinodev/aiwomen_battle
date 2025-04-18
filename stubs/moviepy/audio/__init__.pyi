from typing import Any, Protocol

class AudioClip(Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def to_audiofile(self, filename: str, fps: int = 44100, nbytes: int = 2, buffersize: int = 2000, codec: str = None, bitrate: str = None, ffmpeg_params: list = None, write_logfile: bool = False, verbose: bool = True, threads: int = None, ffmpeg_executable: str = None) -> None: ...

class AudioFileClip(AudioClip):
    def __init__(self, filename: str, fps: int = 44100, nbytes: int = 2, buffersize: int = 2000, codec: str = None, bitrate: str = None, ffmpeg_params: list = None, write_logfile: bool = False, verbose: bool = True, threads: int = None, ffmpeg_executable: str = None) -> None: ... 