import jsonlines
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe(file_path: str) -> str:
    segments, _ = _model.transcribe(file_path)
    return " ".join(segment.text for segment in segments)


def transcribe_all(video_dir: str = "video", output_file: str = "transcriptions.jsonl"):
    with jsonlines.open(output_file, mode="w") as writer:
        for path in Path(video_dir).iterdir():
            if path.is_file():
                writer.write({"file": path.name, "text": transcribe(str(path))})
                writer.file.flush()


if __name__ == "__main__":
    _model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    transcribe_all("video", "documents.jsonl")
