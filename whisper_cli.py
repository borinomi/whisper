import whisper
import torch
from whisper.utils import get_writer
import os
import sys
import re

def merge_short_subtitles(srt_path, min_duration=1.0):
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_text = f.read()

    pattern = re.compile(
        r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)',
        re.DOTALL
    )

    def to_seconds(t):
        h, m, s = t.replace(',', '.').split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)

    def to_timestamp(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}".replace('.', ',')

    subs = []
    for match in pattern.finditer(srt_text):
        subs.append({
            'start': match.group(2),
            'end': match.group(3),
            'text': match.group(4).strip()
        })

    merged = []
    for sub in subs:
        duration = to_seconds(sub['end']) - to_seconds(sub['start'])
        if duration < min_duration and merged:
            merged[-1]['end'] = sub['end']
            merged[-1]['text'] += ' ' + sub['text']
        else:
            merged.append(sub.copy())

    result = ''
    for i, sub in enumerate(merged, 1):
        result += f"{i}\n{sub['start']} --> {sub['end']}\n{sub['text']}\n\n"

    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(result)

    return len(subs) - len(merged)

def get_user_input(prompt, default_val=None):
    if default_val:
        user_in = input(f"{prompt} [{default_val}]: ").strip()
        return user_in if user_in else default_val
    return input(f"{prompt}: ").strip()


def run_whisper_cli():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        print("=" * 50)
        print("Whisper CLI")
        print("=" * 50)
        filepath = input("파일 경로: ").strip().strip('"')

    if not filepath or not os.path.exists(filepath):
        print(f"오류: 파일 없음 - {filepath}")
        input("엔터를 누르면 종료...")
        return

    print(f"\n파일: {os.path.basename(filepath)}")

    print("\n[언어 선택] 엔터 = 자동 감지")
    lang_input = get_user_input("언어 코드 (ko, en, ja, auto)", "auto")
    source_lang = None if lang_input == "auto" else lang_input

    print("\n[자막 옵션]")
    print("1. 단어 수 제한")
    print("2. 글자 수 제한")
    limit_mode = get_user_input("선택 (1/2)", "1")

    writer_options = {
        "max_line_count": 1,
        "highlight_words": False,
        "max_line_width": None,
        "max_words_per_line": None
    }

    if limit_mode == "2":
        writer_options["max_line_width"] = int(get_user_input("줄당 최대 글자 수", "42"))
    else:
        writer_options["max_words_per_line"] = int(get_user_input("줄당 최대 단어 수", "8"))

    writer_options["max_line_count"] = int(get_user_input("자막당 최대 줄 수", "1"))

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n모델 로딩 중... (Device: {device})")

    model = whisper.load_model("turbo", device=device)

    print("분석 중...")

    transcribe_options = {
        "verbose": True,
        "condition_on_previous_text": False,
        "word_timestamps": True
    }

    if source_lang:
        transcribe_options["language"] = source_lang

    result = model.transcribe(filepath, **transcribe_options)
    print(f"\n감지된 언어: {result['language']}")

    print("저장 중...")
    folder_path = os.path.dirname(os.path.abspath(filepath))
    filename_base = os.path.splitext(os.path.basename(filepath))[0]

    output_writer = get_writer("srt", folder_path)

    try:
        output_writer(result, filepath, **writer_options)
    except TypeError:
        output_writer(result, filepath)

    srt_path = os.path.join(folder_path, f"{filename_base}.srt")
    merged_count = merge_short_subtitles(srt_path)
    print(f"1초 미만 자막 {merged_count}개 병합 완료")

    print(f"\n완료: {filename_base}.srt")
    input("엔터를 누르면 종료...")

if __name__ == "__main__":
    try:
        run_whisper_cli()
    except KeyboardInterrupt:
        print("\n취소됨")
    except Exception as e:
        print(f"\n오류: {e}")
        input("엔터를 누르면 종료...")
