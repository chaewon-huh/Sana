import argparse
import json
import os
from pathlib import Path

def create_metadata_jsonl(image_dir_path: str):
    """
    지정된 디렉토리에서 이미지 파일과 해당 텍스트 파일을 찾아
    metadata.jsonl 파일을 생성합니다.

    Args:
        image_dir_path (str): 이미지와 텍스트 파일이 있는 디렉토리 경로.
    """
    image_dir = Path(image_dir_path)
    if not image_dir.is_dir():
        print(f"오류: {image_dir_path}는 유효한 디렉토리가 아닙니다.")
        return

    metadata_list = []
    supported_image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']

    print(f"'{image_dir}' 디렉토리에서 파일 검색 중...")

    for item in image_dir.iterdir():
        if item.is_file() and item.suffix.lower() in supported_image_extensions:
            image_file_name = item.name
            text_file_name = item.stem + ".txt"
            text_file_path = image_dir / text_file_name

            if text_file_path.is_file():
                try:
                    with open(text_file_path, 'r', encoding='utf-8') as f:
                        prompt_text = f.read().strip()
                    
                    if prompt_text: # 프롬프트 내용이 비어있지 않은 경우에만 추가
                        metadata_list.append({
                            "file_name": image_file_name,
                            "text": prompt_text
                        })
                        print(f"  처리됨: 이미지='{image_file_name}', 텍스트='{text_file_name}'")
                    else:
                        print(f"  경고: 텍스트 파일 '{text_file_name}'이 비어있습니다. 건너<0xEB><0x9B><0x84>니다.")
                except Exception as e:
                    print(f"  오류: 텍스트 파일 '{text_file_name}' 처리 중 오류 발생: {e}")
            else:
                print(f"  경고: 이미지 '{image_file_name}'에 해당하는 텍스트 파일 '{text_file_name}'을 찾을 수 없습니다.")

    if metadata_list:
        output_file_path = image_dir / "metadata.jsonl"
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for entry in metadata_list:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')
            print(f"'{output_file_path}' 파일이 성공적으로 생성되었습니다.")
            print(f"총 {len(metadata_list)}개의 항목이 기록되었습니다.")
        except Exception as e:
            print(f"오류: '{output_file_path}' 파일 작성 중 오류 발생: {e}")
    else:
        print("처리할 유효한 이미지-텍스트 쌍을 찾지 못했습니다. 'metadata.jsonl' 파일이 생성되지 않았습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이미지 디렉토리에서 metadata.jsonl 파일을 생성합니다.")
    parser.add_argument(
        "image_directory",
        type=str,
        help="이미지 파일과 프롬프트 텍스트 파일이 포함된 디렉토리 경로입니다."
    )
    args = parser.parse_args()

    create_metadata_jsonl(args.image_directory)
