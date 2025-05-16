import zipfile
# argparse는 현재 사용하지 않으므로 주석 처리하거나 삭제할 수 있습니다.
# import argparse 
import os
import time
# from multiprocessing import Pool, cpu_count # 멀티프로세싱 대신 스레딩 사용
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading # 스레드 로컬 데이터를 위해 필요할 수 있으나, 여기서는 직접 사용 안 함

# --- 경로 설정 ---
# 사용자가 이 부분을 수정하여 zip 파일 경로와 압축 해제 경로를 지정할 수 있습니다.
DEFAULT_ZIP_FILE_PATH = "/workspace/data/baseline_dataset_v3_lora.zip"  # 여기에 압축 해제할 ZIP 파일 경로를 입력하세요.
DEFAULT_EXTRACT_PATH = "/workspace/data/baseline_dataset_v3_lora/"    # 여기에 압축을 해제할 디렉토리 경로를 입력하세요.
MAX_WORKERS = os.cpu_count() * 3 # 스레드 풀의 최대 작업자 수, I/O 바운드 작업이므로 CPU 코어보다 많게 설정 가능
# -----------------

def extract_member_thread(zip_ref, member_name, extract_path):
    """
    공유 ZipFile 객체에서 단일 멤버를 추출합니다.
    ZipFile 객체는 스레드로부터 안전하게 읽기 작업을 수행할 수 있어야 합니다.
    extractall() 메서드가 스레드로부터 안전한 것처럼 extract()도 일반적으로 안전합니다.
    """
    try:
        zip_ref.extract(member_name, extract_path)
        return f"추출 완료: {member_name}"
    except Exception as e:
        # 오류 발생 시 어떤 파일에서 오류가 났는지 명확히 하기 위해 member_name 포함
        return f"추출 오류 ({member_name}): {e}"

def unzip_threaded(zip_file_path, extract_path):
    """
    멀티스레딩을 사용하여 zip 파일을 병렬로 압축 해제하고 진행 상황을 표시합니다.
    """
    start_time = time.time()

    if not os.path.exists(extract_path):
        try:
            os.makedirs(extract_path, exist_ok=True) # exist_ok=True로 경쟁 상태 방지
            print(f"'{extract_path}' 디렉토리를 생성했습니다.")
        except OSError as e:
            print(f"오류: '{extract_path}' 디렉토리 생성 실패: {e}")
            return

    if not os.path.isfile(zip_file_path):
        print(f"오류: 입력된 파일 '{zip_file_path}'를 찾을 수 없거나 파일이 아닙니다.")
        return

    processed_count = 0
    total_files = 0

    try:
        # ZipFile 객체를 한 번만 열고, 스레드에서 공유합니다.
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            members = zip_ref.namelist() # 또는 zip_ref.infolist() 사용 가능
            total_files = len(members)

            if total_files == 0:
                print("zip 파일 내에 압축 해제할 파일이 없습니다.")
                return

            print(f"압축 해제할 파일 수: {total_files}")
            
            # 실제 사용할 스레드 수 (파일 수보다 많을 필요 없음)
            num_threads = min(MAX_WORKERS, total_files)
            print(f"{num_threads}개의 스레드를 사용하여 압축을 해제합니다.")
            print(f"압축 해제를 시작합니다...")

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(extract_member_thread, zip_ref, member, extract_path) for member in members]
                
                for future in as_completed(futures):
                    result_message = future.result()
                    processed_count += 1
                    print(f"[{processed_count}/{total_files}] {result_message}")
    
    except zipfile.BadZipFile:
        print(f"오류: '{zip_file_path}'는 유효한 zip 파일이 아닙니다.")
        return
    except FileNotFoundError:
        print(f"오류: '{zip_file_path}' 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"압축 해제 중 알 수 없는 오류 발생: {e}")
        # 오류 발생 시에도 지금까지 처리된 파일 수를 출력할 수 있도록 finally 블록 고려 가능
        return
    finally:
        end_time = time.time()
        if total_files > 0: # 파일이 있는 경우에만 시간과 결과 출력
             print(f"총 {processed_count}개의 파일 처리 시도 완료 (전체 {total_files}개 중).")
             print(f"소요 시간: {end_time - start_time:.2f}초")
        elif os.path.exists(zip_file_path): # 파일은 존재하나 내용이 없는 경우
            print("압축 해제 작업이 완료되었으나 처리할 파일이 없었습니다.")


if __name__ == "__main__":
    # 스크립트 상단에 정의된 기본 경로 사용
    zip_to_extract = DEFAULT_ZIP_FILE_PATH
    destination_path = DEFAULT_EXTRACT_PATH

    print(f"대상 ZIP 파일: {zip_to_extract}")
    print(f"압축 해제 경로: {destination_path}")
    
    # 사용자가 경로를 설정했는지 간단히 확인 (더 강력한 검증 추가 가능)
    if zip_to_extract == "your_archive.zip":
        print("주의: 스크립트 상단의 'DEFAULT_ZIP_FILE_PATH'를 실제 ZIP 파일 경로로 수정해주세요.")
        # 예시로 실행하지 않도록 여기서 중단하거나, 사용자 입력을 받도록 수정할 수 있습니다.
    elif not os.path.isfile(zip_to_extract):
         print(f"주의: 지정된 ZIP 파일 '{zip_to_extract}'를 찾을 수 없습니다. 경로를 확인해주세요.")
    else:
        unzip_threaded(zip_to_extract, destination_path)
