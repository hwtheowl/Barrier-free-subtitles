# 필요 라이브러리 불러오기
import torch
import whisper
import pandas as pd

# 자막생성 함수
def Create_voice_sub(video, use_lang=None, use_model="small"):    
    try:
        device = "cuda"
        torch.cuda.get_device_properties(0)
    except:
        device = "cpu"
    
    # STT AI 시작
    model = whisper.load_model(use_model, device=device)
    result = model.transcribe(verbose=True, word_timestamps=False, language=use_lang, audio=video)
    voice_df = pd.DataFrame(result["segments"])[["start", "end", "text"]]
    
    # 결과 시간변환
    voice_df["start_time"] = convert_to_srt_time_format(voice_df["start"])
    voice_df["end_time"] = convert_to_srt_time_format(voice_df["end"])
    
    return voice_df

# 시간변환 함수
def convert_to_srt_time_format(timestamps):
    return pd.to_datetime(timestamps, unit='s', origin='unix').dt.strftime('%H:%M:%S,%f').apply(lambda x: x[:-3])

def Create_Barrier_free_sub(video, voice_df, back_df):
    # 데이터프레임 결합
    sub_df = pd.concat([voice_df, back_df])
    sub_df = sub_df.sort_values(by='start_time')
    sub_df = sub_df.reset_index(drop=True)
    sub_df['id'] = sub_df.index
    sub_df['id'] += 1
    
    # srt 자막 변환 및 추출
    sub_df["sub"] = sub_df.index.astype(str) + "\n" + sub_df["start_time"] + " --> " + sub_df["end_time"] + "\n" + sub_df["text"] + "\n"
    sub = sub_df['sub']
    
    # 큰따옴표 제거하기 전에 파일 저장
    srt_filename = f'{video.replace(".mp4", "")}.srt'
    sub.to_csv(srt_filename, header=None, mode='w', index=False, encoding='utf-8')

    # 큰따옴표 제거
    with open(srt_filename, 'r', encoding='utf-8') as file:
        filedata = file.read()

    filedata = filedata.replace('"', '')

    # 큰따옴표가 제거된 내용으로 다시 파일 작성
    with open(srt_filename, 'w', encoding='utf-8') as file:
        file.write(filedata)

# 영상파일명 정의, 사용 모델 및 언어 설정 *언어설정 생략 가능
video = "test.mp4"
use_lang = "ko"

#--- 비음성 자막 데이터(임시) ---#
test_text = {'text':'[테스트 메세지입니다.]', 'start':'1.5', 'end':'5.5'}
back_df = pd.DataFrame([test_text])
back_df["start_time"] = convert_to_srt_time_format(back_df["start"])
back_df["end_time"] = convert_to_srt_time_format(back_df["end"])
#--- 비음성 자막 데이터(임시) ---#



voice_df = Create_voice_sub(video=video, use_lang=use_lang)
Create_Barrier_free_sub(video=video, voice_df=voice_df, back_df=back_df)