# 도다리 Dodari
[NHNDQ](https://huggingface.co/NHNDQ/nllb-finetuned-en2ko) AI 한영/영한 번역기를 일반 사람들도 쉽게 쓸수 있게 만든 로컬 웹서비스 입니다. (based on Gradio)

<img src='https://github.com/vEduardovich/dodari/blob/main/dodari.png' style='display:block;border-radius:10px;text-align:center;' title='도다리'/>

<br/>

## 특징
자신의 컴퓨터에서 제한없이 `한영`-`영한` AI 번역이 가능합니다. 
- 일반 기계번역에 비해 품질이 훨씬 우수합니다.
- 여러 txt파일들을 한번에 번역해줍니다. epub과 pdf번역은 추후 버전에서 가능합니다.
- 번역문(원문).txt 파일과 번역문.txt 파일, 이렇게 두가지 파일로 출력됩니다. 번역이 이상할 경우 원문과 바로 비교할수 있습니다.
- 사용이 아주 쉽습니다. 번역이 필요한 파일들을 드래그한 후 '번역하기' 버튼만 클릭하면 됩니다. 알아서 한<=>영 으로 번역해 줍니다.
- 번역 성능이 뛰어난 모델로 최신 업데이트가 가능합니다 - 현재는 가성비가 가장 좋은 NHNDQ만 사용합니다.
<img src='https://github.com/vEduardovich/dodari/blob/main/dodari_src.jpg' style='display:block;border-radius:10px;text-align:center;' title='도다리 실행화면'/>

<br/>

## 설치 및 실행
초보자라면,

1. <a href='https://github.com/vEduardovich/dodari/archive/refs/heads/main.zip' title='압축 파일 다운로드' style='text-align:center'>압축 파일 다운로드</a> 클릭
2. 압축해제 후 
> - 윈도우 사용자는 start_windows.bat 더블 클릭
> - 맥이나 우분투 사용자는 커맨드 창에서 sh start_mac.sh 실행
3. 처음 실행이라면 프로그램을 자동으로 설치한 후 실행합니다. 이미 설치가 되었다면 바로 실행합니다.

<br/>

고급 사용자라면,
1. git clone https://github.com/vEduardovich/dodari.git
2. cd dodari
3. 실행하기
> - 윈도우는 start_windows.bat 실행
> - 맥, 우분투는 sh start_mac.sh 실행

<span style='color:orange;margin-left:20px;'>첫 실행시 관련 프로그램 설치와 AI 모델을 다운로드 하는데 아주 오랜 시간이 걸립니다!</span>

<br/>

## 업데이트 목표
1. epub과 pdf 번역
2. 다양한 AI 번역 모델을 사용할 수 있게 하기
3. 여러 언어 번역도 가능하게 하기
4. 그러면서도 복잡하기 않고 아주 단순한 UI 유지하기
