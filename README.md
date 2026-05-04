<p align="center">
<img src='https://github.com/user-attachments/assets/6b6a73f6-087c-439c-869f-5e0d0629db92' width='200px' height='200px' title='도다리'/>
<h1 align="center">도다리 Dodari 2</h1>
<p align='center'>
도다리2는 구글의 최신 AI를 활용해 <br/>EPUB, PDF, TXT 문서를 장르와 문맥에 맞게 번역하는 다국어 번역기입니다.<br/>
-------<br/>
<span style='font-size:0.9em;'>*2024년 3월 도다리1 후속버전입니다</span><br/>
<span style='font-size:0.9em;'>2026.05.04 이제 윈도우에서도 도다리2 번역이 가능합니다.</span>
</p>

<img src='https://github.com/user-attachments/assets/2f27d751-b037-4204-8ffc-fd9e16b89015' title='도다리'/>

### 주요기능
1. `EPUB(전자책)`, `PDF`, `TXT` 번역. 
2. _주의!_ 레이아웃 보존을 위해 `PDF`의 번역 결과는 더 이상 `PDF`가 아닌, `EPUB`으로 형식이 바뀌어 저장됨. 복잡한 수식, 표는 이미지로 추가됨.
3. `번역문(원문)` 파일과 `번역문` 파일, 두가지 파일로 출력. 문장단위로 원문과 비교가능.
4. 언어 자동감지 기능
5. `한국어` · `영어` · `일본어` · `중국어` · `프랑스어` · `이탈리아어` · `네덜란드어` · `덴마크어` · `스웨덴어` · `노르웨이어` · `아랍어` · `페르시아어` 교차 번역 가능
6. 도서 장르 자동감지
7. 번역 문체 선택 가능
8. 용어집 - AI가 추출한 주요 용어(이름)등을 일관성있게 적용 가능
9. 용량 무제한

<br/>

### 구동 사양
| AI모델 | Gemma4 e4b 8bit | Gemma4 31b 4bit |
|:---|:---|:---|
| 추천 사양 | 최소 (기본품질) | 권장 (고품질) |
| 모델 설명 | 빠르고 쾌적한 번역 | 깊은 문맥 파악 풍부한 단어선택 |
| 주관적 느낌 | Deepl보다 좋은것같음 | 제미나이만큼 좋은거 같음 |
| 칩 | Apple Silicon M1 이상 | M3 Pro / M4 Max 이상 |
| 통합 메모리 | **8GB~16GB** | **32GB** |
| 저장 공간 | 10GB 이상 여유 | SSD 35GB 이상 |
| macOS | Ventura 13.0 이상 | 최신 버전 권장 |
| Python | 3.11 이상 | — |

<br/>

## 설치 및 실행

초보자라면,
1. <a href='https://github.com/vEduardovich/dodari/archive/refs/heads/main.zip' title='압축 파일 다운로드' style='text-align:center'>압축 파일 다운로드</a> 클릭
2. 압축해제 후 
- 윈도우 사용자는 `start_windows.bat` 더블 클릭
- 맥이나 우분투 사용자는 터미널 창에서 `sh start_mac.sh` 실행
3.  `http://127.0.0.1:7860` 에 접속하면 도다리2가 보입니다.

_첫 실행시 관련 프로그램 설치와 AI 모델을 다운로드 하는데 아주 오랜 시간이 걸립니다!_
_에러 발생시 dodari_env폴더 삭제후 다시 실행해보세요._

<br>
고급 사용자라면,

```bash
git clone https://github.com/vEduardovich/dodari.git
cd dodari
```
- 윈도우는 `start_windows.bat` 실행
- 맥, 우분투는 `sh start_mac.sh` 실행


<br/>

## 프로젝트 구조

```
dodari/
├── dodari_env         # 도다리 실행을 위한 관련 파일들이 설치되는 폴더
├── dodari.py          # 메인 애플리케이션
├── start_mac.sh       # 맥 실행 스크립트
├── start_windows.sh   # 윈도우 실행 스크립트
└── requirements.txt   # 의존성 목록
```

<br>

## 최신 버전으로 업데이트 하기
초보자라면,
1. 위 압축파일을 다시 다운로드하고 압축을 푼후
2. 기존 도다리 폴더안에 덮어쓰면 됩니다.

고급 사용자라면,
1. git pull

<br>

## 번역속도 체감
1. m5max는 워낙 고사양이니 m1pro의 속도만 보시면 됩니다.
2. 소설은 텍스트 뿐이라 epub이나 pdf나 번역속도가 비슷합니다.
3. 그림이나 코드가 많은 책은 pdf 번역속도가 더 빠릅니다. 왜 그런지 저도 잘 모르겠습니다. epub의 경우 표와 상세 플래그까지 모두 번역해서 그런게 아닌가 예상은 합니다. pdf는 이미지로 그냥 잘라낸후 번역없이 첨부하거든요.
<table style="table-layout:auto"><thead><tr><th rowspan="2">책</th><th rowspan="2">맥북</th><th colspan="2">epub</th><th colspan="2">pdf</th></tr><tr><th>기본e4b</th><th>고급31b</th><th>기본e4b</th><th>고급31b</th></tr></thead><tbody><tr><td rowspan="2">1984<br/>(소설)</td><td>m1pro 16g</td><td>133분</td><td>-</td><td>133분</td><td>-</td></tr><tr><td>m5max 128g</td><td>40분</td><td>135분</td><td>41분</td><td>136분</td></tr><tr><td rowspan="2">Pro Git<br/>(IT서적)</td><td>m1pro 16g</td><td>137분</td><td>-</td><td>65분</td><td>-</td></tr><tr><td>m5max 128g</td><td>45분</td><td>159분</td><td>21분</td><td>81분</td></tr></tbody></table>
1. 윈도우에서는 아주 느립니다. <br/>- 2020 LG그램으로 1984 소설의 한페이지를 번역하는데 EPUB은 15분, PDF는 18분 걸렸습니다(pdf는 첫로딩시 20분 걸리기도 합니다)<br/>- 그러니까 일반적인 윈도우 랩탑에서 100페이지짜리 epub전자책을 번역한다면 대략 1500분(25시간)이 걸린다는 이야기입니다. 200페이지면 50시간입니다. 하지만 나를 위해 컴퓨터가 쉬지않고 열심히 일하는 모습은 보기 좋았습니다.


<br/>

## 상세 기능설명
1. 도다리에 접속합니다. `http://127.0.0.1:7860`
2. 순서1에 번역할 파일을 첨부합니다. 파일의 언어를 자동으로 감지합니다. 수동변경도 가능합니다.
<p align="center"><img src='https://github.com/user-attachments/assets/22e4c9d3-5bf5-40f0-9712-6dfe326b54bf' width='360px'></p>

3. 순서2에 번역 목표 언어를 선택합니다.
4. 사용할 모델을 선택합니다. 기본선택 모델은 e4b모델입니다.
<p align="center"><img src='https://github.com/user-attachments/assets/00ae73fb-dcf1-4200-8c3b-e561615eca3a' width='360px'></p>

5. 순서3에 번역출력 방식을 선택합니다. 번역문을 먼저 표기하고 원문을 뒤에 표기할지, 반대로 표기할지 결정하는 겁니다. 학습용으로 번역하실 경우 원문을 먼저 표기하고 번역문을 뒤에 표기하면 좋습니다.
6. `~다`와 `~합니다` 어투를 결정합니다. 번역시 존댓말과 반말이 섞여 나오는 것을 방지합니다.
7. 용어집을 만들수 있습니다. 선택후 `AI용어 자동추출`버튼을 클릭하면 책에서 쓰이는 주요 이름이나 용어들이 뽑아집니다. 원하실 경우 직접 추가 삭제 가능합니다. `용어집 적용` 버튼을 클릭시하시면 적용됩니다.
<p align="center"><img src='https://github.com/user-attachments/assets/01b939af-a7cc-4ed5-8d91-19906a46cabd' width='360px'>

8. 마지막입니다. 순서4에서 `번역 실행하기`를 누르면 번역을 시작합니다.
<p align="center"><img src='https://github.com/user-attachments/assets/c74fc2f5-c83d-4427-927b-5d3ee3e0849b' width='360px'></p>

<br/>

## 삭제하기
### 1. 프로그램 삭제
다운로드한 `dodari` 폴더 전체를 지웁니다.

### 2. AI모델 삭제
#### Mac
`~/.cache/huggingface/hub` 아래의 폴더들을 삭제
<br/>
#### Windows
1. ollama 모델삭제
```bash
ollama rm gemma4:e4b
ollama rm gemma4:31b
```
1. ollama 프로그램 삭제
제어판 → 프로그램 추가/제거 → Ollama 제거

<br/>

## 업데이트 기록
2026.05.04 윈도우 플랫폼 추가, 특수문자관련 번역 오류 수정
<br/>

## 개발 뒷이야기
1. 주식으로 큰돈을 잃으니 개발을 향한 열정이 살아났습니다. 
2. 도다리2는 그렇게 탄생했습니다.
3. 저와 gemini와 claude가 함께 만들었습니다. 솔직히 저의 기여도는 클로드 다음입니다.
4. pdf는 언제봐도 아주 나쁜 파일 형식입니다.
5. 개발기간 일주일보다 이 README.md를 만드는데 더 오랜 시간이 걸렸습니다.
6. AI 도움없이 처음부터 끝까지 제가 만든건 이 파일 하나뿐입니다.
7. 번역 품질이 제 상상을 초월할 정도로 좋았습니다. 도다리1때는 원문없이 도저히 번역문만 읽을수가 없었는데 지금은 번역문 파일만 다운받아서 읽어도 아무 문제가 없습니다. IT책, 사회과학책 뿐만 아니라 소설책까지 말입니다.
8. 도다리1때는 네이버 스마트스토어에서 의뢰를 받아 유료 번역을 하곤 했는데 이제 그 시대는 끝난것 같습니다. 도다리2의 31b모델로 직접 번역하시면 구글 제미나이의 번역 품질 거의 그대로를 즐기실수 있습니다. 그래도 <strong>[네이버🛒스토어](https://smartstore.naver.com/ai_dodari/products/10259674404)</strong>는 계속 운영하겠습니다. 제가 주식 인버스를 탔거든요!
<br/>

---

© 2026 Dodari Project. All rights reserved.
