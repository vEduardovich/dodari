<p align="center">
<img src='https://github.com/user-attachments/assets/6b6a73f6-087c-439c-869f-5e0d0629db92' width='200px' height='200px' title='Dodari'/>
<h1 align="center">Dodari 2</h1>
<p align='center'>
Dodari 2 is a multilingual AI translator that uses Google's latest AI to translate <br/>EPUB, PDF, and TXT documents with genre-aware, context-sensitive accuracy.<br/>
-------<br/>
<span style='font-size:0.9em;'>*Successor to Dodari 1 (released March 2024)</span>
</p>

<img src='https://github.com/user-attachments/assets/835a52f7-c3c4-4ab8-972c-37e299afe316' title='Dodari'/>

### Key Features
1. Translates `EPUB (e-books)`, `PDF`, and `TXT` files.
2. _Note:_ To preserve layout, `PDF` translation output is saved as `EPUB` rather than `PDF`. Complex formulas and tables are embedded as images.
3. Outputs two files: `Translation (Original)` and `Translation only` — allowing sentence-by-sentence comparison with the source.
4. Automatic language detection.
5. Cross-translation between `Korean` · `English` · `Japanese` · `Chinese` · `French` · `Italian` · `Dutch` · `Danish` · `Swedish` · `Norwegian` · `Arabic` · `Persian`.
6. Automatic book genre detection.
7. Selectable translation tone/style.
8. Glossary — extract key terms (names, etc.) with AI and apply them consistently throughout.
9. No file size limit.

<br/>

### System Requirements
<table>
  <thead>
    <tr>
      <th colspan="2">AI Model</th>
      <th>Gemma4 e4b 8bit</th>
      <th>Gemma4 31b 4bit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2">Recommendation</td>
      <td>Minimum (standard quality)</td>
      <td>Recommended (high quality)</td>
    </tr>
    <tr>
      <td colspan="2">Model description</td>
      <td>Fast and comfortable translation</td>
      <td>Deep context, rich vocabulary</td>
    </tr>
    <tr>
      <td colspan="2">Subjective quality</td>
      <td>Feels better than DeepL</td>
      <td>Close to Gemini quality</td>
    </tr>
    <tr>
      <td colspan="2">Storage</td>
      <td>10 GB free space</td>
      <td>35 GB SSD or more</td>
    </tr>
    <tr>
      <td colspan="2">Python</td>
      <td>3.11 or higher</td>
      <td>—</td>
    </tr>
    <tr>
      <td rowspan="3">Mac</td>
      <td>Chip</td>
      <td>Apple Silicon M1 or later</td>
      <td>M3 Pro / M4 Max or later</td>
    </tr>
    <tr>
      <td>Unified Memory</td>
      <td><strong>8 GB – 16 GB</strong></td>
      <td><strong>32 GB</strong></td>
    </tr>
    <tr>
      <td>OS</td>
      <td>macOS Ventura 13.0 or later</td>
      <td>Latest version recommended</td>
    </tr>
    <tr>
      <td rowspan="3">Windows</td>
      <td>GPU</td>
      <td>Not required</td>
      <td>24 GB VRAM or more</td>
    </tr>
    <tr>
      <td>RAM</td>
      <td>8 GB</td>
      <td>64 GB</td>
    </tr>
    <tr>
      <td>Windows</td>
      <td>Windows 10 (22H2) or later</td>
      <td>Windows 11</td>
    </tr>
  </tbody>
</table>

<br/>

## Installation & Setup

For beginners:
1. Click <a href='https://github.com/vEduardovich/dodari/archive/refs/heads/main.zip' title='Download zip' style='text-align:center'>Download ZIP</a>
2. Extract the archive, then:
- **Windows**: double-click `start_windows.bat`
- **Mac**: run `sh start_mac.sh` in a terminal window
3. Open `http://127.0.0.1:7860` in your browser — Dodari 2 will be ready.

_On first run, setup and AI model download will take a long time. Please be patient!_
_If you encounter an error, delete the `dodari_env` folder and run the script again._

<br>
For advanced users:

```bash
git clone https://github.com/vEduardovich/dodari.git
cd dodari
```
- Windows: run `start_windows.bat`
- Mac: run `sh start_mac.sh`


<br/>

## Project Structure

```
dodari/
├── dodari_env         # Folder where runtime dependencies are installed
├── dodari.py          # Main application
├── start_mac.sh       # Mac launch script
├── start_windows.bat  # Windows launch script
└── requirements.txt   # Dependency list
```

<br>

## Updating to the Latest Version
For beginners:
1. Download the ZIP again and extract it.
2. Overwrite the existing Dodari folder with the new files.

For advanced users:
1. `git pull`
<br>

## Translation Speed Reference
1. The M5 Max is a very high-end machine — the M1 Pro numbers are more representative for most users.
2. Novels are text-only, so EPUB and PDF translation speeds are similar.
3. Books with many images or code blocks translate faster as PDF — PDFs skip translating images entirely and embed them as-is, while EPUB translates tables and detailed flags too.
<table style="table-layout:auto"><thead><tr><th rowspan="2">Book</th><th rowspan="2">MacBook</th><th colspan="2">epub</th><th colspan="2">pdf</th></tr><tr><th>e4b (standard)</th><th>31b (high quality)</th><th>e4b (standard)</th><th>31b (high quality)</th></tr></thead><tbody><tr><td rowspan="2">1984<br/>(novel)</td><td>M1 Pro 16 GB</td><td>133 min</td><td>—</td><td>133 min</td><td>—</td></tr><tr><td>M5 Max 128 GB</td><td>40 min</td><td>135 min</td><td>41 min</td><td>136 min</td></tr><tr><td rowspan="2">Pro Git<br/>(IT book)</td><td>M1 Pro 16 GB</td><td>137 min</td><td>—</td><td>65 min</td><td>—</td></tr><tr><td>M5 Max 128 GB</td><td>45 min</td><td>159 min</td><td>21 min</td><td>81 min</td></tr></tbody></table>

4. Windows is considerably slower. <br/>— On a 2020 LG Gram laptop, translating one page of the novel *1984* took 15 minutes for EPUB and 18 minutes for PDF (the first PDF load can take up to 20 minutes).<br/>— So on a typical Windows laptop, a 100-page EPUB would take roughly 1,500 minutes (25 hours). 200 pages = 50 hours. That said, watching your computer work tirelessly for you is strangely satisfying.


<br/>


## Uninstalling
### 1. Remove the program
Delete the entire `dodari` folder you downloaded.

### 2. Remove the AI model
#### Mac
Delete the folders under `~/.cache/huggingface/hub`.
<br/>

#### Windows
1. Remove Ollama models:
```bash
ollama rm gemma4:e4b
ollama rm gemma4:31b
```
2. Uninstall Ollama:
Control Panel → Programs → Uninstall Ollama

<br/>

## Changelog
* 2026.05.04 — Added Windows support; fixed translation errors related to special characters.
* 2026.05.06 Implemented Multilingual Support 
<br/>

---
<p align="center">
<img src='https://github.com/user-attachments/assets/6b6a73f6-087c-439c-869f-5e0d0629db92' width='200px' height='200px' title='도다리'/>
<h1 align="center">도다리 Dodari 2</h1>
<p align='center'>
도다리2는 구글의 최신 AI를 활용해 <br/>EPUB, PDF, TXT 문서를 장르와 문맥에 맞게 번역하는 다국어 번역기입니다.<br/>
-------<br/>
<span style='font-size:0.9em;'>*2024년 3월 도다리1 후속버전입니다</span><br/>
<span style='font-size:0.9em;'>*2026.05.04 이제 윈도우에서도 도다리2 번역이 가능합니다.</span>
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
<table>
  <thead>
    <tr>
      <th colspan="2">AI모델</th>
      <th>Gemma4 e4b 8bit</th>
      <th>Gemma4 31b 4bit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2">추천 사양</td>
      <td>최소 (기본품질)</td>
      <td>권장 (고품질)</td>
    </tr>
    <tr>
      <td colspan="2">모델 설명</td>
      <td>빠르고 쾌적한 번역</td>
      <td>깊은 문맥 파악 풍부한 단어선택</td>
    </tr>
    <tr>
      <td colspan="2">주관적 느낌</td>
      <td>DeepL보다 좋은것같음</td>
      <td>제미나이만큼 좋은거 같음</td>
    </tr>
    <tr>
      <td colspan="2">저장 공간</td>
      <td>10GB 이상 여유</td>
      <td>SSD 35GB 이상</td>
    </tr>
    <tr>
      <td colspan="2">Python</td>
      <td>3.11 이상</td>
      <td>—</td>
    </tr>
    <tr>
      <td rowspan="3">맥북</td>
      <td>칩</td>
      <td>Apple Silicon M1 이상</td>
      <td>M3 Pro / M4 Max 이상</td>
    </tr>
    <tr>
      <td>통합 메모리</td>
      <td><strong>8GB~16GB</strong></td>
      <td><strong>32GB</strong></td>
    </tr>
    <tr>
      <td>OS</td>
      <td>Ventura 13.0 이상</td>
      <td>최신 버전 권장</td>
    </tr>
    <tr>
      <td rowspan="3">윈도우</td>
      <td>GPU</td>
      <td>없어도 됨</td>
      <td>24GB+</td>
    </tr>
    <tr>
      <td>RAM</td>
      <td>8GB</td>
      <td>64GB</td>
    </tr>
    <tr>
      <td>Windows</td>
      <td>Windows 10(22H2) 이상</td>
      <td>Windows 11</td>
    </tr>
  </tbody>
</table>


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

4. 윈도우에서는 아주 느립니다. <br/>- 2020 LG그램으로 1984 소설의 한페이지를 번역하는데 EPUB은 15분, PDF는 18분 걸렸습니다(pdf는 첫로딩시 20분 걸리기도 합니다)<br/>- 그러니까 일반적인 윈도우 랩탑에서 100페이지짜리 epub전자책을 번역한다면 대략 1500분(25시간)이 걸린다는 이야기입니다. 200페이지면 50시간입니다. 하지만 나를 위해 컴퓨터가 쉬지않고 열심히 일하는 모습은 보기 좋았습니다.


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
2. ollama 프로그램 삭제
제어판 → 프로그램 추가/제거 → Ollama 제거

<br/>

## 업데이트 기록
2026.05.04 윈도우 플랫폼 추가, 특수문자관련 번역 오류 수정
2026.05.06 다국어버전 적용
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
