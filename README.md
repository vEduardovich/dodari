<p align="center">
<img src='https://github.com/vEduardovich/dodari/assets/20391482/972aee6d-383e-47ed-90b6-73e0cc513973' title='도다리'/>
<h1 align="center">도다리 Dodari</h1>
<p align='center'>
AI 한영/영한 번역기를<br> 일반 사람들도 쉽게 쓸수 있게 만든 로컬 웹서비스 입니다. (based on Gradio)</p>
</p>

<br>

## 특징
자신의 컴퓨터에서 제한없이 `한영`-`영한` AI 번역이 가능합니다. 
- 일반 구글번역에 비해 품질이 우수합니다.
- `txt`와 `epub(전자책)`, `srt(자막)` 번역이 가능합니다.
- `번역문(원문)` 파일과 `번역문` 파일, 이렇게 두가지 파일로 출력합니다. `번역문(원문)`의 경우 번역이 이상할 경우 원문과 바로 비교할수 있습니다.
- 사용이 아주 쉽습니다. 번역이 필요한 파일들을 드래그한 후 `번역 실행하기` 버튼만 클릭하면 됩니다. 알아서 `한↔영` 으로 번역해 줍니다.
<img src='https://github.com/vEduardovich/dodari/assets/20391482/6d46e5b9-3a49-4950-984f-5bbbbeb5f2b5' style='display:block;border-radius:10px;text-align:center;' title='도다리 실행화면'/>

<br>

## 번역 결과
아래와 같이 두개의 파일이 만들어집니다. 

<p align="center">번역문(원문) 파일</p>
<img src='https://github.com/vEduardovich/dodari/assets/20391482/10a0f93e-ce46-4303-ac7b-d5226b92dbfd' style='border-radius:10px;margin-right:10px;' title='한영 번역화면'/>

<br>

<p align="center" >번역문 파일</p>
<img src='https://github.com/vEduardovich/dodari/assets/20391482/3ae95633-0c8e-4997-99fe-44151b845094' style='border-radius:10px;' title='한글 번역화면'/>

<br>


<p align="center">자막 번역</p>
<p align="center">
<img src='https://github.com/vEduardovich/dodari/assets/20391482/a4f1e7b4-5925-413a-a9f4-93248b106c27' style='border-radius:10px;' title='한글 자막번역'/>
</p>

<p align="center" >(참고) DeepL 번역</p>
<p>"토끼굴 아래로 앨리스는 언니 옆에 앉아서 할 일이 없는 것에 매우 지치기 시작했고, 언니가 읽고 있는 책을 한두 번 들여다봤지만, 그 책에는 그림이나 대화가 없었습니다."그림이나 대화가 없는 책이 무슨 소용이 있을까?"앨리스는 생각했습니다. "그래서 그녀는 데이지 사슬을 만드는 즐거움이 일어나서 데이지를 따는 수고로움의 가치가 있는지, (더운 날이 그녀를 매우 졸리고 멍청하게 만들었 기 때문에 가능한 한 잘 생각했습니다) 마음 속으로 생각하고 있었는데 갑자기 분홍색 눈을 가진 하얀 토끼가 그녀 곁으로 달려갔습니다."거기에는 그렇게 놀라운 것도 없었고 앨리스도 토끼가 스스로 말하는 것을 듣는 것이 그렇게 매우 이상하게 생각하지 않았습니다."오 맙소사! 오, 이런! 너무 늦겠어!" (나중에 곰곰이 생각해보니 앨리스가 이걸 궁금해했어야 했다는 생각이 들었지만, 그 당시에는 모든 것이 아주 자연스러워 보였습니다)</p>
<p align="center" >(참고) 구글 번역</p>
<p>“앨리스는 은행에서 여동생 옆에 앉아 있고 할 일이 없는 것에 매우 지치기 시작했습니다. 한두 번 그녀는 여동생이 읽고 있는 책을 들여다 보았지만 그 안에 그림이나 대화가 전혀 없었습니다. "그림이나 대화가 없는 책이 무슨 소용이 있겠는가?"라고 앨리스는 생각했습니다. 데이지 체인을 만드는 즐거움이 일어나서 데이지를 따는 수고를 할 만큼 가치가 있을지에 대해 그녀 자신의 마음 속에서 (그리고 그녀는 할 수 있는 한, 더운 날 때문에 그녀는 매우 졸리고 멍청하다고 느꼈다) 갑자기 백인이 분홍색 눈을 가진 토끼가 그녀 옆으로 달려왔습니다. 그다지 주목할만한 점은 없었습니다. 앨리스는 토끼가 혼잣말하는 것을 듣고도 별로 이상하다고 생각하지 않았습니다. “오 이런! 이런! 너무 늦을 것 같아요!” (나중에 그녀가 곰곰이 생각해보니, 그녀는 이것에 대해 궁금해했어야 했다는 생각이 들었지만, 당시에는 모든 것이 아주 자연스러워 보였습니다.)</p>

<br>

## 설치 및 실행
공통,
> 1. **먼저 파이썬이 설치되어 있어야 합니다.** 윈도우에는 파이썬이 기본으로 설치되어 있지 않습니다.
> 2. https://wikidocs.net/8 를 참고하여 쉽게 설치하실수 있습니다.

<br>

초보자라면,
1. <a href='https://github.com/vEduardovich/dodari/archive/refs/heads/main.zip' title='압축 파일 다운로드' style='text-align:center'>압축 파일 다운로드</a> 클릭
2. 압축해제 후 
> - 윈도우 사용자는 start_windows.bat 더블 클릭
> - 맥이나 우분투 사용자는 터미널 창에서 sh start_mac.sh 실행
3. 처음 실행이라면 프로그램을 자동으로 설치한 후 실행합니다. 이미 설치가 되었다면 바로 실행합니다.

<br>

고급 사용자라면,
1. git clone https://github.com/vEduardovich/dodari.git
2. cd dodari
3. 아래 방법으로 실행하기
> - 윈도우는 start_windows.bat 실행
> - 맥, 우분투는 sh start_mac.sh 실행

_첫 실행시 관련 프로그램 설치와 AI 모델을 다운로드 하는데 아주 오랜 시간이 걸립니다!</span>_

<br>

## 번역 속도 비교
헤르만 헤세의 싯다르타 text과 이상한 나라의 앨리스(영문판).epub을 번역해봤습니다.
<table>
  <thead>
    <tr>
      <th>사양</th>
      <th>운영체제</th>
      <th>CPU</th>
      <th>GPU</th>
      <th>싯다르타.txt (378kb, 2577문장)</th>
      <th>이상한나라의 앨리스.epub</th>
      <th>srt 자막 (128kb, 1846문장)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LG 2020 그램 17</td>
      <td>윈도우10</td>
      <td>i7 (1.3GHz)</td>
      <td>내장그래픽</td>
      <td>2시간 55분 24초</td>
      <td>3시간 45분 6초</td>
      <td>42분 32초</td>
    </tr>
    <tr>
      <td>Mac Pro M1</td>
      <td>iOS</td>
      <td>10코어</td>
      <td>16코어</td>
      <td>54분 23초</td>
      <td>1시간 13분 5초</td>
      <td>8분 27초</td>
    </tr>
    <tr>
      <td>데스크탑</td>
      <td>Ubuntu22.04</td>
      <td>i9-13900k</td>
      <td>RTX4090 24GB</td>
      <td>5분 25초</td>
      <td>7분 20초</td>
      <td>2분 8초</td>
    </tr>
    <tr>
      <td>데스크탑</td>
      <td>윈도우11</td>
      <td>i9-13900k</td>
      <td>RTX4090 24GB</td>
      <td>11분 49초</td>
      <td>14분 36초</td>
      <td>2분 35초</td>
    </tr>
  </tbody>
</table>

<br>

[![Youtube](http://img.youtube.com/vi/hE-4hXLhlcg/0.jpg)](https://youtu.be/hE-4hXLhlcg)
<br>
_번역영상_

<br>

## 디렉토리 구조
```
dodari
├⎯ models : AI가 다운로드되는 폴더
├⎯ venv   : 도다리 실행을 위한 관련 파일들이 설치되는 폴더
├⎯ imgs   : 도다리 이미지
```

<br>

## 최신 버전으로 업데이트 하기
초보자라면,
1. 위 압축파일을 다시 다운로드하고 압축을 푼후
2. 기존 도다리 폴더안에 덮어쓰면 됩니다.

고급 사용자라면,
1. git pull

<br>

## 삭제하기
- 폴더 전체를 지우면 깨끗하게 지워집니다.

<br>
<br>

## 도다리 일반번역과 고급번역의 차이
- 도다리 일반번역은 가성비가 가장 좋은 <a href='https://huggingface.co/NHNDQ/nllb-finetuned-en2ko' target='_blank'>NHNDQ</a>를 사용합니다. 여기 깃헙에 공개한 도다리가 바로 일반번역 모델입니다.
- 도다리 고급번역은 올해의 AI 언어모델로 선정된 <a href='https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0' target='_blank'>EEVE</a>를 사용합니다.
- 둘의 기능과 소스코드는 동일합니다. 단지 고급번역은 매우 높은 사양의 컴퓨터가 필요한 AI모델을 사용합니다. 이에 직접 서비스를 결정했습니다.

고급번역으로 `이상한나라의 앨리스` 번역했을때
<table>
  <thead>
    <tr>
      <th>상태</th>
      <th>걸린시간</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>vllm 가속 미적용시</td>
      <td>4시간 7분 26초</td>
    </tr>
    <tr>
      <td>vllm 가속 적용시</td>
      <td>8분 18초</td>
    </tr>
  </tbody>
</table>
<br>

## 도다리 고급번역
- DeepL에 준하는 번역품질 (EEVE-Korean-Instruct-10.8B-v1.0 모델 사용)
- AI추론 가속 기술(vllm)을 이용한 고속번역
- <a href='https://moonlit.himion.com/dodari?utm_campaign=goto_moonlit&utm_source=github&utm_medium=link&utm_content=dodari_landing_page' target='_blank'>도다리 고급번역 보러가기</a>

<br>


## 업데이트 예정 사항!
도다리 고급 번역이 어느정도 안정됨에 따라 코드 공개를 결정했습니다.
1. vram이 23이상일때 야놀자 모델사용 가능
2. 운영체제가 리눅스일 경우 vllm(AI 가속 모듈) 사용 가능
3. 번역 프롬프트 커스터마이징
4. 필터 문구 커스터마이징

조금만 기다려주세요.