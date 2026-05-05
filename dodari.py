import os
import sys

if os.name == 'nt':
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
import base64
from typing import List, Union, Sequence
from datetime import timedelta
import logging, warnings
import copy
import re, time, platform, shutil, zipfile, subprocess, json, locale
import requests
import chardet

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
    from docling_core.types.doc.document import ImageRefMode
    DOCLING_AVAILABLE = True
    print('[INFO] docling import successful')
except Exception as _e:
    DOCLING_AVAILABLE = False
    print(f'[WARNING] docling import failed: {_e}')

try:
    import fitz
    FITZ_AVAILABLE = True
    print('[INFO] fitz(PyMuPDF) import successful')
except Exception as _e:
    FITZ_AVAILABLE = False
    print(f'[WARNING] fitz(PyMuPDF) import failed: {_e}')

import ebooklib
from ebooklib import epub
from langdetect import detect, detect_langs, DetectorFactory
DetectorFactory.seed = 0
import nltk

from bs4 import BeautifulSoup
import gradio as gr
import gc
from xml.etree.ElementTree import parse
import atexit

def cleanup_llm_server():
    if platform.system() != 'Windows':
        print("\n[SHUTDOWN] Stopping MLX API server...")
        os.system("pkill -f 'mlx_vlm.server'")

atexit.register(cleanup_llm_server)

def format_korean_time(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds}초'
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f'{minutes}분 {sec}초' if sec else f'{minutes}분'
    hours, min_ = divmod(minutes, 60)
    parts = [f'{hours}시간']
    if min_:
        parts.append(f'{min_}분')
    if sec:
        parts.append(f'{sec}초')
    return ' '.join(parts)

logging.getLogger().disabled = True
logging.raiseExceptions = False
warnings.filterwarnings('ignore')

nltk.download('punkt_tab')
PathType = Union[str, os.PathLike]

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_path = "imgs/dodari.png"
encoded_img = get_base64_image(img_path)
img_src = f"data:image/png;base64,{encoded_img}"

SUPPORTED_LANGUAGES = {
    '한국어':     ('ko', 'Korean'),
    '영어':       ('en', 'English'),
    '일본어':     ('ja', 'Japanese'),
    '중국어':     ('zh', 'Chinese (Simplified)'),
    '프랑스어':   ('fr', 'French'),
    '이탈리아어': ('it', 'Italian'),
    '네덜란드어': ('nl', 'Dutch'),
    '덴마크어':   ('da', 'Danish'),
    '스웨덴어':   ('sv', 'Swedish'),
    '노르웨이어': ('no', 'Norwegian'),
    '아랍어':     ('ar', 'Arabic'),
    '페르시아어': ('fa', 'Persian (Farsi)'),
}
LANG_CODE_TO_NAME = {v[0]: k for k, v in SUPPORTED_LANGUAGES.items()}
LANG_CODE_TO_NAME['zh-cn'] = '중국어'
LANG_CODE_TO_NAME['zh-tw'] = '중국어'

GENRE_CHOICES_KO   = ["IT 및 엔지니어링", "문학 및 소설", "인문 및 사회과학", "비즈니스 및 경제", "영상 및 대본", "일반 문서(기본)"]
TONE_CHOICES_KO    = ["서술체 (~다)", "경어체 (~합니다)"]
BILINGUAL_CHOICES_KO = ["번역문(원문)", "원문(번역문)"]

_UI_CONFIG_PATH = 'ui_config.json'
_UI_LANG_CODES  = ['ko', 'en', 'ja', 'zh', 'fr', 'it', 'nl', 'da', 'sv', 'no', 'ar', 'fa']

LANG_DISPLAY_BY_UI = {
    'ko': {'한국어':'한국어','영어':'영어','일본어':'일본어','중국어':'중국어','프랑스어':'프랑스어','이탈리아어':'이탈리아어','네덜란드어':'네덜란드어','덴마크어':'덴마크어','스웨덴어':'스웨덴어','노르웨이어':'노르웨이어','아랍어':'아랍어','페르시아어':'페르시아어'},
    'en': {'한국어':'Korean','영어':'English','일본어':'Japanese','중국어':'Chinese','프랑스어':'French','이탈리아어':'Italian','네덜란드어':'Dutch','덴마크어':'Danish','스웨덴어':'Swedish','노르웨이어':'Norwegian','아랍어':'Arabic','페르시아어':'Persian'},
    'ja': {'한국어':'韓国語','영어':'英語','일본어':'日本語','중국어':'中国語','프랑스어':'フランス語','이탈리아어':'イタリア語','네덜란드어':'オランダ語','덴마크어':'デンマーク語','스웨덴어':'スウェーデン語','노르웨이어':'ノルウェー語','아랍어':'アラビア語','페르시아어':'ペルシア語'},
    'zh': {'한국어':'韩语','영어':'英语','일본어':'日语','중국어':'中文','프랑스어':'法语','이탈리아어':'意大利语','네덜란드어':'荷兰语','덴마크어':'丹麦语','스웨덴어':'瑞典语','노르웨이어':'挪威语','아랍어':'阿拉伯语','페르시아어':'波斯语'},
    'fr': {'한국어':'Coréen','영어':'Anglais','일본어':'Japonais','중국어':'Chinois','프랑스어':'Français','이탈리아어':'Italien','네덜란드어':'Néerlandais','덴마크어':'Danois','스웨덴어':'Suédois','노르웨이어':'Norvégien','아랍어':'Arabe','페르시아어':'Persan'},
    'it': {'한국어':'Coreano','영어':'Inglese','일본어':'Giapponese','중국어':'Cinese','프랑스어':'Francese','이탈리아어':'Italiano','네덜란드어':'Olandese','덴마크어':'Danese','스웨덴어':'Svedese','노르웨이어':'Norvegese','아랍어':'Arabo','페르시아어':'Persiano'},
    'nl': {'한국어':'Koreaans','영어':'Engels','일본어':'Japans','중국어':'Chinees','프랑스어':'Frans','이탈리아어':'Italiaans','네덜란드어':'Nederlands','덴마크어':'Deens','스웨덴어':'Zweeds','노르웨이어':'Noors','아랍어':'Arabisch','페르시아어':'Perzisch'},
    'da': {'한국어':'Koreansk','영어':'Engelsk','일본어':'Japansk','중국어':'Kinesisk','프랑스어':'Fransk','이탈리아어':'Italiensk','네덜란드어':'Hollandsk','덴마크어':'Dansk','스웨덴어':'Svensk','노르웨이어':'Norsk','아랍어':'Arabisk','페르시아어':'Persisk'},
    'sv': {'한국어':'Koreanska','영어':'Engelska','일본어':'Japanska','중국어':'Kinesiska','프랑스어':'Franska','이탈리아어':'Italienska','네덜란드어':'Holländska','덴마크어':'Danska','스웨덴어':'Svenska','노르웨이어':'Norska','아랍어':'Arabiska','페르시아어':'Persiska'},
    'no': {'한국어':'Koreansk','영어':'Engelsk','일본어':'Japansk','중국어':'Kinesisk','프랑스어':'Fransk','이탈리아어':'Italiensk','네덜란드어':'Nederlandsk','덴마크어':'Dansk','스웨덴어':'Svensk','노르웨이어':'Norsk','아랍어':'Arabisk','페르시아어':'Persisk'},
    'ar': {'한국어':'الكورية','영어':'الإنجليزية','일본어':'اليابانية','중국어':'الصينية','프랑스어':'الفرنسية','이탈리아어':'الإيطالية','네덜란드어':'الهولندية','덴마크어':'الدانماركية','스웨덴어':'السويدية','노르웨이어':'النرويجية','아랍어':'العربية','페르시아어':'الفارسية'},
    'fa': {'한국어':'کره‌ای','영어':'انگلیسی','일본어':'ژاپنی','중국어':'چینی','프랑스어':'فرانسوی','이탈리아어':'ایتالیایی','네덜란드어':'هلندی','덴마크어':'دانمارکی','스웨덴어':'سوئدی','노르웨이어':'نروژی','아랍어':'عربی','페르시아어':'فارسی'},
}

UI_LANG_NAMES = {
    'ko':'한국어','en':'English','ja':'日本語','zh':'中文','fr':'Français','it':'Italiano',
    'nl':'Nederlands','da':'Dansk','sv':'Svenska','no':'Norsk','ar':'العربية','fa':'فارسی',
}

UI_TEXT = {
'ko': {
    'app_title': "AI 다국어 번역기 <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>도다리2</a></span> 입니다",
    'step1':'순서 1','step2':'순서 2','step3':'순서 3','step4':'순서 4','status_tab':'상태창',
    'step1_title':'1. 번역할 파일들 선택',
    'files_label':'파일들',
    'file_limit':'한번에 {n}개까지 첨부가 가능합니다',
    'origin_lang_label':'원본 언어 (자동 감지 · 수동 변경 가능)',
    'target_lang_label':'번역 목표 언어',
    'engine_ollama':'✔ Ollama 번역 엔진 활성화됨',
    'engine_gemma':'✔ Gemma 4 API 번역 사용 중',
    'model_label':"모델 선택 (E4B: 16GB 이하 초고속 추천 · 31B: 32GB 이상 고품질, 교체 시 서버 재시작 소요)",
    'bilingual_label':"이중언어 표기 방식 (학습용은 '원문(번역문)' 추천)",
    'genre_label':'장르 지정 (AI가 자동 추론)',
    'tone_label':'문체 선택 (일관된 어투 유지)',
    'genre_0':'IT 및 엔지니어링','genre_1':'문학 및 소설','genre_2':'인문 및 사회과학',
    'genre_3':'비즈니스 및 경제','genre_4':'영상 및 대본','genre_5':'일반 문서(기본)',
    'tone_0':'서술체 (~다)','tone_1':'경어체 (~합니다)',
    'bilingual_0':'번역문(원문)','bilingual_1':'원문(번역문)',
    'glossary_title':'✨ 용어집',
    'btn_glossary_extract':'🔍 AI 용어 자동 추출',
    'glossary_label':'용어집 (원문: 번역어 형식, 줄바꿈으로 구분)',
    'glossary_placeholder':'James: 제임스\nEldoria: 엘도리아\nDark Magic: 어둠의 마법',
    'btn_glossary_apply':'✅ 용어집 적용',
    'btn_glossary_clear':'🗑️ 용어집 초기화',
    'glossary_count':'현재 적용된 용어: {n}개',
    'glossary_desc':'소설 인물 이름·전문 용어가 페이지마다 달라지는 문제를 방지합니다.\n\n**① 자동 추출:** 버튼을 누르면 AI가 파일에서 중요 용어를 찾아 제안합니다.\n\n**② 직접 입력:** 아래 텍스트박스에 `원문: 번역어` 형식으로 한 줄씩 작성해도 됩니다.\n\n(예: `James: 제임스`, `Eldoria: 엘도리아`)',
    'btn_translate':'번역 실행하기',
    'download_label':'번역결과 다운로드',
    'ui_lang_label':'UI 언어',
    'ui_lang_restart':'UI 언어가 변경되었습니다. 앱을 재시작해 주세요.',
    'status_detecting':"🔍 언어 감지중입니다...",
    'status_ready':"번역준비를 마쳤습니다.\n위에 '번역실행하기' 버튼을 클릭하세요",
    'status_detected':"{lang} 문서가 감지되었습니다. 목표 언어를 선택하고 번역을 시작하세요.",
    'status_image_only':"이미지만 있는 파일입니다. 확인해주세요!",
    'err_file_none':"번역할 파일을 추가하세요",
    'err_file_limit':"한번에 {n}개 이상의 파일을 첨부할수 없습니다 파일을 다시 첨부해주세요",
    'err_file_count':"한번에 {n}개 이상의 파일을 번역할 수 없습니다.",
    'err_lang_same':"원본 언어와 목표 언어가 같습니다 ({lang}).<br>다른 목표 언어를 선택한 후 다시 시도해주세요.",
    'err_lang_detect':"언어 감지가 완료되지 않았습니다.<br>파일을 다시 첨부한 후 언어 확인까지 완료해주세요.",
    'err_server':"[오류] 번역 서버({url})에 연결할 수 없습니다.<br>{guide}",
    'server_guide_mac':"Mac: <code>start_mac.sh</code> 실행 여부를 확인하세요.",
    'server_guide_linux':"Linux: <code>start_ubuntu.sh</code> 또는 vLLM 서버 실행 여부를 확인하세요.",
    'server_guide_windows':"Windows: Ollama가 실행 중인지 확인하세요. (<code>ollama serve</code>)",
    'server_guide_default':"번역 서버 실행 여부를 확인하세요.",
    'err_upload_detect':"어떤 언어인지 알아내는데 실패했습니다.",
    'err_size_exceeded':"제한 용량을 초과했습니다.",
    'translation_complete':"번역완료! 걸린시간 : {t} 하단에서 결과물을 다운로드하세요.",
    'progress_init':"번역 모델을 준비중입니다...",
    'progress_server':"번역 서버 상태 확인 중...",
    'progress_files':'파일로딩',
    'lang_unknown':'알 수 없음',
    'glossary_applied':'✅ **{n}개의 용어가 적용되었습니다.** 이제 번역 시 이 용어들이 우선 사용됩니다.',
    'glossary_empty':'⚠️ 적용된 용어가 없습니다. `원문: 번역어` 형식으로 입력하세요.',
    'glossary_cleared':'용어집이 초기화되었습니다.',
},
'en': {
    'app_title': "AI Multilingual Translator <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>Dodari2</a></span>",
    'step1':'Step 1','step2':'Step 2','step3':'Step 3','step4':'Step 4','status_tab':'Status',
    'step1_title':'1. Select files to translate',
    'files_label':'Files',
    'file_limit':'You can attach up to {n} files at a time',
    'origin_lang_label':'Source language (auto-detected · manually changeable)',
    'target_lang_label':'Target language',
    'engine_ollama':'✔ Ollama translation engine active',
    'engine_gemma':'✔ Gemma 4 API translation active',
    'model_label':"Model selection (E4B: fast for ≤16GB · 31B: high quality for ≥32GB, server restart on change)",
    'bilingual_label':"Bilingual display mode (for learners: 'Original (Translation)' recommended)",
    'genre_label':'Genre (AI auto-inferred)',
    'tone_label':'Tone (consistent style maintained)',
    'genre_0':'IT & Engineering','genre_1':'Literature & Fiction','genre_2':'Humanities & Social Science',
    'genre_3':'Business & Economics','genre_4':'Film & Script','genre_5':'General Document (default)',
    'tone_0':'Narrative (~plain)','tone_1':'Formal (~polite)',
    'bilingual_0':'Translation (Original)','bilingual_1':'Original (Translation)',
    'glossary_title':'✨ Glossary',
    'btn_glossary_extract':'🔍 AI Auto-Extract Terms',
    'glossary_label':'Glossary (source: translation format, one per line)',
    'glossary_placeholder':'James: James\nEldoria: Eldoria\nDark Magic: Dark Magic',
    'btn_glossary_apply':'✅ Apply Glossary',
    'btn_glossary_clear':'🗑️ Clear Glossary',
    'glossary_count':'Applied terms: {n}',
    'glossary_desc':'Prevents character names and technical terms from varying across pages.\n\n**① Auto-extract:** AI scans your file and suggests key terms.\n\n**② Manual input:** Enter terms in `source: translation` format, one per line.\n\n(e.g. `James: James`, `Eldoria: Eldoria`)',
    'btn_translate':'Start Translation',
    'download_label':'Download Results',
    'ui_lang_label':'UI Language',
    'ui_lang_restart':'UI language changed. Please restart the app.',
    'status_detecting':"🔍 Detecting language...",
    'status_ready':"Ready to translate.\nClick the 'Start Translation' button above.",
    'status_detected':"{lang} document detected. Select target language and start translation.",
    'status_image_only':"This file contains only images. Please verify!",
    'err_file_none':"Please add a file to translate",
    'err_file_limit':"Cannot attach more than {n} files at a time. Please re-attach files.",
    'err_file_count':"Cannot translate more than {n} files at a time.",
    'err_lang_same':"Source and target languages are the same ({lang}).<br>Please select a different target language.",
    'err_lang_detect':"Language detection not complete.<br>Please re-attach the file and wait for detection.",
    'err_server':"[Error] Cannot connect to translation server ({url}).<br>{guide}",
    'server_guide_mac':"Mac: Check if <code>start_mac.sh</code> is running.",
    'server_guide_linux':"Linux: Check if <code>start_ubuntu.sh</code> or the vLLM server is running.",
    'server_guide_windows':"Windows: Check if Ollama is running. (<code>ollama serve</code>)",
    'server_guide_default':"Check if the translation server is running.",
    'err_upload_detect':"Failed to detect the language of the file.",
    'err_size_exceeded':"File size limit exceeded.",
    'translation_complete':"Translation complete! Time elapsed: {t} Download the results below.",
    'progress_init':"Preparing translation model...",
    'progress_server':"Checking translation server status...",
    'progress_files':'Loading files',
    'lang_unknown':'Unknown',
    'glossary_applied':'✅ **{n} terms applied.** These will be prioritized during translation.',
    'glossary_empty':'⚠️ No terms applied. Use `source: translation` format.',
    'glossary_cleared':'Glossary cleared.',
},
'ja': {
    'app_title': "AI多言語翻訳機 <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>Dodari2</a></span>",
    'step1':'手順 1','step2':'手順 2','step3':'手順 3','step4':'手順 4','status_tab':'状態',
    'step1_title':'1. 翻訳するファイルを選択',
    'files_label':'ファイル',
    'file_limit':'一度に{n}個まで添付できます',
    'origin_lang_label':'原文言語（自動検出・手動変更可）',
    'target_lang_label':'翻訳先言語',
    'engine_ollama':'✔ Ollama翻訳エンジン有効',
    'engine_gemma':'✔ Gemma 4 API翻訳使用中',
    'model_label':"モデル選択（E4B: 16GB以下高速・31B: 32GB以上高品質、切替時サーバー再起動）",
    'bilingual_label':"対訳表示方式（学習者向け：「原文（訳文）」推奨）",
    'genre_label':'ジャンル指定（AI自動推定）',
    'tone_label':'文体選択（一貫したスタイル維持）',
    'genre_0':'IT・エンジニアリング','genre_1':'文学・小説','genre_2':'人文・社会科学',
    'genre_3':'ビジネス・経済','genre_4':'映像・台本','genre_5':'一般文書（デフォルト）',
    'tone_0':'叙述体（〜だ）','tone_1':'丁寧体（〜です）',
    'bilingual_0':'訳文（原文）','bilingual_1':'原文（訳文）',
    'glossary_title':'✨ 用語集',
    'btn_glossary_extract':'🔍 AI用語自動抽出',
    'glossary_label':'用語集（原文: 訳語 形式、改行区切り）',
    'glossary_placeholder':'James: ジェームズ\nEldoria: エルドリア\nDark Magic: ダークマジック',
    'btn_glossary_apply':'✅ 用語集適用',
    'btn_glossary_clear':'🗑️ 用語集リセット',
    'glossary_count':'適用中の用語: {n}件',
    'glossary_desc':'人物名・専門用語がページごとに変わる問題を防ぎます。\n\n**① 自動抽出:** ボタンを押すとAIがファイルから重要用語を提案します。\n\n**② 手動入力:** `原文: 訳語` 形式で1行ずつ入力してください。\n\n(例: `James: ジェームズ`, `Eldoria: エルドリア`)',
    'btn_translate':'翻訳を実行',
    'download_label':'翻訳結果ダウンロード',
    'ui_lang_label':'UI言語',
    'ui_lang_restart':'UI言語が変更されました。アプリを再起動してください。',
    'status_detecting':"🔍 言語を検出中...",
    'status_ready':"翻訳準備完了。\n上の「翻訳を実行」ボタンをクリックしてください。",
    'status_detected':"{lang}の文書が検出されました。翻訳先言語を選んで翻訳を開始してください。",
    'status_image_only':"画像のみのファイルです。確認してください！",
    'err_file_none':"翻訳するファイルを追加してください",
    'err_file_limit':"一度に{n}個以上のファイルは添付できません。再度添付してください。",
    'err_file_count':"一度に{n}個以上のファイルは翻訳できません。",
    'err_lang_same':"原文言語と翻訳先言語が同じです（{lang}）。<br>別の翻訳先言語を選択してください。",
    'err_lang_detect':"言語検出が完了していません。<br>ファイルを再添付して言語確認を完了してください。",
    'err_server':"[エラー] 翻訳サーバー({url})に接続できません。<br>{guide}",
    'server_guide_mac':"Mac: <code>start_mac.sh</code>が実行中か確認してください。",
    'server_guide_linux':"Linux: <code>start_ubuntu.sh</code>またはvLLMサーバーの起動を確認してください。",
    'server_guide_windows':"Windows: Ollamaが実行中か確認してください。（<code>ollama serve</code>）",
    'server_guide_default':"翻訳サーバーの起動を確認してください。",
    'err_upload_detect':"ファイルの言語検出に失敗しました。",
    'err_size_exceeded':"ファイルサイズ制限を超えています。",
    'translation_complete':"翻訳完了！所要時間: {t} 下でダウンロードしてください。",
    'progress_init':"翻訳モデルを準備中...",
    'progress_server':"翻訳サーバーの状態確認中...",
    'progress_files':'ファイル読込',
    'lang_unknown':'不明',
    'glossary_applied':'✅ **{n}件の用語が適用されました。** 翻訳時にこれらが優先されます。',
    'glossary_empty':'⚠️ 適用された用語がありません。`原文: 訳語` 形式で入力してください。',
    'glossary_cleared':'用語集をリセットしました。',
},
'zh': {
    'app_title': "AI多语言翻译器 <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>Dodari2</a></span>",
    'step1':'步骤 1','step2':'步骤 2','step3':'步骤 3','step4':'步骤 4','status_tab':'状态',
    'step1_title':'1. 选择要翻译的文件',
    'files_label':'文件',
    'file_limit':'每次最多可添加{n}个文件',
    'origin_lang_label':'原始语言（自动检测 · 可手动更改）',
    'target_lang_label':'目标语言',
    'engine_ollama':'✔ Ollama翻译引擎已激活',
    'engine_gemma':'✔ 正在使用Gemma 4 API翻译',
    'model_label':"模型选择（E4B：16GB以下高速·31B：32GB以上高质量，切换时重启服务器）",
    'bilingual_label':"双语显示方式（学习者推荐：'原文（译文）'）",
    'genre_label':'类型指定（AI自动推断）',
    'tone_label':'文体选择（保持一致风格）',
    'genre_0':'IT与工程','genre_1':'文学与小说','genre_2':'人文与社会科学',
    'genre_3':'商业与经济','genre_4':'影视与剧本','genre_5':'一般文档（默认）',
    'tone_0':'叙述体','tone_1':'正式体',
    'bilingual_0':'译文（原文）','bilingual_1':'原文（译文）',
    'glossary_title':'✨ 术语表',
    'btn_glossary_extract':'🔍 AI自动提取术语',
    'glossary_label':'术语表（原文: 译文格式，每行一条）',
    'glossary_placeholder':'James: 詹姆斯\nEldoria: 埃尔多利亚\nDark Magic: 黑暗魔法',
    'btn_glossary_apply':'✅ 应用术语表',
    'btn_glossary_clear':'🗑️ 清空术语表',
    'glossary_count':'已应用术语: {n}条',
    'glossary_desc':'防止人物名称和专业术语在不同页面出现差异。\n\n**① 自动提取:** 点击按钮，AI将从文件中提取重要术语并提供建议。\n\n**② 手动输入:** 以`原文: 译文`格式逐行输入。\n\n(例: `James: 詹姆斯`, `Eldoria: 埃尔多利亚`)',
    'btn_translate':'开始翻译',
    'download_label':'下载翻译结果',
    'ui_lang_label':'界面语言',
    'ui_lang_restart':'界面语言已更改。请重启应用。',
    'status_detecting':"🔍 正在检测语言...",
    'status_ready':"翻译准备完毕。\n请点击上方的「开始翻译」按钮。",
    'status_detected':"检测到{lang}文档。请选择目标语言并开始翻译。",
    'status_image_only':"该文件仅包含图片。请确认！",
    'err_file_none':"请添加要翻译的文件",
    'err_file_limit':"每次不能添加超过{n}个文件，请重新添加。",
    'err_file_count':"每次不能翻译超过{n}个文件。",
    'err_lang_same':"原始语言和目标语言相同（{lang}）。<br>请选择不同的目标语言。",
    'err_lang_detect':"语言检测未完成。<br>请重新添加文件并等待语言检测完成。",
    'err_server':"[错误] 无法连接到翻译服务器({url})。<br>{guide}",
    'server_guide_mac':"Mac: 请检查<code>start_mac.sh</code>是否正在运行。",
    'server_guide_linux':"Linux: 请检查<code>start_ubuntu.sh</code>或vLLM服务器是否正在运行。",
    'server_guide_windows':"Windows: 请检查Ollama是否正在运行。（<code>ollama serve</code>）",
    'server_guide_default':"请检查翻译服务器是否正在运行。",
    'err_upload_detect':"无法检测文件语言。",
    'err_size_exceeded':"超出文件大小限制。",
    'translation_complete':"翻译完成！耗时：{t} 请在下方下载结果。",
    'progress_init':"正在准备翻译模型...",
    'progress_server':"正在检查翻译服务器状态...",
    'progress_files':'加载文件',
    'lang_unknown':'未知',
    'glossary_applied':'✅ **已应用{n}条术语。** 翻译时将优先使用这些术语。',
    'glossary_empty':'⚠️ 未应用任何术语。请使用`原文: 译文`格式输入。',
    'glossary_cleared':'术语表已清空。',
},
'fr': {
    'app_title': "Traducteur multilingue IA <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>Dodari2</a></span>",
    'step1':'Étape 1','step2':'Étape 2','step3':'Étape 3','step4':'Étape 4','status_tab':'Statut',
    'step1_title':'1. Sélectionner les fichiers à traduire',
    'files_label':'Fichiers',
    'file_limit':'Vous pouvez joindre jusqu\'à {n} fichiers à la fois',
    'origin_lang_label':'Langue source (détection auto · modification manuelle possible)',
    'target_lang_label':'Langue cible',
    'engine_ollama':'✔ Moteur de traduction Ollama actif',
    'engine_gemma':'✔ Traduction API Gemma 4 active',
    'model_label':"Sélection du modèle (E4B : rapide ≤16GB · 31B : haute qualité ≥32GB, redémarrage serveur au changement)",
    'bilingual_label':"Mode bilingue (pour apprenants : 'Original (Traduction)' recommandé)",
    'genre_label':'Genre (inféré automatiquement par IA)',
    'tone_label':'Style (maintien d\'un style cohérent)',
    'genre_0':'Informatique & Ingénierie','genre_1':'Littérature & Fiction','genre_2':'Sciences humaines & sociales',
    'genre_3':'Commerce & Économie','genre_4':'Film & Scénario','genre_5':'Document général (défaut)',
    'tone_0':'Narratif (courant)','tone_1':'Formel (soutenu)',
    'bilingual_0':'Traduction (Original)','bilingual_1':'Original (Traduction)',
    'glossary_title':'✨ Glossaire',
    'btn_glossary_extract':'🔍 Extraction auto par IA',
    'glossary_label':'Glossaire (format source: traduction, un par ligne)',
    'glossary_placeholder':'James: James\nEldoria: Eldoria\nDark Magic: Magie Noire',
    'btn_glossary_apply':'✅ Appliquer le glossaire',
    'btn_glossary_clear':'🗑️ Effacer le glossaire',
    'glossary_count':'Termes appliqués : {n}',
    'glossary_desc':'Évite que les noms de personnages et termes techniques varient d\'une page à l\'autre.\n\n**① Extraction auto :** L\'IA scanne le fichier et suggère les termes clés.\n\n**② Saisie manuelle :** Entrez les termes au format `source: traduction`, un par ligne.\n\n(ex : `James: James`, `Eldoria: Eldoria`)',
    'btn_translate':'Lancer la traduction',
    'download_label':'Télécharger les résultats',
    'ui_lang_label':'Langue de l\'interface',
    'ui_lang_restart':'Langue de l\'interface modifiée. Veuillez redémarrer l\'application.',
    'status_detecting':"🔍 Détection de la langue...",
    'status_ready':"Prêt à traduire.\nCliquez sur « Lancer la traduction » ci-dessus.",
    'status_detected':"Document {lang} détecté. Sélectionnez la langue cible et lancez la traduction.",
    'status_image_only':"Ce fichier ne contient que des images. Veuillez vérifier !",
    'err_file_none':"Veuillez ajouter un fichier à traduire",
    'err_file_limit':"Impossible de joindre plus de {n} fichiers. Veuillez les re-joindre.",
    'err_file_count':"Impossible de traduire plus de {n} fichiers à la fois.",
    'err_lang_same':"La langue source et la langue cible sont identiques ({lang}).<br>Veuillez sélectionner une langue cible différente.",
    'err_lang_detect':"Détection de langue incomplète.<br>Veuillez re-joindre le fichier et attendre la détection.",
    'err_server':"[Erreur] Impossible de se connecter au serveur ({url}).<br>{guide}",
    'server_guide_mac':"Mac : Vérifiez que <code>start_mac.sh</code> est en cours d'exécution.",
    'server_guide_linux':"Linux : Vérifiez que <code>start_ubuntu.sh</code> ou le serveur vLLM est lancé.",
    'server_guide_windows':"Windows : Vérifiez qu'Ollama est en cours d'exécution. (<code>ollama serve</code>)",
    'server_guide_default':"Vérifiez que le serveur de traduction est en cours d'exécution.",
    'err_upload_detect':"Échec de la détection de langue.",
    'err_size_exceeded':"Taille de fichier dépassée.",
    'translation_complete':"Traduction terminée ! Durée : {t} Téléchargez les résultats ci-dessous.",
    'progress_init':"Préparation du modèle de traduction...",
    'progress_server':"Vérification du serveur de traduction...",
    'progress_files':'Chargement des fichiers',
    'lang_unknown':'Inconnu',
    'glossary_applied':'✅ **{n} termes appliqués.** Ils seront prioritaires lors de la traduction.',
    'glossary_empty':'⚠️ Aucun terme appliqué. Utilisez le format `source: traduction`.',
    'glossary_cleared':'Glossaire effacé.',
},
'it': {
    'app_title': "Traduttore multilingue IA <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>Dodari2</a></span>",
    'step1':'Fase 1','step2':'Fase 2','step3':'Fase 3','step4':'Fase 4','status_tab':'Stato',
    'step1_title':'1. Seleziona i file da tradurre',
    'files_label':'File',
    'file_limit':'Puoi allegare fino a {n} file alla volta',
    'origin_lang_label':'Lingua sorgente (rilevamento auto · modifica manuale possibile)',
    'target_lang_label':'Lingua di destinazione',
    'engine_ollama':'✔ Motore di traduzione Ollama attivo',
    'engine_gemma':'✔ Traduzione API Gemma 4 attiva',
    'model_label':"Selezione modello (E4B: veloce ≤16GB · 31B: alta qualità ≥32GB, riavvio server al cambio)",
    'bilingual_label':"Modalità bilingue (per studenti: 'Originale (Traduzione)' consigliato)",
    'genre_label':'Genere (inferito automaticamente dall\'IA)',
    'tone_label':'Stile (stile coerente mantenuto)',
    'genre_0':'IT e Ingegneria','genre_1':'Letteratura e Narrativa','genre_2':'Scienze Umane e Sociali',
    'genre_3':'Business ed Economia','genre_4':'Film e Sceneggiatura','genre_5':'Documento Generale (predefinito)',
    'tone_0':'Narrativo (corrente)','tone_1':'Formale (sostenuto)',
    'bilingual_0':'Traduzione (Originale)','bilingual_1':'Originale (Traduzione)',
    'glossary_title':'✨ Glossario',
    'btn_glossary_extract':'🔍 Estrazione auto termini IA',
    'glossary_label':'Glossario (formato sorgente: traduzione, uno per riga)',
    'glossary_placeholder':'James: James\nEldoria: Eldoria\nDark Magic: Magia Oscura',
    'btn_glossary_apply':'✅ Applica glossario',
    'btn_glossary_clear':'🗑️ Cancella glossario',
    'glossary_count':'Termini applicati: {n}',
    'glossary_desc':'Evita che nomi di personaggi e termini tecnici varino da pagina a pagina.\n\n**① Estrazione auto:** L\'IA scansiona il file e suggerisce termini chiave.\n\n**② Inserimento manuale:** Inserisci termini nel formato `sorgente: traduzione`, uno per riga.\n\n(es: `James: James`, `Eldoria: Eldoria`)',
    'btn_translate':'Avvia traduzione',
    'download_label':'Scarica risultati',
    'ui_lang_label':'Lingua interfaccia',
    'ui_lang_restart':'Lingua interfaccia modificata. Riavvia l\'applicazione.',
    'status_detecting':"🔍 Rilevamento lingua...",
    'status_ready':"Pronto per tradurre.\nClicca su «Avvia traduzione» in alto.",
    'status_detected':"Documento {lang} rilevato. Seleziona la lingua di destinazione e avvia la traduzione.",
    'status_image_only':"Il file contiene solo immagini. Verificare!",
    'err_file_none':"Aggiungi un file da tradurre",
    'err_file_limit':"Non è possibile allegare più di {n} file. Riallega i file.",
    'err_file_count':"Non è possibile tradurre più di {n} file alla volta.",
    'err_lang_same':"La lingua sorgente e di destinazione sono uguali ({lang}).<br>Seleziona una lingua di destinazione diversa.",
    'err_lang_detect':"Rilevamento lingua non completato.<br>Riallega il file e attendi il rilevamento.",
    'err_server':"[Errore] Impossibile connettersi al server di traduzione ({url}).<br>{guide}",
    'server_guide_mac':"Mac: Verifica che <code>start_mac.sh</code> sia in esecuzione.",
    'server_guide_linux':"Linux: Verifica che <code>start_ubuntu.sh</code> o il server vLLM sia in esecuzione.",
    'server_guide_windows':"Windows: Verifica che Ollama sia in esecuzione. (<code>ollama serve</code>)",
    'server_guide_default':"Verifica che il server di traduzione sia in esecuzione.",
    'err_upload_detect':"Rilevamento lingua del file fallito.",
    'err_size_exceeded':"Dimensione file superata.",
    'translation_complete':"Traduzione completata! Tempo impiegato: {t} Scarica i risultati qui sotto.",
    'progress_init':"Preparazione modello di traduzione...",
    'progress_server':"Verifica stato server di traduzione...",
    'progress_files':'Caricamento file',
    'lang_unknown':'Sconosciuto',
    'glossary_applied':'✅ **{n} termini applicati.** Saranno prioritari durante la traduzione.',
    'glossary_empty':'⚠️ Nessun termine applicato. Usa il formato `sorgente: traduzione`.',
    'glossary_cleared':'Glossario cancellato.',
},
'nl': {
    'app_title': "AI meertalige vertaler <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>Dodari2</a></span>",
    'step1':'Stap 1','step2':'Stap 2','step3':'Stap 3','step4':'Stap 4','status_tab':'Status',
    'step1_title':'1. Selecteer te vertalen bestanden',
    'files_label':'Bestanden',
    'file_limit':'U kunt maximaal {n} bestanden tegelijk toevoegen',
    'origin_lang_label':'Brontaal (automatisch gedetecteerd · handmatig aanpasbaar)',
    'target_lang_label':'Doeltaal',
    'engine_ollama':'✔ Ollama vertaalmachine actief',
    'engine_gemma':'✔ Gemma 4 API vertaling actief',
    'model_label':"Modelselectie (E4B: snel ≤16GB · 31B: hoge kwaliteit ≥32GB, server herstart bij wisseling)",
    'bilingual_label':"Tweetalige weergave (voor leerlingen: 'Origineel (Vertaling)' aanbevolen)",
    'genre_label':'Genre (automatisch afgeleid door AI)',
    'tone_label':'Stijl (consistente stijl behouden)',
    'genre_0':'IT & Engineering','genre_1':'Literatuur & Fictie','genre_2':'Humaniora & Sociale Wetenschappen',
    'genre_3':'Zakelijk & Economie','genre_4':'Film & Script','genre_5':'Algemeen document (standaard)',
    'tone_0':'Verhalend (gewoon)','tone_1':'Formeel (beleefd)',
    'bilingual_0':'Vertaling (Origineel)','bilingual_1':'Origineel (Vertaling)',
    'glossary_title':'✨ Woordenlijst',
    'btn_glossary_extract':'🔍 AI automatische termijnextractie',
    'glossary_label':'Woordenlijst (bron: vertaling formaat, één per regel)',
    'glossary_placeholder':'James: James\nEldoria: Eldoria\nDark Magic: Donkere Magie',
    'btn_glossary_apply':'✅ Woordenlijst toepassen',
    'btn_glossary_clear':'🗑️ Woordenlijst wissen',
    'glossary_count':'Toegepaste termen: {n}',
    'glossary_desc':'Voorkomt dat namen van personages en technische termen per pagina verschillen.\n\n**① Automatische extractie:** AI scant het bestand en stelt sleuteltermen voor.\n\n**② Handmatige invoer:** Voer termen in het formaat `bron: vertaling` in, één per regel.\n\n(bijv. `James: James`, `Eldoria: Eldoria`)',
    'btn_translate':'Vertaling starten',
    'download_label':'Resultaten downloaden',
    'ui_lang_label':'Interfacetaal',
    'ui_lang_restart':'Interfacetaal gewijzigd. Start de app opnieuw op.',
    'status_detecting':"🔍 Taal detecteren...",
    'status_ready':"Klaar voor vertaling.\nKlik op 'Vertaling starten' hierboven.",
    'status_detected':"{lang} document gedetecteerd. Selecteer doeltaal en start de vertaling.",
    'status_image_only':"Dit bestand bevat alleen afbeeldingen. Controleer dit!",
    'err_file_none':"Voeg een te vertalen bestand toe",
    'err_file_limit':"Kan niet meer dan {n} bestanden toevoegen. Voeg opnieuw toe.",
    'err_file_count':"Kan niet meer dan {n} bestanden tegelijk vertalen.",
    'err_lang_same':"Bron- en doeltaal zijn gelijk ({lang}).<br>Selecteer een andere doeltaal.",
    'err_lang_detect':"Taaldetectie niet voltooid.<br>Voeg het bestand opnieuw toe en wacht op detectie.",
    'err_server':"[Fout] Kan geen verbinding maken met vertaalserver ({url}).<br>{guide}",
    'server_guide_mac':"Mac: Controleer of <code>start_mac.sh</code> actief is.",
    'server_guide_linux':"Linux: Controleer of <code>start_ubuntu.sh</code> of de vLLM-server actief is.",
    'server_guide_windows':"Windows: Controleer of Ollama actief is. (<code>ollama serve</code>)",
    'server_guide_default':"Controleer of de vertaalserver actief is.",
    'err_upload_detect':"Taaldetectie van het bestand mislukt.",
    'err_size_exceeded':"Bestandsgrootte overschreden.",
    'translation_complete':"Vertaling voltooid! Verstreken tijd: {t} Download de resultaten hieronder.",
    'progress_init':"Vertaalmodel voorbereiden...",
    'progress_server':"Status vertaalserver controleren...",
    'progress_files':'Bestanden laden',
    'lang_unknown':'Onbekend',
    'glossary_applied':'✅ **{n} termen toegepast.** Deze hebben prioriteit bij vertaling.',
    'glossary_empty':'⚠️ Geen termen toegepast. Gebruik het formaat `bron: vertaling`.',
    'glossary_cleared':'Woordenlijst gewist.',
},
'da': {
    'app_title': "AI flersproget oversætter <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>Dodari2</a></span>",
    'step1':'Trin 1','step2':'Trin 2','step3':'Trin 3','step4':'Trin 4','status_tab':'Status',
    'step1_title':'1. Vælg filer til oversættelse',
    'files_label':'Filer',
    'file_limit':'Du kan vedhæfte op til {n} filer ad gangen',
    'origin_lang_label':'Kildesprog (automatisk registreret · manuelt redigerbart)',
    'target_lang_label':'Målsprog',
    'engine_ollama':'✔ Ollama oversættelsesmotor aktiv',
    'engine_gemma':'✔ Gemma 4 API oversættelse aktiv',
    'model_label':"Modelvalg (E4B: hurtig ≤16GB · 31B: høj kvalitet ≥32GB, servergenstart ved skift)",
    'bilingual_label':"Tosproget visningstilstand (for lærende: 'Original (Oversættelse)' anbefales)",
    'genre_label':'Genre (automatisk udledt af AI)',
    'tone_label':'Stil (konsistent stil opretholdt)',
    'genre_0':'IT & Ingeniørvidenskab','genre_1':'Litteratur & Fiktion','genre_2':'Humaniora & Samfundsvidenskab',
    'genre_3':'Business & Økonomi','genre_4':'Film & Manuskript','genre_5':'Generelt dokument (standard)',
    'tone_0':'Fortællende (almindelig)','tone_1':'Formel (høflig)',
    'bilingual_0':'Oversættelse (Original)','bilingual_1':'Original (Oversættelse)',
    'glossary_title':'✨ Ordliste',
    'btn_glossary_extract':'🔍 AI automatisk termudtrækning',
    'glossary_label':'Ordliste (kilde: oversættelse format, én per linje)',
    'glossary_placeholder':'James: James\nEldoria: Eldoria\nDark Magic: Mørk Magi',
    'btn_glossary_apply':'✅ Anvend ordliste',
    'btn_glossary_clear':'🗑️ Ryd ordliste',
    'glossary_count':'Anvendte termer: {n}',
    'glossary_desc':'Forhindrer at personnavne og faglige termer varierer fra side til side.\n\n**① Automatisk udtrækning:** AI scanner filen og foreslår nøgletermer.\n\n**② Manuel indtastning:** Indtast termer i formatet `kilde: oversættelse`, én per linje.\n\n(f.eks. `James: James`, `Eldoria: Eldoria`)',
    'btn_translate':'Start oversættelse',
    'download_label':'Download resultater',
    'ui_lang_label':'Grænsefladesprog',
    'ui_lang_restart':'Grænsefladesprog ændret. Genstart venligst appen.',
    'status_detecting':"🔍 Registrerer sprog...",
    'status_ready':"Klar til oversættelse.\nKlik på 'Start oversættelse' ovenfor.",
    'status_detected':"{lang}-dokument registreret. Vælg målsprog og start oversættelse.",
    'status_image_only':"Denne fil indeholder kun billeder. Kontroller venligst!",
    'err_file_none':"Tilføj en fil til oversættelse",
    'err_file_limit':"Kan ikke vedhæfte mere end {n} filer. Vedhæft igen.",
    'err_file_count':"Kan ikke oversætte mere end {n} filer ad gangen.",
    'err_lang_same':"Kilde- og målsprog er ens ({lang}).<br>Vælg et andet målsprog.",
    'err_lang_detect':"Sprogregistrering ikke fuldført.<br>Vedhæft filen igen og vent på registrering.",
    'err_server':"[Fejl] Kan ikke oprette forbindelse til oversættelsesserveren ({url}).<br>{guide}",
    'server_guide_mac':"Mac: Kontroller at <code>start_mac.sh</code> kører.",
    'server_guide_linux':"Linux: Kontroller at <code>start_ubuntu.sh</code> eller vLLM-serveren kører.",
    'server_guide_windows':"Windows: Kontroller at Ollama kører. (<code>ollama serve</code>)",
    'server_guide_default':"Kontroller at oversættelsesserveren kører.",
    'err_upload_detect':"Sprogregistrering af filen mislykkedes.",
    'err_size_exceeded':"Filstørrelse overskredet.",
    'translation_complete':"Oversættelse fuldført! Tid brugt: {t} Download resultaterne nedenfor.",
    'progress_init':"Forbereder oversættelsesmodel...",
    'progress_server':"Kontrollerer oversættelsesserverstatus...",
    'progress_files':'Indlæser filer',
    'lang_unknown':'Ukendt',
    'glossary_applied':'✅ **{n} termer anvendt.** Disse vil have prioritet under oversættelse.',
    'glossary_empty':'⚠️ Ingen termer anvendt. Brug formatet `kilde: oversættelse`.',
    'glossary_cleared':'Ordliste ryddet.',
},
'sv': {
    'app_title': "AI flerspråkig översättare <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>Dodari2</a></span>",
    'step1':'Steg 1','step2':'Steg 2','step3':'Steg 3','step4':'Steg 4','status_tab':'Status',
    'step1_title':'1. Välj filer att översätta',
    'files_label':'Filer',
    'file_limit':'Du kan bifoga upp till {n} filer åt gången',
    'origin_lang_label':'Källspråk (automatiskt detekterat · manuellt ändringsbart)',
    'target_lang_label':'Målspråk',
    'engine_ollama':'✔ Ollama översättningsmotor aktiv',
    'engine_gemma':'✔ Gemma 4 API-översättning aktiv',
    'model_label':"Modellval (E4B: snabb ≤16GB · 31B: hög kvalitet ≥32GB, serveromstart vid byte)",
    'bilingual_label':"Tvåspråkigt visningsläge (för studerande: 'Original (Översättning)' rekommenderas)",
    'genre_label':'Genre (automatiskt härledd av AI)',
    'tone_label':'Stil (konsekvent stil bibehålls)',
    'genre_0':'IT & Teknik','genre_1':'Litteratur & Fiktion','genre_2':'Humaniora & Samhällsvetenskap',
    'genre_3':'Affärer & Ekonomi','genre_4':'Film & Manus','genre_5':'Allmänt dokument (standard)',
    'tone_0':'Berättande (vardaglig)','tone_1':'Formell (artig)',
    'bilingual_0':'Översättning (Original)','bilingual_1':'Original (Översättning)',
    'glossary_title':'✨ Ordlista',
    'btn_glossary_extract':'🔍 AI automatisk termextraktion',
    'glossary_label':'Ordlista (källterm: översättning format, en per rad)',
    'glossary_placeholder':'James: James\nEldoria: Eldoria\nDark Magic: Mörk Magi',
    'btn_glossary_apply':'✅ Tillämpa ordlista',
    'btn_glossary_clear':'🗑️ Rensa ordlista',
    'glossary_count':'Tillämpade termer: {n}',
    'glossary_desc':'Förhindrar att personnamn och facktermer varierar från sida till sida.\n\n**① Automatisk extraktion:** AI skannar filen och föreslår nyckeltermer.\n\n**② Manuell inmatning:** Ange termer i formatet `källterm: översättning`, en per rad.\n\n(t.ex. `James: James`, `Eldoria: Eldoria`)',
    'btn_translate':'Starta översättning',
    'download_label':'Ladda ner resultat',
    'ui_lang_label':'Gränssnittsspråk',
    'ui_lang_restart':'Gränssnittsspråket har ändrats. Starta om appen.',
    'status_detecting':"🔍 Detekterar språk...",
    'status_ready':"Redo att översätta.\nKlicka på 'Starta översättning' ovan.",
    'status_detected':"{lang}-dokument detekterat. Välj målspråk och starta översättning.",
    'status_image_only':"Den här filen innehåller bara bilder. Kontrollera!",
    'err_file_none':"Lägg till en fil att översätta",
    'err_file_limit':"Kan inte bifoga mer än {n} filer. Bifoga igen.",
    'err_file_count':"Kan inte översätta mer än {n} filer åt gången.",
    'err_lang_same':"Käll- och målspråk är samma ({lang}).<br>Välj ett annat målspråk.",
    'err_lang_detect':"Språkdetektering inte slutförd.<br>Bifoga filen igen och vänta på detektering.",
    'err_server':"[Fel] Kan inte ansluta till översättningsservern ({url}).<br>{guide}",
    'server_guide_mac':"Mac: Kontrollera att <code>start_mac.sh</code> körs.",
    'server_guide_linux':"Linux: Kontrollera att <code>start_ubuntu.sh</code> eller vLLM-servern körs.",
    'server_guide_windows':"Windows: Kontrollera att Ollama körs. (<code>ollama serve</code>)",
    'server_guide_default':"Kontrollera att översättningsservern körs.",
    'err_upload_detect':"Språkdetektering av filen misslyckades.",
    'err_size_exceeded':"Filstorleksgräns överskriden.",
    'translation_complete':"Översättning klar! Tid förfluten: {t} Ladda ned resultaten nedan.",
    'progress_init':"Förbereder översättningsmodell...",
    'progress_server':"Kontrollerar översättningsserverns status...",
    'progress_files':'Laddar filer',
    'lang_unknown':'Okänt',
    'glossary_applied':'✅ **{n} termer tillämpade.** Dessa prioriteras vid översättning.',
    'glossary_empty':'⚠️ Inga termer tillämpade. Använd formatet `källterm: översättning`.',
    'glossary_cleared':'Ordlista rensad.',
},
'no': {
    'app_title': "AI flerspråklig oversetter <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>Dodari2</a></span>",
    'step1':'Trinn 1','step2':'Trinn 2','step3':'Trinn 3','step4':'Trinn 4','status_tab':'Status',
    'step1_title':'1. Velg filer som skal oversettes',
    'files_label':'Filer',
    'file_limit':'Du kan legge ved opptil {n} filer om gangen',
    'origin_lang_label':'Kildespråk (automatisk oppdaget · manuelt endringsbart)',
    'target_lang_label':'Målspråk',
    'engine_ollama':'✔ Ollama oversettelsesmotor aktiv',
    'engine_gemma':'✔ Gemma 4 API-oversettelse aktiv',
    'model_label':"Modellvalg (E4B: rask ≤16GB · 31B: høy kvalitet ≥32GB, serveromstart ved bytte)",
    'bilingual_label':"Tospråklig visningsmodus (for elever: 'Original (Oversettelse)' anbefales)",
    'genre_label':'Sjanger (automatisk utledet av AI)',
    'tone_label':'Stil (konsekvent stil opprettholdt)',
    'genre_0':'IT & Ingeniørfag','genre_1':'Litteratur & Fiksjon','genre_2':'Humaniora & Samfunnsvitenskap',
    'genre_3':'Forretning & Økonomi','genre_4':'Film & Manus','genre_5':'Generelt dokument (standard)',
    'tone_0':'Fortellende (vanlig)','tone_1':'Formell (høflig)',
    'bilingual_0':'Oversettelse (Original)','bilingual_1':'Original (Oversettelse)',
    'glossary_title':'✨ Ordliste',
    'btn_glossary_extract':'🔍 AI automatisk termuttrekking',
    'glossary_label':'Ordliste (kilde: oversettelse format, én per linje)',
    'glossary_placeholder':'James: James\nEldoria: Eldoria\nDark Magic: Mørk Magi',
    'btn_glossary_apply':'✅ Bruk ordliste',
    'btn_glossary_clear':'🗑️ Tøm ordliste',
    'glossary_count':'Brukte termer: {n}',
    'glossary_desc':'Forhindrer at personnavn og faglige termer varierer fra side til side.\n\n**① Automatisk uttrekking:** AI skanner filen og foreslår nøkkeltermer.\n\n**② Manuell innføring:** Skriv inn termer i formatet `kilde: oversettelse`, én per linje.\n\n(f.eks. `James: James`, `Eldoria: Eldoria`)',
    'btn_translate':'Start oversettelse',
    'download_label':'Last ned resultater',
    'ui_lang_label':'Grensesnittspråk',
    'ui_lang_restart':'Grensesnittspråket er endret. Start appen på nytt.',
    'status_detecting':"🔍 Oppdager språk...",
    'status_ready':"Klar til oversettelse.\nKlikk på 'Start oversettelse' ovenfor.",
    'status_detected':"{lang}-dokument oppdaget. Velg målspråk og start oversettelse.",
    'status_image_only':"Denne filen inneholder bare bilder. Kontroller!",
    'err_file_none':"Legg til en fil for oversettelse",
    'err_file_limit':"Kan ikke legge ved mer enn {n} filer. Legg ved på nytt.",
    'err_file_count':"Kan ikke oversette mer enn {n} filer om gangen.",
    'err_lang_same':"Kilde- og målspråk er det samme ({lang}).<br>Velg et annet målspråk.",
    'err_lang_detect':"Språkoppdagelse ikke fullført.<br>Legg ved filen på nytt og vent på oppdagelse.",
    'err_server':"[Feil] Kan ikke koble til oversettelsesserveren ({url}).<br>{guide}",
    'server_guide_mac':"Mac: Kontroller at <code>start_mac.sh</code> kjører.",
    'server_guide_linux':"Linux: Kontroller at <code>start_ubuntu.sh</code> eller vLLM-serveren kjører.",
    'server_guide_windows':"Windows: Kontroller at Ollama kjører. (<code>ollama serve</code>)",
    'server_guide_default':"Kontroller at oversettelsesserveren kjører.",
    'err_upload_detect':"Språkoppdagelse av filen mislyktes.",
    'err_size_exceeded':"Filstørrelsesbegrensning overskredet.",
    'translation_complete':"Oversettelse fullført! Tid brukt: {t} Last ned resultater nedenfor.",
    'progress_init':"Forbereder oversettelsesmodell...",
    'progress_server':"Kontrollerer oversettelsesserverstatus...",
    'progress_files':'Laster filer',
    'lang_unknown':'Ukjent',
    'glossary_applied':'✅ **{n} termer brukt.** Disse vil ha prioritet under oversettelse.',
    'glossary_empty':'⚠️ Ingen termer brukt. Bruk formatet `kilde: oversettelse`.',
    'glossary_cleared':'Ordliste tømt.',
},
'ar': {
    'app_title': "مترجم متعدد اللغات بالذكاء الاصطناعي <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>Dodari2</a></span>",
    'step1':'الخطوة 1','step2':'الخطوة 2','step3':'الخطوة 3','step4':'الخطوة 4','status_tab':'الحالة',
    'step1_title':'1. اختر الملفات للترجمة',
    'files_label':'الملفات',
    'file_limit':'يمكنك إرفاق حتى {n} ملفات في المرة الواحدة',
    'origin_lang_label':'لغة المصدر (تحديد تلقائي · تغيير يدوي ممكن)',
    'target_lang_label':'لغة الهدف',
    'engine_ollama':'✔ محرك ترجمة Ollama نشط',
    'engine_gemma':'✔ ترجمة Gemma 4 API نشطة',
    'model_label':"اختيار النموذج (E4B: سريع ≤16GB · 31B: جودة عالية ≥32GB، إعادة تشغيل الخادم عند التبديل)",
    'bilingual_label':"وضع العرض ثنائي اللغة (للمتعلمين: يُنصح بـ 'الأصل (الترجمة)')",
    'genre_label':'النوع الأدبي (يُستنتج تلقائياً بالذكاء الاصطناعي)',
    'tone_label':'الأسلوب (الحفاظ على أسلوب متسق)',
    'genre_0':'تكنولوجيا المعلومات والهندسة','genre_1':'الأدب والروايات','genre_2':'الإنسانيات والعلوم الاجتماعية',
    'genre_3':'الأعمال والاقتصاد','genre_4':'الأفلام والنصوص','genre_5':'وثيقة عامة (افتراضي)',
    'tone_0':'سردي (عادي)','tone_1':'رسمي (مهذب)',
    'bilingual_0':'الترجمة (الأصل)','bilingual_1':'الأصل (الترجمة)',
    'glossary_title':'✨ قاموس المصطلحات',
    'btn_glossary_extract':'🔍 استخراج تلقائي للمصطلحات بالذكاء الاصطناعي',
    'glossary_label':'قاموس المصطلحات (تنسيق المصدر: الترجمة، سطر واحد لكل مصطلح)',
    'glossary_placeholder':'James: جيمس\nEldoria: إلدوريا\nDark Magic: السحر الأسود',
    'btn_glossary_apply':'✅ تطبيق قاموس المصطلحات',
    'btn_glossary_clear':'🗑️ مسح قاموس المصطلحات',
    'glossary_count':'المصطلحات المطبقة: {n}',
    'glossary_desc':'يمنع تغير أسماء الشخصيات والمصطلحات التقنية من صفحة لأخرى.\n\n**① الاستخراج التلقائي:** يقوم الذكاء الاصطناعي بمسح الملف واقتراح المصطلحات الرئيسية.\n\n**② الإدخال اليدوي:** أدخل المصطلحات بتنسيق `المصدر: الترجمة`، سطر واحد لكل مصطلح.\n\n(مثال: `James: جيمس`، `Eldoria: إلدوريا`)',
    'btn_translate':'بدء الترجمة',
    'download_label':'تنزيل النتائج',
    'ui_lang_label':'لغة الواجهة',
    'ui_lang_restart':'تم تغيير لغة الواجهة. يرجى إعادة تشغيل التطبيق.',
    'status_detecting':"🔍 جارٍ اكتشاف اللغة...",
    'status_ready':"جاهز للترجمة.\nانقر على 'بدء الترجمة' في الأعلى.",
    'status_detected':"تم اكتشاف وثيقة {lang}. اختر لغة الهدف وابدأ الترجمة.",
    'status_image_only':"هذا الملف يحتوي على صور فقط. يرجى التحقق!",
    'err_file_none':"أضف ملفاً للترجمة",
    'err_file_limit':"لا يمكن إرفاق أكثر من {n} ملفات. أعد الإرفاق.",
    'err_file_count':"لا يمكن ترجمة أكثر من {n} ملفات في المرة الواحدة.",
    'err_lang_same':"لغة المصدر والهدف متماثلتان ({lang}).<br>اختر لغة هدف مختلفة.",
    'err_lang_detect':"اكتشاف اللغة غير مكتمل.<br>أعد إرفاق الملف وانتظر اكتمال الاكتشاف.",
    'err_server':"[خطأ] لا يمكن الاتصال بخادم الترجمة ({url}).<br>{guide}",
    'server_guide_mac':"Mac: تحقق من تشغيل <code>start_mac.sh</code>.",
    'server_guide_linux':"Linux: تحقق من تشغيل <code>start_ubuntu.sh</code> أو خادم vLLM.",
    'server_guide_windows':"Windows: تحقق من تشغيل Ollama. (<code>ollama serve</code>)",
    'server_guide_default':"تحقق من تشغيل خادم الترجمة.",
    'err_upload_detect':"فشل اكتشاف لغة الملف.",
    'err_size_exceeded':"تم تجاوز حد حجم الملف.",
    'translation_complete':"اكتملت الترجمة! الوقت المستغرق: {t} قم بتنزيل النتائج أدناه.",
    'progress_init':"جارٍ تحضير نموذج الترجمة...",
    'progress_server':"جارٍ التحقق من حالة خادم الترجمة...",
    'progress_files':'جارٍ تحميل الملفات',
    'lang_unknown':'غير معروف',
    'glossary_applied':'✅ **تم تطبيق {n} مصطلح.** ستُعطى هذه الأولوية خلال الترجمة.',
    'glossary_empty':'⚠️ لم يتم تطبيق أي مصطلحات. استخدم تنسيق `المصدر: الترجمة`.',
    'glossary_cleared':'تم مسح قاموس المصطلحات.',
},
'fa': {
    'app_title': "مترجم چندزبانه هوش مصنوعی <span style='color:red;'><a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration:none;color:red;'>Dodari2</a></span>",
    'step1':'مرحله ۱','step2':'مرحله ۲','step3':'مرحله ۳','step4':'مرحله ۴','status_tab':'وضعیت',
    'step1_title':'۱. فایل‌های مورد نظر را انتخاب کنید',
    'files_label':'فایل‌ها',
    'file_limit':'می‌توانید تا {n} فایل را به یکباره پیوست کنید',
    'origin_lang_label':'زبان مبدا (تشخیص خودکار · تغییر دستی ممکن)',
    'target_lang_label':'زبان مقصد',
    'engine_ollama':'✔ موتور ترجمه Ollama فعال است',
    'engine_gemma':'✔ ترجمه Gemma 4 API فعال است',
    'model_label':"انتخاب مدل (E4B: سریع ≤16GB · 31B: کیفیت بالا ≥32GB، راه‌اندازی مجدد سرور هنگام تغییر)",
    'bilingual_label':"حالت نمایش دوزبانه (برای زبان‌آموزان: 'متن اصلی (ترجمه)' توصیه می‌شود)",
    'genre_label':'ژانر (استنتاج خودکار توسط هوش مصنوعی)',
    'tone_label':'سبک (حفظ سبک یکنواخت)',
    'genre_0':'فناوری اطلاعات و مهندسی','genre_1':'ادبیات و داستان','genre_2':'علوم انسانی و اجتماعی',
    'genre_3':'کسب‌وکار و اقتصاد','genre_4':'فیلم و فیلمنامه','genre_5':'سند عمومی (پیش‌فرض)',
    'tone_0':'روایی (معمولی)','tone_1':'رسمی (مؤدبانه)',
    'bilingual_0':'ترجمه (متن اصلی)','bilingual_1':'متن اصلی (ترجمه)',
    'glossary_title':'✨ واژه‌نامه',
    'btn_glossary_extract':'🔍 استخراج خودکار اصطلاحات توسط هوش مصنوعی',
    'glossary_label':'واژه‌نامه (فرمت مبدا: ترجمه، یک در هر خط)',
    'glossary_placeholder':'James: جیمز\nEldoria: الدوریا\nDark Magic: جادوی تاریک',
    'btn_glossary_apply':'✅ اعمال واژه‌نامه',
    'btn_glossary_clear':'🗑️ پاک کردن واژه‌نامه',
    'glossary_count':'اصطلاحات اعمال‌شده: {n}',
    'glossary_desc':'از تغییر نام شخصیت‌ها و اصطلاحات فنی در صفحات مختلف جلوگیری می‌کند.\n\n**① استخراج خودکار:** هوش مصنوعی فایل را اسکن می‌کند و اصطلاحات کلیدی را پیشنهاد می‌دهد.\n\n**② ورود دستی:** اصطلاحات را با فرمت `مبدا: ترجمه` وارد کنید، یک در هر خط.\n\n(مثال: `James: جیمز`، `Eldoria: الدوریا`)',
    'btn_translate':'شروع ترجمه',
    'download_label':'دانلود نتایج',
    'ui_lang_label':'زبان رابط کاربری',
    'ui_lang_restart':'زبان رابط کاربری تغییر یافت. لطفاً برنامه را مجدداً راه‌اندازی کنید.',
    'status_detecting':"🔍 در حال تشخیص زبان...",
    'status_ready':"آماده ترجمه.\nروی 'شروع ترجمه' در بالا کلیک کنید.",
    'status_detected':"سند {lang} شناسایی شد. زبان مقصد را انتخاب کرده و ترجمه را شروع کنید.",
    'status_image_only':"این فایل فقط شامل تصاویر است. لطفاً بررسی کنید!",
    'err_file_none':"یک فایل برای ترجمه اضافه کنید",
    'err_file_limit':"نمی‌توان بیش از {n} فایل پیوست کرد. دوباره پیوست کنید.",
    'err_file_count':"نمی‌توان بیش از {n} فایل را به یکباره ترجمه کرد.",
    'err_lang_same':"زبان مبدا و مقصد یکسان است ({lang}).<br>لطفاً زبان مقصد دیگری انتخاب کنید.",
    'err_lang_detect':"تشخیص زبان کامل نشده است.<br>فایل را دوباره پیوست کنید و منتظر تشخیص بمانید.",
    'err_server':"[خطا] اتصال به سرور ترجمه ({url}) امکان‌پذیر نیست.<br>{guide}",
    'server_guide_mac':"Mac: بررسی کنید <code>start_mac.sh</code> در حال اجراست.",
    'server_guide_linux':"Linux: بررسی کنید <code>start_ubuntu.sh</code> یا سرور vLLM در حال اجراست.",
    'server_guide_windows':"Windows: بررسی کنید Ollama در حال اجراست. (<code>ollama serve</code>)",
    'server_guide_default':"بررسی کنید سرور ترجمه در حال اجراست.",
    'err_upload_detect':"تشخیص زبان فایل ناموفق بود.",
    'err_size_exceeded':"محدودیت حجم فایل رد شد.",
    'translation_complete':"ترجمه کامل شد! زمان سپری‌شده: {t} نتایج را در زیر دانلود کنید.",
    'progress_init':"در حال آماده‌سازی مدل ترجمه...",
    'progress_server':"در حال بررسی وضعیت سرور ترجمه...",
    'progress_files':'در حال بارگذاری فایل‌ها',
    'lang_unknown':'ناشناخته',
    'glossary_applied':'✅ **{n} اصطلاح اعمال شد.** این‌ها در طول ترجمه اولویت خواهند داشت.',
    'glossary_empty':'⚠️ هیچ اصطلاحی اعمال نشده. از فرمت `مبدا: ترجمه` استفاده کنید.',
    'glossary_cleared':'واژه‌نامه پاک شد.',
},
}


def detect_ui_language() -> str:
    try:
        lang_code = locale.getdefaultlocale()[0] or 'en'
        iso = lang_code.split('_')[0].lower()
        if iso.startswith('zh'):
            iso = 'zh'
        return iso if iso in _UI_LANG_CODES else 'en'
    except Exception:
        return 'en'


def load_ui_config() -> str:
    try:
        with open(_UI_CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        lang = data.get('ui_lang', '')
        return lang if lang in _UI_LANG_CODES else ''
    except Exception:
        return ''


def save_ui_config(lang_code: str):
    try:
        with open(_UI_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump({'ui_lang': lang_code}, f)
    except Exception as e:
        print(f'[ui_config] Save failed: {e}')


EPUB_TRANSLATE_TAGS = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'td', 'th', 'blockquote'}

class Dodari:
    def __init__(self):
        self.expire_time = 0
        self.limit_file_count = 100
        self.is_multi = True
        self.is_check_size = False

        self.app = None
        self.max_len = 512
        if platform.system() == 'Windows':
            self.translate_batch_size = 5
            self.translate_workers = 1
        else:
            self.translate_batch_size = 20
            self.translate_workers = 4
        self.kv_bits = 8
        self.temperature = 1
        self.launch_time = None
        self.selected_files = []
        self.upload_msg = None
        self.origin_lang_str = None
        self.target_lang_str = None
        self.origin_lang = None
        self.origin_lang_name = None
        self.target_lang = 'ko'
        self.target_lang_name = '한국어'
        self.target_lang_prompt = 'Korean'

        self.user_glossary: dict = {}

        if platform.system() == 'Windows':
            self.gemma_api_url = 'http://localhost:11434/v1/chat/completions'
            self.gemma_model = 'gemma4:e4b'
        else:
            self.gemma_api_url = 'http://localhost:8000/v1/chat/completions'
            self.gemma_model = 'mlx-community/gemma-4-e4b-it-8bit'

        self.output_folder = 'outputs'
        self.temp_folder_1 = 'temp_1'
        self.temp_folder_2 = 'temp_2'

        self.css = """
            .radio-group .wrap {
                display: float !important;
                grid-template-columns: 1fr 1fr;
            }
            .dodari-hidden { display: none !important; }
            """
        self.start = None
        self.platform = platform.system()
        _saved = load_ui_config()
        self.ui_lang = _saved if _saved else detect_ui_language()

    def _T(self, key: str) -> str:
        lang_dict = UI_TEXT.get(self.ui_lang, UI_TEXT['en'])
        return lang_dict.get(key, UI_TEXT['en'].get(key, key))

    def _genre_choices(self):
        return [(self._T(f'genre_{i}'), GENRE_CHOICES_KO[i]) for i in range(len(GENRE_CHOICES_KO))]

    def _tone_choices(self):
        return [(self._T(f'tone_{i}'), TONE_CHOICES_KO[i]) for i in range(len(TONE_CHOICES_KO))]

    def _bilingual_choices(self):
        return [(self._T(f'bilingual_{i}'), BILINGUAL_CHOICES_KO[i]) for i in range(len(BILINGUAL_CHOICES_KO))]

    def _lang_choices(self):
        disp = LANG_DISPLAY_BY_UI.get(self.ui_lang, LANG_DISPLAY_BY_UI['en'])
        return [(disp[ko], ko) for ko in SUPPORTED_LANGUAGES]

    def launch_interface(self):
        self.remove_folder(self.temp_folder_1)
        self.remove_folder(self.temp_folder_2)

        def _title_html(lc):
            T = lambda k: UI_TEXT.get(lc, UI_TEXT['en']).get(k, UI_TEXT['en'].get(k, k))
            return (
                f"<div style='text-align:center;width:100%;'>"
                f"<a href='https://github.com/vEduardovich/dodari' target='_blank' style='display:inline-block;'>"
                f"<img src='{img_src}' style='display:block;margin:0 auto;width:100px;'></a>"
                f"<h1 style='margin-top:10px;'>{T('app_title')}</h1>"
                f"</div>"
            )

        with gr.Blocks(
            css=self.css,
            title='Dodari',
            theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")
        ) as self.app:
            title_html = gr.HTML(_title_html(self.ui_lang))
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    with gr.Tab(self._T('step1')) as tab1:
                        step1_html = gr.HTML(f"<div style='display:flex;'><h3 style='margin-top:0px;'>{self._T('step1_title')}</h3><span style='margin-left:10px;'>( *.txt, *.epub, *.pdf )</span></div>")
                        file_count = 'multiple' if self.is_multi else 'files'
                        input_window = gr.File(
                            file_count=file_count,
                            file_types=[".txt", ".epub", ".pdf"],
                            label=self._T('files_label')
                        )
                        file_limit_md = gr.Markdown(self._T('file_limit').format(n=self.limit_file_count))
                        lang_msg = gr.HTML(self.upload_msg)
                        self.origin_lang_display = gr.Dropdown(
                            choices=self._lang_choices(),
                            label=self._T('origin_lang_label'),
                            interactive=True,
                            value=None
                        )

                with gr.Column(scale=1, min_width=300):
                    with gr.Tab(self._T('step2')) as tab2:
                        self.target_lang_radio = gr.Radio(
                            choices=self._lang_choices(),
                            value='한국어',
                            label=self._T('target_lang_label')
                        )
                        if platform.system() == 'Windows':
                            _engine_label = self._T('engine_ollama')
                        else:
                            _engine_label = self._T('engine_gemma')
                        engine_html = gr.HTML(f"<p style='color:green;'>{_engine_label}</p>")

                        if platform.system() == 'Windows':
                            _model_choices = ["gemma4:e4b", "gemma4:31b"]
                            _model_default = "gemma4:e4b"
                        else:
                            _model_choices = [
                                "mlx-community/gemma-4-e4b-it-8bit",
                                "mlx-community/gemma-4-31b-it-4bit",
                            ]
                            _model_default = _model_choices[0]
                        self.model_radio = gr.Radio(
                            choices=_model_choices,
                            label=self._T('model_label'),
                            value=_model_default
                        )
                with gr.Column(scale=1, min_width=300):
                    with gr.Tab(self._T('step3')) as tab3:
                        self.bilingual_order_radio = gr.Radio(
                            choices=self._bilingual_choices(),
                            value='번역문(원문)',
                            label=self._T('bilingual_label')
                        )
                        self.genre_radio = gr.Radio(
                            choices=self._genre_choices(),
                            label=self._T('genre_label'),
                            value='일반 문서(기본)'
                        )
                        self.tone_radio = gr.Radio(
                            choices=self._tone_choices(),
                            value='서술체 (~다)',
                            label=self._T('tone_label')
                        )

                        with gr.Accordion(self._T('glossary_title'), open=False) as glossary_accordion:
                            glossary_extract_btn = gr.Button(self._T('btn_glossary_extract'), variant='secondary')
                            glossary_status = gr.Markdown('')
                            self.glossary_textbox = gr.Textbox(
                                label=self._T('glossary_label'),
                                placeholder=self._T('glossary_placeholder'),
                                lines=6,
                                value=''
                            )
                            glossary_apply_btn = gr.Button(self._T('btn_glossary_apply'), variant='primary')
                            glossary_clear_btn = gr.Button(self._T('btn_glossary_clear'), variant='stop')
                            self.glossary_count_md = gr.Markdown(self._T('glossary_count').format(n=0))
                            glossary_desc_md = gr.Markdown(self._T('glossary_desc'))

                        self.model_radio.change(fn=self.reload_llm_server, inputs=[self.model_radio])

                        def on_origin_lang_change(lang_name):
                            if lang_name and lang_name in SUPPORTED_LANGUAGES:
                                iso_code, _ = SUPPORTED_LANGUAGES[lang_name]
                                self.origin_lang = iso_code
                                self.origin_lang_name = lang_name

                        self.origin_lang_display.change(
                            fn=on_origin_lang_change,
                            inputs=[self.origin_lang_display]
                        )

                        def on_target_lang_change(lang_name):
                            iso_code, prompt_name = SUPPORTED_LANGUAGES.get(lang_name, ('ko', 'Korean'))
                            self.target_lang = iso_code
                            self.target_lang_name = lang_name
                            self.target_lang_prompt = prompt_name

                        self.target_lang_radio.change(
                            fn=on_target_lang_change,
                            inputs=[self.target_lang_radio]
                        )

                        def apply_glossary(text: str):
                            glossary = {}
                            if text and text.strip():
                                for line in text.strip().splitlines():
                                    line = line.strip()
                                    if ':' in line:
                                        parts = line.split(':', 1)
                                        src = parts[0].strip()
                                        tgt = parts[1].strip()
                                        if src and tgt:
                                            glossary[src] = tgt
                            self.user_glossary = dict(list(glossary.items())[:50])
                            count = len(self.user_glossary)
                            if count > 0:
                                return self._T('glossary_applied').format(n=count)
                            return self._T('glossary_empty')

                        def clear_glossary():
                            self.user_glossary = {}
                            return '', self._T('glossary_cleared'), self._T('glossary_count').format(n=0)

                        def extract_glossary_with_ai():
                            if not self.selected_files:
                                yield '⚠️ Please attach a file first.', ''
                                return

                            yield '🔍 Scanning file for proper noun candidates...', ''

                            from collections import Counter
                            import re as _re
                            raw_texts = []
                            for file in self.selected_files[:3]:
                                try:
                                    name = file.get('orig_name', '')
                                    _, ext = os.path.splitext(name)
                                    ext = ext.lower()
                                    if ext == '.txt':
                                        with open(file['path'], 'r', encoding='utf-8', errors='ignore') as f:
                                            raw_texts.append(f.read())
                                    elif ext == '.epub':
                                        book = epub.read_epub(file['path'])
                                        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                                            soup = BeautifulSoup(item.get_content(), 'html.parser')
                                            raw_texts.append(soup.get_text(separator=' ', strip=True))
                                    elif ext == '.pdf' and FITZ_AVAILABLE:
                                        doc = fitz.open(file['path'])
                                        for page in doc:
                                            raw_texts.append(page.get_text())
                                        doc.close()
                                except Exception as e:
                                    print(f'[Glossary] File read error ({name}): {e}')

                            full_text = ' '.join(raw_texts)
                            if not full_text.strip():
                                yield '⚠️ Could not read text from file.', ''
                                return

                            stopwords = {
                                'The', 'A', 'An', 'In', 'On', 'At', 'Of', 'And', 'Or', 'But',
                                'Is', 'Was', 'Are', 'Were', 'He', 'She', 'It', 'They', 'We',
                                'I', 'You', 'His', 'Her', 'This', 'That', 'With', 'For', 'To',
                                'From', 'By', 'As', 'Be', 'Have', 'Has', 'Had', 'Not', 'Do',
                            }
                            candidates_raw = _re.findall(r'\b([A-Z][a-zA-Z\']{1,})\b', full_text)
                            freq = Counter(c for c in candidates_raw if c not in stopwords)
                            top_candidates = [word for word, _ in freq.most_common(40)]

                            if not top_candidates:
                                yield '⚠️ No proper noun candidates found.', ''
                                return

                            yield f'✨ Found {len(top_candidates)} candidates. AI is refining...', ''

                            candidate_list_str = ', '.join(top_candidates)
                            extraction_prompt = (
                                f'The following words were frequently found in a document to be translated into {self.target_lang_prompt}. '
                                f'From this list, identify the most important proper nouns (character names, place names, '
                                f'organization names, technical terms) that should be consistently translated. '
                                f'For each selected term, provide the recommended {self.target_lang_prompt} translation. '
                                f'Return ONLY a list in this exact format, one per line: "OriginalWord: Translation". '
                                f'Select at most 25 terms. Ignore common English words.\n\nWord list: {candidate_list_str}'
                            )
                            try:
                                payload = {
                                    'model': self.gemma_model,
                                    'messages': [{'role': 'user', 'content': extraction_prompt}],
                                    'max_tokens': 512,
                                    'temperature': 0.3,
                                }
                                response = requests.post(
                                    self.gemma_api_url,
                                    headers={'Content-Type': 'application/json'},
                                    json=payload,
                                    timeout=60
                                )
                                response.raise_for_status()
                                ai_result = response.json()['choices'][0]['message']['content'].strip()

                                lines = [l.strip() for l in ai_result.splitlines() if ':' in l and l.strip()]
                                valid_lines = [l for l in lines if len(l.split(':', 1)) == 2]

                                if not valid_lines:
                                    yield '⚠️ AI could not extract terms. Please enter manually.', ''
                                    return

                                result_text = '\n'.join(valid_lines)
                                yield f'✅ **Extracted {len(valid_lines)} terms!** Review below and click [Apply Glossary].', result_text

                            except Exception as e:
                                yield f'⚠️ Error during AI extraction: {e}', ''

                        glossary_extract_btn.click(
                            fn=extract_glossary_with_ai,
                            inputs=[],
                            outputs=[glossary_status, self.glossary_textbox]
                        )

                        glossary_apply_btn.click(
                            fn=apply_glossary,
                            inputs=[self.glossary_textbox],
                            outputs=[glossary_status]
                        ).then(
                            fn=lambda: self._T('glossary_count').format(n=len(self.user_glossary)),
                            inputs=[],
                            outputs=[self.glossary_count_md]
                        )

                        glossary_clear_btn.click(
                            fn=clear_glossary,
                            inputs=[],
                            outputs=[self.glossary_textbox, glossary_status, self.glossary_count_md]
                        )

                with gr.Column(scale=2):
                    with gr.Tab(self._T('step4')) as tab4:
                        translate_btn = gr.Button(
                            value=self._T('btn_translate'),
                            size='lg',
                            variant="primary",
                            interactive=True
                        )
                        with gr.Tab(self._T('status_tab')) as status_tab:
                            status_msg = gr.HTML('', visible=True)
                            done_files = gr.File(label=self._T('download_label'), file_count='multiple', interactive=False, visible=True)
                            run_state = gr.State()

                            translate_btn.click(
                                fn=self.execute_translation_pipeline,
                                inputs=[self.genre_radio, self.tone_radio, self.target_lang_radio, self.bilingual_order_radio],
                                outputs=[done_files, run_state]
                            ).then(
                                fn=self.format_result_message,
                                inputs=[run_state],
                                outputs=[status_msg]
                            )

                            input_window.change(
                                fn=self.on_file_upload,
                                inputs=input_window,
                                outputs=[status_msg, done_files, lang_msg, self.genre_radio, self.tone_radio, self.origin_lang_display, self.target_lang_radio, self.glossary_textbox, self.glossary_count_md, self.bilingual_order_radio],
                                preprocess=False,
                                show_progress="hidden"
                            )

            def on_ui_lang_change(lang_code):
                save_ui_config(lang_code)
                self.ui_lang = lang_code
                T = lambda k: UI_TEXT.get(lang_code, UI_TEXT['en']).get(k, UI_TEXT['en'].get(k, k))
                _eng = T('engine_ollama') if platform.system() == 'Windows' else T('engine_gemma')
                return (
                    gr.update(value=_title_html(lang_code)),
                    gr.update(label=T('step1')),
                    gr.update(value=f"<div style='display:flex;'><h3 style='margin-top:0px;'>{T('step1_title')}</h3><span style='margin-left:10px;'>( *.txt, *.epub, *.pdf )</span></div>"),
                    gr.update(label=T('files_label')),
                    gr.update(value=T('file_limit').format(n=self.limit_file_count)),
                    gr.update(label=T('origin_lang_label'), choices=self._lang_choices()),
                    gr.update(label=T('step2')),
                    gr.update(label=T('target_lang_label'), choices=self._lang_choices()),
                    gr.update(value=f"<p style='color:green;'>{_eng}</p>"),
                    gr.update(label=T('model_label')),
                    gr.update(label=T('step3')),
                    gr.update(label=T('bilingual_label'), choices=self._bilingual_choices()),
                    gr.update(label=T('genre_label'), choices=self._genre_choices()),
                    gr.update(label=T('tone_label'), choices=self._tone_choices()),
                    gr.update(label=T('glossary_title')),
                    gr.update(value=T('btn_glossary_extract')),
                    gr.update(label=T('glossary_label'), placeholder=T('glossary_placeholder')),
                    gr.update(value=T('btn_glossary_apply')),
                    gr.update(value=T('btn_glossary_clear')),
                    gr.update(value=T('glossary_count').format(n=len(self.user_glossary))),
                    gr.update(value=T('glossary_desc')),
                    gr.update(label=T('step4')),
                    gr.update(value=T('btn_translate')),
                    gr.update(label=T('status_tab')),
                    gr.update(label=T('download_label')),
                )

            _live_outputs = [
                title_html, tab1, step1_html, input_window, file_limit_md,
                self.origin_lang_display, tab2, self.target_lang_radio, engine_html,
                self.model_radio, tab3, self.bilingual_order_radio, self.genre_radio,
                self.tone_radio, glossary_accordion, glossary_extract_btn,
                self.glossary_textbox, glossary_apply_btn, glossary_clear_btn,
                self.glossary_count_md, glossary_desc_md, tab4, translate_btn,
                status_tab, done_files,
            ]

            with gr.Row():
                gr.HTML("<div></div>", scale=1)
                gr.HTML("<div></div>", scale=1)
                gr.HTML("<div></div>", scale=1)
                gr.HTML("<div></div>", scale=1)
                ui_lang_dropdown = gr.Dropdown(
                    choices=[(UI_LANG_NAMES[c], c) for c in _UI_LANG_CODES],
                    value=self.ui_lang,
                    label="UI Language",
                    container=False,
                    scale=1,
                    min_width=160,
                )
            ui_lang_dropdown.change(fn=on_ui_lang_change, inputs=[ui_lang_dropdown], outputs=_live_outputs)

        self.app.queue().launch(
            share=True,
            inbrowser=True,
            favicon_path='imgs/dodari.png',
            allowed_paths=['.', './outputs']
        )

    def reload_llm_server(self, new_model: str):
        if self.gemma_model == new_model:
            return

        print(f"\n[Model Switch] Loading {new_model} server...")
        gr.Info(f"Switching model to {new_model}. Please wait.")
        self.gemma_model = new_model

        if platform.system() == 'Windows':
            print(f"[Model Switch] Windows(Ollama): no restart needed, switching → {new_model}")
            gr.Info(f"Ollama model switched to {new_model}. (No server restart needed)")
            return

        cleanup_llm_server()
        time.sleep(2)

        mlx_python = os.environ.get('MLX_PYTHON', 'python3')
        cmd = (
            f"{mlx_python} -m mlx_vlm.server "
            f"--model {new_model} "
            f"--kv-bits {self.kv_bits} "
            f"--port 8000"
        )
        print(f"[Model Switch] MLX params: batch={self.translate_batch_size}, workers={self.translate_workers}, kv-bits={self.kv_bits}")
        subprocess.Popen(cmd, shell=True)
        time.sleep(8)

        gr.Info("Model setup complete!")
        print(f"[Model Switch] Done!\n")

    def format_result_message(self, sec_or_msg):
        if not sec_or_msg:
            return ""
        if str(sec_or_msg).startswith("<p"):
            return sec_or_msg
        return self._T('translation_complete').format(t=sec_or_msg)

    def execute_translation_pipeline(self, genre_val, tone_val="서술체 (~다)", target_lang_name="한국어", bilingual_order_val="번역문(원문)", progress=gr.Progress()):
        if self.expire_time:
            during = self.calculate_elapsed_time(self.launch_time, 2)
            if during > self.expire_time:
                over_time = during - self.expire_time
                over_time_str = str(timedelta(seconds=over_time)).split('.')[0]
                print('Time limit exceeded: ', over_time_str)
                return None, f"<p style='color:red;'>도다리 사용시간이 {over_time_str}만큼 지났습니다</p>"

        if self.is_multi and len(self.selected_files) > self.limit_file_count:
            return None, f"<p style='color:red;'>{self._T('err_file_limit').format(n=self.limit_file_count)}</p>"

        if not self.selected_files:
            return None, f"<p style='color:red;'>{self._T('err_file_none')}</p>"

        self.start = time.time()
        print("Start! now.." + str(self.start))
        progress(0, desc=self._T('progress_init'))

        _base_url = self.gemma_api_url.rsplit('/v1/', 1)[0]
        progress(0, desc=self._T('progress_server'))
        server_ok = False
        for attempt in range(5):
            try:
                resp = requests.get(f'{_base_url}/v1/models', timeout=3)
                if resp.status_code == 200:
                    server_ok = True
                    break
            except Exception:
                pass
            print(f'[Server] {_base_url} not responding ({attempt + 1}/5), retrying in 2s...')
            time.sleep(2)

        if not server_ok:
            _guide_key = {
                'Darwin': 'server_guide_mac',
                'Windows': 'server_guide_windows',
            }.get(platform.system(), 'server_guide_default')
            _guide = self._T(_guide_key)
            return (
                None,
                f"<p style='color:red;'>{self._T('err_server').format(url=_base_url, guide=_guide)}</p>"
            )
        print('Gemma API ready for translation')

        if not self.origin_lang:
            return None, f"<p style='color:red;'>{self._T('err_lang_detect')}</p>"

        target_iso, target_prompt = SUPPORTED_LANGUAGES.get(target_lang_name, ('ko', 'Korean'))
        self.target_lang = target_iso
        self.target_lang_name = target_lang_name
        self.target_lang_prompt = target_prompt

        if self.origin_lang == target_iso:
            return None, f"<p style='color:red;'>{self._T('err_lang_same').format(lang=target_lang_name)}</p>"

        origin_abb = self.origin_lang
        target_abb = target_iso
        all_file_path = []

        for file in progress.tqdm(self.selected_files, desc=self._T('progress_files')):
            print(f'file: {file}')
            name, ext = os.path.splitext(file['orig_name'])

            if 'epub' in ext:
                self.extract_epub_contents(self.temp_folder_1, file['path'])
                self.extract_epub_contents(self.temp_folder_2, file['path'])

                opf_file = self.locate_epub_metadata_opf()
                tree = parse(opf_file)
                opf = tree.getroot()

                for child in opf.iter():
                    print(child.tag)
                    if 'language' in child.tag:
                        child.text = target_abb
                        print(child.text)
                        break
                output_opf = open(opf_file, 'wb')
                tree.write(output_opf, encoding='utf-8', xml_declaration=True)
                output_opf.close()
                print('Language metadata updated')

                file_path = self.list_epub_html_files()
                print('File count: ', len(file_path))
                for html_file in progress.tqdm(file_path, desc='Chapter'):
                    print('html_file')
                    print(html_file)
                    try:
                        html_file_2 = html_file.replace(self.temp_folder_1, self.temp_folder_2)

                        input_file_1 = open(html_file, 'r', encoding='utf-8')
                        input_file_2 = open(html_file_2, 'r', encoding='utf-8')

                        soup_1 = BeautifulSoup(input_file_1.read(), 'html.parser')
                        soup_2 = BeautifulSoup(input_file_2.read(), 'html.parser')

                        _skip_epub_types = {'index', 'toc', 'cover', 'lot', 'loi'}
                        _body_tag = soup_1.find('body')
                        if _body_tag and _skip_epub_types.intersection((_body_tag.get('epub:type') or '').split()):
                            continue

                        _leaf_filter = EPUB_TRANSLATE_TAGS - {'span'}
                        p_tags_1 = [
                            tag for tag in soup_1.find_all(EPUB_TRANSLATE_TAGS)
                            if tag.get_text(strip=True)
                            and not tag.find(_leaf_filter)
                            and not (tag.name == 'span' and tag.parent and tag.parent.name in EPUB_TRANSLATE_TAGS)
                        ]
                        p_tags_2 = [
                            tag for tag in soup_2.find_all(EPUB_TRANSLATE_TAGS)
                            if tag.get_text(strip=True)
                            and not tag.find(_leaf_filter)
                            and not (tag.name == 'span' and tag.parent and tag.parent.name in EPUB_TRANSLATE_TAGS)
                        ]

                        only_texts = []
                        whole_particle = []
                        _ch_elapsed = format_korean_time(int(time.time() - self.start))
                        for text_node_1, text_node_2 in progress.tqdm(zip(p_tags_1, p_tags_2), desc=f'Translating paragraphs ({_ch_elapsed} elapsed)'):
                            _bl = text_node_1.find('a', attrs={'role': 'doc-backlink'})
                            if _bl and text_node_1.get_text(strip=True) == _bl.get_text(strip=True):
                                whole_particle.append(0)
                                continue
                            text_str = text_node_1.text.strip()
                            if not text_str or self.contains_no_alphabets(text_str) or len(text_str) <= 1:
                                whole_particle.append(0)
                                continue

                            raw_particle = nltk.sent_tokenize(text_node_1.text)
                            particle = [s for s in raw_particle if sum(1 for c in s if c.isalpha()) >= 2]
                            if not particle:
                                whole_particle.append(0)
                                continue
                            only_texts.extend(particle)
                            particle.append(0)
                            whole_particle.extend(particle)

                        parti_1, parti_2 = self.batch_translate_engine(only_texts, whole_particle, 'epub', genre_val, tone_val, bilingual_order_val)

                        particle_list_1 = []
                        particle_list_2 = []
                        translated_str_1 = ''
                        translated_str_2 = ''
                        for p_1, p_2 in zip(parti_1, parti_2):
                            if p_1:
                                translated_str_1 += ' ' + p_1
                                translated_str_2 += ' ' + p_2
                            else:
                                particle_list_1.append(translated_str_1)
                                particle_list_2.append(translated_str_2)
                                translated_str_1 = ''
                                translated_str_2 = ''

                        for p_1, p_2, text_node_1, text_node_2 in zip(particle_list_1, particle_list_2, p_tags_1, p_tags_2):
                            text_str = text_node_1.text.strip()

                            if not text_str or self.contains_no_alphabets(text_str) or len(text_str) <= 1:
                                continue

                            p_tag_1 = soup_1.new_tag(text_node_1.name)
                            p_tag_2 = soup_2.new_tag(text_node_2.name)

                            for _attr, _val in text_node_1.attrs.items():
                                p_tag_1[_attr] = _val
                                p_tag_2[_attr] = _val

                            if text_node_1.text.strip():
                                p_tag_1.string = p_1
                                p_tag_2.string = p_2
                                for _empty_a in reversed(text_node_1.find_all('a')):
                                    if not _empty_a.get_text(strip=True):
                                        p_tag_1.insert(0, copy.copy(_empty_a))
                                        p_tag_2.insert(0, copy.copy(_empty_a))
                                img_tag = text_node_1.find('img')
                                if img_tag:
                                    print('Adding image tag')
                                    p_tag_1.append(img_tag)
                                    p_tag_2.append(img_tag)

                            text_node_1.replace_with(p_tag_1)
                            text_node_2.replace_with(p_tag_2)

                        input_file_1.close()
                        input_file_2.close()

                        output_file_1 = open(html_file, 'w', encoding='utf-8')
                        output_file_2 = open(html_file_2, 'w', encoding='utf-8')
                        output_file_1.write(str(soup_1))
                        output_file_2.write(str(soup_2))
                        output_file_1.close()
                        output_file_2.close()
                    except Exception as err:
                        print(err)
                        print('HTML loading error, skipping')
                        continue

                for loc_folder in [self.temp_folder_1, self.temp_folder_2]:
                    self.repack_epub_contents(loc_folder, f'{loc_folder}.epub')

                os.makedirs(self.output_folder, exist_ok=True)
                if bilingual_order_val == "원문(번역문)":
                    done_path_1 = os.path.join(self.output_folder, "{name}_{t2}({t3}){ext}".format(name=name, t2=origin_abb, t3=target_abb, ext=ext))
                else:
                    done_path_1 = os.path.join(self.output_folder, "{name}_{t2}({t3}){ext}".format(name=name, t2=target_abb, t3=origin_abb, ext=ext))
                done_path_2 = os.path.join(self.output_folder, "{name}_{t2}{ext}".format(name=name, t2=target_abb, ext=ext))

                all_file_path.extend([done_path_1, done_path_2])

                shutil.move(f'{self.temp_folder_1}.epub', done_path_1)
                shutil.move(f'{self.temp_folder_2}.epub', done_path_2)

                self.remove_folder(self.temp_folder_1)
                self.remove_folder(self.temp_folder_2)

            elif '.pdf' in ext:
                print(f'[PDF Translation] Start: {name}{ext}')
                print(f'[PDF Translation] Lang: {origin_abb} → {target_abb} | Model: {self.gemma_model}')

                if not DOCLING_AVAILABLE:
                    print('[PDF Translation] Error: docling not installed. Run: pip install docling')
                    continue

                try:
                    os.makedirs(self.output_folder, exist_ok=True)
                    progress(0, desc='[PDF] Structuring HTML with Docling...')

                    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                    
                    print(f"\n[PDF] Processing: {file['path']} - Docling CPU mode")
                    progress(0, desc='[PDF] Loading Docling engine...')
                    
                    try:
                        pipeline_options = PdfPipelineOptions()
                        pipeline_options.generate_picture_images = True
                        pipeline_options.images_scale = 2.0
                        pipeline_options.accelerator_options = AcceleratorOptions(
                            num_threads=4 if platform.system() == 'Windows' else 8,
                            device=AcceleratorDevice.CPU
                        )
                        
                        print("[PDF] Initializing DocumentConverter...")
                        converter = DocumentConverter(
                            format_options={'pdf': PdfFormatOption(pipeline_options=pipeline_options)}
                        )

                        print("[PDF] Starting HTML structure extraction (may take 1-3 min depending on PDF size)...")
                        progress(5, desc='[PDF] Extracting HTML structure (CPU)...')
                        result = converter.convert(file['path'])
                        print("[PDF] Structure extraction complete!")
                        
                    finally:
                        if original_cuda_visible:
                            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                        else:
                            del os.environ["CUDA_VISIBLE_DEVICES"]

                    picture_delete = set()
                    picture_skip = set()
                    try:
                        picture_regions = []
                        for pic in getattr(result.document, 'pictures', []):
                            for prov in getattr(pic, 'prov', []):
                                bbox = getattr(prov, 'bbox', None)
                                if bbox is not None:
                                    picture_regions.append((prov.page_no, bbox))

                        if picture_regions:
                            margin_up    = 150
                            margin_down  = 15
                            margin_horiz = 150
                            for entry in result.document.iterate_items():
                                item = entry[0] if isinstance(entry, (tuple, list)) else entry
                                item_text = getattr(item, 'text', None)
                                if not item_text or not item_text.strip():
                                    continue
                                for tprov in getattr(item, 'prov', []):
                                    bbox = getattr(tprov, 'bbox', None)
                                    if bbox is None:
                                        continue
                                    tb = bbox
                                    for (pp, pb) in picture_regions:
                                        if tprov.page_no != pp:
                                            continue
                                        horiz_ok = (tb.l >= pb.l - margin_horiz and
                                                    tb.r <= pb.r + margin_horiz)
                                        if not horiz_ok:
                                            break
                                        if tb.t >= pb.b and tb.t <= pb.t + margin_up:
                                            picture_delete.add(item_text.strip())
                                        elif pb.b - margin_down <= tb.t < pb.b:
                                            picture_skip.add(item_text.strip())
                                        break

                        print(f'[PDF] Delete text: {len(picture_delete)} / Skip text: {len(picture_skip)}')
                    except Exception as pe:
                        import traceback as _tb
                        print(f'[PDF] Image text collection error: {pe}')
                        _tb.print_exc()

                    code_block_images = []
                    try:
                        import io as _io
                        import base64 as _b64
                        import pypdfium2 as _pdfium

                        pdf_doc = _pdfium.PdfDocument(file['path'])
                        rendered_pages = {}

                        for entry in result.document.iterate_items():
                            item = entry[0] if isinstance(entry, (tuple, list)) else entry
                            item_label = str(getattr(item, 'label', '')).upper()
                            if 'CODE' not in item_label:
                                continue
                            for prov in getattr(item, 'prov', []):
                                bbox = getattr(prov, 'bbox', None)
                                if bbox is None:
                                    continue
                                page_no = prov.page_no
                                if page_no not in rendered_pages:
                                    pdf_page = pdf_doc[page_no - 1]
                                    pt_w = pdf_page.get_width()
                                    pt_h = pdf_page.get_height()
                                    bitmap = pdf_page.render(scale=2.0)
                                    pil_img = bitmap.to_pil()
                                    rendered_pages[page_no] = (pil_img, pt_w, pt_h)
                                pil_img, pt_w, pt_h = rendered_pages[page_no]
                                img_w, img_h = pil_img.size
                                sx = img_w / pt_w
                                sy = img_h / pt_h
                                pad = 10
                                x1 = max(0, int(bbox.l * sx) - pad)
                                y1 = max(0, int((pt_h - bbox.t) * sy) - pad)
                                x2 = min(img_w, int(bbox.r * sx) + pad)
                                y2 = min(img_h, int((pt_h - bbox.b) * sy) + pad)
                                cropped = pil_img.crop((x1, y1, x2, y2))
                                buf = _io.BytesIO()
                                cropped.save(buf, format='PNG')
                                b64str = _b64.b64encode(buf.getvalue()).decode('utf-8')
                                code_block_images.append(f'data:image/png;base64,{b64str}')
                                break

                        pdf_doc.close()
                        print(f'[PDF] {len(code_block_images)} code block image(s) cropped')
                    except Exception as ce:
                        import traceback as _tb
                        print(f'[PDF] Code block image conversion error: {ce}')
                        _tb.print_exc()

                    WIDE_TABLE_COL_THRESHOLD = 5
                    wide_table_images = {}

                    try:
                        pdf_doc_t = _pdfium.PdfDocument(file['path'])
                        rendered_pages_t = {}
                        table_idx = 0

                        for entry in result.document.iterate_items():
                            item = entry[0] if isinstance(entry, (tuple, list)) else entry
                            item_label = str(getattr(item, 'label', '')).upper()
                            if 'TABLE' not in item_label:
                                continue

                            col_count = 0
                            table_data = getattr(item, 'data', None)
                            if table_data is not None:
                                col_count = getattr(table_data, 'num_cols', 0)
                                if col_count == 0 and hasattr(table_data, 'grid') and table_data.grid:
                                    col_count = len(table_data.grid[0]) if table_data.grid[0] else 0

                            if col_count >= WIDE_TABLE_COL_THRESHOLD:
                                for prov in getattr(item, 'prov', []):
                                    bbox = getattr(prov, 'bbox', None)
                                    if bbox is None:
                                        continue
                                    page_no = prov.page_no
                                    if page_no not in rendered_pages_t:
                                        pdf_page = pdf_doc_t[page_no - 1]
                                        pt_w = pdf_page.get_width()
                                        pt_h = pdf_page.get_height()
                                        bitmap = pdf_page.render(scale=2.0)
                                        pil_img = bitmap.to_pil()
                                        rendered_pages_t[page_no] = (pil_img, pt_w, pt_h)

                                    pil_img, pt_w, pt_h = rendered_pages_t[page_no]
                                    img_w, img_h = pil_img.size
                                    sx = img_w / pt_w
                                    sy = img_h / pt_h
                                    pad = 8
                                    x1 = max(0, int(bbox.l * sx) - pad)
                                    y1 = max(0, int((pt_h - bbox.t) * sy) - pad)
                                    x2 = min(img_w, int(bbox.r * sx) + pad)
                                    y2 = min(img_h, int((pt_h - bbox.b) * sy) + pad)
                                    cropped = pil_img.crop((x1, y1, x2, y2))
                                    buf = _io.BytesIO()
                                    cropped.save(buf, format='PNG')
                                    b64str = _b64.b64encode(buf.getvalue()).decode('utf-8')
                                    wide_table_images[table_idx] = f'data:image/png;base64,{b64str}'
                                    break

                            table_idx += 1

                        pdf_doc_t.close()
                        print(f'[PDF] {len(wide_table_images)} wide table image(s) cropped (total tables: {table_idx})')

                    except Exception as te:
                        import traceback as _tb
                        print(f'[PDF] Wide table image conversion error: {te}')
                        wide_table_images = {}
                        _tb.print_exc()

                    formula_images = []
                    try:
                        pdf_doc_fm = _pdfium.PdfDocument(file['path'])
                        rendered_pages_fm = {}
                        for entry in result.document.iterate_items():
                            item = entry[0] if isinstance(entry, (tuple, list)) else entry
                            if 'FORMULA' not in str(getattr(item, 'label', '')).upper():
                                continue
                            for prov in getattr(item, 'prov', []):
                                bbox = getattr(prov, 'bbox', None)
                                if bbox is None:
                                    continue
                                page_no = prov.page_no
                                if page_no not in rendered_pages_fm:
                                    pdf_page = pdf_doc_fm[page_no - 1]
                                    pt_w = pdf_page.get_width()
                                    pt_h = pdf_page.get_height()
                                    pil_img = pdf_page.render(scale=2.0).to_pil()
                                    rendered_pages_fm[page_no] = (pil_img, pt_w, pt_h)
                                pil_img, pt_w, pt_h = rendered_pages_fm[page_no]
                                img_w, img_h = pil_img.size
                                sx = img_w / pt_w
                                sy = img_h / pt_h
                                pad = 8
                                x1 = max(0, int(bbox.l * sx) - pad)
                                y1 = max(0, int((pt_h - bbox.t) * sy) - pad)
                                x2 = min(img_w, int(bbox.r * sx) + pad)
                                y2 = min(img_h, int((pt_h - bbox.b) * sy) + pad)
                                cropped = pil_img.crop((x1, y1, x2, y2))
                                buf = _io.BytesIO()
                                cropped.save(buf, format='PNG')
                                formula_images.append(f'data:image/png;base64,{_b64.b64encode(buf.getvalue()).decode()}')
                                break
                        pdf_doc_fm.close()
                        print(f'[PDF] {len(formula_images)} formula image(s) cropped')
                    except Exception as fe:
                        import traceback as _tb
                        print(f'[PDF] Formula image conversion error: {fe}')
                        _tb.print_exc()

                    html_content = result.document.export_to_html(image_mode=ImageRefMode.EMBEDDED)
                    print('[PDF Translation] HTML structure complete. Starting text translation pipeline (BeautifulSoup)...')

                    soup_1 = BeautifulSoup(html_content, 'html.parser')
                    soup_2 = BeautifulSoup(html_content, 'html.parser')

                    for soup_obj in [soup_1, soup_2]:
                        pre_tags = soup_obj.find_all('pre')
                        for i, pre in enumerate(pre_tags):
                            if i < len(code_block_images):
                                img_tag = soup_obj.new_tag(
                                    'img',
                                    src=code_block_images[i],
                                    style='display:block;max-width:100%;margin:1em 0;'
                                )
                                pre.replace_with(img_tag)
                            else:
                                pass

                    for soup_obj in [soup_1, soup_2]:
                        table_tags = soup_obj.find_all('table')
                        for i, table in enumerate(table_tags):
                            if i in wide_table_images:
                                img_tag = soup_obj.new_tag(
                                    'img',
                                    src=wide_table_images[i],
                                    style='display:block;max-width:100%;margin:1.5em auto;border:1px solid #ddd;'
                                )
                                table.replace_with(img_tag)
                            else:
                                table['style'] = (
                                    'width:100%;border-collapse:collapse;'
                                    'font-size:0.82em;word-break:break-word;'
                                    'table-layout:fixed;'
                                )

                    if formula_images:
                        _fm_re = re.compile(r'formula\s+not\s+decoded', re.IGNORECASE)
                        for soup_obj in [soup_1, soup_2]:
                            _fm_idx = 0
                            for _el in list(soup_obj.find_all(['p', 'div', 'span', 'section'])):
                                if _fm_idx >= len(formula_images):
                                    break
                                if _fm_re.search(_el.get_text(separator=' ', strip=True)) and not _el.find(['p', 'div']):
                                    _el.replace_with(soup_obj.new_tag('img', src=formula_images[_fm_idx],
                                        style='display:block;max-width:100%;margin:1em auto;'))
                                    _fm_idx += 1

                    block_tag_names = {'p', 'div', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'figcaption'}
                    target_tags = list(block_tag_names)
                    tags_1 = soup_1.find_all(target_tags)
                    tags_2 = soup_2.find_all(target_tags)

                    only_texts = []
                    whole_particle = []
                    valid_tags_1 = []
                    valid_tags_2 = []

                    for t_idx, tag_1 in enumerate(tags_1):
                        if any(tag_1.find(bt) for bt in block_tag_names):
                            continue
                        if tag_1.find_parent('figure') or tag_1.find('img'):
                            continue

                        text = tag_1.get_text(separator=' ').strip()
                        if picture_delete and text in picture_delete:
                            tag_1.decompose()
                            tags_2[t_idx].decompose()
                            continue
                        if picture_skip and text in picture_skip:
                            for s_tag in [tag_1, tags_2[t_idx]]:
                                s_tag['style'] = 'font-style:italic;text-align:center;margin-top:2px;font-size:0.9em;'
                            continue
                        if re.match(r'^\d+\s+https?://', text):
                            continue
                        if len(text) > 1 and any(c.isalpha() for c in text):
                            raw_sentences = nltk.sent_tokenize(text)
                            sentences = [s for s in raw_sentences if sum(1 for c in s if c.isalpha()) >= 2]
                            if not sentences:
                                continue
                            only_texts.extend(sentences)

                            p_with_marker = list(sentences)
                            p_with_marker.append(0)
                            whole_particle.extend(p_with_marker)

                            valid_tags_1.append(tag_1)
                            valid_tags_2.append(tags_2[t_idx])
                            
                    if only_texts:
                        progress(0.7, desc=f'[PDF] Batch translating text... ({format_korean_time(int(time.time() - self.start))} elapsed)')
                        parti_1, parti_2 = self.batch_translate_engine(only_texts, whole_particle, 'epub', genre_val, tone_val, bilingual_order_val)

                        assembled_1 = []
                        assembled_2 = []
                        translated_str_1 = ''
                        translated_str_2 = ''
                        
                        for p_1, p_2 in zip(parti_1, parti_2):
                            if p_1:
                                translated_str_1 += ' ' + p_1
                                translated_str_2 += ' ' + p_2
                            else:
                                assembled_1.append(translated_str_1.strip())
                                assembled_2.append(translated_str_2.strip())
                                translated_str_1 = ''
                                translated_str_2 = ''
                        
                        for t_idx, valid_tag_1 in enumerate(progress.tqdm(valid_tags_1, desc='HTML Reassembly')):
                            trans_1 = assembled_1[t_idx]
                            trans_2 = assembled_2[t_idx]
                            
                            valid_tag_1.clear()
                            valid_tag_1.string = trans_1

                            valid_tags_2[t_idx].clear()
                            valid_tags_2[t_idx].string = trans_2
                            
                    progress(0.95, desc=f'[PDF] EPUB packaging... (elapsed: {format_korean_time(int(time.time() - self.start))})')
                    if bilingual_order_val == "원문(번역문)":
                        done_path_1 = os.path.join(self.output_folder, f"{name}_{origin_abb}({target_abb}).epub")
                    else:
                        done_path_1 = os.path.join(self.output_folder, f"{name}_{target_abb}({origin_abb}).epub")
                    done_path_2 = os.path.join(self.output_folder, f"{name}_{target_abb}.epub")

                    self.build_epub_from_soup(soup_1, done_path_1, name, lang_code=target_abb)
                    self.build_epub_from_soup(soup_2, done_path_2, name, lang_code=target_abb)

                    all_file_path.extend([done_path_1, done_path_2])
                    print(f'[PDF Translation] Success! EPUB created: {done_path_1}, {done_path_2}')
                    
                except Exception as err:
                    import traceback
                    print(f'[PDF Translation] Error: {err}')
                    traceback.print_exc()
                    continue

            else:
                output_file_1, output_file_2, book = self.initialize_output_files(origin_abb, target_abb, name, ext, file, bilingual_order_val)
                book_raw = book.read()
                sentences = book_raw.split(sep='\n')

                only_texts = []
                whole_particle = []
                for sen in progress.tqdm(sentences, desc='Paragraph'):
                    particle = nltk.sent_tokenize(sen)
                    particle = self.clean_text_spacing(particle)

                    only_texts.extend(particle)
                    particle.append(0)
                    whole_particle.extend(particle)

                particle_list_1, particle_list_2 = self.batch_translate_engine(only_texts, whole_particle, 'txt', genre_val, tone_val, bilingual_order_val)

                translated_particle_1 = ' '.join(particle_list_1)
                translated_particle_2 = ' '.join(particle_list_2)
                output_file_1.write(translated_particle_1)
                output_file_2.write(translated_particle_2)
                all_file_path.extend([output_file_1.name, output_file_2.name])
                self.finalize_file_streams(book, output_file_1, output_file_2)

        sec = self.reset_session_and_gc()

        return all_file_path, sec

    def get_genre_prompt_extension(self, genre_val: str) -> str:
        genre_map = {
            "IT 및 엔지니어링": "IT & Engineering",
            "문학 및 소설": "Literature & Fiction",
            "인문 및 사회과학": "Humanities & Social Sciences",
            "비즈니스 및 경제": "Business & Economy",
            "영상 및 대본": "Subtitles & Scripts"
        }
        english_genre = genre_map.get(genre_val, "")
        if english_genre:
            return f"\nThe text is from a {english_genre}. Adapt the terminology, context, and tone appropriately.\n"
        return ""

    def get_tone_prompt_extension(self, tone_val: str) -> str:
        is_formal = "경어체" in tone_val
        lang = self.target_lang

        if lang == 'ko':
            if is_formal:
                return (
                    "모든 나레이션은 반드시 정중한 경어체(~합니다/~입니다)로 종결하십시오. "
                    "단, 대화문은 원문의 캐릭터 뉘앙스를 살려 자연스럽게 번역하십시오. "
                )
            else:
                return (
                    "모든 나레이션(서술 및 묘사)은 반드시 격식 있는 서술체(~다/~라)로 종결하십시오. "
                    "단, 큰따옴표(\" \") 내의 대화문은 캐릭터 간의 관계와 상황에 맞는 자연스러운 어투(존댓말, 반말 등)를 유지하십시오. "
                )

        elif lang == 'ja':
            if is_formal:
                return (
                    "地の文（ナレーション）はすべて丁寧語（です・ます体）で統一してください。"
                    "ただし、会話文（「」内）はキャラクターの関係性や状況に合った自然な話し方を維持してください。"
                )
            else:
                return (
                    "地の文（ナレーション）はすべて普通体（だ・である体）で統一してください。"
                    "ただし、会話文はキャラクターのニュアンスを活かした自然な表現を維持してください。"
                )

        elif lang == 'zh':
            if is_formal:
                return (
                    "所有叙述性文字（旁白、描写）请使用正式规范的书面语风格。"
                    "对话部分请根据角色关系和情境，保持自然流畅的表达。"
                )
            else:
                return (
                    "所有叙述性文字（旁白、描写）请使用简洁流畅的现代白话文风格。"
                    "对话部分请根据角色性格和情境，保持生动自然的语气。"
                )

        elif lang == 'fr':
            if is_formal:
                return (
                    "Use a formal literary register throughout the narration, with 'vous' for second-person address. "
                    "For dialogue, preserve each character's natural voice and relationship dynamics. "
                )
            else:
                return (
                    "Use a clear, natural narrative prose style with 'tu' for informal address. "
                    "For dialogue, preserve each character's authentic tone and relationships. "
                )

        elif lang == 'it':
            if is_formal:
                return (
                    "Use a formal literary register throughout the narration, with 'Lei' for second-person address. "
                    "For dialogue, preserve each character's natural voice and relationship dynamics. "
                )
            else:
                return (
                    "Use a natural, flowing narrative prose style with 'tu' for informal address. "
                    "For dialogue, preserve each character's authentic tone and relationships. "
                )

        elif lang == 'nl':
            if is_formal:
                return (
                    "Use a formal, polished written register throughout the narration, with 'u' for second-person address. "
                    "For dialogue, preserve each character's natural voice and relationship dynamics. "
                )
            else:
                return (
                    "Use a natural, clear narrative prose style with 'jij/je' for informal address. "
                    "For dialogue, preserve each character's authentic tone. "
                )

        elif lang in ('da', 'sv', 'no'):
            lang_name = {'da': 'Danish', 'sv': 'Swedish', 'no': 'Norwegian'}[lang]
            if is_formal:
                return (
                    f"Use a formal, elevated literary prose register appropriate for polished {lang_name} writing. "
                    "For dialogue, preserve each character's natural voice and social dynamics. "
                )
            else:
                return (
                    f"Use a clear, modern and natural prose style appropriate for contemporary {lang_name}. "
                    "For dialogue, preserve each character's authentic voice. "
                )

        elif lang == 'ar':
            if is_formal:
                return (
                    "استخدم أسلوب اللغة العربية الفصحى الرسمية في جميع أجزاء السرد والوصف. "
                    "أما في الحوارات، فحافظ على الأسلوب الطبيعي المناسب لشخصية كل متحدث وعلاقاته. "
                )
            else:
                return (
                    "استخدم أسلوب اللغة العربية المعاصرة الواضحة والسلسة في السرد. "
                    "أما في الحوارات، فحافظ على الأسلوب الطبيعي والحيوي لكل شخصية. "
                )

        elif lang == 'fa':
            if is_formal:
                return (
                    "در تمام بخش‌های روایی از نثر رسمی و ادبی با ضمیر محترمانه «شما» استفاده کنید. "
                    "در دیالوگ‌ها، لحن طبیعی و شخصیت هر کاراکتر را حفظ کنید. "
                )
            else:
                return (
                    "در بخش‌های روایی از نثر روان و طبیعی زبان فارسی معاصر استفاده کنید. "
                    "در دیالوگ‌ها، لحن اصیل و شخصیت هر کاراکتر را حفظ کنید. "
                )

        else:
            if is_formal:
                return (
                    "Use a formal, polished prose style throughout the narration. "
                    "For dialogue, preserve each character's natural voice and relationship dynamics. "
                )
            else:
                return (
                    "Use a clear, natural narrative prose style throughout. "
                    "For dialogue, preserve each character's authentic voice and tone. "
                )

    def request_gemma_api_single(self, text: str, genre_val: str, tone_val: str = "서술체 (~다)") -> str:
        genre_instruction = self.get_genre_prompt_extension(genre_val)
        tone_instruction = self.get_tone_prompt_extension(tone_val)
        prompt = (
            f"Translate the following text into {self.target_lang_prompt}. "
            f"{genre_instruction}"
            f"{tone_instruction}"
            f"Output only the translation, nothing else.\n\n{text}"
        )
        payload = {
            "model": self.gemma_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_len,
            "temperature": self.temperature,
            "top_k": 64,
            "top_p": 0.95,
        }
        try:
            response = requests.post(
                self.gemma_api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as err:
            print(f'Single API call failed: {err}')
            return text

    def _parse_llm_response(self, raw: str, expected_count: int) -> list:
        result = {}
        current_num = None
        buffer = []

        for line in raw.splitlines():
            m = re.match(r'^\s*(\d+)[.)]\s*(.*)', line)
            if m:
                if current_num is not None:
                    result[current_num] = ' '.join(buffer).strip()
                current_num = int(m.group(1))
                buffer = [m.group(2)] if m.group(2).strip() else []
            elif current_num is not None and line.strip():
                buffer.append(line.strip())

        if current_num is not None:
            result[current_num] = ' '.join(buffer).strip()

        return [result.get(i + 1, '') for i in range(expected_count)]

    def request_gemma_api_batch(self, texts: list, genre_val: str, tone_val: str = "서술체 (~다)") -> list:
        if not texts:
            return []

        genre_instruction = self.get_genre_prompt_extension(genre_val)
        tone_instruction = self.get_tone_prompt_extension(tone_val)

        glossary_instruction = ''
        if self.user_glossary:
            terms = ', '.join(f'"{src}" → "{tgt}"' for src, tgt in self.user_glossary.items())
            glossary_instruction = f'TERMINOLOGY (Strictly enforce — no exceptions): {terms}. '

        numbered_input = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        prompt = (
            f"Translate each numbered sentence below into {self.target_lang_prompt}. "
            f"{glossary_instruction}"
            f"{genre_instruction}"
            f"{tone_instruction}"
            f"Return ONLY the numbered translations in exactly the same format (1. 2. 3. ...). "
            f"Do not add any explanation or extra text.\n\n{numbered_input}"
        )
        batch_max_tokens = min(self.max_len * len(texts), 1024)
        
        batch_timeout = max(300, 120 * len(texts))
        
        payload = {
            "model": self.gemma_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": batch_max_tokens,
            "temperature": self.temperature,
            "top_k": 64,
            "top_p": 0.95,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.gemma_api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=batch_timeout,
                )
                response.raise_for_status()
                raw = response.json()['choices'][0]['message']['content'].strip()
                parsed = self._parse_llm_response(raw, len(texts))
                
                if len(parsed) == len(texts) and all(parsed):
                    return parsed
                
                print(f'  [WARNING] Batch parse partial miss ({sum(1 for p in parsed if p)}/{len(texts)}), retrying missing items individually...')
                for i, (p, original) in enumerate(zip(parsed, texts)):
                    if not p:
                        parsed[i] = self.request_gemma_api_single(original, genre_val, tone_val)
                return parsed

            except Exception as err:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f'  [ERROR] Batch API call failed ({attempt + 1}/{max_retries}): {err}. Retrying in {wait_time}s...')
                    time.sleep(wait_time)
                else:
                    print(f'  [FATAL] Batch API failed 3 times. Falling back to individual calls.')
                    return [self.request_gemma_api_single(t, genre_val, tone_val) for t in texts]

        return [self.request_gemma_api_single(t, genre_val, tone_val) for t in texts]

    def _process_translation_batch(self, args: tuple) -> list:
        idx, total_chunks, chunk, genre_val, tone_val = args
        preview = chunk[0][:40].replace('\n', ' ')
        print(f'  [Batch {idx+1}/{total_chunks}] Start ({len(chunk)} sentences) | Genre: {genre_val} | Tone: {tone_val} | "{preview}..."')
        t0 = time.time()
        result = self.request_gemma_api_batch(chunk, genre_val, tone_val)
        elapsed = time.time() - t0
        first_out = result[0][:40].replace('\n', ' ') if result else '-'
        print(f'  [Batch {idx+1}/{total_chunks}] Done {elapsed:.1f}s | → "{first_out}..."')
        return result

    def batch_translate_engine(self, only_texts, whole_particle, what, genre_val="일반 문서(기본)", tone_val="서술체 (~다)", bilingual_order="번역문(원문)"):
        particle_list_1 = []
        particle_list_2 = []

        processed_texts = list(only_texts)

        from concurrent.futures import ThreadPoolExecutor

        total = len(processed_texts)
        chunks = [
            processed_texts[i: i + self.translate_batch_size]
            for i in range(0, total, self.translate_batch_size)
        ]

        total_chunks = len(chunks)
        print(f'▶ Translation start: {total} sentences → {total_chunks} batches × {self.translate_workers} concurrent')
        t_start = time.time()

        chunk_results = []
        for _g_start in range(0, total_chunks, self.translate_workers):
            _group = chunks[_g_start: _g_start + self.translate_workers]
            _args = [(_g_start + i, total_chunks, c, genre_val, tone_val) for i, c in enumerate(_group)]
            with ThreadPoolExecutor(max_workers=len(_group)) as executor:
                _group_results = list(executor.map(self._process_translation_batch, _args))
            chunk_results.extend(_group_results)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        translated_list = [item for sublist in chunk_results for item in sublist]

        total_elapsed = time.time() - t_start
        speed = f'{total/total_elapsed:.1f} sent/s' if total_elapsed > 0 else '-'
        print(f'▶ Translation complete: {total} sentences / total {total_elapsed:.1f}s ({speed})')

        text_idx = 0

        for output_idx, whole in enumerate(whole_particle):
            if whole:
                generated_text = translated_list[output_idx - text_idx]

                if bilingual_order == "원문(번역문)":
                    translated_text_1 = "{t2} ({t1})".format(t1=generated_text, t2=whole_particle[output_idx])
                else:
                    translated_text_1 = "{t1} ({t2})".format(t1=generated_text, t2=whole_particle[output_idx])
                particle_list_1.append(translated_text_1)
                translated_text_2 = generated_text
                particle_list_2.append(translated_text_2)
            else:
                text_idx += 1
                if 'epub' in what:
                    particle_list_1.append(0)
                    particle_list_2.append(0)
                else:
                    particle_list_1.append('\n')
                    particle_list_2.append('\n')

        print('Translation reassembly complete')
        return particle_list_1, particle_list_2

    def auto_detect_genre(self, filename: str) -> str:
        prompt = (
            f"The title of a book or document is '{filename}'. Which literary genre does it most likely belong to?\n"
            "Choose EXACTLY ONE from this list: [IT 및 엔지니어링, 문학 및 소설, 인문 및 사회과학, 비즈니스 및 경제, 영상 및 대본, 일반 문서(기본)].\n"
            "Respond ONLY with the chosen genre keyword and nothing else."
        )
        payload = {
            "model": self.gemma_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 15,
            "temperature": 0.1,
        }
        try:
            res = requests.post(self.gemma_api_url, json=payload, timeout=5)
            ans = res.json()['choices'][0]['message']['content'].strip()
            for g in ["IT 및 엔지니어링", "문학 및 소설", "인문 및 사회과학", "비즈니스 및 경제", "영상 및 대본", "일반 문서(기본)"]:
                if g in ans:
                    return g
        except Exception as err:
            print(f'Genre inference failed: {err}')
        return "일반 문서(기본)"

    def on_file_upload(self, files: Sequence):
        _gd0 = GENRE_CHOICES_KO[-1]
        _td0 = TONE_CHOICES_KO[0]
        _bd0 = BILINGUAL_CHOICES_KO[0]
        try:
            print('File upload received')
            if not self.launch_time:
                self.launch_time = time.time()
            self.selected_files = files
            obj = '', None

            if not files:
                self.user_glossary = {}
                yield obj[0], obj[1], self.upload_msg, _gd0, _td0, None, gr.update(), '', self._T('glossary_count').format(n=0), _bd0
                return
            print('Attached files: ', len(files))
            if self.is_multi and len(files) > self.limit_file_count:
                self.user_glossary = {}
                yield obj[0], obj[1], f"<p style='text-align:center;color:red;'>{self._T('err_file_count').format(n=self.limit_file_count)}</p>", _gd0, _td0, None, gr.update(), '', self._T('glossary_count').format(n=0), _bd0
                return

            yield gr.update(), gr.update(), f"<p style='text-align:center;'>{self._T('status_detecting')}</p>", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            aBook = files[0]
            name, ext = os.path.splitext(aBook['orig_name'])

            print(f"[{name}] Inferring genre...")
            inferred_genre = self.auto_detect_genre(name)
            print(f"Inferred genre: {inferred_genre}")

            if '.epub' in ext:
                file = epub.read_epub(aBook['path'])
                lang = file.get_metadata('DC', 'language')
                if lang:
                    check_lang = lang[0][0]
                else:
                    print("EPUB has no language tag. Detecting from text.")
                    check_lang = 'en'
                    for item in file.get_items():
                        if item.get_type() == ebooklib.ITEM_DOCUMENT:
                            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                            all_tags = soup.find_all('p')
                            if not all_tags:
                                continue
                            text_tags = [tag.text for tag in all_tags if tag.text.strip()]
                            if not text_tags:
                                continue
                            lang_str = ' '.join(text_tags)
                            try:
                                langs = detect_langs(lang_str[0:500])
                                top = langs[0]
                                detected = top.lang if top.prob >= 0.8 else 'en'
                            except Exception:
                                detected = 'en'
                            normalized = 'zh' if detected.startswith('zh') else detected
                            if normalized in LANG_CODE_TO_NAME:
                                check_lang = normalized
                                break

            elif '.pdf' in ext:
                try:
                    if FITZ_AVAILABLE:
                        doc = fitz.open(aBook['path'])
                        text = doc[0].get_text() if len(doc) > 0 else ''
                        doc.close()
                        if text.strip():
                            try:
                                langs = detect_langs(text[:500])
                                top = langs[0]
                                check_lang = top.lang if top.prob >= 0.8 else 'en'
                            except Exception:
                                check_lang = 'en'
                        else:
                            check_lang = 'en'
                        print(f'[PDF] Language detected: {check_lang}')
                    else:
                        print('[PDF] fitz not installed — defaulting to English')
                        check_lang = 'en'
                except Exception as e:
                    print(f'[PDF] Language detection failed: {e} — defaulting to English')
                    check_lang = 'en'

            else:
                aBook_path = aBook['path']
                if self.is_check_size:
                    file_size = os.path.getsize(aBook_path) / 1024
                    if file_size > 500:
                        self.selected_files = None
                        self.user_glossary = {}
                        yield obj[0], obj[1], "<p style='text-align:center;color:red;'>제한 용량을 초과했습니다.</p><p style='text-align:center;color:skyblue;'>첨부하신 파일용량이 500kb를 넘습니다.</p><p style='text-align:center;color:skyblue;'>제한용량을 늘리기 위해서는 추가옵션 결제가 필요합니다</p>", "일반 문서(기본)", "서술체 (~다)", None, gr.update(), '', '현재 적용된 용어: 0개', "번역문(원문)"
                        return
                book = self.open_text_with_detection(aBook_path)
                raw_text = book.read()[0:1000]
                try:
                    langs = detect_langs(raw_text)
                    top = langs[0]
                    check_lang = top.lang if top.prob >= 0.8 else 'en'
                    print(f'[TXT] Language detected: {top.lang} (confidence {top.prob:.2f}) → {check_lang}')
                except Exception:
                    check_lang = 'en'

            normalized_lang = 'zh' if check_lang.startswith('zh') else check_lang
            self.origin_lang = normalized_lang
            self.origin_lang_name = LANG_CODE_TO_NAME.get(normalized_lang, f"{self._T('lang_unknown')} ({check_lang})")
            origin_dropdown_val = self.origin_lang_name if normalized_lang in LANG_CODE_TO_NAME else None
            _disp = LANG_DISPLAY_BY_UI.get(self.ui_lang, LANG_DISPLAY_BY_UI['en'])
            lang_info_display = _disp.get(self.origin_lang_name, self.origin_lang_name) if origin_dropdown_val else f"{self._T('lang_unknown')} ({check_lang})"
            auto_target = '영어' if normalized_lang == 'ko' else '한국어'
            auto_iso, auto_prompt = SUPPORTED_LANGUAGES[auto_target]
            self.target_lang = auto_iso
            self.target_lang_name = auto_target
            self.target_lang_prompt = auto_prompt
            self.user_glossary = {}
            _status_ready = self._T('status_ready').replace('\n', '<br>')
            _lang_msg = f"<p style='text-align:center;'><span style='color:skyblue;font-size:1.5em;'>{self._T('status_detected').format(lang=lang_info_display)}</span></p>"
            yield f"<p>{_status_ready}</p>", obj[1], _lang_msg, inferred_genre, _td0, origin_dropdown_val, auto_target, '', self._T('glossary_count').format(n=0), _bd0
        except Exception as err:
            print(err)
            self.user_glossary = {}
            yield obj[0], obj[1], f"<p style='text-align:center;color:red;'>{self._T('err_upload_detect')}</p>", _gd0, _td0, None, gr.update(), '', self._T('glossary_count').format(n=0), _bd0

    def open_text_with_detection(self, file_name: str):
        try:
            check_encoding = open(file_name, 'rb')
            result = chardet.detect(check_encoding.read(10000))
            print(result)
            input_file = open(file_name, 'r', encoding=result['encoding'])
            return input_file
        except:
            return None

    def initialize_output_files(self, origin_abb, target_abb, name, ext, file, bilingual_order="번역문(원문)"):
        if bilingual_order == "원문(번역문)":
            bilingual_filename = "{name}_{t2}({t3}){ext}".format(name=name, t2=origin_abb, t3=target_abb, ext=ext)
        else:
            bilingual_filename = "{name}_{t2}({t3}){ext}".format(name=name, t2=target_abb, t3=origin_abb, ext=ext)
        output_file_1 = self.create_output_file_stream(bilingual_filename)
        output_file_2 = self.create_output_file_stream(
            "{name}_{t2}{ext}".format(name=name, t2=target_abb, ext=ext)
        )
        book = self.open_text_with_detection(file['path'])
        return output_file_1, output_file_2, book

    def create_output_file_stream(self, file_name: str):
        saveDir = self.output_folder
        if not (os.path.isdir(saveDir)):
            os.makedirs(os.path.join(saveDir))
        file = os.path.join(saveDir, file_name)
        output_file = open(file, 'w', encoding='utf-8')
        return output_file

    def remove_folder(self, temp_folder: PathType):
        if os.path.exists(temp_folder):
            if os.name == 'nt':
                import gc
                gc.collect()
            shutil.rmtree(temp_folder, ignore_errors=True)

    def extract_epub_contents(self, folder_path: PathType, epub_file: PathType):
        try:
            zip_module = zipfile.ZipFile(epub_file, 'r')
            os.makedirs(folder_path, exist_ok=True)
            zip_module.extractall(folder_path)
            zip_module.close()
        except:
            print('Invalid EPUB file')
            pass

    def repack_epub_contents(self, folder_path: PathType, epub_name: PathType):
        try:
            zip_module = zipfile.ZipFile(epub_name, 'w', zipfile.ZIP_DEFLATED)
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_module.write(file_path, os.path.relpath(file_path, folder_path))
            zip_module.close()
        except Exception as err:
            print('EPUB file creation failed.')
            print(err)
            pass

    def list_epub_html_files(self) -> List:
        file_path = []
        for root, _, files in os.walk(self.temp_folder_1):
            for file in files:
                if file.endswith(('xhtml', 'html', 'htm')):
                    file_path.append(os.path.join(root, file))
        return file_path

    def locate_epub_metadata_opf(self):
        opf_path = None
        for root, _, files in os.walk(self.temp_folder_1):
            for file in files:
                if file.endswith('opf'):
                    opf_path = os.path.join(root, file)
                    return opf_path

    def calculate_elapsed_time(self, start_time, what):
        end = time.time()
        during = end - start_time
        sec = str(timedelta(seconds=during)).split('.')[0]
        return sec if what == 1 else during

    def reset_session_and_gc(self) -> str:
        gc.collect()
        sec = self.calculate_elapsed_time(self.start, 1)
        print(f'{sec}')
        self.start = None
        return sec

    def build_epub_from_soup(self, soup: BeautifulSoup, epub_path: str, title: str, lang_code: str = 'ko') -> None:
        book = epub.EpubBook()
        book.set_identifier(f'dodari-pdf-{int(time.time())}')
        book.set_title(title)
        book.set_language(lang_code)

        img_idx = 0
        for img_tag in soup.find_all('img'):
            src = img_tag.get('src', '')
            if not src.startswith('data:image/'):
                continue

            try:
                header, b64data = src.split(',', 1)
                mime_match = re.search(r'data:(image/\w+);base64', header)
                if not mime_match:
                    continue
                mime = mime_match.group(1)
                ext = mime.split('/')[1]

                img_bytes = base64.b64decode(b64data)
                img_filename = f'images/img_{img_idx:03d}.{ext}'

                epub_img = epub.EpubItem(
                    uid=f'img_{img_idx:03d}',
                    file_name=img_filename,
                    media_type=mime,
                    content=img_bytes
                )
                book.add_item(epub_img)
                img_tag['src'] = img_filename
                img_idx += 1

            except Exception as img_err:
                print(f'[EPUB] Image extraction failed (img_{img_idx:03d}): {img_err}')
                img_tag.decompose()

        print(f'[EPUB] {img_idx} image(s) extracted')

        css_content = (
            'body { font-family: serif; line-height: 1.8; margin: 2em; }\n'
            'h1, h2, h3, h4, h5, h6 { font-family: sans-serif; }\n'
            'img { max-width: 100%; height: auto; display: block; margin: 1em auto; }\n'
            'pre { background: #f4f4f4; padding: 1em; overflow-x: auto; font-size: 0.85em; }\n'
            'figcaption { font-style: italic; text-align: center; font-size: 0.9em; }\n'
        )
        css_item = epub.EpubItem(
            uid='style_main',
            file_name='styles/main.css',
            media_type='text/css',
            content=css_content.encode('utf-8')
        )
        book.add_item(css_item)

        body_content = str(soup.body) if soup.body else str(soup)

        chapter = epub.EpubHtml(
            title=title,
            file_name='content.xhtml',
            lang=lang_code
        )
        chapter.content = body_content
        chapter.add_item(css_item)

        book.add_item(chapter)
        book.toc = [epub.Link('content.xhtml', title, 'content')]
        book.spine = ['nav', chapter]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        epub.write_epub(epub_path, book, {})
        print(f'[EPUB] Packaging complete: {epub_path}')

    def finalize_file_streams(self, book, output_file_1, output_file_2):
        book.close()
        output_file_1.close()
        output_file_2.close()

    def clean_text_spacing(self, particle):
        new_particle = []
        pattern = r'\s+'
        for s in particle:
            if s.strip():
                s = re.sub(pattern, ' ', s)
                new_particle.append(s)
        return new_particle

    def contains_no_alphabets(self, text):
        pattern = r'^[^a-zA-Z]*$'
        return bool(re.match(pattern, text))

if __name__ == "__main__":
    dodari = Dodari()
    print(f'Time limit: {dodari.expire_time/60} min')
    print('File count limit:', dodari.limit_file_count)
    print('Multi-file enabled: ', dodari.is_multi)
    print('Text file size check: ', dodari.is_check_size)
    print()

    dodari.launch_interface()
