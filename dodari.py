import os
import base64
from typing import List, Union, Sequence
from datetime import timedelta
import logging, warnings
import copy
import re, time, platform, shutil, zipfile, subprocess
import requests
import chardet

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
    from docling_core.types.doc.document import ImageRefMode
    DOCLING_AVAILABLE = True
    print('[INFO] docling 임포트 성공')
except Exception as _e:
    DOCLING_AVAILABLE = False
    print(f'[WARNING] docling 임포트 실패: {_e}')

try:
    import fitz
    FITZ_AVAILABLE = True
    print('[INFO] fitz(PyMuPDF) 임포트 성공')
except Exception as _e:
    FITZ_AVAILABLE = False
    print(f'[WARNING] fitz(PyMuPDF) 임포트 실패: {_e}')

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
    print("\n[시스템 종료] MLX API 서버를 안전하게 종료합니다...")
    os.system("pkill -f 'mlx_vlm.server'")

atexit.register(cleanup_llm_server)

def format_korean_time(seconds: int) -> str:
    """초(int)를 '1시간 2분 30초' 형태의 한국어 문자열로 변환. 0인 단위는 생략."""
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

EPUB_TRANSLATE_TAGS = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'td', 'th', 'blockquote'}

class Dodari:
    def __init__(self):
        self.expire_time = 0
        self.limit_file_count = 100
        self.is_multi = True
        self.is_check_size = False

        self.app = None
        self.max_len = 512
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
            """
        self.start = None

    def launch_interface(self):
        self.remove_folder(self.temp_folder_1)
        self.remove_folder(self.temp_folder_2)

        with gr.Blocks(
            css=self.css,
            title='Dodari',
            theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")
        ) as self.app:
            gr.HTML(f"""
                <div style="text-align: center; width: 100%;">
                    <a href='https://github.com/vEduardovich/dodari' target='_blank' style='display: inline-block;'>
                    <img src='{img_src}' style='display: block; margin: 0 auto; width: 100px;'>
                    </a>
                    <h1 style='margin-top: 10px;'>
                    AI 다국어 번역기 <span style='color: red;'>
                        <a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration: none; color: red;'>도다리2</a>
                    </span> 입니다
                    </h1>
                </div>
            """)
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    with gr.Tab('순서 1'):
                        gr.HTML(f"<div style='display:flex;'><h3 style='margin-top:0px;'>1. 번역할 파일들 선택</h3><span  style='margin-left:10px;'>( *.txt, *.epub, *.pdf )</span></div>")
                        file_count = 'multiple' if self.is_multi else 'files'
                        input_window = gr.File(
                            file_count=file_count,
                            file_types=[".txt", ".epub", ".pdf"],
                            label='파일들'
                        )
                        gr.Markdown(f'한번에 {self.limit_file_count}개까지 첨부가 가능합니다')
                        lang_msg = gr.HTML(self.upload_msg)
                        self.origin_lang_display = gr.Dropdown(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            label="원본 언어 (자동 감지 · 수동 변경 가능)",
                            interactive=True,
                            value=None
                        )

                with gr.Column(scale=1, min_width=300):
                    with gr.Tab('순서 2'):
                        self.target_lang_radio = gr.Radio(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            value='한국어',
                            label='번역 목표 언어'
                        )
                        gr.HTML("<p style='color:green;'>✔ Apple MLX 고속 번역 엔진 활성화됨</p>")

                        _model_choices = [
                            "mlx-community/gemma-4-e4b-it-8bit",
                            "mlx-community/gemma-4-31b-it-4bit",
                        ]
                        self.model_radio = gr.Radio(
                            choices=_model_choices,
                            label="모델 선택 (E4B: 16GB 이하 권장 · 31B: 32GB 이상 고품질, 교체 시 서버 재시작 소요)",
                            value=_model_choices[0]
                        )
                with gr.Column(scale=1, min_width=300):
                    with gr.Tab('순서 3'):
                        self.bilingual_order_radio = gr.Radio(
                            choices=["번역문(원문)", "원문(번역문)"],
                            value="번역문(원문)",
                            label="이중언어 표기 방식 (학습용은 '원문(번역문)' 추천)"
                        )
                        self.genre_radio = gr.Radio(
                            choices=["IT 및 엔지니어링", "문학 및 소설", "인문 및 사회과학", "비즈니스 및 경제", "영상 및 대본", "일반 문서(기본)"],
                            label="장르 지정 (AI가 자동 추론)",
                            value="일반 문서(기본)"
                        )
                        self.tone_radio = gr.Radio(
                            choices=["서술체 (~다)", "경어체 (~합니다)"],
                            value="서술체 (~다)",
                            label="문체 선택 (일관된 어투 유지)"
                        )

                        with gr.Accordion('✨ 용어집', open=False):
                            glossary_extract_btn = gr.Button('🔍 AI 용어 자동 추출', variant='secondary')
                            glossary_status = gr.Markdown('')
                            self.glossary_textbox = gr.Textbox(
                                label='용어집 (원문: 번역어 형식, 줄바꿈으로 구분)',
                                placeholder='James: 제임스\nEldoria: 엘도리아\nDark Magic: 어둠의 마법',
                                lines=6,
                                value=''
                            )
                            glossary_apply_btn = gr.Button('✅ 용어집 적용', variant='primary')
                            glossary_clear_btn = gr.Button('🗑️ 용어집 초기화', variant='stop')
                            self.glossary_count_md = gr.Markdown('현재 적용된 용어: 0개')
                            gr.Markdown(
                                '소설 인물 이름·전문 용어가 페이지마다 달라지는 문제를 방지합니다.\n\n'
                                '**① 자동 추출:** 버튼을 누르면 AI가 파일에서 중요 용어를 찾아 제안합니다.\n\n'
                                '**② 직접 입력:** 아래 텍스트박스에 `원문: 번역어` 형식으로 한 줄씩 작성해도 됩니다.\n\n'
                                '(예: `James: 제임스`, `Eldoria: 엘도리아`)'
                            )

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
                                return f'✅ **{count}개의 용어가 적용되었습니다.** 이제 번역 시 이 용어들이 우선 사용됩니다.'
                            return '⚠️ 적용된 용어가 없습니다. `원문: 번역어` 형식으로 입력하세요.'

                        def clear_glossary():
                            self.user_glossary = {}
                            return '', '용어집이 초기화되었습니다.', '현재 적용된 용어: 0개'

                        def extract_glossary_with_ai():
                            if not self.selected_files:
                                yield '⚠️ 파일을 먼저 첨부해 주세요.', ''
                                return

                            yield '🔍 파일에서 고유명사 후보를 스캔 중입니다...', ''

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
                                    print(f'[용어 추출] 파일 읽기 오류 ({name}): {e}')

                            full_text = ' '.join(raw_texts)
                            if not full_text.strip():
                                yield '⚠️ 텍스트를 읽을 수 없습니다.', ''
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
                                yield '⚠️ 번역할 고유명사 후보를 찾지 못했습니다.', ''
                                return

                            yield f'✨ {len(top_candidates)}개의 후보를 발견했습니다. AI가 정제 중입니다...', ''

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
                                    yield '⚠️ AI가 용어를 추출하지 못했습니다. 직접 입력해 주세요.', ''
                                    return

                                result_text = '\n'.join(valid_lines)
                                yield f'✅ **{len(valid_lines)}개의 용어를 추출했습니다!** 아래 내용을 확인하고 [용어집 적용]을 눌러주세요.', result_text

                            except Exception as e:
                                yield f'⚠️ AI 추출 중 오류가 발생했습니다: {e}', ''

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
                            fn=lambda: f'현재 적용된 용어: {len(self.user_glossary)}개',
                            inputs=[],
                            outputs=[self.glossary_count_md]
                        )

                        glossary_clear_btn.click(
                            fn=clear_glossary,
                            inputs=[],
                            outputs=[self.glossary_textbox, glossary_status, self.glossary_count_md]
                        )

                with gr.Column(scale=2):
                    with gr.Tab('순서 4'):
                        translate_btn = gr.Button(
                            value="번역 실행하기",
                            size='lg',
                            variant="primary",
                            interactive=True
                        )
                        with gr.Tab('상태창'):
                            status_msg = gr.HTML('', visible=True)
                            done_files = gr.File(label='번역결과 다운로드', file_count='multiple', interactive=False, visible=True)
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

        self.app.queue().launch(
            share=True,
            inbrowser=True,
            favicon_path='imgs/dodari.png',
            allowed_paths=['.', './outputs']
        )

    def reload_llm_server(self, new_model: str):
        if self.gemma_model == new_model:
            return

        print(f"\n[모델 교체] {new_model} 서버 로딩을 시작합니다...")
        gr.Info(f"모델을 {new_model}로 교체 중입니다. 잠시 기다려주세요.")
        self.gemma_model = new_model

        cleanup_llm_server()
        time.sleep(2)

        mlx_python = os.environ.get('MLX_PYTHON', 'python3')
        cmd = (
            f"{mlx_python} -m mlx_vlm.server "
            f"--model {new_model} "
            f"--kv-bits {self.kv_bits} "
            f"--port 8000"
        )
        print(f"[모델 교체] MLX 파라미터: batch={self.translate_batch_size}, workers={self.translate_workers}, kv-bits={self.kv_bits}")
        subprocess.Popen(cmd, shell=True)
        time.sleep(8)

        gr.Info("모델 셋업이 완료되었습니다!")
        print(f"[모델 교체] 완료!\n")

    def format_result_message(self, sec_or_msg):
        if not sec_or_msg:
            return ""
        if str(sec_or_msg).startswith("<p"):
            return sec_or_msg
        return f"번역 완료! 걸린시간 : {sec_or_msg} 하단에서 결과물을 다운로드하세요."

    def execute_translation_pipeline(self, genre_val, tone_val="서술체 (~다)", target_lang_name="한국어", bilingual_order_val="번역문(원문)", progress=gr.Progress()):
        if self.expire_time:
            during = self.calculate_elapsed_time(self.launch_time, 2)
            if during > self.expire_time:
                over_time = during - self.expire_time
                over_time_str = str(timedelta(seconds=over_time)).split('.')[0]
                print('제한 시간이 지남 : ', over_time_str)
                return None, f"<p style='color:red;'>도다리 사용시간이 {over_time_str}만큼 지났습니다</p>"

        if self.is_multi and len(self.selected_files) > self.limit_file_count:
            return None, f"<p style='color:red;'>한번에 {self.limit_file_count}개 이상의 파일을 첨부할수 없습니다 파일을 다시 첨부해주세요</p>"

        if not self.selected_files:
            return None, "<p style='color:red;'>번역할 파일을 추가하세요</p>"

        self.start = time.time()
        print("Start! now.." + str(self.start))
        progress(0, desc="번역 모델을 준비중입니다...")

        progress(0, desc="번역 서버 상태 확인 중...")
        server_ok = False
        for attempt in range(5):
            try:
                resp = requests.get('http://localhost:8000/v1/models', timeout=3)
                if resp.status_code == 200:
                    server_ok = True
                    break
            except Exception:
                pass
            print(f'[서버 대기] localhost:8000 응답 없음 ({attempt + 1}/5), 2초 후 재시도...')
            time.sleep(2)

        if not server_ok:
            return (
                None,
                "<p style='color:red;'>[오류] 번역 서버(localhost:8000)에 연결할 수 없습니다.<br>"
                "<code>start_mac.sh</code> 실행 여부를 확인하세요.</p>"
            )
        print('Gemma API 번역 준비 완료')

        if not self.origin_lang:
            return None, "<p style='color:red;'>언어 감지가 완료되지 않았습니다.<br>파일을 다시 첨부한 후 언어 확인까지 완료해주세요.</p>"

        target_iso, target_prompt = SUPPORTED_LANGUAGES.get(target_lang_name, ('ko', 'Korean'))
        self.target_lang = target_iso
        self.target_lang_name = target_lang_name
        self.target_lang_prompt = target_prompt

        if self.origin_lang == target_iso:
            return None, f"<p style='color:red;'>원본 언어와 목표 언어가 같습니다 ({target_lang_name}).<br>다른 목표 언어를 선택한 후 다시 시도해주세요.</p>"

        origin_abb = self.origin_lang
        target_abb = target_iso
        all_file_path = []

        for file in progress.tqdm(self.selected_files, desc='파일로딩'):
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
                print('언어수정완료')

                file_path = self.list_epub_html_files()
                print('파일개수: ', len(file_path))
                for html_file in progress.tqdm(file_path, desc='챕터'):
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
                        for text_node_1, text_node_2 in progress.tqdm(zip(p_tags_1, p_tags_2), desc=f'단락 번역 중 (경과: {_ch_elapsed})'):
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
                                    print('이미지 태그도 추가해준다')
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
                        print('html로딩 중 문제가 생겼지만 넘어간다')
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
                print(f'[PDF 번역] 시작: {name}{ext}')
                print(f'[PDF 번역] 언어: {origin_abb} → {target_abb} | 모델: {self.gemma_model}')

                if not DOCLING_AVAILABLE:
                    print('[PDF 번역] 오류: docling 패키지가 설치되지 않았습니다. pip install docling 실행 필요.')
                    continue

                try:
                    os.makedirs(self.output_folder, exist_ok=True)
                    progress(0, desc='[PDF] Docling을 이용한 HTML 구조화 중...')

                    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                    
                    print(f"\n[PDF 처리 시작] {file['path']} - Docling 가동 (CPU 전용 모드)")
                    progress(0, desc='[PDF] Docling 엔진 로딩 중...')
                    
                    try:
                        pipeline_options = PdfPipelineOptions()
                        pipeline_options.generate_picture_images = True
                        pipeline_options.images_scale = 2.0
                        pipeline_options.accelerator_options = AcceleratorOptions(
                            num_threads=8,
                            device=AcceleratorDevice.CPU
                        )
                        
                        print("[PDF] DocumentConverter 초기화 중...")
                        converter = DocumentConverter(
                            format_options={'pdf': PdfFormatOption(pipeline_options=pipeline_options)}
                        )
                        
                        print("[PDF] HTML 구조 추출 시작 (이 과정은 PDF 용량에 따라 1~3분 정도 소요될 수 있습니다)...")
                        progress(5, desc='[PDF] HTML 구조 추출 중 (CPU)...')
                        result = converter.convert(file['path'])
                        print("[PDF] 구조 추출 완료!")
                        
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

                        print(f'[PDF] 삭제 텍스트 {len(picture_delete)}개 / 번역생략 텍스트 {len(picture_skip)}개')
                    except Exception as pe:
                        import traceback as _tb
                        print(f'[PDF] 이미지 텍스트 수집 오류: {pe}')
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
                        print(f'[PDF] 코드 블록 이미지 {len(code_block_images)}개 크롭 완료')
                    except Exception as ce:
                        import traceback as _tb
                        print(f'[PDF] 코드 블록 이미지 변환 오류: {ce}')
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
                        print(f'[PDF] 넓은 표 이미지 {len(wide_table_images)}개 크롭 완료 (전체 표: {table_idx}개)')

                    except Exception as te:
                        import traceback as _tb
                        print(f'[PDF] 넓은 표 이미지 변환 오류: {te}')
                        wide_table_images = {}
                        _tb.print_exc()

                    html_content = result.document.export_to_html(image_mode=ImageRefMode.EMBEDDED)
                    print('[PDF 번역] HTML 구조화 완료. 텍스트 번역 파이프라인(BeautifulSoup) 시작...')

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
                        progress(0.7, desc=f'[PDF] 문맥 기반 텍스트 배치 번역 중... (경과: {format_korean_time(int(time.time() - self.start))})')
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
                        
                        for t_idx, valid_tag_1 in enumerate(progress.tqdm(valid_tags_1, desc='HTML 재조립')):
                            trans_1 = assembled_1[t_idx]
                            trans_2 = assembled_2[t_idx]
                            
                            valid_tag_1.clear()
                            valid_tag_1.string = trans_1

                            valid_tags_2[t_idx].clear()
                            valid_tags_2[t_idx].string = trans_2
                            
                    progress(0.95, desc=f'[PDF] EPUB 패키징 중... (경과: {format_korean_time(int(time.time() - self.start))})')
                    if bilingual_order_val == "원문(번역문)":
                        done_path_1 = os.path.join(self.output_folder, f"{name}_{origin_abb}({target_abb}).epub")
                    else:
                        done_path_1 = os.path.join(self.output_folder, f"{name}_{target_abb}({origin_abb}).epub")
                    done_path_2 = os.path.join(self.output_folder, f"{name}_{target_abb}.epub")

                    self.build_epub_from_soup(soup_1, done_path_1, name, lang_code=target_abb)
                    self.build_epub_from_soup(soup_2, done_path_2, name, lang_code=target_abb)

                    all_file_path.extend([done_path_1, done_path_2])
                    print(f'[PDF 번역] 성공! EPUB 생성: {done_path_1}, {done_path_2}')
                    
                except Exception as err:
                    import traceback
                    print(f'[PDF 번역] 오류 발생: {err}')
                    traceback.print_exc()
                    continue

            else:
                output_file_1, output_file_2, book = self.initialize_output_files(origin_abb, target_abb, name, ext, file, bilingual_order_val)
                book_raw = book.read()
                sentences = book_raw.split(sep='\n')

                only_texts = []
                whole_particle = []
                for sen in progress.tqdm(sentences, desc='단락'):
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
            print(f'단일 API 호출 실패: {err}')
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
                
                print(f'  [경고] 배치 파싱 일부 누락 ({sum(1 for p in parsed if p)}/{len(texts)}), 누락 항목 개별 보충 중...')
                for i, (p, original) in enumerate(zip(parsed, texts)):
                    if not p:
                        parsed[i] = self.request_gemma_api_single(original, genre_val, tone_val)
                return parsed

            except Exception as err:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f'  [오류] 배치 API 호출 실패 ({attempt + 1}/{max_retries}): {err}. {wait_time}초 후 재시도...')
                    time.sleep(wait_time)
                else:
                    print(f'  [치명적] 배치 API 3회 모두 실패. 개별 호출 폴백 모드 진입.')
                    return [self.request_gemma_api_single(t, genre_val, tone_val) for t in texts]

        return [self.request_gemma_api_single(t, genre_val, tone_val) for t in texts]

    def _process_translation_batch(self, args: tuple) -> list:
        idx, total_chunks, chunk, genre_val, tone_val = args
        preview = chunk[0][:40].replace('\n', ' ')
        print(f'  [배치 {idx+1}/{total_chunks}] 시작 ({len(chunk)}문장) | 장르: {genre_val} | 문체: {tone_val} | "{preview}..."')
        t0 = time.time()
        result = self.request_gemma_api_batch(chunk, genre_val, tone_val)
        elapsed = time.time() - t0
        first_out = result[0][:40].replace('\n', ' ') if result else '-'
        print(f'  [배치 {idx+1}/{total_chunks}] 완료 {elapsed:.1f}s | → "{first_out}..."')
        return result

    def batch_translate_engine(self, only_texts, whole_particle, what, genre_val="일반 문서(기본)", tone_val="서술체 (~다)", bilingual_order="번역문(원문)"):
        particle_list_1 = []
        particle_list_2 = []

        processed_texts = []
        for text in only_texts:
            text = re.sub(r'^[^\w\s]', '', text)
            text = re.sub(r':', '-', text)
            processed_texts.append(text)

        from concurrent.futures import ThreadPoolExecutor

        total = len(processed_texts)
        chunks = [
            processed_texts[i: i + self.translate_batch_size]
            for i in range(0, total, self.translate_batch_size)
        ]

        total_chunks = len(chunks)
        print(f'▶ 번역 시작: {total}문장 → {total_chunks}배치 × 동시{self.translate_workers}개')
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
        print(f'▶ 전체 번역 완료: {total}문장 / 총 {total_elapsed:.1f}s ({total/total_elapsed:.1f}문장/s)')

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

        print('번역 재조립 완료')
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
            print(f'장르 추론 실패: {err}')
        return "일반 문서(기본)"

    def on_file_upload(self, files: Sequence):
        try:
            print('들어옴')
            if not self.launch_time:
                self.launch_time = time.time()
            self.selected_files = files
            obj = '', None

            if not files:
                self.user_glossary = {}
                yield obj[0], obj[1], self.upload_msg, "일반 문서(기본)", "서술체 (~다)", None, gr.update(), '', '현재 적용된 용어: 0개', "번역문(원문)"
                return
            print('첨부 파일 수 : ', len(files))
            if self.is_multi and len(files) > self.limit_file_count:
                self.user_glossary = {}
                yield obj[0], obj[1], f"<p style='text-align:center;color:red;'>한번에 {self.limit_file_count}개 이상의 파일을 번역할 수 없습니다.</p>", "일반 문서(기본)", "서술체 (~다)", None, gr.update(), '', '현재 적용된 용어: 0개', "번역문(원문)"
                return

            yield gr.update(), gr.update(), "<p style='text-align:center;'>🔍 언어 감지중입니다...</p>", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            aBook = files[0]
            name, ext = os.path.splitext(aBook['orig_name'])

            print(f"[{name}] 의 장르를 추론 중입니다...")
            inferred_genre = self.auto_detect_genre(name)
            print(f"추론된 장르: {inferred_genre}")

            if '.epub' in ext:
                file = epub.read_epub(aBook['path'])
                lang = file.get_metadata('DC', 'language')
                if lang:
                    check_lang = lang[0][0]
                else:
                    print("언어 설정이 되어있지 않은 epub입니다. 텍스트에서 언어를 감지합니다.")
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
                        print(f'[PDF] 언어 감지 결과: {check_lang}')
                    else:
                        print('[PDF] fitz 미설치 — 영어 원문 기본값 적용')
                        check_lang = 'en'
                except Exception as e:
                    print(f'[PDF] 언어 감지 실패: {e} — 영어 원문 기본값 적용')
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
                    print(f'[TXT] 언어 감지: {top.lang} (신뢰도 {top.prob:.2f}) → {check_lang}')
                except Exception:
                    check_lang = 'en'

            normalized_lang = 'zh' if check_lang.startswith('zh') else check_lang
            self.origin_lang = normalized_lang
            self.origin_lang_name = LANG_CODE_TO_NAME.get(normalized_lang, f'알 수 없음 ({check_lang})')
            origin_dropdown_val = self.origin_lang_name if normalized_lang in LANG_CODE_TO_NAME else None
            lang_info = self.origin_lang_name if origin_dropdown_val else f'알 수 없음 ({check_lang})'
            auto_target = '영어' if normalized_lang == 'ko' else '한국어'
            auto_iso, auto_prompt = SUPPORTED_LANGUAGES[auto_target]
            self.target_lang = auto_iso
            self.target_lang_name = auto_target
            self.target_lang_prompt = auto_prompt
            self.user_glossary = {}
            yield "<p>번역준비를 마쳤습니다.</p><p>위에 '번역실행하기' 버튼을 클릭하세요</p>", obj[1], f"<p style='text-align:center;'><span style='color:skyblue;font-size:1.5em;'>{lang_info}</span><span> 문서가 감지되었습니다. 목표 언어를 선택하고 번역을 시작하세요.</span></p>", inferred_genre, "서술체 (~다)", origin_dropdown_val, auto_target, '', '현재 적용된 용어: 0개', "번역문(원문)"
        except Exception as err:
            print(err)
            self.user_glossary = {}
            yield obj[0], obj[1], "<p style='text-align:center;color:red;'>어떤 언어인지 알아내는데 실패했습니다.</p>", "일반 문서(기본)", "서술체 (~다)", None, gr.update(), '', '현재 적용된 용어: 0개', "번역문(원문)"

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
            shutil.rmtree(temp_folder)

    def extract_epub_contents(self, folder_path: PathType, epub_file: PathType):
        try:
            zip_module = zipfile.ZipFile(epub_file, 'r')
            os.makedirs(folder_path, exist_ok=True)
            zip_module.extractall(folder_path)
            zip_module.close()
        except:
            print('잘못된 epub파일입니다')
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
            print('epub 파일을 생성하는데 실패했습니다.')
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
        """번역된 soup를 EPUB 파일로 패키징한다.

        Phase A: Base64 인라인 이미지 → 독립 EpubItem 분리 (e-Reader 메모리 과부하 해소)
        Phase B: 기본 CSS + 단일 챕터 HTML을 ebooklib으로 EPUB 구조 조립
        """
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
                print(f'[EPUB] 이미지 추출 실패 (img_{img_idx:03d}): {img_err}')
                img_tag.decompose()

        print(f'[EPUB] 이미지 {img_idx}개 분리 완료')

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
        print(f'[EPUB] 패키징 완료: {epub_path}')

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
    print(f'시간제한: {dodari.expire_time/60}분')
    print('파일 개수 제한수:', dodari.limit_file_count)
    print('멀티 파일 가능여부: ', dodari.is_multi)
    print('텍스트 파일용량 제한: ', dodari.is_check_size)
    print()

    dodari.launch_interface()
