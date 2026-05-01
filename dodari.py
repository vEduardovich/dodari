# rtx4090용으로 mlx --> vlm으로 전환하기

import os
import base64
from typing import List, Union, Sequence
from datetime import timedelta
import logging, warnings
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
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
    print('[INFO] fitz(PyMuPDF) 임포트 성공')
except Exception as _e:
    FITZ_AVAILABLE = False
    print(f'[WARNING] fitz(PyMuPDF) 임포트 실패: {_e}')

import ebooklib
from ebooklib import epub
from langdetect import detect, detect_langs, DetectorFactory
DetectorFactory.seed = 0  # 결정적 감지 결과 보장 (확률적 오탐 방지)
import nltk

from bs4 import BeautifulSoup
import gradio as gr
import gc
from xml.etree.ElementTree import parse
import atexit

def cleanup_llm_server():
    # 앱 종료 시 MLX 서버 프로세스를 안전하게 정리
    print("\n[시스템 종료] MLX API 서버를 안전하게 종료합니다...")
    os.system("pkill -f 'mlx_vlm.server'")

atexit.register(cleanup_llm_server)

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

# 지원 언어 목록 — 표시명: (ISO 639-1 코드, LLM 프롬프트 언어명)
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
# langdetect 반환 코드 → 표시명 역매핑 (원본 언어 자동 감지 표시용)
LANG_CODE_TO_NAME = {v[0]: k for k, v in SUPPORTED_LANGUAGES.items()}
LANG_CODE_TO_NAME['zh-cn'] = '중국어'  # 중국어 간체 변형
LANG_CODE_TO_NAME['zh-tw'] = '중국어'  # 중국어 번체 변형

# EPUB 번역 대상 허용 태그 목록 (블록·인라인 모두 포함)
# h1~h6: 챕터/섹션 제목, li: 목차, span: 인라인 강조, td/th: 표, blockquote: 인용문
EPUB_TRANSLATE_TAGS = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'td', 'th', 'blockquote'}

class Dodari:
    def __init__(self):
        self.expire_time = 0  # xxxx초 후에 서비스 종료된다! 0이라면 제한 x
        self.limit_file_count = 100
        self.is_multi = True  # 여러파일 번역 가능하게하려면 True로 바꿔준다
        self.is_check_size = False  # 텍스트 파일용량을 제한할 것인지 체크

        self.app = None
        self.max_len = 512
        # Apple Silicon MLX Continuous Batching 최적화
        # E4B 경량 모델 기본값: 배치 크기 상향 → 속도 개선
        self.translate_batch_size = 20  # E4B 경량 모델 기본값으로 15→20 상향
        self.translate_workers = 4      # MLX는 동시 요청 증가 시 스케줄링 오버헤드 발생
        self.kv_bits = 8
        self.temperature = 1
        self.launch_time = None
        self.selected_files = []
        self.upload_msg = None
        self.origin_lang_str = None
        self.target_lang_str = None
        self.origin_lang = None        # 원본 언어 ISO 코드 (langdetect 감지 결과)
        self.origin_lang_name = None   # 원본 언어 표시명 (LANG_CODE_TO_NAME 변환 결과)
        self.target_lang = 'ko'        # 목표 언어 ISO 코드 (기본: 한국어)
        self.target_lang_name = '한국어'  # 목표 언어 표시명
        self.target_lang_prompt = 'Korean'  # LLM 프롬프트용 언어명 (기본: 한국어)

        # 용어집 — 사용자가 확정한 {원문: 번역어} 딕셔너리
        # 비어있으면 용어집 프롬프트를 주입하지 않음 (선택 기능)
        self.user_glossary: dict = {}

        # Apple MLX LLM API 서버 설정 (localhost:8000, OpenAI 호환)
        self.gemma_api_url = 'http://localhost:8000/v1/chat/completions'
        self.gemma_model = 'mlx-community/gemma-4-e4b-it-8bit'  # 16GB RAM 이하 사용자를 위한 경량 기본값

        self.output_folder = 'outputs'  # 번역 결과가 저장되는 폴더
        self.temp_folder_1 = 'temp_1'  # 임시로 압축 풀 폴더
        self.temp_folder_2 = 'temp_2'  # 임시로 압축 풀 폴더

        self.css = """
            .radio-group .wrap {
                display: float !important;
                grid-template-columns: 1fr 1fr;
            }
            """
        self.start = None  # 시간 측정 시작

    def launch_interface(self):
        self.remove_folder(self.temp_folder_1)  # 시작하기 전에 temp폴더들 모두 삭제하기
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
                        <a href='https://github.com/vEduardovich/dodari' target='_blank' style='text-decoration: none; color: red;'>도다리</a>
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
                        # 원본 언어 — 자동 감지 후 드롭다운에 표시, 오감지 시 수동 변경 가능
                        self.origin_lang_display = gr.Dropdown(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            label="원본 언어 (자동 감지 · 수동 변경 가능)",
                            interactive=True,
                            value=None
                        )

                with gr.Column(scale=1, min_width=300):
                    with gr.Tab('순서 2'):
                        # 목표 언어 — 라디오 버튼으로 한눈에 전체 선택지 표시
                        self.target_lang_radio = gr.Radio(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            value='한국어',
                            label='번역 목표 언어'
                        )
                        # Apple Silicon 전용 고속 추론 엔진
                        gr.HTML("<p style='color:green;'>✔ Apple MLX 고속 번역 엔진 활성화됨</p>")

                        # 모델 선택: E4B(경량·기본) / 31B(고품질·고사양)
                        _model_choices = [
                            "mlx-community/gemma-4-e4b-it-8bit",   # 16GB RAM 이하 권장
                            "mlx-community/gemma-4-31b-it-4bit",   # 32GB RAM 이상 권장
                        ]
                        self.model_radio = gr.Radio(
                            choices=_model_choices,
                            label="모델 선택 (E4B: 16GB 이하 권장 · 31B: 32GB 이상 고품질, 교체 시 서버 재시작 소요)",
                            value=_model_choices[0]
                        )
                with gr.Column(scale=1, min_width=300):
                    with gr.Tab('순서 3'):
                        # 이중언어 표기 순서 선택 — 기본은 번역문(원문), 학습자는 원문(번역문) 추천
                        self.bilingual_order_radio = gr.Radio(
                            choices=["번역문(원문)", "원문(번역문)"],
                            value="번역문(원문)",
                            label="이중언어 표기 방식 (공부용은 '원문(번역문)' 추천)"
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

                        # 용어집 섹션 — 선택 기능이므로 접을 수 있는 Accordion으로 배치
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

                        # 원본 언어 수동 변경 시 인스턴스 변수 갱신 (오감지 보정용)
                        def on_origin_lang_change(lang_name):
                            if lang_name and lang_name in SUPPORTED_LANGUAGES:
                                iso_code, _ = SUPPORTED_LANGUAGES[lang_name]
                                self.origin_lang = iso_code
                                self.origin_lang_name = lang_name

                        self.origin_lang_display.change(
                            fn=on_origin_lang_change,
                            inputs=[self.origin_lang_display]
                        )

                        # 목표 언어 변경 시 인스턴스 변수 즉시 갱신 (번역 프롬프트 자동 반영)
                        def on_target_lang_change(lang_name):
                            iso_code, prompt_name = SUPPORTED_LANGUAGES.get(lang_name, ('ko', 'Korean'))
                            self.target_lang = iso_code
                            self.target_lang_name = lang_name
                            self.target_lang_prompt = prompt_name

                        self.target_lang_radio.change(
                            fn=on_target_lang_change,
                            inputs=[self.target_lang_radio]
                        )

                        # ── 용어집 관련 함수 ──────────────────────────────────────
                        def apply_glossary(text: str):
                            # 텍스트박스의 줄바꿈 구분 용어 목록을 파싱하여 self.user_glossary에 저장
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
                            # 최대 50개 — 초과 시 프롬프트가 과도하게 길어지는 것을 방지
                            self.user_glossary = dict(list(glossary.items())[:50])
                            count = len(self.user_glossary)
                            if count > 0:
                                return f'✅ **{count}개의 용어가 적용되었습니다.** 이제 번역 시 이 용어들이 우선 사용됩니다.'
                            return '⚠️ 적용된 용어가 없습니다. `원문: 번역어` 형식으로 입력하세요.'

                        def clear_glossary():
                            # 용어집을 완전히 초기화한다
                            self.user_glossary = {}
                            return '', '용어집이 초기화되었습니다.', '현재 적용된 용어: 0개'

                        def extract_glossary_with_ai():
                            # 1단계: 파일에서 대문자 고유명사 후보를 고속 수집
                            # 2단계: Gemma 4로 핵심 20~30개를 정제하고 번역어를 제안
                            if not self.selected_files:
                                yield '⚠️ 파일을 먼저 첨부해 주세요.', ''
                                return

                            yield '🔍 파일에서 고유명사 후보를 스캔 중입니다...', ''

                            # --- 1단계: 파일에서 텍스트 추출 ---
                            from collections import Counter
                            import re as _re
                            raw_texts = []
                            for file in self.selected_files[:3]:  # 최대 3개 파일만 스캔 (속도 보장)
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
                                        # PyMuPDF로 빠른 텍스트 추출 (Docling 미사용)
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

                            # 영문 대문자 시작 단어 후보 추출 (2글자 이상, 일반 관사·전치사 제외)
                            stopwords = {
                                'The', 'A', 'An', 'In', 'On', 'At', 'Of', 'And', 'Or', 'But',
                                'Is', 'Was', 'Are', 'Were', 'He', 'She', 'It', 'They', 'We',
                                'I', 'You', 'His', 'Her', 'This', 'That', 'With', 'For', 'To',
                                'From', 'By', 'As', 'Be', 'Have', 'Has', 'Had', 'Not', 'Do',
                            }
                            candidates_raw = _re.findall(r'\b([A-Z][a-zA-Z\']{1,})\b', full_text)
                            freq = Counter(c for c in candidates_raw if c not in stopwords)
                            # 빈도수 상위 40개만 선별하여 AI에게 전달 (토큰 절약)
                            top_candidates = [word for word, _ in freq.most_common(40)]

                            if not top_candidates:
                                yield '⚠️ 번역할 고유명사 후보를 찾지 못했습니다.', ''
                                return

                            yield f'✨ {len(top_candidates)}개의 후보를 발견했습니다. AI가 정제 중입니다...', ''

                            # --- 2단계: Gemma 4로 핵심 용어 정제 및 번역어 제안 ---
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
                                    'temperature': 0.3,  # 용어 추출은 낮은 온도로 일관성 유지
                                }
                                response = requests.post(
                                    self.gemma_api_url,
                                    headers={'Content-Type': 'application/json'},
                                    json=payload,
                                    timeout=60
                                )
                                response.raise_for_status()
                                ai_result = response.json()['choices'][0]['message']['content'].strip()

                                # 결과 파싱 — "원문: 번역어" 형식 줄만 추출
                                lines = [l.strip() for l in ai_result.splitlines() if ':' in l and l.strip()]
                                valid_lines = [l for l in lines if len(l.split(':', 1)) == 2]

                                if not valid_lines:
                                    yield '⚠️ AI가 용어를 추출하지 못했습니다. 직접 입력해 주세요.', ''
                                    return

                                result_text = '\n'.join(valid_lines)
                                yield f'✅ **{len(valid_lines)}개의 용어를 추출했습니다!** 아래 내용을 확인하고 [용어집 적용]을 눌러주세요.', result_text

                            except Exception as e:
                                yield f'⚠️ AI 추출 중 오류가 발생했습니다: {e}', ''

                        # 용어 자동 추출 버튼 — generator 함수라 스트리밍 상태 업데이트 가능
                        glossary_extract_btn.click(
                            fn=extract_glossary_with_ai,
                            inputs=[],
                            outputs=[glossary_status, self.glossary_textbox]
                        )

                        # 용어집 적용 버튼
                        glossary_apply_btn.click(
                            fn=apply_glossary,
                            inputs=[self.glossary_textbox],
                            outputs=[glossary_status]
                        ).then(
                            fn=lambda: f'현재 적용된 용어: {len(self.user_glossary)}개',
                            inputs=[],
                            outputs=[self.glossary_count_md]
                        )

                        # 용어집 초기화 버튼
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

        # 기존 MLX 프로세스 종료 후 새 서버 기동 (포트 충돌 방지)
        cleanup_llm_server()
        time.sleep(2)

        # start_mac.sh에서 export한 MLX_PYTHON 경로 사용 (venv 격리 문제 우회)
        mlx_python = os.environ.get('MLX_PYTHON', 'python3')
        cmd = (
            f"{mlx_python} -m mlx_vlm.server "
            f"--model {new_model} "
            f"--kv-bits {self.kv_bits} "
            f"--port 8000"
        )
        print(f"[모델 교체] MLX 파라미터: batch={self.translate_batch_size}, workers={self.translate_workers}, kv-bits={self.kv_bits}")
        subprocess.Popen(cmd, shell=True)
        time.sleep(8)  # MLX 모델 로딩 대기

        gr.Info("모델 셋업이 완료되었습니다!")
        print(f"[모델 교체] 완료!\n")

    def format_result_message(self, sec_or_msg):
        if not sec_or_msg:
            return ""
        if str(sec_or_msg).startswith("<p"):
            return sec_or_msg
        return f"번역 완료! 걸린시간 : {sec_or_msg} 하단에서 결과물을 다운로드하세요."

    def execute_translation_pipeline(self, genre_val, tone_val="서술체 (~다)", target_lang_name="한국어", bilingual_order_val="번역문(원문)", progress=gr.Progress()):
        if self.expire_time:  # 시간 제한이 있다면
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

        # API 서버(localhost:8000) 응답 여부 확인 — 최대 5회 재시도 (총 10초 대기)
        # /v1/models 엔드포인트로 헬스체크 (OpenAI 호환 표준)
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

        # 선택된 목표 언어의 ISO 코드와 LLM 프롬프트명을 SUPPORTED_LANGUAGES에서 조회
        target_iso, target_prompt = SUPPORTED_LANGUAGES.get(target_lang_name, ('ko', 'Korean'))
        self.target_lang = target_iso
        self.target_lang_name = target_lang_name
        self.target_lang_prompt = target_prompt

        # 원본 언어와 목표 언어가 같으면 번역 의미 없음 → 경고 후 중단
        if self.origin_lang == target_iso:
            return None, f"<p style='color:red;'>원본 언어와 목표 언어가 같습니다 ({target_lang_name}).<br>다른 목표 언어를 선택한 후 다시 시도해주세요.</p>"

        # ISO 639-1 코드를 파일명 접두사로 사용 (예: "ko", "en", "ja")
        origin_abb = self.origin_lang
        target_abb = target_iso
        all_file_path = []

        for file in progress.tqdm(self.selected_files, desc='파일로딩'):
            print(f'file: {file}')
            name, ext = os.path.splitext(file['orig_name'])

            if 'epub' in ext:  # epub 파일 처리
                self.extract_epub_contents(self.temp_folder_1, file['path'])  # temp 폴더 아래 압축풀기
                self.extract_epub_contents(self.temp_folder_2, file['path'])

                # opf 언어 수정하기
                opf_file = self.locate_epub_metadata_opf()
                tree = parse(opf_file)
                opf = tree.getroot()

                for child in opf.iter():
                    print(child.tag)
                    if 'language' in child.tag:
                        child.text = target_abb  # 선택된 목표 언어의 ISO 639-1 코드 적용
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

                        # EPUB의 다양한 태그 구조에서 실제 텍스트를 담고 있는 리프 노드만 추출
                        # h1~h6(제목), li(목차), span(인라인 강조) 등 EPUB_TRANSLATE_TAGS 전체 탐색
                        # 리프 노드 조건: 텍스트가 존재하고 하위에 다른 블록 태그를 포함하지 않는 것만
                        _leaf_filter = EPUB_TRANSLATE_TAGS - {'span'}  # span은 블록 판별에서 제외
                        p_tags_1 = [
                            tag for tag in soup_1.find_all(EPUB_TRANSLATE_TAGS)
                            if tag.get_text(strip=True)
                            and not tag.find(_leaf_filter)
                        ]
                        p_tags_2 = [
                            tag for tag in soup_2.find_all(EPUB_TRANSLATE_TAGS)
                            if tag.get_text(strip=True)
                            and not tag.find(_leaf_filter)
                        ]

                        # 1. 번역할 문구들만 빼온다
                        only_texts = []  # 줄바꿈이 빠진 오직 원문 텍스트만
                        whole_particle = []  # particle 리스트들을 모두 합친 리스트
                        for text_node_1, text_node_2 in progress.tqdm(zip(p_tags_1, p_tags_2), desc='단락수'):
                            text_str = text_node_1.text.strip()
                            # [핵심 변경] 단어 1~2개 짧은 텍스트도 번역 대상 — Gemma 4는 환각 없이 처리 가능
                            # 알파벳/한글이 1자도 없는 경우(순수 숫자·특수문자)와 1글자 이하만 스킵
                            if not text_str or self.contains_no_alphabets(text_str) or len(text_str) <= 1:
                                whole_particle.append(0)
                                continue

                            particle = nltk.sent_tokenize(text_node_1.text)
                            only_texts.extend(particle)
                            particle.append(0)
                            whole_particle.extend(particle)

                        # 2. 가져온 문구들을 번역 시작한다
                        parti_1, parti_2 = self.batch_translate_engine(only_texts, whole_particle, 'epub', genre_val, tone_val, bilingual_order_val)

                        # 3. 번역 완료한 문구들을 다시 원래의 p태그에 맞게 replace해준다. 이게 핵심이다
                        particle_list_1 = []
                        particle_list_2 = []
                        translated_str_1 = ''
                        translated_str_2 = ''
                        for p_1, p_2 in zip(parti_1, parti_2):  # 단락으로 다시 묶는다
                            if p_1:  # 줄바꿈이 없는 하나의 단락이라면 계속 하나의 문장으로 묶는다
                                translated_str_1 += ' ' + p_1
                                translated_str_2 += ' ' + p_2
                            else:  # 줄바꿈을 만나면 p배열에 추가한다. 이렇게 p태그를 맞춘다
                                particle_list_1.append(translated_str_1)
                                particle_list_2.append(translated_str_2)
                                translated_str_1 = ''
                                translated_str_2 = ''

                        # 4. ['문장1 문장2','문장3','문장4 문장5 문장6']을 하나씩 풀어 p에 붙인다
                        for p_1, p_2, text_node_1, text_node_2 in zip(particle_list_1, particle_list_2, p_tags_1, p_tags_2):
                            text_str = text_node_1.text.strip()

                            # 수집 루프와 동일한 스킵 조건 — 번역하지 않은 노드는 원본 그대로 유지
                            if not text_str or self.contains_no_alphabets(text_str) or len(text_str) <= 1:
                                continue  # img태그만 있다면 태그 안에 유지된다. 숫자나 특수문자로만 구성되어 있다면 넘어간다

                            # 원본 태그 이름(h2, li 등) 보존 — 항상 p로 덮어쓰면 EPUB 구조와 스타일이 깨짐
                            p_tag_1 = soup_1.new_tag(text_node_1.name)
                            p_tag_2 = soup_2.new_tag(text_node_2.name)

                            try:
                                if text_node_1.attrs and text_node_1.attrs['class']:
                                    p_tag_1['class'] = text_node_1.attrs['class']
                                    p_tag_2['class'] = text_node_1.attrs['class']
                            except:
                                pass

                            if text_node_1.text.strip():  # 번역할 문구가 있을때만 string을 바꾼다. ★★★ p태그안에 img가 있을때 살려야하기 때문이다
                                p_tag_1.string = p_1
                                p_tag_2.string = p_2
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

                # temp.epub으로 압축
                for loc_folder in [self.temp_folder_1, self.temp_folder_2]:
                    self.repack_epub_contents(loc_folder, f'{loc_folder}.epub')

                os.makedirs(self.output_folder, exist_ok=True)
                # 원문(번역문) 모드면 파일명도 원문 언어가 앞에 오도록 표기
                if bilingual_order_val == "원문(번역문)":
                    done_path_1 = os.path.join(self.output_folder, "{name}_{t2}({t3}){ext}".format(name=name, t2=origin_abb, t3=target_abb, ext=ext))
                else:
                    done_path_1 = os.path.join(self.output_folder, "{name}_{t2}({t3}){ext}".format(name=name, t2=target_abb, t3=origin_abb, ext=ext))
                done_path_2 = os.path.join(self.output_folder, "{name}_{t2}{ext}".format(name=name, t2=target_abb, ext=ext))

                all_file_path.extend([done_path_1, done_path_2])

                shutil.move(f'{self.temp_folder_1}.epub', done_path_1)
                shutil.move(f'{self.temp_folder_2}.epub', done_path_2)

                # 번역이 끝날때 더 이상 임시 폴더를 삭제하지 않는다. 실서비스를 위해서는 이미지 여부등 중간 확인이 필요하기 때문이다
                self.remove_folder(self.temp_folder_1)
                self.remove_folder(self.temp_folder_2)

            elif '.pdf' in ext:  # PDF 파일 처리 (Docling + HTML 변환 번역)
                print(f'[PDF 번역] 시작: {name}{ext}')
                print(f'[PDF 번역] 언어: {origin_abb} → {target_abb} | 모델: {self.gemma_model}')

                if not DOCLING_AVAILABLE:
                    print('[PDF 번역] 오류: docling 패키지가 설치되지 않았습니다. pip install docling 실행 필요.')
                    continue

                try:
                    os.makedirs(self.output_folder, exist_ok=True)
                    progress(0, desc='[PDF] Docling을 이용한 HTML 구조화 중...')

                    # 1. Docling으로 PDF → HTML 변환. 이미지는 base64로 임베드해 원본 보존
                    # Docling 기동 직전에 CUDA 장치를 은폐 — CPU 전용 모드로 강제 격리
                    # os는 파일 상단에서 임포트됨 — 함수 내 재선언 금지 (UnboundLocalError 방지)
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
                        # Docling 작업 종료 후 환경 변수 복구 (타 시스템 영향 최소화)
                        if original_cuda_visible:
                            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                        else:
                            del os.environ["CUDA_VISIBLE_DEVICES"]

                    # 이미지 관련 텍스트 분류
                    # picture_delete: 이미지 내부/상단 레이블 → HTML에서 완전 삭제 (이미지에 이미 시각적으로 포함)
                    # picture_skip:   이미지 바로 아래 캡션 → 번역 생략, 원문 그대로 유지
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
                            margin_up    = 150  # 이미지 상단 위쪽 레이블 포함 범위
                            margin_down  = 15   # 이미지 바로 아래 캡션 허용 범위 (본문은 제외)
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
                                        # 이미지 내부 or 상단(레이블): 삭제
                                        if tb.t >= pb.b and tb.t <= pb.t + margin_up:
                                            picture_delete.add(item_text.strip())
                                        # 이미지 바로 아래(캡션): 번역만 생략
                                        elif pb.b - margin_down <= tb.t < pb.b:
                                            picture_skip.add(item_text.strip())
                                        break

                        print(f'[PDF] 삭제 텍스트 {len(picture_delete)}개 / 번역생략 텍스트 {len(picture_skip)}개')
                    except Exception as pe:
                        import traceback as _tb
                        print(f'[PDF] 이미지 텍스트 수집 오류: {pe}')
                        _tb.print_exc()

                    # 코드 블록 영역을 pypdfium2로 직접 렌더링하여 크롭 → base64 이미지
                    # (Docling 내부 page.image.pil_image는 캐시 버그로 None이 됨)
                    code_block_images = []  # 순서대로 저장 (HTML의 <pre> 순서와 1:1 매칭)
                    try:
                        import io as _io
                        import base64 as _b64
                        import pypdfium2 as _pdfium

                        pdf_doc = _pdfium.PdfDocument(file['path'])
                        rendered_pages = {}  # page_no → (pil_img, pt_w, pt_h)

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
                                    pdf_page = pdf_doc[page_no - 1]  # 0-indexed
                                    pt_w = pdf_page.get_width()
                                    pt_h = pdf_page.get_height()
                                    bitmap = pdf_page.render(scale=2.0)
                                    pil_img = bitmap.to_pil()
                                    rendered_pages[page_no] = (pil_img, pt_w, pt_h)
                                pil_img, pt_w, pt_h = rendered_pages[page_no]
                                img_w, img_h = pil_img.size
                                sx = img_w / pt_w
                                sy = img_h / pt_h
                                # BOTTOMLEFT 좌표계 → 이미지 좌표 변환 (y축 반전)
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
                                break  # prov 하나만 처리

                        pdf_doc.close()
                        print(f'[PDF] 코드 블록 이미지 {len(code_block_images)}개 크롭 완료')
                    except Exception as ce:
                        import traceback as _tb
                        print(f'[PDF] 코드 블록 이미지 변환 오류: {ce}')
                        _tb.print_exc()

                    # 넓은 표(Wide Table) 영역을 pypdfium2로 직접 렌더링하여 크롭 → base64 이미지
                    # 코드 블록 크롭과 동일한 방식. pdf_doc는 이미 close()됐으므로 별도 인스턴스 사용
                    WIDE_TABLE_COL_THRESHOLD = 5  # 이 열 수 이상인 표는 이미지로 변환
                    wide_table_images = {}  # table_idx(HTML <table> 순서) → base64 이미지 문자열

                    try:
                        pdf_doc_t = _pdfium.PdfDocument(file['path'])
                        rendered_pages_t = {}  # 표 전용 페이지 렌더 캐시
                        table_idx = 0  # HTML <table> 태그 순서와 1:1 매칭

                        for entry in result.document.iterate_items():
                            item = entry[0] if isinstance(entry, (tuple, list)) else entry
                            item_label = str(getattr(item, 'label', '')).upper()
                            if 'TABLE' not in item_label:
                                continue

                            # 표의 열 수 확인 — Docling TableData의 num_cols 또는 grid 첫 행 길이
                            col_count = 0
                            table_data = getattr(item, 'data', None)
                            if table_data is not None:
                                col_count = getattr(table_data, 'num_cols', 0)
                                if col_count == 0 and hasattr(table_data, 'grid') and table_data.grid:
                                    col_count = len(table_data.grid[0]) if table_data.grid[0] else 0

                            if col_count >= WIDE_TABLE_COL_THRESHOLD:
                                # 임계값 이상인 넓은 표만 이미지로 변환
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
                                    # BOTTOMLEFT 좌표계 → 이미지 픽셀 좌표 변환 (y축 반전)
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
                                    break  # prov 하나만 처리 (표는 단일 위치)

                            table_idx += 1  # 넓든 좁든 HTML <table> 순서는 항상 증가

                        pdf_doc_t.close()
                        print(f'[PDF] 넓은 표 이미지 {len(wide_table_images)}개 크롭 완료 (전체 표: {table_idx}개)')

                    except Exception as te:
                        import traceback as _tb
                        print(f'[PDF] 넓은 표 이미지 변환 오류: {te}')
                        wide_table_images = {}
                        _tb.print_exc()

                    html_content = result.document.export_to_html(image_mode=ImageRefMode.EMBEDDED)
                    print('[PDF 번역] HTML 구조화 완료. 텍스트 번역 파이프라인(BeautifulSoup) 시작...')

                    # 2. BeautifulSoup으로 파싱하여 기존 EPUB 파이프라인 재활용
                    soup_1 = BeautifulSoup(html_content, 'html.parser')
                    soup_2 = BeautifulSoup(html_content, 'html.parser')

                    # <pre><code> 블록 → 크롭된 페이지 이미지로 교체
                    # Docling OCR 텍스트보다 원본 이미지가 레이아웃을 완벽히 보존함
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
                                # 크롭 이미지가 없으면 <pre> 태그 그대로 유지
                                pass

                    # 넓은 <table> → 크롭된 PDF 원본 이미지로 교체
                    # 열 겹침 레이아웃 깨짐 해결: EPUB 뷰어는 가로 스크롤 미지원이므로 이미지가 가장 안전함
                    # 좁은 표(임계값 미만)는 <table> 유지 + CSS 보강으로 가독성 개선
                    for soup_obj in [soup_1, soup_2]:
                        table_tags = soup_obj.find_all('table')
                        for i, table in enumerate(table_tags):
                            if i in wide_table_images:
                                # 넓은 표 → 이미지로 완전 교체
                                img_tag = soup_obj.new_tag(
                                    'img',
                                    src=wide_table_images[i],
                                    style='display:block;max-width:100%;margin:1.5em auto;border:1px solid #ddd;'
                                )
                                table.replace_with(img_tag)
                            else:
                                # 좁은 표 → <table> 유지, table-layout:fixed로 열 넘침 방지
                                table['style'] = (
                                    'width:100%;border-collapse:collapse;'
                                    'font-size:0.82em;word-break:break-word;'
                                    'table-layout:fixed;'
                                )

                    # 리프 텍스트 태그만 추출 — 블록 자식을 가진 컨테이너는 제외해 중복 처리 방지
                    # (find_all이 부모<div>와 자식<p>를 동시에 반환하면 같은 텍스트가 두 번 처리됨)
                    block_tag_names = {'p', 'div', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'figcaption'}
                    target_tags = list(block_tag_names)
                    tags_1 = soup_1.find_all(target_tags)
                    tags_2 = soup_2.find_all(target_tags)

                    only_texts = []
                    whole_particle = []
                    valid_tags_1 = []
                    valid_tags_2 = []

                    for t_idx, tag_1 in enumerate(tags_1):
                        # 블록 레벨 자식이 있으면 컨테이너 태그이므로 건너뜀 (자식이 별도로 처리됨)
                        if any(tag_1.find(bt) for bt in block_tag_names):
                            continue
                        # <figure> 내부 태그는 이미지 영역 — 번역하지 않고 원본 보존
                        if tag_1.find_parent('figure') or tag_1.find('img'):
                            continue

                        text = tag_1.get_text(separator=' ').strip()
                        # 이미지 내부 레이블: 완전 삭제 (이미지에 이미 시각적으로 포함됨)
                        if picture_delete and text in picture_delete:
                            tag_1.decompose()
                            tags_2[t_idx].decompose()
                            continue
                        # 이미지 캡션(바로 아래): 이탤릭 스타일 적용 후 번역 생략
                        if picture_skip and text in picture_skip:
                            for s_tag in [tag_1, tags_2[t_idx]]:
                                s_tag['style'] = 'font-style:italic;text-align:center;margin-top:2px;font-size:0.9em;'
                            continue
                        # 각주(footnote): 숫자 + URL 패턴 — 번역 없이 원문 그대로 유지
                        if re.match(r'^\d+\s+https?://', text):
                            continue
                        if len(text) > 1 and any(c.isalpha() for c in text):
                            sentences = nltk.sent_tokenize(text)
                            only_texts.extend(sentences)

                            p_with_marker = list(sentences)
                            p_with_marker.append(0)
                            whole_particle.extend(p_with_marker)

                            valid_tags_1.append(tag_1)
                            valid_tags_2.append(tags_2[t_idx])
                            
                    if only_texts:
                        progress(0, desc='[PDF] 문맥 기반 텍스트 배치 번역 중...')
                        # Yanolja LLM 배칭 번역 호출 (EPUB과 동일하게 문장 단위로 넘김)
                        parti_1, parti_2 = self.batch_translate_engine(only_texts, whole_particle, 'epub', genre_val, tone_val, bilingual_order_val)

                        # 번역 완료한 문구들을 다시 원래의 태그(단락)에 맞게 재조립
                        assembled_1 = []
                        assembled_2 = []
                        translated_str_1 = ''
                        translated_str_2 = ''
                        
                        for p_1, p_2 in zip(parti_1, parti_2):
                            if p_1:  # 문장인 경우
                                translated_str_1 += ' ' + p_1
                                translated_str_2 += ' ' + p_2
                            else:    # 0 마커(태그 종료)인 경우
                                assembled_1.append(translated_str_1.strip())
                                assembled_2.append(translated_str_2.strip())
                                translated_str_1 = ''
                                translated_str_2 = ''
                        
                        # 재조립된 텍스트를 HTML 태그에 반영
                        for t_idx, valid_tag_1 in enumerate(progress.tqdm(valid_tags_1, desc='HTML 재조립')):
                            # assembled 리스트는 valid_tags와 1:1 매칭됨
                            trans_1 = assembled_1[t_idx]
                            trans_2 = assembled_2[t_idx]
                            
                            # soup_1 (이중언어본): 원문 제거 후 번역문(원문) 형식으로 교체
                            valid_tag_1.clear()
                            valid_tag_1.string = trans_1

                            # soup_2 (단일번역본): 원문 제거 후 번역문만 교체
                            valid_tags_2[t_idx].clear()
                            valid_tags_2[t_idx].string = trans_2
                            
                    # 3. EPUB 파일로 패키징 (Base64 이미지 분리 후 ebooklib으로 빌드)
                    progress(0, desc='[PDF] EPUB 패키징 중...')
                    # 원문(번역문) 모드면 파일명도 원문 언어가 앞에 오도록 표기
                    if bilingual_order_val == "원문(번역문)":
                        done_path_1 = os.path.join(self.output_folder, f"{name}_{origin_abb}({target_abb}).epub")
                    else:
                        done_path_1 = os.path.join(self.output_folder, f"{name}_{target_abb}({origin_abb}).epub")
                    done_path_2 = os.path.join(self.output_folder, f"{name}_{target_abb}.epub")

                    # target_abb는 이미 ISO 639-1 코드 ("ko", "en", "ja" 등) — 그대로 사용
                    self.build_epub_from_soup(soup_1, done_path_1, name, lang_code=target_abb)
                    self.build_epub_from_soup(soup_2, done_path_2, name, lang_code=target_abb)

                    all_file_path.extend([done_path_1, done_path_2])
                    print(f'[PDF 번역] 성공! EPUB 생성: {done_path_1}, {done_path_2}')
                    
                except Exception as err:
                    import traceback
                    print(f'[PDF 번역] 오류 발생: {err}')
                    traceback.print_exc()
                    continue

            else:  # 일반 txt파일 처리
                output_file_1, output_file_2, book = self.initialize_output_files(origin_abb, target_abb, name, ext, file, bilingual_order_val)
                book_raw = book.read()
                sentences = book_raw.split(sep='\n')  # 줄바꿈을 기준으로 리스트를 만들어 레이아웃을 유지시키고

                only_texts = []   # 줄바꿈이 빠진 오직 원문 텍스트만
                whole_particle = []  # particle 리스트들을 모두 합친 리스트. 원본 리스트.
                for sen in progress.tqdm(sentences, desc='단락'):
                    particle = nltk.sent_tokenize(sen)
                    particle = self.clean_text_spacing(particle)

                    only_texts.extend(particle)
                    particle.append(0)  # 줄바꿈 정보를 위해 0을 넣어준다
                    whole_particle.extend(particle)

                # epub은 파일 하나에 하나의 배열을 만들지만 text는 기존 방식대로 단락으로 가져온다
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
        # 목표 언어별 문체 지침 — 나레이션은 선택한 문체로 통일, 대화문은 캐릭터 뉘앙스 보존
        is_formal = "경어체" in tone_val
        lang = self.target_lang  # ISO 639-1 코드

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
            # 스칸디나비아 3개국어 — 현대 사용에서 T-V 구분은 사라졌으나 문어체 격식은 유효
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
            # 영어 및 기타 — 기본 격식/비격식 레지스터
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
        # 단일 문장 번역 — 배치 파싱 실패 시 폴백용으로 사용
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
            return text  # 실패 시 원문 반환으로 HTML 구조 붕괴 방지

    def _parse_llm_response(self, raw: str, expected_count: int) -> list:
        # "1. ...\n2. ..." 형식 응답을 파싱해 리스트로 반환한다
        # 번호 뒤 . 또는 ) 모두 허용, 여러 줄에 걸친 항목도 하나로 합친다
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
        # 여러 문장을 번호 목록으로 묶어 한 번의 API 호출로 번역한다
        # [안정성 강화] API 실패 시 즉시 포기하지 않고 3회 재시도 (Exponential Backoff)
        if not texts:
            return []

        genre_instruction = self.get_genre_prompt_extension(genre_val)
        tone_instruction = self.get_tone_prompt_extension(tone_val)

        # 용어집 프롬프트 생성 — user_glossary가 비어있으면 빈 문자열 반환 (기존 로직 동일)
        # genre_instruction보다 앞에 배치하여 LLM이 최우선으로 주목하게 함
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
        
        # 배치 번역 타임아웃을 넉넉하게 설정 (최소 300초)
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
                
                # 파싱 결과 검증 — 개수 불일치 시 일부 재시도
                if len(parsed) == len(texts) and all(parsed):
                    return parsed
                
                # 일부 누락 시 개별 재호출로 보충
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
        # 배치 인덱스와 청크를 받아 번역하고 완료 시 터미널에 로그를 찍는다
        idx, total_chunks, chunk, genre_val, tone_val = args
        preview = chunk[0][:40].replace('\n', ' ')
        print(f'  [배치 {idx+1}/{total_chunks}] 시작 ({len(chunk)}문장) | 장르: {genre_val} | 문체: {tone_val} | "{preview}..."')
        t0 = time.time()
        result = self.request_gemma_api_batch(chunk, genre_val, tone_val)
        elapsed = time.time() - t0
        first_out = result[0][:40].replace('\n', ' ') if result else '-'
        print(f'  [배치 {idx+1}/{total_chunks}] 완료 {elapsed:.1f}s | → "{first_out}..."')
        return result

    # Gemma API 배치 + 병렬 번역 엔진
    def batch_translate_engine(self, only_texts, whole_particle, what, genre_val="일반 문서(기본)", tone_val="서술체 (~다)", bilingual_order="번역문(원문)"):
        particle_list_1 = []
        particle_list_2 = []

        # 특수문자/콜론 전처리 — 환각 방지
        processed_texts = []
        for text in only_texts:
            text = re.sub(r'^[^\w\s]', '', text)  # 특수문자로 시작하면 환각 유발, 제거
            text = re.sub(r':', '-', text)  # 콜론은 LLM이 특수하게 해석하므로 하이픈으로 치환
            processed_texts.append(text)

        # translate_batch_size 단위로 청크를 만들고, translate_workers 개수만큼 동시에 서버에 던진다
        # MLX Continuous Batching으로 동시 요청을 한꺼번에 처리 → 통합 메모리 활용률↑
        # executor.map은 입력 순서대로 결과를 보장하므로 인덱스 정합성이 유지된다
        from concurrent.futures import ThreadPoolExecutor

        total = len(processed_texts)
        chunks = [
            processed_texts[i: i + self.translate_batch_size]
            for i in range(0, total, self.translate_batch_size)
        ]

        total_chunks = len(chunks)
        print(f'▶ 번역 시작: {total}문장 → {total_chunks}배치 × 동시{self.translate_workers}개')
        t_start = time.time()

        # (인덱스, 전체배치수, 청크) 튜플로 묶어 래퍼에 전달 — 완료 즉시 로그 출력
        with ThreadPoolExecutor(max_workers=self.translate_workers) as executor:
            chunk_results = list(executor.map(
                self._process_translation_batch,
                [(i, total_chunks, c, genre_val, tone_val) for i, c in enumerate(chunks)]
            ))

        # 배치별 결과를 원래 순서대로 하나의 리스트로 합친다
        translated_list = [item for sublist in chunk_results for item in sublist]

        total_elapsed = time.time() - t_start
        print(f'▶ 전체 번역 완료: {total}문장 / 총 {total_elapsed:.1f}s ({total/total_elapsed:.1f}문장/s)')

        # epub이나 txt: whole_particle 내 0(줄바꿈 마커)을 기준으로 원본 단락 구조를 복원한다
        text_idx = 0  # ★★★ 이걸로 원본과 번역문의 인덱스를 맞춰준다!

        for output_idx, whole in enumerate(whole_particle):
            if whole:  # 원문이 있는 위치라면
                # 0(줄바꿈 마커)을 건너뛴 실제 번역 인덱스로 접근
                # 배열 길이나 순서가 어긋나면 replace_with 구조가 붕괴되므로 주의
                generated_text = translated_list[output_idx - text_idx]

                # 이중언어 표기 순서 분기 — 학습자 모드는 원문(번역문), 기본은 번역문(원문)
                if bilingual_order == "원문(번역문)":
                    translated_text_1 = "{t2} ({t1})".format(t1=generated_text, t2=whole_particle[output_idx])
                else:
                    translated_text_1 = "{t1} ({t2})".format(t1=generated_text, t2=whole_particle[output_idx])
                particle_list_1.append(translated_text_1)
                translated_text_2 = generated_text  # 번역문만 저장
                particle_list_2.append(translated_text_2)
            else:  # 원문이 없고 줄바꿈 해야하는 인덱스라면
                # ★★ 줄바꿈 되었으므로 원본에서 건너띄어야 할 인덱스를 추가해준다
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
        # 파일이 추가되면 listen한다 (generator: 감지중 메시지 → 결과 순서로 yield)
        try:
            print('들어옴')
            if not self.launch_time:
                self.launch_time = time.time()
            self.selected_files = files
            # yield로 Component를 새로 생성해 던지면 Gradio 5/6에서 프론트엔드가 고장나며 다운로드 창이 뜨지 않는 버그가 발생하므로 데이터만 리턴합니다
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

            # 감지 시작 전 즉시 로딩 메시지 표시
            yield gr.update(), gr.update(), "<p style='text-align:center;'>🔍 언어 감지중입니다...</p>", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            aBook = files[0]
            name, ext = os.path.splitext(aBook['orig_name'])

            # API를 이용해 번역품질을 올리기 위한 장르 추론
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
                    check_lang = 'en'  # 감지 실패 시 기본값
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
                            # 지원 언어이거나 중국어 변형이면 채택하고 탈출
                            normalized = 'zh' if detected.startswith('zh') else detected
                            if normalized in LANG_CODE_TO_NAME:
                                check_lang = normalized
                                break

            elif '.pdf' in ext:
                # PyMuPDF로 첫 페이지 텍스트를 추출해 langdetect로 언어 판별
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
                    # detect_langs로 신뢰도까지 확인 — 짧은 텍스트 오탐 방지
                    langs = detect_langs(raw_text)
                    top = langs[0]
                    check_lang = top.lang if top.prob >= 0.8 else 'en'
                    print(f'[TXT] 언어 감지: {top.lang} (신뢰도 {top.prob:.2f}) → {check_lang}')
                except Exception:
                    check_lang = 'en'

            # 감지된 언어 코드를 정규화 (zh-cn, zh-tw → zh)
            normalized_lang = 'zh' if check_lang.startswith('zh') else check_lang
            # ISO 코드와 표시명을 인스턴스 변수에 저장
            self.origin_lang = normalized_lang
            self.origin_lang_name = LANG_CODE_TO_NAME.get(normalized_lang, f'알 수 없음 ({check_lang})')
            # 드롭다운에는 SUPPORTED_LANGUAGES 키만 유효 — 미지원 언어면 None(빈 선택) 반환
            origin_dropdown_val = self.origin_lang_name if normalized_lang in LANG_CODE_TO_NAME else None
            lang_info = self.origin_lang_name if origin_dropdown_val else f'알 수 없음 ({check_lang})'
            # 원본이 한국어면 목표를 영어로, 그 외 모든 언어면 목표를 한국어로 자동 설정
            auto_target = '영어' if normalized_lang == 'ko' else '한국어'
            auto_iso, auto_prompt = SUPPORTED_LANGUAGES[auto_target]
            self.target_lang = auto_iso
            self.target_lang_name = auto_target
            self.target_lang_prompt = auto_prompt
            # 파일이 바뀌었으므로 용어집 초기화
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
        # txt에서 공통으로 쓰는 파일처리 함수
        # 원문(번역문) 모드면 파일명도 원문 언어가 앞에 오도록 표기
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
        # epub 압축풀기
        try:
            zip_module = zipfile.ZipFile(epub_file, 'r')
            os.makedirs(folder_path, exist_ok=True)
            zip_module.extractall(folder_path)
            zip_module.close()
        except:
            print('잘못된 epub파일입니다')
            pass

    def repack_epub_contents(self, folder_path: PathType, epub_name: PathType):
        # epub으로 다시 압축하기
        try:
            zip_module = zipfile.ZipFile(epub_name, 'w', zipfile.ZIP_DEFLATED)
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_module.write(file_path, os.path.relpath(file_path, folder_path))  # temp을 제외한 상대경로로 압축한다
            zip_module.close()
        except Exception as err:
            print('epub 파일을 생성하는데 실패했습니다.')
            print(err)
            pass

    def list_epub_html_files(self) -> List:
        # 텍스트가 있는 html들만 가져오기
        file_path = []
        for root, _, files in os.walk(self.temp_folder_1):
            for file in files:
                if file.endswith(('xhtml', 'html', 'htm')):
                    file_path.append(os.path.join(root, file))
        return file_path

    def locate_epub_metadata_opf(self):
        # opf파일 가져오기
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

    def reset_session_and_gc(self) -> str:  # 번역 완료 후 마무리 처리
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

        # Phase A: Base64 인라인 이미지를 EPUB 내부 독립 파일로 분리
        img_idx = 0
        for img_tag in soup.find_all('img'):
            src = img_tag.get('src', '')
            if not src.startswith('data:image/'):
                continue  # 외부 URL 이미지는 건너뜀

            try:
                # MIME 타입과 확장자 파싱 (예: "data:image/png;base64,..." → mime="image/png", ext="png")
                header, b64data = src.split(',', 1)
                mime_match = re.search(r'data:(image/\w+);base64', header)
                if not mime_match:
                    continue
                mime = mime_match.group(1)
                ext = mime.split('/')[1]

                img_bytes = base64.b64decode(b64data)
                img_filename = f'images/img_{img_idx:03d}.{ext}'

                # EpubItem으로 래핑하여 EPUB 컨테이너에 등록
                epub_img = epub.EpubItem(
                    uid=f'img_{img_idx:03d}',
                    file_name=img_filename,
                    media_type=mime,
                    content=img_bytes
                )
                book.add_item(epub_img)
                img_tag['src'] = img_filename  # src를 EPUB 내부 상대 경로로 교체
                img_idx += 1

            except Exception as img_err:
                # 이미지 하나의 실패가 전체 EPUB 생성을 막지 않도록 격리 처리
                print(f'[EPUB] 이미지 추출 실패 (img_{img_idx:03d}): {img_err}')
                img_tag.decompose()  # 깨진 이미지 태그 제거하여 xhtml 파싱 오류 방지

        print(f'[EPUB] 이미지 {img_idx}개 분리 완료')

        # Phase B: 기본 CSS 스타일시트 (e-Reader 가독성 개선)
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

        # Phase C: 번역 완료 HTML을 단일 챕터로 조립
        # soup.body가 있으면 body 내용만 사용, 없으면 전체 soup 직렬화
        body_content = str(soup.body) if soup.body else str(soup)

        chapter = epub.EpubHtml(
            title=title,
            file_name='content.xhtml',
            lang=lang_code
        )
        chapter.content = body_content
        chapter.add_item(css_item)  # CSS 연결

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
        # 정규식 패턴: \s+는 하나 이상의 공백 문자 (스페이스, 탭, 개행 등)
        new_particle = []
        pattern = r'\s+'
        for s in particle:
            if s.strip():
                s = re.sub(pattern, ' ', s)
                new_particle.append(s)
        return new_particle

    def contains_no_alphabets(self, text):
        # 문장이 알파벳이 아닌 (숫자, 공백, 특수문자)로만 구성되어 있다면 true리턴
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
