import os
from typing import List, Union, Optional
from time import time
from datetime import timedelta
import logging
import warnings
import platform
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from ebooklib import epub, ITEM_DOCUMENT
from langdetect import detect
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from bs4 import BeautifulSoup
import gradio as gr

logging.getLogger().disabled = True
logging.raiseExceptions = False
warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)

PathType = Union[str, os.PathLike]


class Dodari:
    def __init__(self, max_len: int = 512):
        self.max_len = max_len
        self.selected_files = []

        self.upload_msg = None
        self.origin_lang_str = None
        self.target_lang_str = None
        self.origin_lang = None
        self.target_lang = None

        self.upload_files = None

        self.selected_model = None
        self.model = None
        self.tokenizer = None
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
        self.platform = platform.system()

    @staticmethod
    def remove_folder(temp_folder: PathType):
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

    def main(self):
        self.remove_folder(self.temp_folder_1)
        self.remove_folder(self.temp_folder_2)

        with gr.Blocks(
            css=self.css,
            title='Dodari',
            theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")
        ) as app:
            gr.HTML("""
            <div align='center'>
                <a href='https://github.com/vEduardovich/dodari' target='_blank'>
                <img src='file/imgs/dodari.png' style='display: block; width: 100px;'>
                </a>
                <h1 style='margin-top: 10px;'>AI 한영/영한 번역기
                <span style='color: red'>
                <a href='https://github.com/vEduardovich/dodari' target='_blank'>도다리</a>
                </span> 입니다.
                </h1>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    with gr.Tab('순서 1'):
                        gr.Markdown("<h3>1. 번역할 파일들 선택</h3>")
                        input_window = gr.File(
                            file_count="multiple",
                            file_types=[".txt", ".epub"],
                            label='파일들'
                        )
                        lang_msg = gr.HTML(self.upload_msg)
                        input_window.change(
                            fn=self.change_upload,
                            inputs=input_window,
                            outputs=lang_msg,
                            preprocess=False
                        )

                with gr.Column(scale=2):
                    with gr.Tab('순서 2'):
                        translate_btn = gr.Button(
                            value="번역 실행하기(NHNDQ 모델)",
                            size='lg',
                            variant="primary",
                            interactive=True
                        )

                        gr.HTML("""
                        <div style='text-align: right'>
                            <p style='color: grey;'>처음 실행 시 모델을 다운받는데 아주 오랜 시간이 걸립니다.</p>
                            <p style='color: grey;'>컴퓨터 사양이 좋다면 번역 속도가 빨라집니다.</p>
                            <p style='color: grey;'>맥 M1 이상에서는 MPS를 이용하여 가속합니다.</p>
                        </div>
                        """)

                        with gr.Row():
                            status_msg = gr.Textbox(
                                label="상태 정보",
                                scale=4,
                                value='번역 대기 중...'
                            )
                            translate_btn.click(fn=self.translate_fn, outputs=status_msg)

                            btn_openfolder = gr.Button(
                                value='📂 번역 완료한 파일들 보기',
                                scale=1,
                                variant="secondary"
                            )
                            btn_openfolder.click(
                                fn=lambda: self.open_folder(),
                                inputs=None,
                                outputs=None
                            )

        app.queue().launch(
            inbrowser=True,
            favicon_path='imgs/dodari.png',
            allowed_paths=["."]
        )

    def finalize_fn(self) -> str:
        sec = self.check_time()
        self.start = None

        return sec

    def get_translator(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.selected_model,
            cache_dir=os.path.join("models", "tokenizers")
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.selected_model,
            cache_dir=os.path.join("models")
        )

        gpu_count = torch.cuda.device_count()
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        if gpu_count > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(gpu_count)))
            torch.multiprocessing.set_start_method('spawn')
        self.model.to(device)

        translator = pipeline(
            'translation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            src_lang=self.origin_lang,
            tgt_lang=self.target_lang,
            max_length=self.max_len
        )

        return translator

    def translate_fn(self, progress=gr.Progress()) -> str:
        if not self.selected_files:
            return "번역할 파일을 추가하세요."

        self.start = time()
        progress(0, desc="번역 모델을 준비중입니다...")

        translator = self.get_translator()

        origin_abb = self.origin_lang.split(sep='_')[0]
        target_abb = self.target_lang.split(sep='_')[0]

        for file in progress.tqdm(self.selected_files, desc='파일 로딩'):
            name, ext = os.path.splitext(file['orig_name'])

            if 'epub' in ext:
                self.zip_extract(self.temp_folder_1, file['path'])
                self.zip_extract(self.temp_folder_2, file['path'])

                file_path = self.get_html_list()
                for html_file in progress.tqdm(file_path, desc='챕터'):

                    html_file_2 = html_file.replace(self.temp_folder_1, self.temp_folder_2)

                    input_file_1 = open(html_file, 'r', encoding='utf-8')
                    input_file_2 = open(html_file_2, 'r', encoding='utf-8')

                    soup_1 = BeautifulSoup(input_file_1.read(), 'html.parser')
                    soup_2 = BeautifulSoup(input_file_2.read(), 'html.parser')

                    p_tags_1 = soup_1.find_all('p')
                    p_tags_2 = soup_2.find_all('p')
                    ahtml_text = p_tags_1[0].text.strip() if p_tags_1 else None

                    if not ahtml_text:
                        p_tags_1 = soup_1.find_all('div')
                        p_tags_2 = soup_2.find_all('div')
                        for p_tag_1, p_tag_2 in zip(p_tags_1, p_tags_2):
                            if not p_tag_1.find('div'):
                                ahtml_text = p_tag_1.text.strip()
                                if ahtml_text:
                                    p_tag_1.name = 'p'
                                    p_tag_2.name = 'p'
                                else:
                                    p_tags_1 = soup_1.find_all('p')
                        p_tags_2 = soup_2.find_all('p')

                    for text_node_1, text_node_2 in progress.tqdm(zip(p_tags_1, p_tags_2), desc='단락 수'):
                        if not text_node_1.text.strip():
                            continue

                        p_tag_1 = soup_1.new_tag('p')
                        p_tag_2 = soup_2.new_tag('p')

                        try:
                            if text_node_1.attrs and text_node_1.attrs['class']:
                                p_tag_1['class'] = text_node_1.attrs['class']
                                p_tag_2['class'] = text_node_1.attrs['class']
                        except (Exception,):
                            # TODO: need to resolve the usage of broad exception
                            pass

                        particle = nltk.sent_tokenize(text_node_1.text)
                        particle_list_1 = []
                        particle_list_2 = []
                        for text in progress.tqdm(particle, desc='문장 수'):
                            output = translator(text, max_length=self.max_len)
                            translated_text_1 = f"{output[0]['translation_text']} ({text}) "
                            particle_list_1.append(translated_text_1)

                            translated_text_2 = output[0]['translation_text']
                            particle_list_2.append(translated_text_2)

                        translated_particle_1 = ' '.join(particle_list_1)
                        translated_particle_2 = ' '.join(particle_list_2)
                        p_tag_1.string = translated_particle_1
                        p_tag_2.string = translated_particle_2

                        img_tag = text_node_1.find('img')
                        if img_tag:
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

                for loc_folder in [self.temp_folder_1, self.temp_folder_2]:
                    self.zip_folder(loc_folder, f'{loc_folder}.epub')
                os.makedirs(self.output_folder, exist_ok=True)
                shutil.move(
                    f'{self.temp_folder_1}.epub',
                    os.path.join(
                        self.output_folder,
                        f"{name}_{target_abb}({origin_abb}){ext}"
                    )
                )
                shutil.move(
                    f'{self.temp_folder_2}.epub',
                    os.path.join(
                        self.output_folder,
                        f"{name}_{target_abb}{ext}"
                    )
                )

                self.remove_folder(self.temp_folder_1)
                self.remove_folder(self.temp_folder_2)

            else:
                output_file_bi = self.write_filename(
                    f"{name}_{target_abb}({origin_abb}){ext}"
                )
                output_file = self.write_filename(
                    f"{name}_{target_abb}{ext}"
                )

                book = self.get_filename(file['path'])
                book_list = book.split(sep='\n')
                for book in progress.tqdm(book_list, desc='단락'):
                    particle = nltk.sent_tokenize(book)

                    for text in progress.tqdm(particle, desc='문장'):
                        output = translator(text, max_length=self.max_len)
                        output_file_bi.write(
                            f"{output[0]['translation_text']} ({text}) "
                        )
                        output_file.write(output[0]['translation_text'])
                    output_file_bi.write('\n')
                    output_file.write('\n')
                output_file_bi.close()
                output_file.close()

        sec = self.finalize_fn()

        return f"번역 완료! 걸린 시간: {sec}"

    def change_upload(self, files: List) -> Optional[str]:
        check_lang = ''
        try:
            self.selected_files = files
            if not files:
                return self.upload_msg
            a_book = files[0]
            name, ext = os.path.splitext(a_book['path'])
            if '.epub' in ext:
                file = epub.read_epub(a_book['path'])
                lang = file.get_metadata('DC', 'language')
                if lang:
                    check_lang = lang[0][0]
                else:
                    for item_idx, item in enumerate(file.get_items()):
                        if item.get_type() == ITEM_DOCUMENT:
                            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                            all_tags = soup.find_all('p')
                            if not all_tags:
                                continue

                            text_tags = [tag.text for tag in all_tags if tag.text.strip()]
                            lang_str = ' '.join(text_tags)
                            check_lang = detect(lang_str[:500])
                            if 'en' in check_lang or 'ko' in check_lang:
                                break
                            else:
                                return """
                                <p style='text-align: center; color: red;'>표준 규격을 벗어난 epub입니다.
                                <a href='https://moonlit.himion.com/info/contactUs'>이곳</a>을 이용해
                                해당 epub 파일을 첨부해서 보내주시면 바로 해결해드립니다.
                                번역에 실패했습니다.
                                </p>
                                """

            else:
                book = self.get_filename(a_book['path'])
                check_lang = detect(book[:200])

            self.origin_lang_str = '영어' if 'en' in check_lang else "한국어"
            self.target_lang_str = '한국어' if 'en' in check_lang else "영어"
            self.origin_lang = "eng_Latn" if 'en' in check_lang else "kor_Hang"
            self.target_lang = "kor_Hang" if 'en' in check_lang else "eng_Latn"
            self.selected_model = 'NHNDQ/nllb-finetuned-en2ko' if 'en' in check_lang else 'NHNDQ/nllb-finetuned-ko2en'

            return f"""
            <p style='text-align: center;'>
            <span style='color: skyblue; font-size: 1.5em;'>{self.origin_lang_str}</span><span>를 </span>
            <span style='color: red; font-size: 1.5em;'> {self.target_lang_str}</span>
            <span>로 번역합니다.</span>
            </p>
            """

        except (Exception,):
            # TODO: need to resolve the usage of broad exception
            return "<p style='text-align: center; color: red;'>어떤 언어인지 알아내는데 실패했습니다.</p>"

    @staticmethod
    def get_filename(file_name: PathType) -> str:
        try:
            input_file = open(file_name, 'r', encoding='utf-8')
            return input_file.read()

        except (Exception,):
            # TODO: need to resolve the usage of broad exception
            try:
                input_file = open(file_name, 'r', encoding='euc-kr')
                return input_file.read()
            except (Exception,):
                # TODO: need to resolve the usage of broad exception
                input_file = open(file_name, 'r', encoding='cp949', errors='ignore')
                return input_file.read()

    def write_filename(self, file_name: str):
        save_dir = self.output_folder
        if not (os.path.isdir(save_dir)):
            os.makedirs(os.path.join(save_dir))

        file = os.path.join(save_dir, file_name)
        output_file = open(file, 'w', encoding='utf-8')

        return output_file

    def open_folder(self):
        save_dir = self.output_folder
        open_command_dict = {
            'Windows': 'start',
            'Darwin': 'open',
            'Linux': 'nautilus'
        }

        if not (os.path.isdir(save_dir)):
            os.makedirs(save_dir)
        if self.platform in open_command_dict.keys():
            print(f"Detected the platform as: {self.platform}")
            command_to_open = f"{open_command_dict[self.platform]} {save_dir}"
            os.system(command_to_open)
        else:
            raise NotImplementedError

    @staticmethod
    def zip_extract(folder_path: PathType, epub_file: PathType):
        try:
            zip_module = ZipFile(epub_file, 'r')
            os.makedirs(folder_path, exist_ok=True)
            zip_module.extractall(folder_path)
            zip_module.close()

        except (Exception,):
            # TODO: need to resolve the usage of broad exception
            print('잘못된 epub 파일입니다')
            pass

    @staticmethod
    def zip_folder(folder_path: PathType, epub_name: PathType):
        try:
            zip_module = ZipFile(epub_name, 'w', ZIP_DEFLATED)
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_module.write(file_path, os.path.relpath(file_path, folder_path))
            zip_module.close()

        except (Exception,):
            # TODO: need to resolve the usage of broad exception
            print('epub 파일을 생성하는데 실패했습니다.')
            pass

    def get_html_list(self) -> List:
        file_paths = []
        for root, _, files in os.walk(self.temp_folder_1):
            for file in files:
                if file.endswith(('xhtml', 'html', 'htm')):
                    file_paths.append(os.path.join(root, file))

        return file_paths

    def check_time(self) -> str:
        end = time()
        during = end - self.start
        sec = str(timedelta(seconds=during)).split('.')[0]

        return sec


if __name__ == "__main__":
    dodari = Dodari()
    dodari.main()
