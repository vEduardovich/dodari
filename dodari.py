

import os, time, re, datetime, platform
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
nltk.download('punkt')
from langdetect import detect
from torch import cuda
from tqdm import tqdm
import tkinter as tk
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger().disabled = True 
from gradio.themes.base import Base
from gradio.themes.utils import colors, sizes

class Dodari:
    def __init__(self):
        self.max_len = 512
        self.selected_files = []
        self.upload_msg = "<div style='text-align:right;color:grey;'><p>폴더 전체를 업로드 하시려면</p><p>'클릭해서 업로드하기' -> 폴더선택 -> upload클릭 하세요.</p></div>"
        self.origin_lang = None
        self.target_lang = None

        self.selected_model = None
        self.model = None
        self.tokenizer = None
        self.css = """
            .radio-group .wrap {
                display: float !important;
                grid-template-columns: 1fr 1fr;
            }
            """
        self.start = '' 
        self.platform = platform.system()

    def main(self):
        with gr.Blocks(css=self.css, theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")) as app:
            gr.HTML("<h1>AI 한영/영한 번역기 '<span style='color:red'>도다리</span>' 입니다 </h1>")
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    with gr.Tab('순서 1'):
                        gr.Markdown("<h3>1. 번역할 파일들 선택</h3>")
                        input_window = gr.File(file_count="directory", label='파일들')
                        lang_msg= gr.HTML(self.upload_msg)
                        input_window.change(fn=self.change_upload, inputs=input_window, outputs=lang_msg)

                with gr.Column():
                    with gr.Tab('순서 2'):
                        gr.HTML("<p>현재 <a href='https://huggingface.co/NHNDQ/nllb-finetuned-ko2en' target='_blank'>NHNDQ/nllb-finetuned</a> AI 번역모델을 사용합니다</p>")
                        translate_btn = gr.Button(value="번역 실행하기", size='lg', variant="primary")
                        gr.HTML("<div style='text-align:right'><p style = 'color:grey;'>처음 실행시 모델을 다운받는데 아주 오랜 시간이 걸립니다.</p><p style='color:grey;'>컴퓨터 사양이 좋다면 번역 속도가 빨라집니다.</p><p style='color:grey;'>맥에서는 cpu만 쓰기때문에 번역속도가 느립니다</p></div>")                     
                        with gr.Row():
                            
                            msg = gr.Textbox(label="상태 정보", scale=4, value='번역 대기중..')
                            translate_btn.click(fn=self.translateFn, outputs=msg)
                            btn_openfolder = gr.Button(value='📂 완료 폴더 열기', scale=1, variant="secondary")
                            btn_openfolder.click(fn=lambda: self.open_folder(), inputs=None, outputs=None)

        app.launch(inbrowser=True)
    def translateFn(self, progress=gr.Progress(track_tqdm=True)):
        self.start = time.time()
        
        if not self.selected_files : return "번역할 파일을 추가하세요."

        device = 0 if cuda.is_available() else -1 

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.selected_model, cache_dir=os.path.join("models", "tokenizers"))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=self.selected_model, cache_dir=os.path.join("models"))

        translator = pipeline('translation', model=self.model, tokenizer=self.tokenizer, device= device, src_lang=self.origin_lang, tgt_lang=self.target_lang, max_length=self.max_len)

        for idx, file in enumerate(tqdm(self.selected_files, desc='파일전체')):
            book = self.get_filename(file);
            file_name = file.split(sep='\\')[-1] if self.platform == 'Windows' else file.split(sep='/')[-1]
            
            name = file_name.split(sep='.')[0]
            ext = file_name.split(sep='.')[1]
            origin_abb = self.origin_lang.split(sep='_')[0]
            target_abb = self.target_lang.split(sep='_')[0]
            output_file_bi = self.write_filename( "{name}_{t2}({t3}).{ext}".format(name=name, t2=target_abb, t3=origin_abb, ext = ext) )
            output_file = self.write_filename( "{name}_{t2}.{ext}".format(name=name, t2=target_abb, ext = ext) )

            book_list = book.split(sep='\n')
            for book in book_list:
                aBook = nltk.sent_tokenize(book)
                
                for num, text in enumerate(tqdm( aBook, desc='문장수' )):
                    output = translator(text, max_length=self.max_len)
                    output_file_bi.write("{t1} ({t2}) ".format(t1=output[0]['translation_text'], t2=text) )
                    output_file.write(output[0]['translation_text'])
                output_file_bi.write('\n')
                output_file.write('\n')

        output_file_bi.close()
        output_file.close()
        sec = self.check_time()
        return "번역완료! 걸린시간 : {t1}".format(t1=sec)

    def change_upload(self, files):
        self.selected_files = files
        if not files : return self.upload_msg
        book = self.get_filename(files[0]);
        check_lang = detect(book[0:100])
        origin_lang_str = '영어' if check_lang == 'en' else "한국어"
        target_lang_str = '한국어' if check_lang == 'en' else "영어"
        self.origin_lang = "eng_Latn" if check_lang == 'en' else "kor_Hang"
        self.target_lang = "kor_Hang" if check_lang == 'en' else "eng_Latn"
        self.selected_model = 'NHNDQ/nllb-finetuned-en2ko' if check_lang == 'en' else 'NHNDQ/nllb-finetuned-ko2en'
        
        return "<p style='text-align:center;'><span style='color:skyblue;font-size:1.5em;'>{t1}</span><span>를 </span> <span style='color:red;font-size:1.5em;'> {t2}</span><span>로 번역합니다.</span></p>".format(t1=origin_lang_str, t2 = target_lang_str)
    def get_filename(self, fileName):
        input_file = open(fileName, 'r', encoding='utf-8')
        return input_file.read()
    def write_filename(self, file_name):
        saveDir = os.path.join(os.getcwd(), 'outputs')
        if not(os.path.isdir(saveDir)): 
            os.makedirs(os.path.join(saveDir)) 

        file = saveDir + '/' + file_name
        output_file = open(file, 'w', encoding='utf-8')
        return output_file

    def open_folder(self):
        saveDir = os.path.join(os.getcwd(), 'outputs')
        if not(os.path.isdir(saveDir)): 
            os.makedirs(os.path.join(saveDir)) 
        if  self.platform == 'Windows': os.system(f"start {saveDir}")
        elif self.platform == 'Darwin': os.system(f"open {saveDir}")
        elif self.platform == 'Linux': os.system(f"nautilus {saveDir}")
        
    def check_time(self):
        end = time.time()
        during = end - self.start
        sec = str(datetime.timedelta(seconds=during)).split('.')[0]
        return sec
if __name__ == "__main__":
    dodari = Dodari()
    dodari.main()
