import os, time, datetime, platform
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
nltk.download('punkt', quiet=True)
import ebooklib
from ebooklib import epub
import shutil
import zipfile
from langdetect import detect

import torch
from bs4 import BeautifulSoup
import gradio as gr

import logging
logging.getLogger().disabled = True 
logging.raiseExceptions = False
import warnings
warnings.filterwarnings('ignore')
class Dodari:
    def __init__(self):
        self.max_len = 512
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

    def remove_folder(self, temp_folder):
        if os.path.exists(temp_folder): shutil.rmtree(temp_folder)

    def main(self):
        self.remove_folder(self.temp_folder_1)
        self.remove_folder(self.temp_folder_2)
        
        with gr.Blocks(css=self.css, title='Dodari', theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")) as app:
            gr.HTML("<div align='center'><a href='https://github.com/vEduardovich/dodari' target='_blank'><img src='file/imgs/dodari.png' style='display:block;width:100px;'></a> <h1 style='margin-top:10px;'>AI 한영/영한 번역기 <span style='color:red'><a href='https://github.com/vEduardovich/dodari' target='_blank'>도다리</a></span> 입니다 </h1></div>")
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    with gr.Tab('순서 1'):
                        gr.Markdown("<h3>1. 번역할 파일들 선택</h3>")
                        input_window = gr.File(file_count="multiple", file_types=[".txt",".epub"], label='파일들' )
                        lang_msg= gr.HTML(self.upload_msg)
                        input_window.change(fn=self.change_upload, inputs=input_window, outputs=lang_msg, preprocess=False)

                with gr.Column(scale=2):
                    with gr.Tab('순서 2'):
                        translate_btn = gr.Button(value="번역 실행하기(NHNDQ 모델)", size='lg', variant="primary", interactive = True)

                        gr.HTML("<div style='text-align:right'><p style = 'color:grey;'>처음 실행시 모델을 다운받는데 아주 오랜 시간이 걸립니다.</p><p style='color:grey;'>컴퓨터 사양이 좋다면 번역 속도가 빨라집니다.</p><p style='color:grey;'>맥m1이상에서는 mps를 이용하여 가속합니다</p></div>")

                        with gr.Row():
                            status_msg = gr.Textbox(label="상태 정보", scale=4, value='번역 대기중..')
                            translate_btn.click(fn=self.translateFn, outputs=status_msg )
                            
                            btn_openfolder = gr.Button(value='📂 번역 완료한 파일들 보기', scale=1, variant="secondary")
                            btn_openfolder.click(fn=lambda: self.open_folder(), inputs=None, outputs=None)

        app.queue().launch(inbrowser=True, favicon_path = 'imgs/dodari.png', allowed_paths=["."])

    def finalize_fn(self):
        sec = self.check_time()
        self.start = None
        return sec

    def get_translator(self):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.selected_model, cache_dir=os.path.join("models", "tokenizers"))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=self.selected_model, cache_dir=os.path.join("models"))

        
        gpu_count = torch.cuda.device_count()
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" )

        if gpu_count > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(gpu_count)))
            torch.multiprocessing.set_start_method('spawn')
        self.model.to(device)

        translator = pipeline('translation', model=self.model, tokenizer=self.tokenizer, device=device, src_lang=self.origin_lang, tgt_lang=self.target_lang, max_length=self.max_len)
        return translator


    def translateFn(self, progress=gr.Progress() ):
        if not self.selected_files : return "번역할 파일을 추가하세요."
        
        self.start = time.time()
        progress(0, desc="번역 모델을 준비중입니다...")

        translator = self.get_translator()

        origin_abb = self.origin_lang.split(sep='_')[0]
        target_abb = self.target_lang.split(sep='_')[0]

        for file in progress.tqdm(self.selected_files, desc='파일로딩'):
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
                                if ahtml_text :
                                    p_tag_1.name = 'p'
                                    p_tag_2.name = 'p'
                                else: p_tags_1 = soup_1.find_all('p')
                        p_tags_2 = soup_2.find_all('p')

                    for text_node_1, text_node_2 in progress.tqdm( zip(p_tags_1, p_tags_2), desc='단락수' ): 
                        if not text_node_1.text.strip(): continue

                        p_tag_1 = soup_1.new_tag('p')
                        p_tag_2 = soup_2.new_tag('p')

                        try:
                            if text_node_1.attrs and text_node_1.attrs['class']:
                                p_tag_1['class'] = text_node_1.attrs['class']
                                p_tag_2['class'] = text_node_1.attrs['class']
                        except: pass

                        particle = nltk.sent_tokenize(text_node_1.text)
                        particle_list_1 = []
                        particle_list_2 = []
                        for text in progress.tqdm( particle, desc='문장수') :
                            output = translator(text, max_length=self.max_len)
                            translated_text_1 = "{t1} ({t2}) ".format(t1=output[0]['translation_text'], t2=text) 
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

                    
                    output_file_1.write( str(soup_1) )
                    output_file_2.write( str(soup_2) )
                    output_file_1.close()
                    output_file_2.close()

                self.zip_folder(self.temp_folder_1, f'{self.temp_folder_1}.epub')
                self.zip_folder(self.temp_folder_2, f'{self.temp_folder_2}.epub')
                os.makedirs(self.output_folder, exist_ok=True)
                shutil.move(f'{self.temp_folder_1}.epub', os.path.join(self.output_folder, "{name}_{t2}({t3}).{ext}".format(name=name, t2=target_abb, t3=origin_abb, ext = ext) ) )
                shutil.move(f'{self.temp_folder_2}.epub', os.path.join(self.output_folder, "{name}_{t2}.{ext}".format(name=name, t2=target_abb, ext = ext) ) )

                self.remove_folder(self.temp_folder_1)
                self.remove_folder(self.temp_folder_2)

            else:
                output_file_bi = self.write_filename( "{name}_{t2}({t3}).{ext}".format(name=name, t2=target_abb, t3=origin_abb, ext = ext) )
                output_file = self.write_filename( "{name}_{t2}.{ext}".format(name=name, t2=target_abb, ext = ext) )

                book = self.get_filename(file['path']);
                book_list = book.split(sep='\n')
                for book in progress.tqdm(book_list, desc='단락'):
                    particle = nltk.sent_tokenize(book)
                    
                    for text in progress.tqdm( particle, desc='문장' ):
                        output = translator(text, max_length=self.max_len)
                        output_file_bi.write("{t1} ({t2}) ".format(t1=output[0]['translation_text'], t2=text) )
                        output_file.write(output[0]['translation_text'])
                    output_file_bi.write('\n')
                    output_file.write('\n')
                output_file_bi.close()
                output_file.close()

        sec = self.finalize_fn()
        return "번역완료! 걸린시간 : {t1}".format(t1=sec)


    def change_upload(self, files):
        try:
            self.selected_files = files
            if not files : return self.upload_msg
            aBook = files[0]
            name, ext = os.path.splitext(aBook['path'])
            if '.epub' in ext:
                file = epub.read_epub(aBook['path'])
                lang = file.get_metadata('DC', 'language')
                if lang:
                    check_lang = lang[0][0]
                else:
                    for item_idx, item in enumerate(file.get_items()):
                        if item.get_type() == ebooklib.ITEM_DOCUMENT:
                            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                            all_tags = soup.find_all('p')
                            if not all_tags: continue

                            text_tags = [tag.text for tag in all_tags if tag.text.strip()]
                            lang_str = ' '.join(text_tags)
                            check_lang = detect(lang_str[0:500])
                            if 'en' in check_lang or 'ko' in check_lang: break
                            else:
                                return "<p style='text-align:center;color:red;'>표준 규격을 벗어난 epub입니다. <a href='https://moonlit.himion.com/info/contactUs'>이곳</a>을 이용해 해당 epub파일을 첨부해 보내주시면 바로 해결해드립니다. 번역에 실패했습니다.</p>"

            else:
                book = self.get_filename(aBook['path']);
                check_lang = detect(book[0:200])

            self.origin_lang_str = '영어' if 'en' in check_lang else "한국어"
            self.target_lang_str = '한국어' if 'en' in check_lang else "영어"
            self.origin_lang = "eng_Latn" if 'en' in check_lang else "kor_Hang"
            self.target_lang = "kor_Hang" if 'en' in check_lang else "eng_Latn"
            self.selected_model = 'NHNDQ/nllb-finetuned-en2ko' if 'en' in check_lang else 'NHNDQ/nllb-finetuned-ko2en'

            return "<p style='text-align:center;'><span style='color:skyblue;font-size:1.5em;'>{t1}</span><span>를 </span> <span style='color:red;font-size:1.5em;'> {t2}</span><span>로 번역합니다.</span></p>".format(t1=self.origin_lang_str, t2 = self.target_lang_str)
        except Exception as err:
            return "<p style='text-align:center;color:red;'>어떤 언어인지 알아내는데 실패했습니다.</p>"

    def get_filename(self, fileName):
        try:
            input_file = open(fileName, 'r', encoding='utf-8')
            return input_file.read()
        except:
            try :
                input_file = open(fileName, 'r', encoding='euc-kr')
                return input_file.read()
            except :
                input_file = open(fileName, 'r', encoding='cp949', errors='ignore')
                return input_file.read()

    def write_filename(self, file_name):
        
        saveDir = self.output_folder
        if not(os.path.isdir(saveDir)): 
            os.makedirs(os.path.join(saveDir)) 

        file = saveDir + '/' + file_name
        output_file = open(file, 'w', encoding='utf-8')
        return output_file

    def open_folder(self):
        
        saveDir = self.output_folder
        if not(os.path.isdir(saveDir)): 
            os.makedirs(saveDir) 
        if  self.platform == 'Windows': os.system(f"start {saveDir}")
        elif self.platform == 'Darwin': os.system(f"open {saveDir}")
        elif self.platform == 'Linux': os.system(f"nautilus {saveDir}")
        

    def zip_extract(self, folder_path, epub_file):
        try:
            zip_module = zipfile.ZipFile(epub_file, 'r')
            os.makedirs(folder_path, exist_ok=True)
            zip_module.extractall(folder_path)
            zip_module.close()
        except:
            print('잘못된 epub파일입니다')
            pass

    
    def zip_folder(self, folder_path, epub_name):
        try:
            zip_module = zipfile.ZipFile(epub_name, 'w', zipfile.ZIP_DEFLATED)
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_module.write(file_path, os.path.relpath(file_path, folder_path))
            zip_module.close()
        except Exception as err:
            print('epub 파일을 생성하는데 실패했습니다.')
            pass

    
    def get_html_list(self):
        file_path = []
        for root, _, files in os.walk(self.temp_folder_1):
            for file in files:
                if file.endswith('xhtml') or file.endswith('html'):
                    file_path.append(os.path.join(root, file))
        return file_path;
    def check_time(self):
        end = time.time()
        during = end - self.start
        sec = str(datetime.timedelta(seconds=during)).split('.')[0]
        return sec

if __name__ == "__main__":
    dodari = Dodari()
    dodari.main()