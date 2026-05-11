<p align="center">
<img src='https://github.com/user-attachments/assets/6b6a73f6-087c-439c-869f-5e0d0629db92' width='200px' height='200px' title='Dodari'/>
<h1 align="center">Dodari 2</h1>
<p align='center'>
English | <a href='https://github.com/vEduardovich/dodari/README.ko.md'>한국어</a> <br/>
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

4. Windows is considerably slower.
* _On a 2020 LG Gram laptop, translating one page of the novel *1984* took 15 minutes for EPUB and 18 minutes for PDF (the first PDF load can take up to 20 minutes)_
* _So on a typical Windows laptop, a 100-page EPUB would take roughly 1,500 minutes (25 hours). 200 pages = 50 hours. That said, watching your computer work tirelessly for you is strangely satisfying_


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

© 2026 Dodari Project. All rights reserved.