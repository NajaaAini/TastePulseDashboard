##RUN THISSS
## THE ACCURACY ACCURATE

import streamlit as st
import pandas as pd
import numpy as np
import re
import malaya
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import pipeline
from nltk.util import ngrams
from collections import Counter
import tensorflow as tf
import pickle
import plotly.graph_objects as go
from streamlit_js_eval import get_geolocation
from geopy.distance import geodesic # To calculate distance between you and cafes
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim


# --- CONFIGURATION ---#
st.set_page_config(page_title="TastePulse Strategic Dashboard", layout="wide")

st.markdown("""
    <style>
    
    /* 1. GLOBAL CANVAS & TYPOGRAPHY */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #cfdae6 !important; /* Pure White Background */
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    .main .block-container {{
        zoom: {zoom_factor} !important;
        -moz-transform: scale({zoom_factor}); /* Fallback support for Firefox engines */
        -moz-transform-origin: top left;
    }}

    /* 2. EXECUTIVE SIDEBAR (NAVY BLUE) */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important; /* Deep Navy Blue */
        border-right: 2px solid #E2E8F0;
    }

    /* Sidebar Text Elements - Crisp White */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* 3. TITLES & HEADERS (NAVY BLUE) */
    h1, h2, h3, .stSubheader {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    /* 4. CORPORATE METRIC CARDS (WHITE WITH NAVY ACCENTS) */
    [data-testid="stMetric"] {
        background-color: #F8FAFC !important; /* Very light blue-grey tint */
        border: 1px solid #E2E8F0 !important;
        border-top: 4px solid #1E3A8A !important; /* Navy Blue Accent Top Border */
        padding: 1.5rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }

    [data-testid="stMetricLabel"] {
        color: #475569 !important; /* Slate Grey */
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 1.0rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #1E3A8A !important; /* Vibrant Navy Blue */
        font-weight: 800 !important;
    }

    /* 5. TABS STYLING (NAVY THEME) */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #E2E8F0 !important;
    }

    .stTabs [data-baseweb="tab"] p {
        color: #64748B !important;
        font-weight: 600 !important;
    }

    .stTabs [aria-selected="true"] p {
        color: #1E3A8A !important; /* Navy for active tab */
    }

    /* 6. BUTTONS */
    .stButton>button {
        background-color: #1E3A8A !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #0F172A !important; /* Darker navy on hover */
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }

    /* 7. DATA TABLES */
    [data-testid="stTable"], [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
    }
    
    /* Input Widgets Black text for visibility on white background */
    div[data-testid="stWidgetLabel"] p, label {
        color: #0F172A !important;
    }
    
    /* FIXED ROUNDED WHITE CARD CONTROLLER (Safely ignores components nested within expanders to prevent breaking them) */
    [data-testid="column"] div[data-testid="stVerticalBlock"] > div.element-container:has(div[data-testid="stBlockContainer"] .stElementContainer),
    [data-testid="column"] > div[data-testid="stVerticalBlock"] > div.element-container:not(:has(div[data-testid="stExpander"])) {
        background-color: #FFFFFF !important; /* Pure White Card Canvas */
        border-radius: 14px !important;      /* Softer, more premium corner radius */
        padding: 1.75rem !important;          /* Generous breathing room inside the cards */
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05) !important; /* Soft depth shadow */
        border: 1px solid #CBD5E1 !important; /* Clean corporate boundary line */
        margin-bottom: 1.5rem !important;     /* Spacing before the next row */
    }

    /* Ensure inner components don't clip through the rounded borders */
    div[data-testid="stPlotlyChart"], .stAlert, img {
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    /* FORCE COLUMNS TO MATCH HEIGHT EQUALLY */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: stretch !important; /* Forces side-by-side columns to be identical height */
    }

    [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }

    /* CARDS STRETCH TO FILL THE WHOLE COLUMN SAFELY WITHOUT EXPANDER COLLISION */
    [data-testid="column"] .stVBox:not(:has(div[data-testid="stExpander"])) {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important; /* Pushes interpretation notes to the bottom */
        background-color: #FFFFFF !important;
        border-radius: 14px !important;
        padding: 1.75rem !important;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05) !important;
        border: 1px solid #CBD5E1 !important;
    }

    /* Ensure interpretation / info blocks anchor at the bottom */
    [data-testid="column"] .stAlert {
        margin-top: auto !important; /* Pushes the interpretation notes to match baselines */
    }

    /* ============================================
       8. EXPANDER (STREAMLIT NATIVE FIXED)
       ============================================ */
    div[data-testid="stExpander"] {
        border: none !important;
        background-color: transparent !important;
        margin-bottom: 1.25rem !important;
        width: 100% !important;
    }

    /* Clean corporate expander header style */
    div[data-testid="stExpander"] details summary {
        background-color: #FFFFFF !important;
        color: #1B3F5E !important;
        font-weight: 600 !important;
        border: 1px solid #B2D8E8 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.25rem !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
    }

    /* When expanded, dynamically shift content border boundaries */
    div[data-testid="stExpander"] details[open] summary {
        border-bottom-left-radius: 0px !important;
        border-bottom-right-radius: 0px !important;
        border-bottom: 1px dashed #B2D8E8 !important;
    }

    /* Target the inner vertical container box holding text inside expanders */
    div[data-testid="stExpander"] details div[data-testid="stVerticalBlock"] {
        background-color: #FFFFFF !important;
        border: 1px solid #B2D8E8 !important;
        border-top: none !important;
        border-bottom-left-radius: 10px !important;
        border-bottom-right-radius: 10px !important;
        padding: 1.5rem !important;
        
        /* Force text inside expanders to wrap cleanly on zoom */
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: normal !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# --- BRAND HEADER & LOGO GRID ---
col_logo, col_title = st.columns([0.15, 0.85], vertical_alignment="center")

with col_logo:
    try:
        # Renders the logo. use_container_width ensures it resizes gracefully on zoom
        st.image("logo.png", use_container_width=True)
    except Exception:
        # Fallback icon box in case logo.png is temporarily missing from your folder
        st.subheader("🍽️") 

with col_title:
    # Your Main Application Header Title
    st.title("TastePulse Strategic Dashboard")

# Add a clean boundary rule before the rest of your app content begins
st.divider()

#LOCATION
CAFE_LOCATIONS = {
    "Dusun Riffa, Changlun": [6.423946, 100.473019],
    "Dapoq Tok Su Lijah, Sintok": [6.434437, 100.436695],
    "Pahit Cafe, Sintok": [6.443910, 100.446854],
    "Dapoq Mak Kami, Jitra": [6.320635, 100.447478],
    "Potlepak, Changlun": [6.437034, 100.421312],
    "Achik Steamboat & Shellout, Jitra": [6.275820, 100.420783],
    "Pak Man Cafe, Jitra": [6.255540, 100.467795],
    "Forestspot Cafe, Jitra": [6.261284, 100.446172],
    "Sifu Tomyam, Jitra": [6.325441, 100.422638],
    "Damia Ayam Gunting, Jitra": [6.254032, 100.416826],
    "Lubuk Makan, Changlun": [6.436497, 100.445877],
    "Warong Chabok, Jitra": [6.254564, 100.412890],
    "Dua Rasa Timur Utara, Changlun": [6.429412, 100.428787],
    "Dekya Authentic Thai, Changlun": [6.436720, 100.420474],
    "Kopi Logika, Jitra": [6.262017, 100.421661],
    "Nasi Goreng Taman Ehsan, Jitra": [6.258984, 100.423266],
    "Nasi Lemak Panas Oh Tajul, Jitra": [6.285971, 100.420002],
    "Warung Pokok Getah, Changlun": [6.455182, 100.418269],
    "Thai Celup, Jitra": [6.291287, 100.387583],
    "MD2 Botanic Cafe, Changlun": [6.429378, 100.474998],
    "Rasta Cafe, Jitra": [6.277133, 100.410817],
    "D'Kayangan Palace, Jitra": [6.217136, 100.416871],
    "Breaktrits Cafe, Changlun" : [6.423364075687903, 100.4040927239452],
    "Chuacha, Changlun" : [6.447188316106749, 100.41549057456629],
    "Det's Pizzeria,Changlun" : [6.4219539381823045, 100.39345050859986],
    "THE BRO’s CAFE, Changlun" : [6.450483348574081, 100.47733391549053],
    "HillSide Cafe, Sintok" : [6.440964666826916, 100.45127421550666],
    "Flourcrown Patisserie Cafe, Jitra" : [6.23743167183583, 100.42225547790913],
    "Restaurant Pastalia, Jitra" : [6.2431167386548045, 100.420703972276],
    "The Weekend Cafe, Jitra" : [6.331111477884146, 100.3654122969632],
    "Dekukis Cafe, Sintok" : [6.443523015945437, 100.47858889088961],
    "Santap Tokwan, Sintok" : [6.438243839960452, 100.44737114670794],
    "Cik Jah Nasi Ayam Bakar, Sintok" : [6.43693982403718, 100.44350670489115],
    "Mum's Heritage House, Changlun" : [6.434987642215916, 100.42882025243937],
    "Thai Kitchen Kafeteria, Jitra" : [6.27334839701289, 100.40754765274774],
    "Layan Cafe, UUM" : [6.457713250701544, 100.50271028838606],
    "Sepagi Kopitiam, UUM" : [6.4622290633983415, 100.49922289420716],
    "S'Antik Cafe, UUM" : [6.460650489134661, 100.49934960001983],
    "RootsBase, UUM" : [6.470632086408734, 100.50098613813057],
    "Chavista, UUM" : [6.457827998106925, 100.5034232404532],
    "Pandan Cendol, UUM" : [6.470727480115344, 100.50064103858125],
    "Eyskem, UUM" : [6.457149909892413, 100.5046496037505],
    "Ch4tter Coffee, UUM" : [6.46396198822407, 100.50076769576155]
}


def get_alert_keywords(text_series, n=2):
    try:
        # Create bigrams from the filtered text
        vec = CountVectorizer(ngram_range=(2, 2)).fit(text_series)
        bag_of_words = vec.transform(text_series)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        # Return top N results formatted as a string
        return ", ".join([f"'{x[0]}'" for x in words_freq[:n]])
    except:
        return "general negative feedback"


# ===================================
#      PREPROC - RESOURCES
# ===================================
kedah_dict = {
    "hang": "kamu","hangpaa": "kamu semua","hangpa": "kamu semua","hampa": "kamu semua",
    "awat": "kenapa","mai": "mari","cek": "saya","sat": "sekejap","dak": "tak","pa": "apa",
    "mcm": "macam","sbb": "sebab","xtau": "tak tahu","tk": "tak","x": "tak","mmg": "memang",
    "gheti": "pandai","kito": "kita", "tq": "terima kasih", "p": "pergi","pi": "pergi","tuju": "pergi",
    "ni": "ini", "besok": "esok", "setaq": "setar","jgn": "jangan","yg": "yang","kat": "di",
    "segan": "malu","kompom": "confirm","bwk": "bawak","org":"orang","sgup": "sanggup","maap": "maaf","dap": "sedap",
    "sedp": "sedap","shedap": "sedap","pi":"pergi","n": "dan","xkan": "tidak akan","6x" : "6kali","nk": "nak", "shakap" : "cakap",
    "mknan" : "makanan","bes" : "best", "habaq" : "bagitau", "trasa" : "terasa", "pernh" : "pernah",
    "xdpt" : "tidak dapat", "rifa" : "riffa", "sodap" : "sedap", "qamai" : "ramai", "nk" :"nak", "keja" : "kerja", 
    "boking" : "booking","sgt" : "sangat", "mkn" : "makan", "nti" : "nanti", "lgi" : "lagi",
    "dh" : "dah", "byk" : "banyak", "tkdak" : "tiada", "hrga" : "harga", "skli" : "sekali", "kelia" : "bagus",
    "habih" : "habis","lembab" : "lambat", "sdp" : "sedap", "x" : "tak", "xsedap" : "tak sedap",
    "beghatoq" : "beratur" , "baghu" : "baru" , "pyh" : "payah", "bulih" : "boleh",
    "awai" : "awal", "belambak" : "banyak", "tapaw" : "bungkus", "kyg" : "kenyang", "byk" : "banyak"
}

stopwords = {
     "1000K","2025","4Upage","4U","6Pagi","Abcxyz","1Year","Xbyzca","Xcyzba","Xeeprac","Xetaidien","Xfaktor","Xmax250V3",
    "Xuhuong","Xybca","Xzybca","yang","dan","di","ke","dari","itu","ini","untuk","dengan","pada","dalam",
    "ada","atau","saya","awak","kami","kita","mereka","dia","tu","lah","pun",
    "ia","kau","engkau","korang","korg","ko","aku",
    "nak","takde","takdak","xdak","xde","tidak","bukan",
    "lebih","kurang","semua","setiap","banyak","sedikit",
    "oleh","sebab","kerana","k kerana","sebabtu","jadi",
    "akan","boleh","mampu","harus","patut","jika","kalau","walaupun",
    "dalam","luar","atas","bawah","antara","semasa","selepas","sebelum",
    "bagaimana","apa","siapa","mana","kenapa","mengapa",
    "buat","jadi","macam","mcm","je","aja","saja","jer","ja",
    "punya","punye","punyew","punyaaa","â",
    "lah","la","leh","kan","pun","nya","tu","ni","eh","oh","ha","haa",
    "orang","org","sorang","seorang",
    "bagi","dapat","perlu","mesti","sudah","belum","pusingan",
    "haha","hahaha","kah","weh","wei","eh","ya","ye","ok","okay",
    "ohh","hmm","emm", "fyp", "terima kasih", "alor setar", "viral",
    "fypviral","makan","kedahfodie","kedahfoodie","kedah", "fodie", "jcmkedah","hari","the","in","a",
    "to","fod","mari","dekat","sini","in","hatyai","foryou","thailand","datang","try",
    "makananviral","baru", "fod","alor","setar","fyp","um","mukbang","sungai petani","kamu","alorsetar","tau","capcut","jangan","lupa","eating","fypage",
    "replying","um","depa","hidengem","makanlokal","kedai","jom","and","dah","for","1",
    "lagi","of","menu","malaysia","with","0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", 
    "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", 
    "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", 
    "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", 
    "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate",
    "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", 
    "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az",
    "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", 
    "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", 
    "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", 
    "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", 
    "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", 
    "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", 
    "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", 
    "contains", "corresponding","capcut","Capcut", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz",
}

import streamlit as st
import pandas as pd
import numpy as np
import re
import malaya
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- AI ENGINES ---
@st.cache_resource
def load_bilstm_resources():
    # Load trained tokenizer from your training phase
    try:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    except:
        # Fallback if pickle is missing
        tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")

    # Build BiLSTM Architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(5000, 64, input_length=100),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Load trained weights
    try:
        model.load_weights("bilstm_sentiment_model.h5")
    except:
        st.warning("Model weights not found. Running with uninitialized weights.")

    demoji_tool = malaya.preprocessing.demoji()
    return model, tokenizer, demoji_tool

bilstm_model, tokenizer, demoji_tool = load_bilstm_resources()

def get_bilstm_preds(test_df):
    texts = test_df['text_cleaned'].astype(str).tolist()
    if not texts: return [], []

    # Use trained tokenizer ONLY (do not use fit_on_texts to avoid leakage)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

    predictions = bilstm_model.predict(padded)
    classes = ['negative', 'neutral', 'positive']
    
    preds = [classes[np.argmax(p)] for p in predictions]
    scores = [np.max(p) for p in predictions]
    return preds, scores

# --- NB MODEL ---
def train_nb_model(train_df):
    vec = TfidfVectorizer(max_features=1000)
    X_train = vec.fit_transform(train_df['text_cleaned'])
    model = MultinomialNB()
    model.fit(X_train, train_df['sentiment_label'])
    return model, vec

def predict_nb(model, vec, test_df):
    X_test = vec.transform(test_df['text_cleaned'])
    preds = model.predict(X_test)
    probs = np.max(model.predict_proba(X_test), axis=1)
    return preds, probs

# --- PREPROCESSING PIPELINE ---
@st.cache_data
def run_full_preprocessing(df):
    df = df.copy()
    
    df.columns = df.columns.str.strip().str.lower()
    
    if 'text' not in df.columns:
        st.error("Uploaded CSV is missing a 'text' column!")
        return df
        
    df = df.dropna(subset=["text"])
    
    target_date_col = 'createtimeiso' if 'createtimeiso' in df.columns else 'date'
    if target_date_col in df.columns:
        df['date'] = pd.to_datetime(df[target_date_col], errors='coerce')
        df['date'] = df['date'].fillna(pd.Timestamp('2024-01-01')).dt.date

    def refined_cleaner(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'<.*?>|https?://\S+|www\.\S+', '', text)
        emoji_dict = demoji_tool.demoji(text)
        for emoji_sym, description in emoji_dict.items():
            text = text.replace(emoji_sym, f" {description} ")
        text = re.sub(r'[^\w\s]', ' ', text).lower()
        
        # Kedah Slang & Stopword Logic
        words = text.split()
        cleaned_words = [kedah_dict.get(w, w) for w in words if kedah_dict.get(w, w) not in stopwords]
        return " ".join(cleaned_words).strip()

    with st.spinner("Cleaning and normalizing Kedah dialect..."):
        df['text_cleaned'] = df['text'].apply(refined_cleaner)
    
    # Fill defaults for missing data
    df['sentiment_label'] = df.get('sentiment_label', pd.Series(['neutral']*len(df))).fillna('neutral').str.lower().str.strip()
    df['place_name'] = df.get('place_name', pd.Series(['Unknown Venue']*len(df))).fillna('Unknown Venue').str.replace("'", "").str.strip()
    
    return df

with st.sidebar:
    from sklearn.model_selection import train_test_split

    st.title("DATA CONTROL")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    # 1. CHECK IF FILE IS UPLOADED
    if uploaded_file is not None:
        path = uploaded_file
        
        # Clean visual banner displaying the file name in white font
        st.markdown(
            f"""
            <div style="padding: 12px; background-color: #1E1E1E; border-left: 5px solid #00E676; border-radius: 4px; margin: 15px 0;">
                <span style="color: white; font-family: monospace; font-size: 14px;">📁 <b> File:</b> {uploaded_file.name}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.info("**No Data Available:** Please upload a valid CSV file.")
        st.stop() # Softly halts the execution of code below this point on the current page

    # 2. RUN DATA PIPELINE SECURELY
    try:
        raw_df = pd.read_csv(path)
        df = run_full_preprocessing(raw_df)
        
        # 1. Reliable Stream (100% of data - Used for Influencers)
        df_full_for_metrics = df.copy() 
        
        # 2. Scientific Stream (20% Split - Used for Accuracy and general Charts)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Clean test set for realistic evaluation
        train_df = train_df.drop_duplicates(subset=['text_cleaned', 'sentiment_label'])
        test_df = test_df.drop_duplicates(subset=['text_cleaned', 'sentiment_label'])
        test_df = test_df[~test_df['text_cleaned'].isin(set(train_df['text_cleaned']))]
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
        
    # 🔍 Accessibility Notice Block
    st.markdown("### 🔍 Display Accessibility")
    st.markdown(
        """
        <div style="color: white;">
            <ul>
                <li><strong>Zoom In:</strong> Press <code>Ctrl</code> + <code>+</code></li>
                <li><strong>Zoom Out:</strong> Press <code>Ctrl</code> + <code>-</code></li>
                <li><strong>Reset View:</strong> Press <code>Ctrl</code> + <code>0</code></li>
            </ul>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.subheader("🧠 Model Selection")
    engine_choice = st.radio("Primary Model:", ["MultiNB (Machine Learning)", "BiLSTM (Deep Learning)"])
    
    st.subheader("📊 Sentiment Filter")
    sentiment_filter = st.selectbox("View Sentiment:", ["All", "Positive", "Neutral", "Negative"])

    st.divider()
    
    # --- DATE & VENUE FILTERS ---
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_val = st.date_input("Select Date Range", [min_date, max_date])

    if isinstance(date_val, (list, tuple)) and len(date_val) == 2:
        start_date, end_date = date_val
    else:
        start_date, end_date = min_date, max_date

    cafes = sorted([str(x) for x in df['place_name'].unique() if pd.notna(x)])
    sel_cafes = st.multiselect("📍 Filter Venues", ["All"] + cafes, default=["All"])

# --- APPLY MASKS ---
mask_full = (df_full_for_metrics['date'] >= start_date) & (df_full_for_metrics['date'] <= end_date)
mask_test = (test_df['date'] >= start_date) & (test_df['date'] <= end_date)

if "All" not in sel_cafes:
    mask_full &= (df_full_for_metrics['place_name'].isin(sel_cafes))
    mask_test &= (test_df['place_name'].isin(sel_cafes))

reliable_filtered_df = df_full_for_metrics[mask_full].copy() # 100%
base_filtered_df = test_df[mask_test].copy()               # 20%

if reliable_filtered_df.empty or base_filtered_df.empty:
    st.warning("⚠️ No data found for the selected criteria.")
    st.stop()

# --- PREDICTIONS & ACCURACY ---
nb_model, nb_vec = train_nb_model(train_df)

# This ensures that models ONLY execute once during a data upload, completely eliminating filter lag.
if 'ml_pred' not in test_df.columns:
    with st.spinner("Initializing baseline Machine Learning predictions..."):
        test_df['ml_pred'], _ = predict_nb(nb_model, nb_vec, test_df)
        test_df['dl_pred'], _ = get_bilstm_preds(test_df)
        test_df['true_label'] = test_df['sentiment_label']

if 'ml_pred' not in df_full_for_metrics.columns:
    with st.spinner("Initializing Deep Learning influencer stream pipeline..."):
        df_full_for_metrics['ml_pred'], _ = predict_nb(nb_model, nb_vec, df_full_for_metrics)
        df_full_for_metrics['dl_pred'], _ = get_bilstm_preds(df_full_for_metrics)

# --- RE-APPLY FILTERS ON TOP OF PRE-COMPUTED DATA (Happens instantly) ---
mask_full = (df_full_for_metrics['date'] >= start_date) & (df_full_for_metrics['date'] <= end_date)
mask_test = (test_df['date'] >= start_date) & (test_df['date'] <= end_date)

if "All" not in sel_cafes:
    mask_full &= (df_full_for_metrics['place_name'].isin(sel_cafes))
    mask_test &= (test_df['place_name'].isin(sel_cafes))

# Dynamically slice our pre-calculated rows into the active dashboard subsets
reliable_filtered_df = df_full_for_metrics[mask_full].copy()
base_filtered_df = test_df[mask_test].copy()

if reliable_filtered_df.empty or base_filtered_df.empty:
    st.warning("⚠️ No data found for the selected criteria.")
    st.stop()

# A. Process the 20% Stream (Accuracy & Standard Charts)
ml_preds_test = base_filtered_df['ml_pred']
dl_preds_test = base_filtered_df['dl_pred']

# Select model for 20% stream
if engine_choice == "MultiNB (Machine Learning)":
    base_filtered_df['sentiment_label'] = ml_preds_test
else:
    base_filtered_df['sentiment_label'] = dl_preds_test

# Calculate Accuracy
nb_acc = accuracy_score(base_filtered_df['true_label'], ml_preds_test)
dl_acc = accuracy_score(base_filtered_df['true_label'], dl_preds_test)

# B. Process the 100% Stream (Influencer Analysis)
if engine_choice == "MultiNB (Machine Learning)":
    reliable_filtered_df['sentiment_label'] = reliable_filtered_df['ml_pred']
else:
    reliable_filtered_df['sentiment_label'] = reliable_filtered_df['dl_pred']

# --- FINAL DATASET ASSIGNMENT ---

# 1. Standard Charts Filter (20% Data) - This affects Word Clouds, Bar Charts, etc.
if sentiment_filter != "All":
    filtered_test_df = base_filtered_df[base_filtered_df['sentiment_label'] == sentiment_filter.lower()]
else:
    filtered_test_df = base_filtered_df

# 2. Influencer Filter (100% Data) - This is optional, but keep it for other pages
if sentiment_filter != "All":
    influencer_df = reliable_filtered_df[reliable_filtered_df['sentiment_label'] == sentiment_filter.lower()]
else:
    influencer_df = reliable_filtered_df

# Initialize session state for navigation
if 'active_page' not in st.session_state:
    st.session_state.active_page = "Overview"

# Create columns: 5 for main, 4 for the gap, 1 for About
c1, c2, c3, c4, c5,c6, c_end = st.columns(7)

with c1:
    if st.button("Overview", use_container_width=True): st.session_state.active_page = "Overview"
with c2:
    if st.button("Cafe/Restaurant", use_container_width=True): st.session_state.active_page = "Cafes"
with c3:
    if st.button("Word Clouds", use_container_width=True): st.session_state.active_page = "Clouds"
with c4:
    if st.button("Influencer Analysis", use_container_width=True): st.session_state.active_page = "Influencers"
with c5:
    if st.button("Strategic Mapping", use_container_width=True): st.session_state.active_page = "Strategy"
with c6:
    if st.button("Model Comparison", use_container_width=True): st.session_state.active_page = "Model"

with c_end:
    if st.button("About", use_container_width=True): st.session_state.active_page = "About"

st.divider()

##################################################
##################################################
# PAGE 1

if st.session_state.active_page == "Overview":    #st.subheader("📄 Strategic Management Center")

    if not filtered_test_df.empty:
        # 1. Create temporary copy for calculation
            inf_sync_df = filtered_test_df.copy()

        # 2. Extract Handle (ADDED 'videoweburl' to the list)
            url_col = next((c for c in ['videoweburl', 'webVideoUrl', 'videoWebUrl'] if c in inf_sync_df.columns), None)
        
            if url_col:
                inf_sync_df['author_handle'] = inf_sync_df[url_col].apply(
                    lambda x: str(x).split('@')[1].split('/')[0] if pd.notna(x) and '@' in str(x) else "Unknown"
                )
            else:
                # SAFETY FALLBACK: If no URL column is found, use uniqueid or just mark as Unknown
                if 'uniqueid' in inf_sync_df.columns:
                    inf_sync_df['author_handle'] = inf_sync_df['uniqueid'].astype(str)
                else:
                    inf_sync_df['author_handle'] = "Unknown"
            
            # 3. Standardize Metrics (Handling lowercase 'diggcount', etc.)
            # This maps your CSV's lowercase names to the names used in your math
            metric_map = {
                'playcount': 'playCount', 
                'diggcount': 'diggCount', 
                'sharecount': 'shareCount', 
                'commentcount': 'commentCount'
            }
            
            for old_name, new_name in metric_map.items():
                if old_name in inf_sync_df.columns:
                    inf_sync_df[new_name] = pd.to_numeric(inf_sync_df[old_name], errors='coerce').fillna(0)
                elif new_name not in inf_sync_df.columns:
                    inf_sync_df[new_name] = 0

            # 4. Group & Filter "Unknown"
            inf_stats = inf_sync_df.groupby('author_handle').agg({
                'commentCount': 'sum',
                'diggCount': 'sum',
                'playCount': 'sum',
                'shareCount': 'sum'
            }).rename(columns={'commentCount': 'Total Comments'})

            # Remove the noise
            inf_stats = inf_stats[~inf_stats.index.str.contains('unknown|nan|none', case=False, na=False)]

            # 5. Calculate Score (Weighting 'Fame' over 'Frequency')
            if not inf_stats.empty:
                inf_stats['Influence Score'] = (
                    (inf_stats['playCount'] * 1.5) + 
                    (inf_stats['diggCount'] * 2.0) + 
                    (inf_stats['shareCount'] * 5.0) +
                    (inf_stats['Total Comments'] * 0.1) # low weight for comment frequency
                )
                
                inf_stats = inf_stats.sort_values(by='Influence Score', ascending=False)

                # 6. Define the Global Variables for the UI
                global_top_handle = f"@{inf_stats.index[0]}" if not str(inf_stats.index[0]).startswith('@') else inf_stats.index[0]
                inf_plot_df = inf_stats.reset_index().rename(columns={'index': 'author_handle'})
            else:
                global_top_handle = "N/A"

         # Download Button & Red Flag Alert in two columns to save space
    head_col, btn_col = st.columns([4, 1])
    
    with head_col:
        st.header("📄 Strategic Management Center")
       
    with btn_col:
        # Move the download logic here
        csv = filtered_test_df[['date', 'place_name', 'text', 'sentiment_label']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export CSV",
            data=csv,
            file_name='MPKP_Sentiment_Report.csv',
            mime='text/csv',
            use_container_width=True
        )

    if not filtered_test_df.empty:
        alert_stats = filtered_test_df.groupby('place_name').agg(
            Total=('text', 'count'),
            Pos_Rate=('sentiment_label', lambda x: (x == 'positive').mean() * 100)
        ).reset_index()
        critical_cafes = alert_stats[(alert_stats['Pos_Rate'] < 15) & (alert_stats['Total'] > 2)]
            
        if not critical_cafes.empty:
            st.markdown(f"""
                <div style="
                background:#FFF8E6;
                padding:18px;
                border-left:6px solid #FFB703;
                border-radius:12px;
                margin-bottom:15px;
                ">
                <h4>⚠️ Attention Required</h4>
                <p><strong>{len(critical_cafes)}</strong> venues show consistently low positive sentiment.</p>
                </div>
                """, unsafe_allow_html=True)
            cols = st.columns(len(critical_cafes))
            for i, (_, row) in enumerate(critical_cafes.iterrows()):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div style='background:#DC2626;color:white;
                        padding:17px;border-radius:12px;
                        height:140px;text-align:center'>
                            <b>{row['place_name']}</b>
                            <h2 style='color:white;margin-top:10px;'>
                                {row['Pos_Rate']:.1f}%
                            </h2>
                        </div>
                        """,
                        unsafe_allow_html=True)
            st.info("💡 **Action:** These places have a Positive Score below 15%, meaning most feedback is negative. Review and improve service or quality.")


    # --- 2. KPI METRICS ---
    
    k1, k2, k3, k4, k5 = st.columns(5,gap="medium")
    k1.metric(
        label="Total Food Posts", 
        value=f"{len(filtered_test_df):,}", 
        help="This count represents the Testing Data (20% of the total dataset) used to verify model accuracy after filtering."
    )
    
    k2.metric(
        label="Top Cafe", 
        value=filtered_test_df['place_name'].value_counts().idxmax() if not filtered_test_df.empty else "N/A",
        help="The venue with the highest number of reviews within the filtered test set."
    )
    
    k3.metric(
        label="Engine Active", 
        value=engine_choice,
        help="The AI model currently being used to generate the sentiment labels shown below."
    )
    
    sent_counts = filtered_test_df['sentiment_label'].value_counts(normalize=True) * 100
    pos_pct = sent_counts.get('positive', 0)
    k4.metric(
        label="Positive Sentiment", 
        value=f"{pos_pct:.1f}%",
        help="The percentage of reviews in the test set classified as 'positive' by the active AI engine."
    )
   
    k5.metric(
        label="Top Influencer", 
        value=global_top_handle, 
        help=f"Calculated using {len(reliable_filtered_df)} total records (100% data stream)."    )

    st.divider()

    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        st.subheader("📊 Sentiment Distribution")
        
        # 1. Prepare Data
        sentiment_counts = filtered_test_df['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Total']
        
        # 2. Toggle for Chart Type
        chart_type = st.radio("Select View:", ["Bar Chart (Total)", "Pie Chart (%)"], horizontal=True, label_visibility="collapsed", key="sent_toggle_main")

        if chart_type == "Bar Chart (Total)":
            fig_sent = px.bar(
                sentiment_counts, x='Sentiment', y='Total', 
                color='Sentiment', 
                color_discrete_map={'negative': '#EF553B', 'neutral': '#636EFA', 'positive': '#00CC96'},
                text='Total'
            )
            fig_sent.update_layout(showlegend=False)
            fig_sent.update_xaxes(showgrid=False, zeroline=False, showline=False)
            fig_sent.update_yaxes(showgrid=False, zeroline=False, showline=False)
        else:
            fig_sent = px.pie(
                sentiment_counts, values='Total', names='Sentiment',
                color='Sentiment',
                color_discrete_map={'negative': '#EF553B', 'neutral': '#636EFA', 'positive': '#00CC96'},
                hole=0.4
            )
            fig_sent.update_traces(textinfo='percent+label')

        # Unified Height
        fig_sent.update_layout(height=400, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig_sent, use_container_width=True)
        
        # YOUR ORIGINAL INTERPRETATION
        dominant_sent = sentiment_counts.iloc[0]['Sentiment']
        st.info(f"**💡 Interpretation:** The community mood is primarily **{dominant_sent}**. A high 'Positive' bar indicates successful food tourism, while 'Negative' peaks suggest a need for quality control intervention.")

    with r1_col2:
        st.subheader("🌡️ Regional Heatmap")
        heat_data = []
        for name, coords in CAFE_LOCATIONS.items():
            neg_ratio = filtered_test_df[filtered_test_df['place_name'] == name]['sentiment_label'].value_counts(normalize=True).get('negative', 0)
            heat_data.append([coords[0], coords[1], neg_ratio])
        
        m_heat = folium.Map(location=[6.4550, 100.5050], zoom_start=12)
        from folium.plugins import HeatMap
        HeatMap(heat_data).add_to(m_heat)
        
        # Unified Height
        st_folium(
            m_heat, 
            height=450, 
            use_container_width=True, 
            key="heatmap_overview_top"
        )
        # YOUR ORIGINAL INTERPRETATION
        st.info("💡 **Interpretation:** Red hotspots show areas with more negative feedback. These locations may need service or facility improvements.") 

    st.divider()

   # --- ROW 2: WORD CLOUD & DAILY POSTS PER VENUE ---
  # --- ROW 2: WORD CLOUD & DAILY POSTS PER VENUE ---
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        st.subheader("☁️ Word Cloud")
        wc_limit = st.slider("Word Density", 10, 100, 30, key="wc_slider_overview_row2")

        tokens = " ".join(filtered_test_df['text_cleaned'].astype(str)).split()
        
        if tokens:
            wc = WordCloud(
                background_color="white", 
                width=800, height=400, 
                max_words=wc_limit
            ).generate(" ".join(tokens))
            
            fig_wc, ax = plt.subplots(figsize=(8, 4.5)) 
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")  # Hides traditional chart axes entirely for the image block
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)
            
            # YOUR ORIGINAL INTERPRETATION
            st.info("💡 **Interpretation:** Bigger words appear more often in reviews, showing what people talk about most \t\t (exp: sedap).")         
        else:
            st.warning("No text data available.")

    # --- PLACED TREND_COL2 CONTENT HERE ---
    with r2_col2:
        st.subheader("🏢 Venue Engagement Activity")
        timeline_df = filtered_test_df.copy()
        timeline_df['date'] = pd.to_datetime(timeline_df['date'])
        timeline_df['date_only'] = timeline_df['date'].dt.date
        
        # Venue Trends Chart (Moved to Row 2, Right)
        venue_timeline = timeline_df.groupby(['date_only', 'place_name']).size().reset_index(name='Post Count')
        fig_venue_line = px.line(
            venue_timeline, 
            x='date_only', 
            y='Post Count', 
            color='place_name', 
            markers=True,
            title="Daily Posts per Venue",
            # --- RENAMED AXES ---
            labels={
                'date_only': 'Timeline Date',
                'Post Count': 'Total Review Count',
                'place_name': 'Establishment Name'
            }
        )
        
        # Format layout to match white card & strip grid/false lines
        fig_venue_line.update_layout(
            height=535, 
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF'
        )
        fig_venue_line.update_xaxes(showgrid=False, zeroline=False, showline=False)
        fig_venue_line.update_yaxes(showgrid=False, zeroline=False, showline=False)
        
        st.plotly_chart(fig_venue_line, use_container_width=True)
        st.info("💡 **Interpretation:** Shows how each venue is performing. A cafe with rising activity is likely trending.")

    st.divider()

    # --- 5. BOTTOM SECTION: TRENDS SIDE-BY-SIDE ---
    st.subheader("📅 Detailed Food Tourism Engagement Trends")
    
    # Create two columns for the trend charts
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        # Engagement Line Chart (Left)
        daily_counts = timeline_df.groupby('date_only').size().reset_index(name='Total Posts')
        fig_line = px.line(
            daily_counts, 
            x='date_only', 
            y='Total Posts', 
            markers=True, 
            title="Total Daily Post Volume", # Internal Plotly Title
            labels={
                'date_only': 'Timeline Date',
                'Total Posts': 'Volume of Submissions'
            }
        )
        
        # Format layout to match white card & strip grid/false lines
        fig_line.update_layout(
            height=360, 
            margin=dict(l=45, r=15, t=50, b=35), # Standardized bounding padding
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF'
        )
        fig_line.update_xaxes(showgrid=False, zeroline=False, showline=False)
        fig_line.update_yaxes(showgrid=False, zeroline=False, showline=False)
        
        st.plotly_chart(fig_line, use_container_width=True)
        st.info("**💡 Interpretation:** Spikes usually correlate with school holidays, local festivals, or viral social media posts.")
        
    with trend_col2:
        # ⚠️ REMOVED st.subheader() HERE TO PREVENT TOP CONTAINER BLOWOUT
        
        timeline_df['day_name'] = timeline_df['date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = timeline_df['day_name'].value_counts().reindex(day_order).fillna(0)
        
        # Engagement Peaks by Day (Right)
        fig_peaks = px.line(
            x=day_counts.index, y=day_counts.values,
            markers=True,
            title="Weekly Post Breakdown by Day", # Matches Plotly's title canvas strip on the left
            labels={
                'x': 'Day of the Week', 
                'y': 'Total Activity Volume'
            }
        )
        fig_peaks.update_traces(line_color='#636EFA', line_width=3)
        
        # Format layout to match white card & strip grid/false lines
        fig_peaks.update_layout(
            height=360, 
            margin=dict(l=45, r=15, t=50, b=35), # Mirror padding matching trend_col1 precisely
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF'
        )
        fig_peaks.update_xaxes(showgrid=False, zeroline=False, showline=False)
        fig_peaks.update_yaxes(showgrid=False, zeroline=False, showline=False)
        
        st.plotly_chart(fig_peaks, use_container_width=True)
        
        # YOUR ORIGINAL INTERPRETATION
        peak_day = day_counts.idxmax()
        st.info(f"💡 **Interpretation:** More visitors come on **{peak_day}s**.")
    st.divider()
    # --- 7. RAW DATA LOG ---
    st.subheader("📑 Detailed Sentiment Log")
    display_df = filtered_test_df.copy().sort_values(by='date', ascending=False)
    display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%d-%b-%Y')
    
    # --- RENAME COLUMNS TO SIMPLE TERMS ---
    display_df = display_df.rename(columns={
        'date': 'Date',
        'place_name': 'Place Name',
        'text': 'Review',
        'sentiment_label': 'Sentiment',
        'sentiment_score': 'Score'
    })
    
    # Pass the simple column names to the dataframe viewer
    st.dataframe(
        display_df[['Date', 'Place Name', 'Review', 'Sentiment', 'Score']], 
        use_container_width=True,
        height=300
    )
    st.caption("🔍 **Data Audit:** This log provides transparency, allowing officers to verify individual reviews that triggered the alerts above.")

##################################################
##################################################
# PAGE 2




elif st.session_state.active_page == "Cafes":
    st.header("🍴 Cafe & Restaurant Performance Analytics")
    
    st.caption("Strategic insight into district-wide food sentiment and service quality.")
    st.divider()
    # --- PRE-CALCULATIONS ---
    cafe_stats = filtered_test_df.groupby('place_name').agg(
        Total_Reviews=('text', 'count'),
        Avg_Sentiment=('sentiment_score', 'mean'),
        Pos_Reviews=('sentiment_label', lambda x: (x == 'positive').sum()),
        Neg_Reviews=('sentiment_label', lambda x: (x == 'negative').sum())
    ).reset_index()
    
    cafe_stats['Pos_Rate'] = (cafe_stats['Pos_Reviews'] / cafe_stats['Total_Reviews']) * 100
    cafe_stats['Neg_Rate'] = (cafe_stats['Neg_Reviews'] / cafe_stats['Total_Reviews']) * 100    


    # --- ROW 1: KPI CARDS WITH SMART TOOLTIPS ---
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
    # This CSS targets ONLY the first metric found in the app (kpi1)
        venue_count = filtered_test_df['place_name'].nunique()
        
        # Custom HTML to mimic st.metric but with massive font
        st.markdown(f"""
            <div style="display: flex; flex-direction: column;">
                <p style="font-size: 18px; margin-bottom: 0px;">🏪 Total Venues Tracked</p>
                <p style="font-size: 120px; font-weight: 800; margin: 0px; line-height: 1; color: #1f77b4;">
                    {venue_count}
                </p>
                <p style="font-size: 14px; color: gray; margin-top: -10px;">📍 Coverage: Kubang Pasu District</p>
            </div>
            """, unsafe_allow_html=True)
            
    #st.metric(
    #    label="🏪 Total Venues Tracked", 
    #    value=f"{filtered_test_df['place_name'].nunique()}",
    #    help="This is the count of unique business names found in the dataset after filtering."
    #    )
    #st.caption("📍 *Coverage: Kubang Pasu District*")

    with kpi2:
        st.write("**🏆 Top 3 (High Satisfaction)**")
        top_3 = cafe_stats.sort_values(by=['Pos_Rate', 'Total_Reviews'], ascending=False).head(3)
        
        for _, row in top_3.iterrows():
            # Clean green box for top performers
            st.success(f"{row['place_name']}: {row['Pos_Rate']:.1f}%", icon="✅")
            # Interactive explanation below the box
            #st.caption(f"💡 {int(row['Pos_Reviews'])} happy customers out of {int(row['Total_Reviews'])} reviews.")
    with kpi3:
        st.write("**🚩 Bottom 3 (Highest Negativity)**")
        
        # Sort by highest negative rate
        bottom_3 = cafe_stats.sort_values(by=['Neg_Rate', 'Total_Reviews'], ascending=False).head(3)
        
        for _, row in bottom_3.iterrows():
            neg_count = int(row['Neg_Reviews'])
            
            # Show the red alert box
            st.error(f"{row['place_name']}: {row['Neg_Rate']:.1f}% Negative", icon="🚨")
            
            # Caption showing the raw count of negative reviews
            #st.caption(f"🚨 Critical: {neg_count} users explicitly left negative feedback.")
            
    # --- NEW: INTERACTIVE INTERPRETATION EXPANDER ---
    with st.expander("❓ How these rankings decided?"):
        st.write("""
        **The Evaluation Process:**
        1. **Data Mining:** We look at all TikTok comments for each cafe.
        2. **Sentiment Analysis:** Labels each comment as Positive, Neutral, or Negative.
        3. **Math Logic:** We calculate the percentage:  
        $(Positive \ Reviews \div Total \ Reviews) \\times 100$
        
        **Why is a percentage high or low?**
        * **High %:** Found many 'praise' keywords (e.g., *sedap, mantap, worth it*).
        * **Low %:** Found 'complaint' keywords (e.g., *lambat, mahal, kureng*) or many neutral 'okay-only' ratings.
        """)

    # --- ROW 2: SENTIMENT GAUGES ---
    st.subheader("🌡️ Sentiment Intensity Analysis")
    st.info("💡 **Interpretation:** These gauges show how strong the emotions are. A high negative value means customers are very unhappy.")
    pos_avg = filtered_test_df[filtered_test_df['sentiment_label'] == 'positive']['sentiment_score'].mean() or 0
    neu_avg = filtered_test_df[filtered_test_df['sentiment_label'] == 'neutral']['sentiment_score'].mean() or 0
    neg_avg = filtered_test_df[filtered_test_df['sentiment_label'] == 'negative']['sentiment_score'].mean() or 0

    # 2. CONVERT TO PERCENTAGE (Multiply by 100)
    pos_pct = pos_avg * 100
    neu_pct = neu_avg * 100
    neg_pct = neg_avg * 100

    def create_gauge(value, title, color):
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            number = {'suffix': "%", 'font': {'size': 24}}, # Add % suffix
            title = {'text': title, 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 100]}, # Scale is now 0 to 100
                'bar': {'color': "#2c3e50"},
                'steps': [
                    {'range': [0, 100], 'color': color}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    # 3. Display with scaled values
    m1, m2, m3 = st.columns(3)
    m1.plotly_chart(create_gauge(pos_pct, "Positive Strength", "#00CC96"), use_container_width=True)
    m2.plotly_chart(create_gauge(neu_pct, "Neutral Strength", "#636EFA"), use_container_width=True)
    m3.plotly_chart(create_gauge(neg_pct, "Negative Strength", "#EF553B"), use_container_width=True)
    st.divider()

    # --- ROW 3: DEEP DIVE (LOCATION + REVIEWS) ---
    st.subheader("📍 Venue Spotlight & Geographic Mapping")

    # --- LOCATION DETECTION ---
    user_location = get_geolocation()
    u_lat, u_lon = (user_location['coords']['latitude'], user_location['coords']['longitude']) if user_location and user_location.get('coords') else (6.4550, 100.5050)
    if not user_location: 
        st.caption("ℹ️ Using default campus coordinates (GPS not detected).")

    selected_cafe = st.selectbox("Select Cafe for detailed insights:", ["All"] + list(cafe_stats['place_name'].unique()), key="global_cafe_selector")

    if selected_cafe != "All":
        specific_df = filtered_test_df[filtered_test_df['place_name'].str.strip() == selected_cafe.strip()].copy()
        
        if not specific_df.empty:
            # --- NEW SECTION: VIDEO AND MAP SIDE-BY-SIDE ONLY ---
            col_video, col_map = st.columns([1.0, 1.0])
            
            # ==================== COL 1: TIKTOK VIDEO EMBED & LINK TRIGGER ====================
            with col_video:
                st.write("**📱 TikTok Video Review**")
                
                # Mengesan lajur URL video yang wujud dalam dataset anda
                url_variants = ['videoweburl', 'webVideoUrl', 'video_url', 'url']
                found_url_col = next((c for c in url_variants if c in filtered_test_df.columns), None)
                
                if found_url_col:
                    # Ambil URL video yang pertama (paling atas/terbaru)
                    first_video_url = filtered_test_df[filtered_test_df['place_name'].str.strip() == selected_cafe.strip()][found_url_col].iloc[0]
                    
                    if pd.notna(first_video_url) and "video/" in str(first_video_url):
                        # Bulletproof parsing rules to extract the exact Video ID
                        video_id = str(first_video_url).split("/video/")[-1].split("?")[0]
                        
                        # Clean corporate embed iframe format matching your Enforcement Center style
                        st.components.v1.html(f"""
                            <iframe src="https://www.tiktok.com/embed/{video_id}" 
                                    style="width: 100%; height: 350px; border: none; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.06);" 
                                    allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
                                    allowfullscreen>
                            </iframe>
                        """, height=355)
                        
                        # Direct HTML Action Button styled natively to jump open source links
                        st.markdown(f"""
                            <div style="text-align: center; margin-top: 8px;">
                                <a href="{first_video_url}" target="_blank" style="text-decoration: none;">
                                    <button style="background-color: #FE2C55; color: white; border: none; padding: 10px 20px; font-weight: 700; border-radius: 8px; cursor: pointer; width: 100%; font-size: 1.0rem; box-shadow: 0 4px 6px rgba(254, 44, 85, 0.2);">
                                        Open in TikTok 
                                    </button>
                                </a>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ No playable link structure found in records.")
                else:
                    st.info("ℹ️ Video data system column missing.")

            # ==================== COL 2: GEOGRAPHIC NAVIGATION MAP ====================
            with col_map:
                st.write("**🗺️ Navigation & Location**")
                coords = CAFE_LOCATIONS.get(selected_cafe)
                if coords:
                    lat, lon = coords
                    dist = geodesic((u_lat, u_lon), (lat, lon)).km
                    st.success(f"📍 Distance: **{dist:.2f} km**")
                    m = folium.Map(location=[lat, lon], zoom_start=15)
                    folium.Marker([lat, lon], popup=selected_cafe, icon=folium.Icon(color="red", icon="cutlery", prefix="fa")).add_to(m)
                    folium.Marker([u_lat, u_lon], popup="You", icon=folium.Icon(color="blue", icon="user", prefix="fa")).add_to(m)
                    # Height scaled to 350px to perfectly visually align with the iframe video container
                    st_folium(m, height=350, use_container_width=True, key=f"map_{selected_cafe}")
                else:
                    st.warning("📍 Coordinates not found in CAFE_LOCATIONS.")

            # ==================== NEW SECTION: FULL-WIDTH REVIEWS TABLE BELOW ====================
            st.write("")
            st.write(f"**💬 Recent Reviews for {selected_cafe}**")
            specific_df['date'] = pd.to_datetime(specific_df['date'], errors='coerce').dt.strftime('%d-%b-%Y')
            
            # --- RENAME COLUMNS TO SIMPLE TERMS (SPECIFIC VIEW) ---
            specific_df = specific_df.rename(columns={
                'date': 'Date',
                'text': 'Review',
                'sentiment_label': 'Sentiment'
            })
            st.dataframe(specific_df[['Date', 'Review', 'Sentiment']], height=300, use_container_width=True)

            # --- AI INTERPRETATION BOX ---
            st.subheader("🤖 Performance Summary")
            # Backend matches renamed column keys safely by calling the raw dataframe target
            pos_score = (filtered_test_df[filtered_test_df['place_name'].str.strip() == selected_cafe.strip()]['sentiment_label'] == 'positive').mean() * 100
            if pos_score >= 80:
                st.success(f"**Great Job!** {selected_cafe} is a viral favorite. People love the vibe and food here. Keep it up!")
            elif pos_score >= 50:
                st.warning(f"**Doing OK.** {selected_cafe} has mixed reviews. Some people like it, some don't. Management should check for slow service.")
            else:
                st.error(f"**Alert!** Many customers are unhappy here. MPKP might need to check if the food quality or prices are making people complain on TikTok.")
        else:
            st.error("❌ No data found for this venue.")
    else:
        # --- ALL PLACES VIEW ---
        st.write("**🗺️ District-wide Sentiment Map**")
        m_all = folium.Map(location=[u_lat, u_lon], zoom_start=12)
        folium.Marker([u_lat, u_lon], popup="Me", icon=folium.Icon(color="blue", icon="user", prefix="fa")).add_to(m_all)
        for name, coords in CAFE_LOCATIONS.items():
            folium.Marker(location=coords, popup=name, icon=folium.Icon(color="red", icon="store", prefix="fa")).add_to(m_all)
        st_folium(m_all, height=400, use_container_width=True, key="overview_map")
        
        st.write("**💬 Review Database Preview**")
        all_display = filtered_test_df.copy()
        all_display['date'] = pd.to_datetime(all_display['date'], errors='coerce').dt.strftime('%d-%b-%Y')
        
        # --- RENAME COLUMNS TO SIMPLE TERMS (ALL VIEW) ---
        all_display = all_display.rename(columns={
            'date': 'Date',
            'place_name': 'Place Name',
            'text': 'Review',
            'sentiment_label': 'Sentiment'
        })
        st.dataframe(all_display[['Date', 'Place Name', 'Review', 'Sentiment']], height=300, use_container_width=True)
    st.divider()
    
    
##################################################
##################################################
# PAGE 3

elif st.session_state.active_page == "Clouds":
    from collections import Counter
    from nltk.util import ngrams
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    st.header("☁️ Linguistic Intelligence")
    st.write("Analyze common words and phrases across different venues.")
    st.divider()
    
    # 1. Dashboard Controls
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        search_query = st.text_input("🔍 Search specific keywords (e.g., 'pedas', 'servis'):", "")
    with ctrl_col2:
        # Limit adjusted to 100, help removed
        word_limit = st.slider("Word Density", 1, 100, 30)
    
    # Caption added below the controls
    st.caption("ℹ️ *Adjust the slider to control the number of words displayed in the clouds. Higher density reveals more detailed feedback but may appear cluttered.*")
    
    # Filter the data based on search
    if search_query:
        cloud_df = filtered_test_df[filtered_test_df['text_cleaned'].str.contains(search_query, case=False, na=False)]
    else:
        cloud_df = filtered_test_df

    # 2. Extract unique venues
    venues_to_show = sorted([str(x) for x in cloud_df['place_name'].unique() if pd.notna(x)])

    if not venues_to_show:
        st.warning("No data matches your search query.")
    else:
        for venue in venues_to_show:
            with st.expander(f"🏢 {venue} - Unigram & Bigram Analysis", expanded=False):
                v_df = cloud_df[cloud_df['place_name'] == venue]
                
                # Combine text and handle empty strings
                combined_text = " ".join(v_df['text_cleaned'].astype(str)).strip()
                v_tokens = combined_text.split()
                
                if len(v_tokens) >= 2:
                    col_uni, col_bi = st.columns(2)
                    
                    # --- UNIGRAMS (Single Words) ---
                    with col_uni:
                        st.markdown("#### ✨ Top Keywords")
                        uni_counts = Counter(v_tokens)
                        
                        # Generate WordCloud
                        wc_uni = WordCloud(
                            background_color="white", 
                            width=600, height=400,
                            max_words=word_limit
                        ).generate_from_frequencies(uni_counts)
                        
                        fig_uni, ax_uni = plt.subplots(figsize=(5, 3))
                        ax_uni.imshow(wc_uni, interpolation='bilinear')
                        ax_uni.axis("off")
                        st.pyplot(fig_uni)
                        plt.close(fig_uni) 
                        
                        st.dataframe(pd.DataFrame(uni_counts.most_common(5), columns=['Word', 'Count']), use_container_width=True)

                    # --- BIGRAMS (Word Pairs) ---
                    with col_bi:
                        st.markdown("#### 🗣️ Popular Phrases")
                        bigram_list = [" ".join(gram) for gram in ngrams(v_tokens, 2)]
                        v_bigrams = Counter(bigram_list)
                        
                        if v_bigrams:
                            wc_bi = WordCloud(
                                background_color="white", 
                                width=600, height=400,
                                max_words=word_limit
                            ).generate_from_frequencies(v_bigrams)
                            
                            fig_bi, ax_bi = plt.subplots(figsize=(5, 3))
                            ax_bi.imshow(wc_bi, interpolation='bilinear')
                            ax_bi.axis("off")
                            st.pyplot(fig_bi)
                            plt.close(fig_bi) 
                            
                            st.dataframe(pd.DataFrame(v_bigrams.most_common(5), columns=['Phrase', 'Count']), use_container_width=True)
                        else:
                            st.info("Not enough unique pairs for a Bigram cloud.")
                else:
                    st.write("💡 Not enough text data for this venue.")

##################################################
##################################################
# PAGE 4

elif st.session_state.active_page == "Influencers":
    st.header("👤 Influencer Engagement Analysis")
    st.markdown("---")

    if not df_full_for_metrics.empty:
        # 1. Create a working copy
        inf_df = df_full_for_metrics.copy()

        # 2. Extract Handle
        url_variants = ['videoweburl', 'webVideoUrl', 'video_url', 'url']
        found_url_col = next((c for c in url_variants if c in inf_df.columns), None)
        
        if found_url_col:
            inf_df['author_handle'] = inf_df[found_url_col].apply(
                lambda x: str(x).split('@')[1].split('/')[0] if pd.notna(x) and '@' in str(x) else "Unknown"
            )
        else:
            inf_df['author_handle'] = inf_df['uniqueid'] if 'uniqueid' in inf_df.columns else "Unknown"

        # 3. Standardize Metrics (Rename lowercase to CamelCase)
        mapping = {
            'diggcount': 'diggCount',
            'replycommenttotal': 'replyCommentTotal',
            'playcount': 'playCount',
            'sharecount': 'shareCount'
        }
        
        for old_col, new_col in mapping.items():
            if old_col in inf_df.columns:
                inf_df.rename(columns={old_col: new_col}, inplace=True)

        # Ensure columns exist as numbers to prevent crashes
        for col in ['playCount', 'diggCount', 'shareCount', 'replyCommentTotal']:
            if col in inf_df.columns:
                inf_df[col] = pd.to_numeric(inf_df[col], errors='coerce').fillna(0)
            else:
                inf_df[col] = 0

        # 4. Aggregation (FIXED: Added replyCommentTotal here)
        text_col = 'text_cleaned' if 'text_cleaned' in inf_df.columns else 'text'
        
        influencer_stats = inf_df.groupby('author_handle').agg({
            'playCount': 'sum',
            'diggCount': 'sum',
            'shareCount': 'sum',
            'replyCommentTotal': 'sum',  # THIS LINE FIXES THE KEYERROR
            text_col: 'count'
        }).rename(columns={text_col: 'Total Comments'})

        # 5. Score & Filter
        influencer_stats['Influence Score'] = (
            (influencer_stats['playCount'] * 1.0) + 
            (influencer_stats['diggCount'] * 1.0) + 
            (influencer_stats['shareCount'] * 2.0) +
            (influencer_stats['Total Comments'] * 50.0)  # High weight on post frequency
        )
        
        influencer_stats = influencer_stats[~influencer_stats.index.str.contains('unknown|nan|none|sksksk', case=False, na=False)]
        influencer_stats = influencer_stats.sort_values(by='Influence Score', ascending=False).reset_index()
        # --- 6. DISPLAY TOP INFLUENCER ---
        if not influencer_stats.empty:
            top_inf = influencer_stats.iloc[0]
            
            st.subheader("🥇 District Top Influencer")
            inf_c1, inf_c2, inf_c3 = st.columns(3)
            
            # Ensure the @ symbol is present
            handle_display = f"@{top_inf['author_handle']}" if not str(top_inf['author_handle']).startswith('@') else top_inf['author_handle']
            
            inf_c1.metric("Handle", handle_display)
            # Formatting to 1 decimal place for the weighted score
            inf_c2.metric("Influence Score", f"{top_inf['Influence Score']:,.1f}")
            inf_c3.metric("Total Posts", f"{int(top_inf['Total Comments']):,}")
            
            
        # FIXED HELP SECTION
        formula_display = (
            f"**Formula Applied:** Influence Score ({int(top_inf['Influence Score']):,}) = "
            f"Plays ({int(top_inf['playCount']):,}) + Likes ({int(top_inf['diggCount']):,}) + "
            f"Comments ({int(top_inf['Total Comments']):,}) + Shares ({int(top_inf['shareCount']):,})"
        )
        st.markdown(formula_display)
        
        st.info(f"💡 **Interpretation:** @ {top_inf['author_handle']} gets the most engagement. This account helps attract attention and can be useful for tourism promotion.")
        st.divider()

        # --- 6. METRIC SPECIFIC LEADERS ---
        m1, m2, m3 = st.columns(3)
        
        most_likes = influencer_stats.sort_values(by='diggCount', ascending=False).iloc[0]
        most_comments = influencer_stats.sort_values(by='Total Comments', ascending=False).iloc[0]
        most_replies = influencer_stats.sort_values(by='replyCommentTotal', ascending=False).iloc[0]
                
        with m1:
            row = influencer_stats.sort_values(by='diggCount', ascending=False).iloc[0]            
            st.info(f"**Most Comment Likes**\n\n**@{most_likes['author_handle']}**\n\n({int(most_likes['diggCount']):,} likes)")
            st.link_button(f"Visit @{most_likes['author_handle']}", f"https://www.tiktok.com/@{most_likes['author_handle']}")
            
        with m2:
            row = influencer_stats.sort_values(by='Total Comments', ascending=False).iloc[0]            
            st.success(f"**Most Comments Received**\n\n**@{most_comments['author_handle']}**\n\n({int(most_comments['Total Comments']):,} comments)")
            st.link_button(f"Visit @{most_comments['author_handle']}", f"https://www.tiktok.com/@{most_comments['author_handle']}")
            
        with m3:
            row = influencer_stats.sort_values(by='replyCommentTotal', ascending=False).iloc[0]            
            st.warning(f"**Highest Response Rate**\n\n**@{most_replies['author_handle']}**\n\n({int(most_replies['replyCommentTotal']):,} replies)")
            st.link_button(f"Visit @{most_replies['author_handle']}", f"https://www.tiktok.com/@{most_replies['author_handle']}")

        st.info("**💡 Interpretation:** These metrics highlight specialized impact. Profiles linked above represent the key community voices for Kubang Pasu food tourism.")
        st.divider()

        # --- 7. DETAILED REPORT & SEARCH ---
        st.subheader("🔍 Influencer Detailed Report")
        all_handles = influencer_stats['author_handle'].tolist()
        selected_handle = st.selectbox("Select or Search Influencer Handle:", all_handles)

        if selected_handle:
            report_data = influencer_stats[influencer_stats['author_handle'] == selected_handle].iloc[0]
            rep_c1, rep_c2 = st.columns([1, 2])
            
            with rep_c1:
                st.info(f"**Influencer Profile**\n\n**@{selected_handle}**")
                # Using a standard markdown link as an alternative
                st.markdown(f"[🔗 Open TikTok Profile](https://www.tiktok.com/@{selected_handle})")
            
            with rep_c2:
                st.write(f"### Performance Metrics for @{selected_handle}")
                rm1, rm2, rm3 = st.columns(3)
                rm1.metric("Influence Score", f"{int(report_data['Influence Score']):,}")
                rm2.metric("Comments Received", f"{int(report_data['Total Comments']):,}")
                rm3.metric("Comment Likes", f"{int(report_data['diggCount']):,}")

            st.info(f"**💡 Report Interpretation:** The data suggests that **@{selected_handle}** has a strong reach in the community. "
                    f"Their content generates a sentiment trend that MPKP can monitor to understand how this specific influencer "
                    f"shapes the public's perception of Kubang Pasu.")
        
        # --- 8. TOP 10 RANKING CHART ---
        st.subheader("📊 Top 10 Influencers Ranking")
        top_10 = influencer_stats.head(10)

        fig_inf = px.bar(
            top_10, 
            x='author_handle', 
            y='Influence Score',
            color='Total Comments',
            text='Influence Score',
            title="Top 10 Influencers by Influence Score",
            labels={'author_handle': 'Influencer Handle', 'Influence Score': 'Influence Score'},
            color_continuous_scale=['#4C9AFF', '#0052CC'],
        )
        fig_inf.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig_inf.update_xaxes(showgrid=False, zeroline=False, showline=False)
        fig_inf.update_yaxes(showgrid=False, zeroline=False, showline=False)
        st.plotly_chart(fig_inf, use_container_width=True)

        st.info("💡 **Interpretation:** This chart shows the top 10 most active voices in Kubang Pasu. Darker colors mean higher engagement. It helps MPKP see who is consistently influential and who only appears occasionally.")

        # --- 9. LEADERBOARD TABLE ---
        st.subheader("📑 Influencer Leaderboard")
        # --- RENAME COLUMNS TO SIMPLE TERMS ---
        top_10_display = top_10.rename(columns={
            'author_handle': 'Username',
            'Total Comments': 'Comments',
            'diggCount': 'Likes',
            'replyCommentTotal': 'Replies',
            'Influence Score': 'Score'
        })
        
        st.dataframe(
            top_10_display[['Username', 'Comments', 'Likes', 'Replies', 'Score']],
            use_container_width=True,
            hide_index=True
        )
        st.info(f"""
        💡 **Insight:** This shows who drives the most attention in food tourism.  
        @{top_inf['author_handle']} is currently the top influencer.

        **For MPKP:**
        - High engagement: This account attracts the most interest in Kubang Pasu cafes.  
        - Promotion opportunity: Collaborating with this influencer can help boost awareness for events or food campaigns.
        """)
        #else:
        #    st.warning("⚠️ No valid influencers found after filtering 'Unknown' accounts.")
    else:
            st.warning("⚠️ No data found. Please check your filters.")

##################################################
##################################################
# PAGE 5

elif st.session_state.active_page == "Strategy":
    st.header("🎯 AI-Powered Promotional & Strategic Recommendations")
    st.markdown("""
     This dashboard turns customer feedback into useful insights for decision-making. 
                It analyses language patterns to help MPKP (Majlis Perbandaran Kubang Pasu) make better, data-driven decisions and improve the local food sector.
    """)
    st.divider()

    
    
      
        # 1. Define critical keywords for hygiene and quality validation
    danger_keywords = ['basi', 'kotor', 'lipas', 'rambut', 'lalat', 'mahal', 'yahudi', 'kotoq', 'kotor','busuk']

    # 2. Identify venues with multiple high-intensity negative reviews
    # We group by venue and calculate the average score and count
    negative_summary = filtered_test_df[filtered_test_df['sentiment_label'] == 'negative'].groupby('place_name').agg(
        neg_count=('text', 'count'),
        avg_severity=('sentiment_score', 'mean'),
        all_text=('text_cleaned', lambda x: " ".join(x.astype(str)).lower())
    ).reset_index()

    # 3. Filter for REAL crises: At least 3 separate negative comments AND > 80% average severity
    critical_incidents = negative_summary[(negative_summary['neg_count'] >= 3) & (negative_summary['avg_severity'] > 0.80)]
    
    st.info("#### 🚨 Alert: Critical Quality & Hygiene Intervention")

    if not critical_incidents.empty:

        # Select the venue with highest severity
        top_crisis = critical_incidents.sort_values(by='avg_severity', ascending=False).iloc[0]
        problem_venue = top_crisis['place_name']
        severity_val = top_crisis['avg_severity'] * 100

        # --- KEYWORD COUNT ---
        import re
        all_text_content = top_crisis['all_text']
        keyword_counts = {}

        for word in set(danger_keywords):
            matches = re.findall(rf'\b{word}\b', all_text_content)
            if len(matches) > 0:
                keyword_counts[word] = len(matches)

        # Format evidence (optional use)
        if keyword_counts:
            evidence_str = ", ".join([f"`{k}` (x{v})" for k, v in keyword_counts.items()])
        else:
            evidence_str = "General Dissatisfaction"

        # --- LAYOUT ---
        v_col1, v_col2 = st.columns([1.5, 2])

        with v_col1:
            fig_critical = go.Figure(go.Indicator(
                mode="gauge+number",
                value=severity_val,
                title={'text': "Trend Severity (%)", 'font': {'color': 'red', 'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 75], 'color': "#ffcccc"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': 90}
                }
            ))

            fig_critical.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=40, b=50)
            )

            st.plotly_chart(fig_critical, use_container_width=True)

        with v_col2:
            st.markdown('<div style="margin-top: 25px;">', unsafe_allow_html=True)

            st.error(f"**🔴 URGENT ALERT: {problem_venue}**")

            st.markdown(f"""
            - **Complaint Group:** Found **{top_crisis['neg_count']}** separate negative reviews
            """)

            # --- FILTER REVIEWS ---
            venue_reviews = filtered_test_df[
                (filtered_test_df['place_name'] == problem_venue) &
                (filtered_test_df['sentiment_label'] == 'negative')
            ].head(3).copy()

            # --- INTERPRET FUNCTION ---
            def interpret_review(text):
                text = str(text).lower()

                if any(word in text for word in ["harga", "mahal", "expensive", "yahudi", "mahai"]):
                    return "Pricing Issue"

                elif any(word in text for word in ["lambat", "slow", "wait", "service", "ramai", "lama"]):
                    return "Service Issue"

                elif any(word in text for word in ["kotor", "dirty", "lipas", "lalat", "rambut", "busuk"]):
                    return "Hygiene Issue"

                elif any(word in text for word in ["tak sedap", "bad taste", "tawar", "masin"]):
                    return "Food Quality Issue"

                elif any(word in text for word in ["tak datang", "x mau", "never come", "tak repeat"]):
                    return "Non-returning Customers"

                else:
                    return "General Complaint"

            # --- APPLY ---
            venue_reviews.loc[:, 'interpretation'] = venue_reviews['text'].apply(interpret_review)

            # --- SHOW REVIEWS ---
            st.markdown("### 💬 Flagged Customer Reviews")
            for idx, row in venue_reviews.iterrows():
                st.caption(f"Reviewer #{idx+1} (Severity: {round(row['sentiment_score']*100, 1)}%)")
                st.warning(f"\"{row['text']}\"")

            # --- TABLE (ONLY ONCE) ---
            st.markdown("### 🧠 Interpretation")
            st.dataframe(
                venue_reviews[['text', 'interpretation']],
                use_container_width=True,
                height=150
            )

            # --- SUMMARY (ONLY ONCE) ---
            st.markdown("#### 📊 Key Complaint Themes")
            issue_summary = venue_reviews['interpretation'].value_counts()

            for issue, count in issue_summary.items():
                st.write(f"- {issue}: {count} review(s)")

            # --- ACTION ---
            st.markdown("""
            **What MPKP should do:**
            1. Spot Check: Send a health officer immediately  
            2. Official Warning: Issue formal notice to improve standards  
            """)

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.success("✅ Sector Health Check: No concentrated negative trends detected")

    st.divider()
    ########################

    # --- SECTION 2: ISSUE MONITORING (KOTOR & MAHAL) ---
    st.subheader("🔍 Automated Issue Category Detection")
    st.write("This module specifically tracks hygiene and pricing perception across the Kubang Pasu district.")
    
    issue_df = filtered_test_df.copy()
    issue_df['text_cleaned'] = issue_df['text_cleaned'].astype(str).str.lower()
    
    # Specific detection for 'kotor' and 'mahal' keywords
    issue_df['is_kotor'] = issue_df['text_cleaned'].apply(lambda x: 1 if any(w in x for w in ['kotor', 'tak bersih', 'dirty', 'unhygienic', 'lalat', 'tikus']) else 0)
    issue_df['is_mahal'] = issue_df['text_cleaned'].apply(lambda x: 1 if any(w in x for w in ['mahal', 'harga', 'expensive', 'cekik darah', 'pricey']) else 0)

    kotor_stats = issue_df.groupby('place_name')['is_kotor'].sum().reset_index()
    mahal_stats = issue_df.groupby('place_name')['is_mahal'].sum().reset_index()

    col_k, col_m = st.columns(2)

    with col_k:
        # Mengekalkan pembolehubah asal anda
        top_kotor_place = kotor_stats.loc[kotor_stats['is_kotor'].idxmax()]
        st.error(f" **Hygiene Analysis: 'kotor'**")
        st.write(f"The venue **{top_kotor_place['place_name']}** has the most complaints about cleanliness and hygiene.")        
        st.metric(label="Total 'kotor' Mentions", value=f"{int(top_kotor_place['is_kotor'])}", delta="Review Density")

        # --- TAMBAHAN: SURAT AMARAN KEBERSIHAN ---
        with st.popover("📄 Issue Official Notice"):
            st.markdown(f"###  MPKP Enforcement: {top_kotor_place['place_name']}")
            ref_k = st.text_input("Fail Rujukan:", f"MPKP/JKPS/2026/AMR/{top_kotor_place['place_name'][:3].upper()}")
            
            draft_k = (f"Notis Amaran: Pihak MPKP mengesan kegagalan pematuhan tahap kebersihan "
                    f"di {top_kotor_place['place_name']} berdasarkan {int(top_kotor_place['is_kotor'])} aduan digital.")

            st.markdown(f"""
            <div style="background-color: #FFFFFF; padding: 15px; border: 1px solid #ff4b4b; border-radius: 8px; color: #000000; font-family: 'Times New Roman', serif;">
                <div style="text-align: center; font-weight: bold; border-bottom: 2px solid #000000; color: #000000;">MAJLIS PERBANDARAN KUBANG PASU</div>
                <div style="text-align: right; font-size: 0.8rem; margin-top: 5px; color: #000000;">Ruj: {ref_k}</div>
                <p style="color: #000000;"><b>Kepada: Pengurusan {top_kotor_place['place_name']}</b></p>
                <p style="text-align: justify; font-size: 0.9rem; color: #000000;">{draft_k} Sila ambil tindakan pembersihan segera dalam tempoh 14 hari bagi mengelakkan kompaun.</p>
            </div>
            """, unsafe_allow_html=True)
            st.download_button("📥 Download Notis", data=draft_k, file_name=f"Notis_Kotor_{top_kotor_place['place_name']}.txt")

        fig_k = px.bar(
            kotor_stats[kotor_stats['is_kotor'] > 0], 
            x='place_name', 
            y='is_kotor', 
            color_discrete_sequence=['#C53030'], # Deep corporate crimson
            labels={
                'place_name': 'Venue Name',
                'is_kotor': 'Hygiene Complaints Count'
            }
        )
        
        # Match professional background transparency, margins, and typography
        fig_k.update_layout(
            height=200, 
            margin=dict(l=40, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=10, color="#475569")
        )
        
        fig_k.update_yaxes(showgrid=False, zeroline=False, showline=False)
        fig_k.update_xaxes(showgrid=False, zeroline=False, showline=False)
        
        st.plotly_chart(fig_k, use_container_width=True)

    with col_m:
        # Mengekalkan pembolehubah asal anda
        top_mahal_place = mahal_stats.loc[mahal_stats['is_mahal'].idxmax()]
        st.warning(f" **Pricing Analysis: 'mahal'**")
        st.write(f"The venue **{top_mahal_place['place_name']}** is often mentioned as being expensive by visitors.")        
        st.metric(label="Total 'mahal' Mentions", value=f"{int(top_mahal_place['is_mahal'])}", delta="Review Density")

        # --- TAMBAHAN: SURAT SIASATAN HARGA ---
        with st.popover("📄 Issue Pricing Inquiry"):
            st.markdown(f"### MPKP Enforcement: {top_mahal_place['place_name']}")
            ref_m = st.text_input("Fail Rujukan:", f"MPKP/JKPS/2026/NTS/{top_mahal_place['place_name'][:3].upper()}")
            
            draft_m = (f"Notis Siasatan: Pihak MPKP menerima {int(top_mahal_place['is_mahal'])} aduan mengenai harga tidak munasabah "
                    f"di {top_mahal_place['place_name']}. Sila kemukakan senarai harga menu rasmi.")

            st.markdown(f"""
            <div style="background-color: #FFFFFF; padding: 15px; border: 1px solid #ffa500; border-radius: 8px; color: #000000; font-family: 'Times New Roman', serif;">
                <div style="text-align: center; font-weight: bold; border-bottom: 2px solid #000000; color: #000000;">MAJLIS PERBANDARAN KUBANG PASU</div>
                <div style="text-align: right; font-size: 0.8rem; margin-top: 5px; color: #000000;">Ruj: {ref_m}</div>
                <p style="color: #000000;"><b>Kepada: Pengurusan {top_mahal_place['place_name']}</b></p>
                <p style="text-align: justify; font-size: 0.9rem; color: #000000;">{draft_m} Kegagalan memberi kerjasama boleh menyebabkan tindakan pemeriksaan mengejut dijalankan.</p>
            </div>
            """, unsafe_allow_html=True)
            st.download_button("📥 Download Notis", data=draft_m, file_name=f"Notis_Harga_{top_mahal_place['place_name']}.txt")

        fig_m = px.bar(
            mahal_stats[mahal_stats['is_mahal'] > 0], 
            x='place_name', 
            y='is_mahal', 
            color_discrete_sequence=['#E67E22'], # Deep corporate amber/orange
            labels={
                'place_name': 'Venue Name',
                'is_mahal': 'Pricing Complaints Count'
            }
        )
        
        # Match professional background transparency, margins, and typography
        fig_m.update_layout(
            height=200, 
            margin=dict(l=40, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=10, color="#475569")
        )
        
        # ❌ REMOVED ALL LINES (Gridlines, background lines, and zero baselines)
        fig_m.update_yaxes(showgrid=False, zeroline=False, showline=False)
        fig_m.update_xaxes(showgrid=False, zeroline=False, showline=False)
        
        st.plotly_chart(fig_m, use_container_width=True)

    st.divider()

    # --- SECTION 3: PROMOTIONAL STRATEGY ---
    st.markdown("""
    <div style="background: linear-gradient(90deg, #EDF7ED 0%, #FFFFFF 100%); padding: 16px 20px; border-radius: 6px; border-left: 5px solid #1E4620; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.25rem;">📊</span>
            <span style="color: #1E4620; font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 1.15rem; font-weight: 700; letter-spacing: 0.3px; text-transform: uppercase;">
                Strategic Initiative 1: Targeted Operations & Vendor Training
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    service_keywords = ['lambat', 'tunggu', 'lama', 'servis', 'service', 'slow']
    price_keywords = ['mahal', 'harga', 'price', 'cekik darah', 'mahai']
    full_text_corpus = " ".join(filtered_test_df['text_cleaned'].astype(str)).lower()
    service_count = sum(full_text_corpus.count(word) for word in service_keywords)
    price_count = sum(full_text_corpus.count(word) for word in price_keywords)
    venue_scores = filtered_test_df.groupby('place_name')['sentiment_score'].mean()

    rec_col1, rec_col2 = st.columns([1.5, 1]) 
    with rec_col1:
        st.write("**Marketing Insights:**")

        if service_count > price_count and service_count > 0:
            st.write("""
            * **Main Issue:** Slow service is the biggest complaint from visitors.  
            * **What to do:** Train vendors to improve service speed and efficiency.  
            * **Why it matters:** Faster service improves customer satisfaction and table turnover.
            """)

        elif price_count > 0:
            st.write("""
            * **Main Issue:** Some visitors feel prices are too high.  
            * **What to do:** Improve price transparency and promote value-for-money options.  
            * **Why it matters:** Builds trust and attracts more tourists.
            """)

        else:
            st.write("""
            * **Insight:** No major issues detected.  
            * **Action:** Continue promoting food tourism as usual.
            """)

    with rec_col2:
        pattern_data = pd.DataFrame({"Issue": ["Service", "Price"], "Count": [service_count, price_count]})
    
    # 1. Apply professional English axis labels and a refined corporate slate palette
        fig_pattern = px.bar(
            pattern_data, 
            x="Count", 
            y="Issue", 
            orientation='h', 
            height=150, 
            color="Issue",
            color_discrete_map={"Service": "#1E3A8A", "Price": "#3B82F6"}, # Professional Deep and Slate Blues
            labels={
                "Count": "Total Feedback Volume",
                "Issue": "Core Operational Category"
            }
        )
        
        # 2. Match background transparency, margins, and typography
        fig_pattern.update_layout(
            showlegend=False, 
            margin=dict(l=50, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=10, color="#475569")
        )
        
        fig_pattern.update_xaxes(showgrid=False, zeroline=False, showline=False)
        fig_pattern.update_yaxes(showgrid=False, zeroline=False, showline=False)
        
        st.plotly_chart(fig_pattern, use_container_width=True)

            # --- SECTION 4: VENUE PARTNERSHIPS ---
    st.markdown("""
        <div style="background: linear-gradient(90deg, #FFF9E6 0%, #FFFFFF 100%); padding: 16px 20px; border-radius: 6px; border-left: 5px solid #B7791F; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.25rem;">🤝</span>
                <span style="color: #744210; font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 1.15rem; font-weight: 700; letter-spacing: 0.3px; text-transform: uppercase;">
                    Strategic Initiative 2: Commercial Synergy & Venue Selection
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    if not venue_scores.empty:
            top_ai_venue = venue_scores.idxmax()
            top_score = venue_scores.max()
                
            amb_col1, amb_col2, amb_col3 = st.columns([0.8, 1.2, 1.5])
            with amb_col1:
                st.metric(label="⭐ Best Brand Strength", value=top_ai_venue, delta=f"{top_score:.2f} Avg Score")
            with amb_col2:
                st.write("**Ambassador Strategy:**")
                st.write(f"""
                Based on the analysis, **{top_ai_venue}** is a highly trusted venue with very positive feedback.
                It is recommended as a main location for the official 'Taste of Kedah' video series.
                """)
            with amb_col3:
                top_3_venues = venue_scores.nlargest(3).reset_index()
                
                # Professional corporate English labels and deep green palette
                fig_top3 = px.bar(
                    top_3_venues, 
                    x='place_name', 
                    y='sentiment_score', 
                    color='sentiment_score', 
                    color_continuous_scale=['#4C9AFF', '#0052CC'], # Gradient green
                    labels={
                        'place_name': 'Venue Name',
                        'sentiment_score': 'Public Sentiment Index'
                    }
                )
                
                fig_top3.update_layout(
                    height=170, 
                    margin=dict(l=40, r=10, t=10, b=10), 
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif", size=10, color="#475569")
                )
                
                fig_top3.update_coloraxes(showscale=False) 
                
                # ❌ REMOVED ALL LINES (Gridlines, background lines, and zero baselines)
                fig_top3.update_yaxes(showgrid=False, zeroline=False, showline=False)
                fig_top3.update_xaxes(showgrid=False, zeroline=False, showline=False)
                
                st.plotly_chart(fig_top3, use_container_width=True)

    if not filtered_test_df.empty:
        # 1. Create temporary copy for calculation
        inf_sync_df = filtered_test_df.copy()

        # 2. Extract Handle (ADDED 'videoweburl' to the list)
        url_col = next((c for c in ['videoweburl', 'webVideoUrl', 'videoWebUrl'] if c in inf_sync_df.columns), None)
        
        if url_col:
            inf_sync_df['author_handle'] = inf_sync_df[url_col].apply(
                lambda x: str(x).split('@')[1].split('/')[0] if pd.notna(x) and '@' in str(x) else "Unknown"
            )
        else:
            # SAFETY FALLBACK: If no URL column is found, use uniqueid or just mark as Unknown
            if 'uniqueid' in inf_sync_df.columns:
                inf_sync_df['author_handle'] = inf_sync_df['uniqueid'].astype(str)
            else:
                inf_sync_df['author_handle'] = "Unknown"
        
        # 3. Standardize Metrics (Handling lowercase 'diggcount', etc.)
        # This maps your CSV's lowercase names to the names used in your math
        metric_map = {
            'playcount': 'playCount', 
            'diggcount': 'diggCount', 
            'sharecount': 'shareCount', 
            'commentcount': 'commentCount'
        }
        
        for old_name, new_name in metric_map.items():
            if old_name in inf_sync_df.columns:
                inf_sync_df[new_name] = pd.to_numeric(inf_sync_df[old_name], errors='coerce').fillna(0)
            elif new_name not in inf_sync_df.columns:
                inf_sync_df[new_name] = 0

        # 4. Group & Filter "Unknown"
        inf_stats = inf_sync_df.groupby('author_handle').agg({
            'commentCount': 'sum',
            'diggCount': 'sum',
            'playCount': 'sum',
            'shareCount': 'sum'
        }).rename(columns={'commentCount': 'Total Comments'})

        # Remove the noise
        inf_stats = inf_stats[~inf_stats.index.str.contains('unknown|nan|none', case=False, na=False)]

        # 5. Calculate Score (Weighting 'Fame' over 'Frequency')
        if not inf_stats.empty:
            inf_stats['Influence Score'] = (
                (inf_stats['playCount'] * 1.5) + 
                (inf_stats['diggCount'] * 2.0) + 
                (inf_stats['shareCount'] * 5.0) +
                (inf_stats['Total Comments'] * 0.1) # low weight for comment frequency
            )
            
            inf_stats = inf_stats.sort_values(by='Influence Score', ascending=False)

            # 6. Define the Global Variables for the UI
            global_top_handle = f"@{inf_stats.index[0]}" if not str(inf_stats.index[0]).startswith('@') else inf_stats.index[0]
            inf_plot_df = inf_stats.reset_index().rename(columns={'index': 'author_handle'})
        else:
            global_top_handle = "N/A"
            
        st.markdown("""
    <div style="background: linear-gradient(90deg, #EBF8FF 0%, #FFFFFF 100%); padding: 16px 20px; border-radius: 6px; border-left: 5px solid #2B6CB0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.25rem;">📈</span>
            <span style="color: #2B6CB0; font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 1.15rem; font-weight: 700; letter-spacing: 0.3px; text-transform: uppercase;">
                Strategic Initiative 3: Digital Voice Integration & Advocacy
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

        # Assuming global_top_handle was calculated in the logic above
        if 'global_top_handle' in locals() and global_top_handle != "N/A":
            inf_col1, inf_col2, inf_col3 = st.columns([0.8, 1.2, 1.5])
                
            with inf_col1:
                # Using the same metric style as your venue recommendation
                st.metric(
                    label="🏆 Top Digital Voice", 
                    value=global_top_handle, 
                    delta="High Engagement",
                    help="This handle has the highest combined score of plays, likes, and comments."
                )
                    
            with inf_col2:
                st.write("**Creator Strategy:**")
                st.write(f"""
                Data indicates **{global_top_handle}** acts as the primary catalyst for local food trends. 
                **Strategic Action:** Contract this creator to produce a 'Hidden Gems' miniseries featuring the 
                top-rated venues identified in Recommendation 2.
                """)
                    
            with inf_col3:
                # Display the top 3 influencers by score to show alternative partners
                if 'inf_stats' in locals():
                    top_3_inf = inf_stats.sort_values(by='Influence Score', ascending=False).head(3).reset_index()
                    
                    # Professional corporate English labels and deep purple palette
                    fig_inf_top = px.bar(
                        top_3_inf, 
                        x='author_handle', 
                        y='Influence Score', 
                        color='Influence Score', 
                        color_continuous_scale=['#4C9AFF', '#0052CC'],
                        labels={
                            'author_handle': 'Creator Handle (TikTok)',
                            'Influence Score': 'Digital Impact Score'
                        }
                    )
                    
                    fig_inf_top.update_layout(
                        height=170, 
                        margin=dict(l=40, r=10, t=10, b=10), 
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif", size=10, color="#475569")
                    )
                    
                    fig_inf_top.update_coloraxes(showscale=False) 
                    
                    fig_inf_top.update_yaxes(showgrid=False, zeroline=False, showline=False)
                    fig_inf_top.update_xaxes(showgrid=False, zeroline=False, showline=False)
                    
                    st.plotly_chart(fig_inf_top, use_container_width=True)
        else:
            st.warning("Influencer matrix analytics data is currently unavailable for recommendation.")
                
    # --- SECTION 5: MPKP ACTION & GOVERNANCE CENTER (NEW) ---
    st.divider()
    st.subheader("⚖️ MPKP Official Action & Governance Center")
    st.write("Translate insights into official municipal action. Generate formal documentation for enforcement or rewards.")

    # Create two clear action paths based on AI analytics
    st.markdown('<p style="color: black; font-weight: bold; margin-bottom: -10px;">Select Governance Action Path:</p>', unsafe_allow_html=True)
    
    with st.container():
        # Global CSS injection to target Streamlit's internal markdown/label engine
        st.markdown("""
            <style>
                /* Targets the text label of the radio component */
                div[data-testid="stRadio"] label p {
                    color: #000000 !important;
                    font-weight: 500 !important;
                }
                /* Targets the main title label of the radio group if present */
                div[data-testid="stRadio"] > label {
                    color: #000000 !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
        action_type = st.radio(
            label="Select Governance Action Path:", 
            options=[ "Issue Appreciation Letter (Top Performance)", "Invite Influencer (Collaboration)"], 
            horizontal=True
        )

    if action_type == "Issue Appreciation Letter (Top Performance)":
        if not venue_scores.empty:
            top_venues_list = venue_scores.nlargest(5).index.tolist()
            target_top_venue = st.selectbox("🎯 Select Top Performing Venue:", top_venues_list)
            v_score = venue_scores[target_top_venue]
            v_percentage = v_score * 100
            top_col1, top_col2 = st.columns([1.2, 1.8])

            with top_col1:
                st.markdown("### 🏅 MPKP Digital Badge")
                st.success(f"**{target_top_venue}** qualifies for the MPKP Strategic Food Tourism Recognition Program.")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%); padding: 30px; border-radius: 15px; text-align: center; color: white; border: 3px solid #F59E0B; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);">
                    <div style="font-size: 3rem; margin-bottom: 10px;">⭐</div>
                    <h3 style="color: #F59E0B; margin: 0; font-size: 1.2rem; font-weight: 800;">PREMIS PILIHAN MPKP</h3>
                    <p style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em; margin: 5px 0 15px 0;">Peringkat Daerah Kubang Pasu</p>
                    <div style="background-color: rgba(255,255,255,0.1); padding: 8px; border-radius: 8px; font-size: 1.1rem; font-weight: bold; color: #FBBF24;">
                         Satisfaction Index: {v_percentage:.1f}%
                    </div>
                    <p style="font-size: 0.7rem; color: #CBD5E1; margin-top: 15px; font-style: italic;">"Diiktiraf Melalui Analisis Maklum Balas Pintar"</p>
                </div>
                """, unsafe_allow_html=True)

            with top_col2:
                st.markdown("### 📄 Surat Penghargaan MPKP Generator")
                ref_number_top = st.text_input("Rujukan Fail MPKP:", f"MPKP/{target_top_venue[:3].upper()}")
                officer_name_top = st.text_input("Nama Pegawai Penguatkuasa:", "Siti Aminah Binti Ahmad")
                
                draft_text_top = st.text_area("Isi Kandungan Surat Penghargaan (Editable):", 
                    f"Pihak Majlis Perbandaran Kubang Pasu (MPKP) ingin merakamkan setinggi-tinggi penghargaan kepada pihak pengurusan premis atas pencapaian cemerlang "
                    f"mengekalkan mutu kualiti hidangan dan kepuasan pelanggan yang tinggi dengan skor indeks purata {v_percentage:.2f} melalui sistem pemantauan data digital.")

                st.markdown(f"""
                <div style="background-color: #FFFFFF; padding: 25px; border: 1px solid #CBD5E1; border-radius: 8px; color: #000000; font-family: 'Times New Roman', serif;">
                    <div style="text-align: center; font-weight: bold; border-bottom: 2px solid #F59E0B; padding-bottom: 10px;">
                        MAJLIS PERBANDARAN KUBANG PASU<br>
                        <span style="font-size: 0.8rem; font-weight: normal; color: #475569;">Kompleks Pentadbiran MPKP, Pekan Asun, 06000 Jitra, Kedah</span>
                    </div>
                    <div style="text-align: right; margin-top: 10px; font-size: 0.9rem; color: #000000;">Ruj. Kami: {ref_number_top}</div>
                    <div style="margin-top: 20px; color: #000000;"><b>Kepada: Pengurusan {target_top_venue}</b></div>
                    <p style="margin-top: 15px; color: #000000;">Tuan/Puan,</p>
                    <p style="color: #1E3A8A;"><b>SURAT PENGHARGAAN: KECEMERLANGAN MUTU PERKHIDMATAN DAN KEPUASAN PENGGUNA</b></p>
                    <p style="text-align: justify; color: #000000;">{draft_text_top}</p>
                    <p style="color: #000000;">Pihak Majlis berharap agar premis tuan/puan terus mengekalkan standard tinggi ini dan menjadi model contoh bagi sektor pelancongan makanan (Food Tourism) di daerah Kubang Pasu.</p>
                    <p style="color: #000000;">Tahniah dan terima kasih.</p>
                    <p style="color: #000000;"><b>"KEDAH SEJAHTERA AMAN MAKMUR"</b><br><br><br><b>{officer_name_top.upper()}</b><br>Bahagian Kesihatan & Lesen MPKP</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.download_button(
                    label="📥 Download Letter (Text)",
                    data=f"RUJ: {ref_number_top}\nKEPADA: {target_top_venue}\n\n{draft_text_top}",
                    file_name=f"MPKP_Penghargaan_{target_top_venue.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
                
    elif action_type == "Invite Influencer (Collaboration)":
        # Dynamic variable fallback fix: Check local variable, fallback to dataframe if missing
        has_influencers = 'inf_stats' in locals() and not inf_stats.empty
        
        if has_influencers:
            top_influencers_list = inf_stats.head(3).index.tolist()
            target_influencer = st.selectbox("🎯 Select Target Digital Voice:", [f"@{h}" if not str(h).startswith('@') else h for h in top_influencers_list])
            
            inf_letter_col1, inf_letter_col2 = st.columns([1.2, 1.8])
            
            with inf_letter_col1:
                st.markdown("### 📈 Engagement Strength")
                
                # Clean the handle to make sure there's no double @@ symbols
                clean_handle = target_influencer.replace('@', '')
                
                # Displaying the profile box exactly like your working pattern
                st.info(f"**Influencer Profile**\n\n**@{clean_handle}**")
                st.markdown(f"[🔗 Open TikTok Profile](https://www.tiktok.com/@{clean_handle})")
                
                # --- SAFELY ATTEMPT EMBED OR FALLBACK ---
                if 'filtered_test_df' in locals() and url_col in filtered_test_df.columns:
                    # Dynamically find the author/username column used in your main dataframe
                    possible_user_cols = ['author_name', 'author', 'username', 'user_handle', 'nickname']
                    user_col_detected = next((col for col in possible_user_cols if col in filtered_test_df.columns), None)
                    
                    if user_col_detected:
                        inf_data = filtered_test_df[filtered_test_df[user_col_detected].astype(str).str.contains(clean_handle, case=False, na=False)]
                        
                        if not inf_data.empty:
                            inf_video_url = inf_data['videoweburl'].iloc[0]
                            if pd.notna(inf_video_url):
                                inf_video_id = inf_video_url.split('/')[-1].split('?')[0]
                                st.components.v1.html(f"""
                                    <iframe src="https://www.tiktok.com/embed/v2/{inf_video_id}" 
                                            style="width: 100%; height: 320px; border: none; border-radius: 12px;" 
                                            allowfullscreen>
                                    </iframe>
                                """, height=330)

                # Campaign Brief Summary Badge Card remains below
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #7C3AED 0%, #4C1D95 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; border: 1px solid #DDD6FE; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-top: 15px;">
                    <div style="font-size: 1.8rem; margin-bottom: 2px;">🎥</div>
                    <h4 style="color: #FBBF24; margin: 0; font-size: 1rem; font-weight: 800;">KEMPEN 'TASTE OF KEDAH'</h4>
                    <p style="font-size: 0.7rem; color: #C084FC; text-transform: uppercase; letter-spacing: 0.1em; margin: 2px 0 10px 0;">Kolaborasi Kreator MPKP</p>
                    <div style="background-color: rgba(255,255,255,0.15); padding: 8px; border-radius: 6px; text-align: left; font-size: 0.75rem; line-height: 1.4;">
                        • <b>Sasaran:</b> Promosi Premis Pilihan<br>
                        • <b>Platform:</b> TikTok Short-form Video<br>
                        • <b>Impak:</b> Memperkasa Ekonomi Lokal
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with inf_letter_col2:
                st.markdown("### 📄 Surat Jemputan Kolaborasi")
                clean_handle = target_influencer.replace('@','')
                ref_number_inf = st.text_input("Rujukan Fail MPKP:", f"MPKP/{clean_handle.upper()[:3]}")
                officer_name_inf = st.text_input("Nama Pegawai Pengendali:", "Siti Aminah Binti Ahmad")
                
                draft_text_inf = st.text_area("Isi Kandungan Surat Jemputan (Editable):", 
                    f"Pihak Majlis Perbandaran Kubang Pasu (MPKP) tertarik dengan impak digital dan kreativiti kandungan media sosial tuan/puan. "
                    f"Sehubungan dengan itu, pihak kami ingin menjemput tuan/puan secara rasmi untuk bekerjasama sebagai Duta Kempen Pelancongan Makanan Pintar Daerah Kubang Pasu.")

                st.markdown(f"""
                <div style="background-color: #FFFFFF; padding: 25px; border: 1px solid #CBD5E1; border-radius: 8px; color: #000000; font-family: 'Times New Roman', serif;">
                    <div style="text-align: center; font-weight: bold; border-bottom: 2px solid #7C3AED; padding-bottom: 10px;">
                        MAJLIS PERBANDARAN KUBANG PASU<br>
                        <span style="font-size: 0.8rem; font-weight: normal; color: #475569;">Kompleks Pentadbiran MPKP, Pekan Asun, 06000 Jitra, Kedah</span>
                    </div>
                    <div style="text-align: right; margin-top: 10px; font-size: 0.9rem; color: #000000;">Ruj. Kami: {ref_number_inf}</div>
                    <div style="margin-top: 20px; color: #000000;"><b>Kepada: Pengurusan Media / Pemilik Akaun {target_influencer}</b></div>
                    <p style="margin-top: 15px; color: #000000;">Tuan/Puan,</p>
                    <p style="color: #7C3AED;"><b>JEMPUTAN JALINAN KERJASAMA STRATEGIK: DIGITAL INFLUENCER AMBASSADOR PROGRAMME</b></p>
                    <p style="text-align: justify; color: #000000;">{draft_text_inf}</p>
                    <p style="text-align: justify; color: #000000;">Melalui program ini, pihak tuan/puan akan diberi pelepasan khas dan pembiayaan MPKP untuk membuat liputan kreatif eksklusif bagi siri mini 'Hidden Gems' melibatkan premis-premis makanan gred tinggi daerah.</p>
                    <p style="color: #000000;">Kerjasama pihak tuan/puan amat kami hargai demi memajukan sektor pelancongan domestik.</p>
                    <p style="color: #000000;"><b>"KEDAH SEJAHTERA AMAN MAKMUR"</b><br><br><br><b>{officer_name_inf.upper()}</b><br>Bahagian Korporat & Pelancongan MPKP</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.download_button(
                    label="📥 Download Invitation (Text)",
                    data=f"RUJ: {ref_number_inf}\nKEPADA: {target_influencer}\n\n{draft_text_inf}",
                    file_name=f"MPKP_Jemputan_{clean_handle}.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Data kreator media sosial tidak ditemui dalam sistem untuk diproses.")
            
    st.divider()
    st.subheader("Project Alignment Matrix")
    st.table({
        "Objective": ["OBJ 1: Data Gathering", "OBJ 2: Analysis", "OBJ 3: Visual Insights", "OBJ 4: Promoting Sector"],
        "Dashboard Implementation": ["CSV Data Mapped", f" Models ({engine_choice})", "Sentiment Visuals", " Strategy Section"]
    })

##################################################
##################################################
#PAGE 6

elif st.session_state.active_page == "Model":
    ### MODEL COMPARISON [5]
    st.header("🧠 Model Performance Analysis")
    st.write("This section compares how well our two AI 'brains' understand TikTok comments based on Ground Truth (True Labels).")
    st.divider()

    # --- 1. KEY METRICS ---
    # Using 'true_label' for ground truth to avoid logic confusion
    acc_nb = accuracy_score(base_filtered_df['true_label'], base_filtered_df['ml_pred'])
    acc_bilstm = accuracy_score(base_filtered_df['true_label'], base_filtered_df['dl_pred'])
    
    # --- 1. KEY METRICS (REMOVED DELTA) ---
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("MultiNB Accuracy", f"{acc_nb:.2%}")
    # We removed the delta=f"{(acc_bilstm - acc_nb):.2%}" part here:
    col_m2.metric("BiLSTM Accuracy", f"{acc_bilstm:.2%}")

    # --- 2. ACCURACY BAR CHART ---
    st.subheader("📊 Accuracy Benchmarking")
    comp_df = pd.DataFrame({
        "Model": ["MultiNB (Baseline)", "BiLSTM (Deep Learning)"], 
        "Accuracy": [acc_nb, acc_bilstm]
    })
    
    fig_acc = px.bar(
        comp_df, x="Model", y="Accuracy", 
        text_auto='.2%', color="Model",
        color_discrete_sequence=["#0C147F", "#4C9AFF"]
    )
    fig_acc.update_layout(yaxis_range=[0, 1]) # Keep scale 0 to 100%
    st.plotly_chart(fig_acc, use_container_width=True)

    with st.expander("💡 Why is BiLSTM scoring higher?"):
        st.write(f"""
        **BiLSTM ({acc_bilstm:.2%})** uses 'contextual memory'. It reads the sentence forwards and backwards, which is crucial for 
        detecting negations like *"Tak"* or *"Kureng"*. **MultiNB ({acc_nb:.2%})** only counts word frequencies, often 
        getting confused by complex sentences.
        """)

    st.divider()

    # --- 3. SENTIMENT DISTRIBUTION COMPARISON ---
    st.subheader("📈 Prediction Distribution")
    st.write("Comparing how many comments each model assigned to each category.")

    ml_dist = base_filtered_df['ml_pred'].value_counts().reset_index()
    ml_dist.columns = ['Sentiment', 'Count']
    ml_dist['Model'] = 'MultiNB'

    dl_dist = base_filtered_df['dl_pred'].value_counts().reset_index()
    dl_dist.columns = ['Sentiment', 'Count']
    dl_dist['Model'] = 'BiLSTM'

    total_dist = pd.concat([ml_dist, dl_dist])
    
    fig_dist = px.bar(
        total_dist, x="Sentiment", y="Count", color="Model",
        barmode="group", text_auto=True,
        color_discrete_map={"MultiNB": "#0C147F", "BiLSTM": "#4C9AFF"}
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # --- 4. CONFUSION MATRICES ---
    st.subheader("🎯 Error Detection (Confusion Matrix)")
    st.write("Diagonal cells show correct predictions. Off-diagonal cells show specific mistakes.")
    
    col_nb, col_bi = st.columns(2)
    labels = ['negative', 'neutral', 'positive']
    short_labels = ['Neg', 'Neu', 'Pos']
    
    with col_nb:
        st.markdown("<center><b>MultiNB Confusion</b></center>", unsafe_allow_html=True)
        cm_nb = confusion_matrix(base_filtered_df['true_label'], base_filtered_df['ml_pred'], labels=labels)
        fig_cm_nb = ff.create_annotated_heatmap(cm_nb, x=short_labels, y=short_labels, colorscale='Blues')
        
        # --- REMOVE ZERO & FALSE LINES ---
        fig_cm_nb.update_xaxes(showgrid=False, zeroline=False, showline=False)
        fig_cm_nb.update_yaxes(showgrid=False, zeroline=False, showline=False, autorange="reversed") # Keeps matrix orientation right
        
        st.plotly_chart(fig_cm_nb, use_container_width=True)

    with col_bi:
        st.markdown("<center><b>BiLSTM Confusion</b></center>", unsafe_allow_html=True)
        cm_bi = confusion_matrix(base_filtered_df['true_label'], base_filtered_df['dl_pred'], labels=labels)
        
        # Switched 'Greens' to 'Blues' to match your new corporate blue theme
        fig_cm_bi = ff.create_annotated_heatmap(cm_bi, x=short_labels, y=short_labels, colorscale='Blues')
        
        # --- REMOVE ZERO & FALSE LINES ---
        fig_cm_bi.update_xaxes(showgrid=False, zeroline=False, showline=False)
        fig_cm_bi.update_yaxes(showgrid=False, zeroline=False, showline=False, autorange="reversed")
        
        st.plotly_chart(fig_cm_bi, use_container_width=True)

    st.divider()

    # --- USER-FRIENDLY STRATEGIC ANALYSIS ---
    st.subheader("🧐 Which model should MPKP use?")
    col_a, col_b = st.columns(2)

    with col_a:
        st.success("### 📖 Multinomial Naive Bayes (ML) Insight")
        st.write(f"""
        **Best for:** Fast, simple feedback.
        
        * **How it works:** It acts like a dictionary. If it sees the word **'Sedap'**, it automatically thinks the person is happy.
        * **The Weakness:** It can be easily tricked! If a user says **"Tak sedap langsung"**, it might still see 'sedap' and think it's positive because it doesn't understand the word **'Tak'** changes everything.
        * **Verdict:** Great for checking general trends very quickly.
        """)
        
    with col_b:
        st.info("### 🧠  BiLSTM (DL) Insight")
        st.write(f"""
        **Best for:** Sarcasm and Kedah slang.
        
        * **How it works:** It reads the sentence **forwards and backwards**. It knows that the word **'Tak'** at the start changes the meaning of the whole sentence.It correctly identifies **"Tak sedap langsung"** as negative because it links the negation to the adjective.
        * **The Strength:** It understands local slang and context much better. It is harder to trick.
        * **Verdict:** This is the 'gold standard' for MPKP to truly understand if tourists are frustrated or happy.
        """)

elif st.session_state.active_page == "About":

    # ===== HEADER =====
    st.markdown("""
    <h2 style='color:#1F3A5F; border-bottom:2px solid #E5E7EB; padding-bottom:8px;'>
    About TastePulse
    </h2>
    """, unsafe_allow_html=True)

    # ===== OVERVIEW =====
    st.markdown("""
    <h4 style='color:#1F3A5F;'>System Overview</h4>
    """, unsafe_allow_html=True)

    st.write("""
    TastePulse is an intelligent analytics dashboard developed to analyze food tourism trends 
    and customer sentiment within the Kubang Pasu region, including Sintok, Changlun, and Jitra.

    Social media platforms such as TikTok contain valuable customer feedback. However, these 
    reviews often include informal expressions, local dialects, and Kedah slang (e.g., 'kotoq', 'mahai'), 
    which are difficult for traditional systems to interpret.

    This system applies Natural Language Processing (NLP) and Machine Learning techniques to transform 
    unstructured text into structured insights. It supports local authorities (MPKP) and business owners 
    in monitoring public perception and improving service quality.
    """)
    st.divider()
    # ===== OBJECTIVES =====
    st.markdown("""
    <h4 style='color:#1F3A5F;'>Project Objectives</h4>
    """, unsafe_allow_html=True)

    st.markdown("""
    - Analyze customer sentiment from social media reviews  
    - Identify popular and underperforming food venues  
    - Support data-driven decision making  
    - Improve food tourism quality in Kubang Pasu  
    """)
    st.divider()
    # ===== KDD PROCESS =====
    st.markdown("""
    <h4 style='color:#1F3A5F;'>System Methodology: KDD Process</h4>
    """, unsafe_allow_html=True)

    st.markdown("""
    The system follows the Knowledge Discovery in Databases (KDD) process to transform raw data into meaningful insights.
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Selection",
        "Data Preprocessing",
        "Data Transformation",
        "Data Mining",
        "Knowledge Evaluation"
    ])

    with tab1:
        st.markdown("""
        **Purpose:** Identify relevant data  

        Social media data is collected using location-based hashtags such as  
        #KubangPasu and #JitraFood to ensure relevance to local food tourism.
        """)

    with tab2:
        st.markdown("""
        **Purpose:** Clean raw data  

        Removes noise such as URLs, emojis, and irrelevant characters.  
        Standardizes slang and spelling variations for consistency.
        """)

    with tab3:
        st.markdown("""
        **Purpose:** Convert text into structured format  

        Text is transformed into numerical features (e.g., TF-IDF)  
        so machine learning models can process it.
        """)

    with tab4:
        st.markdown("""
        **Purpose:** Extract insights  

        Machine learning models classify sentiment into  
        positive, negative, or neutral categories.
        """)

    with tab5:
        st.markdown("""
        **Purpose:** Present insights  

        Results are visualized through charts, alerts, and dashboards  
        to support decision-making.
        """)

    # ===== KDD IMAGE =====
    st.write("")
    col1, col2, col3 = st.columns([1.5, 2.0, 1.5])
    with col2:
        st.image("KDD.png", caption="Data Processing Workflow", use_column_width=True)
    st.divider()
    # ===== SYSTEM VALUE =====
    st.markdown("""
    <h4 style='color:#1F3A5F;'>System Value</h4>
    """, unsafe_allow_html=True)

    st.markdown("""
    - Local Authorities (MPKP): Monitor food quality and sentiment  
    - Business Owners: Improve services based on feedback  
    - Researchers: Analyze consumer behavior trends  
    """)
    st.divider()
    st.markdown("##Project Profiles")
    st.write("")

    # ==================== ROW 1: LEADER & DEVELOPER ====================
 row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center; margin-top: 0; color: #000000; font-weight: 800; font-size: 1.6rem;'>📋 Project Supervisor</h3>", unsafe_allow_html=True)
            st.divider()
            
            # Increased side column weights to constrain the center image size
            img_left, img_mid, img_right = st.columns([1.5, 1.0, 1.5])
            with img_mid:
                st.image("juhaida.jpeg", use_column_width=True)
                
            st.markdown("""
            <div style="text-align: center; margin-top: 15px;">
                <h3 style="margin: 5px 0; color: #000000; font-size: 1.4rem; font-weight: 800;">ASSOC. PROF. TS. DR. JUHAIDA BINTI ABU BAKAR</h3>
                <p style="font-size: 1.25rem; margin: 6px 0; font-weight: 800; color: #000000;">Project Supervisor</p>
                <p style="font-size: 1.15rem; margin: 4px 0; font-weight: 700; color: #000000;"><b>Institution:</b> Universiti Utara Malaysia (UUM)</p>
                <p style="font-size: 1.15rem; margin: 4px 0; font-weight: 700; color: #000000;"><b>Focus:</b> Data Analytics, NLP, Intelligent Systems</p>
            </div>
            """, unsafe_allow_html=True)

    with row1_col2:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center; margin-top: 0; color: #000000; font-weight: 800; font-size: 1.6rem;'>👨‍💻 System Developer</h3>", unsafe_allow_html=True)
            st.divider()
            
            img_left, img_mid, img_right = st.columns([1.5, 1.0, 1.5])
            with img_mid:
                st.image("najaa.png", use_column_width=True)
                
                # --- CENTERING THE BUTTON HERE ---
                # Create a small sub-grid to isolate and center the button element
                btn_l, btn_m, btn_r = st.columns([1, 1.2, 1])
                with btn_m:
                    if st.button("Contact"):
                    # Wrap the email address in a clickable mailto anchor link
                        st.markdown("""
                            <p style='text-align: center; color: #000000; font-weight: 700; font-size: 1.0rem; margin-top: 5px;'>
                                Email: <a href="mailto:nur_najaa_aini@soc.uum.edu.my" style="color: #1E3A8A; text-decoration: underline; font-weight: 800;">nur_najaa_aini@soc.uum.edu.my</a>
                            </p>
                        """, unsafe_allow_html=True)
                    
            st.markdown("""
            <div style="text-align: center; margin-top: 15px;">
                <h3 style="margin: 5px 0; color: #000000; font-size: 1.4rem; font-weight: 800;">NUR NAJAA AINI BINTI MOHD PUZI</h3>
                <p style="font-size: 1.25rem; margin: 6px 0; font-weight: 800; color: #000000;">Final Year Student</p>
                <p style="font-size: 1.15rem; margin: 4px 0; font-weight: 700; color: #000000;"><b>Project - </b> TastePulse: Sentiment Analysis of Food Tourism in Northern Community</p>
            </div>
            """, unsafe_allow_html=True)


            
    st.write("")
    st.divider()
    st.write("")


    # ==================== ROW 2: OTHER SUPERVISED PROJECTS ====================
    st.markdown("### 📚 Other Supervised Projects")
    st.write("")

    st.markdown("### 📚 Other Supervised Projects")
    st.write("")
    
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # Removed container border
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.image("faiz.jpeg", width=150) # Use width or use_column_width
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 10px;">
            <h4 style="margin: 5px 0; color: #000000; font-size: 1.25rem; font-weight: 800;">NUR FAIZLYANA BINTI MOHD KAMARUL ARIFFIN</h4>
            <p style="font-size: 1.1rem; margin: 4px 0; font-weight: 700; color: #000000;">Final Year Project</p>
            <p style="font-size: 1.05rem; margin: 4px 0; font-weight: 700; color: #000000; font-style: italic;">Title - TastePulse: Sentiment Analysis of Food Tourism in Northern Community</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row2_col2:
        # Removed container border
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.image("mak.jpeg", width=150) # Fixed the 'column=True' typo
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 10px;">
            <h4 style="margin: 5px 0; color: #000000; font-size: 1.25rem; font-weight: 800;">MAK SHEI WEN</h4>
            <p style="font-size: 1.1rem; margin: 4px 0; font-weight: 700; color: #000000;">Final Year Project</p>
            <p style="font-size: 1.05rem; margin: 4px 0; font-weight: 700; color: #000000; font-style: italic;">Title - ThemePulse: Topic Modeling of Food Tourism in Northern Community</p>
        </div>
        """, unsafe_allow_html=True)


# ==================== FOOTER ====================
st.write("")
st.write("")
st.markdown("<p style='text-align: center; color: #000000; font-weight: 700; font-size: 1.0rem;'>© 2026 TastePulse System | Designed for MPKP & Kubang Pasu Food Tourism Management</p>", unsafe_allow_html=True)
