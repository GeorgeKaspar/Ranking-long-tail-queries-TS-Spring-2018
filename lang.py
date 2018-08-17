import polyglot
from polyglot.detect import Detector
import langdetect
import numpy as np
from nltk.corpus import words as nltk_words


def detect_lang_v2(text):
    detector = Detector(text, quiet=True)
    lang = detector.languages[0]
    return lang.code


def detect_lang_v1(text):
    return langdetect.detect(text)

detect_lang = detect_lang_v2


letters_dict = {'q':'й', 'w':'ц', 'e':'у', 'r':'к', 't':'е', 'y':'н', \
                'u':'г', 'i':'ш', 'o':'щ', 'p':'з', '[':'х', ']':'ъ', \
                'a':'ф', 's':'ы', 'd':'в', 'f':'а', 'g':'п', 'h':'р', \
                'j':'о', 'k':'л', 'l':'д', ';':'ж', '\'':'э', 'z':'я', \
                'x':'ч', 'c':'с', 'v':'м', 'b':'и', 'n':'т', 'm':'ь',\
                ',':'б', '.':'ю', '`':'ё', '?':',', ' ':' ', 'е':'e', \
                'й':'й', 'ц':'ц', 'у':'у', 'к':'к', 'н':'н', 'г':'г', \
                'ш':'ш', 'щ':'щ', 'з':'з', 'х':'х', 'ъ':'ъ', 'ф':'ф', \
                'ы':'ы', 'в':'в', 'а':'а', 'п':'п', 'р':'р', 'о':'о', \
                'л':'л', 'д':'д', 'ж':'ж', 'э':'э', 'я':'я', 'ч':'ч', \
                'с':'с', 'м':'м', 'и':'и', 'т':'т', 'ь':'ь', 'б':'б', \
                'ю':'ю', 'ё':'ё', '\u0456':u'\u0456', }

def change_keyboard_layout(text):
    # language1 = detect_lang_v1(text)
    # language2 = detect_lang_v2(text)
    # true_langs = ['ru', 'uk', 'en']
    #  "LATCHSTRING"

    consonants_en = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'z']
    vowels_en = ['a', 'i', 'e', 'o', 'u', 'y']

    consonants_ru = ['б', 'в', 'г', 'д', 'ж', 'з', 'й', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т', 'ф', 'х', 'ц', 'ч', 'ш', 'щ']
    vowels_ru = ['а', 'о', 'э', 'и', 'у', 'ы', 'е', 'ё', 'ю', 'я']
    en = consonants_en + vowels_en
    ru = consonants_ru + vowels_ru

    if set(text).intersection(set(ru)):
        return text

    def sound_map(x):
        if x in consonants_en:
            return 1
        return 0

    sounds = np.fromiter(map(lambda x: sum(map(sound_map, x)), text.split()), np.int32)
    if np.any(sounds >= 4) or 'rfr' in text or 'xnj' in text or 'relf' in text:
        return ''.join(map(lambda x: letters_dict.get(x, x), text))

    return text

    '''
    ind = text.find(' d ')
    if ind == -1:
        return text
    else: 
        return text[:ind] + ''.join(map(lambda x: letters_dict.get(x, x), text[ind:]))
    '''

    