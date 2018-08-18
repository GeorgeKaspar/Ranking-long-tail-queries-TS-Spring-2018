import num2words

ordinal_to_cardinal = {
    'первый': 'один',
    'второй': 'два',
    'третий': 'три',
    'четвертый': 'четыре',
    'пятый': 'пять',
    'шестой': 'шесть',
    'седьмой': 'семь',
    'восьмой': 'восемь',
    'девятый': 'девять',
    'десятый': 'десять',
    'одиннадцатый': 'одиннадцать',
    'двенадцатый': 'двенадцать',
    'тринадцатый': 'тринадцать',
    'четырнадцатый': 'четырнадцать',
    'пятнадцатый': 'пятнадцать',
    'шестнадцатый': 'шестнадцать',
    'семнадцатый': 'семнадцать',
    'восемнадцатый': 'восемнадцать',
    'девятнадцатый': 'девятнадцать',
    'двадцатый': 'двадцать',
    'тридцатый': 'тридцать',
    'сороковой': 'сорок',
    'пятидесятый': 'пятьдесят',
    'шестидесятый': 'шестьдесят',
    'семидесятый': 'семьдесят',
    'восьмидесятый': 'восемьдесят',
    'девяностый': 'девяносто',
    'сотый': 'сто',
    'двухсотый': 'двести',
    'трехсотый': 'триста',
    'четырехсотый': 'четыреста',
    'пятисотый': 'пятьсот',
    'шестясотый': 'шестьсот',
    'семисотый': 'семьсот',
    'восьмисотый': 'восемьсот',
    'девятисотый': 'девятьсот',
    'тысячный': 'тысяча',
    'двухтысячный': 'два тысяча',
    'трехтысячный': 'три тысяча',
    'четырехтысячный': 'четыре тысяча',
    'пятитысячный': 'пять тысяча',
    'шеститысячный': 'шесть тысяча',
    'семитысячный': 'семь тысяча',
    'восьмитысячный': 'восемь тысяча',
    'девятитысячный': 'девять тысяча',
    'стотысячный': 'сто тысяча',
    'миллионный': 'миллион',
}


def _num2words(x):
    try:
        return num2words.num2words(x, lang='ru')
    except:
        return x


def _ordinal_to_cardinal(x):
    global ordinal_to_cardinal
    return ordinal_to_cardinal.get(x, x)
