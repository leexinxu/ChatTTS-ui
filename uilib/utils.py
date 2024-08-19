import os,sys
import requests
import time
import re
import webbrowser
from pathlib import Path
import pandas as pd
# ref: https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
from .zh_normalization import TextNormalizer
from .cfg import SPEAKER_DIR
from functools import partial
from num2words import num2words

def openweb(url):
    time.sleep(3)
    try:
        webbrowser.open(url)
    except Exception:
        pass

def get_parameter(request, param, default, cast_type):
    #  先request.args 后request.form 然后转换cast_type=int|float类型。
    for method in [request.args.get, request.form.get]:
        value = method(param, "").strip()
        if value:
            try:
                return cast_type(value)
            except ValueError:
                break  # args转换失败，退出尝试form
    return default  # 失败，返回默认值。


# 数字转为英文读法
def num_to_english(num):
    num_str = str(num)
    english_digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    big_units = ["", "thousand", "million", "billion", "trillion"]
    teen_map = {
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
        16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"
    }
    tens_map = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    result = []
    need_and = False
    part = []

    # 将数字按三位分组
    while num_str:
        part.append(num_str[-3:])
        num_str = num_str[:-3]

    part.reverse()

    for i, p in enumerate(part):
        p_str = []
        digit_len = len(p)
        hundreds_digit = int(p) // 100 if digit_len == 3 else None
        tens_digit = int(p[-2:]) if digit_len >= 2 else int(p[-1])

        # 处理百位数
        if hundreds_digit is not None and hundreds_digit != 0:
            p_str.append(english_digits[hundreds_digit] + " hundred")
            if tens_digit != 0:
                p_str.append("and")

        # 处理十位和个位数
        if 10 < tens_digit < 20:  # 处理11到19的特殊情况
            p_str.append(teen_map[tens_digit])
        else:
            tens_val = tens_digit // 10
            ones_val = tens_digit % 10
            if tens_val >= 2:
                p_str.append(tens_map[tens_val])
                if ones_val != 0:
                    p_str.append(english_digits[ones_val])
            elif tens_digit != 0:
                p_str.append(english_digits[ones_val])

        if p_str:
            result.append(" ".join(p_str))
            if i < len(part) - 1 and int(p) != 0:
                result.append(big_units[len(part) - i - 1])

    return " ".join(result).capitalize()



def get_lang(text):
    # 定义中文标点符号的模式
    chinese_punctuation = "[。？！，、；：‘’“”（）《》【】…—\u3000]"
    # 使用正则表达式替换所有中文标点为""
    cleaned_text = re.sub(chinese_punctuation, "", text)
    # 使用正则表达式来匹配中文字符范围
    return "zh" if re.search('[\u4e00-\u9fff]', cleaned_text) is not None else "en"

def fraction_to_words(match):
    numerator, denominator = match.groups()
    # 这里只是把数字直接拼接成了英文分数的形式, 实际上应该使用某种方式将数字转换为英文单词
    # 例如: "1/2" -> "one half", 这里仅为展示目的而直接返回了 "numerator/denominator"
    return numerator + " over " + denominator



# 数字转为英文读法
def num2text(text):
    numtext=[' zero ',' one ',' two ',' three ',' four ',' five ',' six ',' seven ',' eight ',' nine ']
    point=' point '
    text = re.sub(r'(\d)\,(\d)', r'\1\2', text)
    text = re.sub(r'(\d+)\s*\+', r'\1 plus ', text)
    text = re.sub(r'(\d+)\s*\-', r'\1 minus ', text)
    text = re.sub(r'(\d+)\s*[\*x]', r'\1 times ', text)
    text = re.sub(r'((?:\d+\.)?\d+)\s*/\s*(\d+)', fraction_to_words, text)

    # 取出数字 number_list= [('1000200030004000.123', '1000200030004000', '123'), ('23425', '23425', '')]
    number_list=re.findall(r'((\d+)(?:\.(\d+))?%?)', text)
    if len(number_list)>0:
        #dc= ('1000200030004000.123', '1000200030004000', '123','')
        for m,dc in enumerate(number_list):
            if len(dc[1])>16:
                continue
            int_text= num_to_english(dc[1])
            if len(dc)>2 and dc[2]:
                int_text+=point+"".join([numtext[int(i)] for i in dc[2]])
            if dc[0][-1]=='%':
                int_text=f' the pronunciation of  {int_text}'
            text=text.replace(dc[0],int_text)


    return text.replace('1',' one ').replace('2',' two ').replace('3',' three ').replace('4',' four ').replace('5',' five ').replace('6',' six ').replace('7','seven').replace('8',' eight ').replace('9',' nine ').replace('0',' zero ').replace('=',' equals ')



def remove_brackets(text):
    # 正则表达式
    text=re.sub(r'\[(uv_break|laugh|lbreak|break)\]',r' \1 ',text,re.I|re.S|re.M)

    # 使用 re.sub 替换掉 [ ] 对
    newt=re.sub(r'\[|\]|！|：|｛|｝', '', text)
    return    re.sub(r'\s(uv_break|laugh|lbreak|break)(?=\s|$)', r' [\1] ', newt)


# 中英文数字转换为文字，特殊符号处理
def split_text(text_list):
    
    tx = TextNormalizer()
    haserror=False
    result=[]
    for i,text in enumerate(text_list):
        text=remove_brackets(text)
        if get_lang(text)=='zh':
            tmp="".join(tx.normalize(text))
        elif haserror:
            tmp=num2text(text)
        else:
            try:
                # 先尝试使用 nemo_text_processing 处理英文
                from nemo_text_processing.text_normalization.normalize import Normalizer
                fun = partial(Normalizer(input_case='cased', lang="en").normalize, verbose=False, punct_post_process=True)
                tmp=fun(text)
                print(f'使用nemo处理英文ok')
            except Exception as e:
                print(f"nemo处理英文失败，改用自定义预处理")
                print(e)
                haserror=True
                tmp=convert_numbers_to_words(text)
        tmp = tmp.replace("·", "").replace("？", " ").replace("?", " ")  # 去掉和转换文本中ChatTTS会读错乱的符号
        if len(tmp)>200:
            tmp_res=split_text_by_punctuation(tmp)
            result=result+tmp_res
        else:
            result.append(tmp)
    return result

# 切分长行 200 150
def split_text_by_punctuation(text):
    # 定义长度限制
    min_length = 150
    punctuation_marks = "。？！，、；：”’》」』）】…—"
    english_punctuation = ".?!,:;)}…"

    # 结果列表
    result = []
    # 起始位置
    pos = 0

    # 遍历文本中的每个字符
    text_length=len(text)
    for i, char in enumerate(text):
        if char in punctuation_marks or char in english_punctuation:
            if  char=='.' and i< text_length-1 and re.match(r'\d',text[i+1]):
                continue
            # 当遇到标点时，判断当前分段长度是否超过120
            if i - pos > min_length:
                # 如果长度超过120，将当前分段添加到结果列表中
                result.append(text[pos:i+1])
                # 更新起始位置到当前标点的下一个字符
                pos = i+1
    #print(f'{pos=},{len(text)=}')

    # 如果剩余文本长度超过120或没有更多标点符号可以进行分割，将剩余的文本作为一个分段添加到结果列表
    if pos < len(text):
        result.append(text[pos:])

    return result


# 获取../static/wavs目录中的所有文件和目录并清理wav
def ClearWav(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    if not files:
        return False, "wavs目录内无wav文件"

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"已删除文件: {file_path}")
            elif os.path.isdir(file_path):
                print(f"跳过文件夹: {file_path}")
        except Exception as e:
            print(f"文件删除错误 {file_path}, 报错信息: {e}")
            return False, str(e)
    return True, "所有wav文件已被删除."



# 加载音色
# 参考 https://github.com/craii/ChatTTS_WebUI/blob/main/utils.py
def load_speaker(name):
    speaker_path = f"{SPEAKER_DIR}/{name}.csv" if not name.endswith('.csv') else f"{SPEAKER_DIR}/{name}"
    if not os.path.exists(speaker_path):
        return None
    try:
        import torch
        d_s = pd.read_csv(speaker_path, header=None).iloc[:, 0]
        tensor = torch.tensor(d_s.values)
    except Exception as e:
        print(e)
        return None
    return tensor


# 获取 speaker_dir下的所有csv pt文件
def get_speakers():
    result=[]
    for it in os.listdir(SPEAKER_DIR):
        if it.endswith('.pt'):
            result.append(it)
    return result

# 判断是否可以连接外网
def is_network():
    try:
        import requests
        requests.head('https://baidu.com')
    except Exception:
        return False
    else:
        return True
    return False



def is_chinese_os():
    import subprocess
    try:
        import locale
        # Windows系统
        if sys.platform.startswith('win'):
            lang = locale.getdefaultlocale()[0]
            return lang.startswith('zh_CN') or lang.startswith('zh_TW') or lang.startswith('zh_HK')
        # macOS系统
        elif sys.platform == 'darwin':
            process = subprocess.Popen(['defaults', 'read', '-g', 'AppleLocale'], stdout=subprocess.PIPE)
            output, error = process.communicate()
            if error:
                # 若默认方法出错，则尝试环境变量
                return os.getenv('LANG', '').startswith('zh_')
            locale = output.decode().strip()
            return locale.startswith('zh_')
        # 类Unix系统
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            return os.getenv('LANG', '').startswith('zh_')
        # 其他系统
        else:
            return False

    except Exception as e:
        # 输出异常到控制台，实际应用中应该使用日志记录异常
        print(e)
        return False



def modelscope_status():
    #return False
    try:
        res=requests.head("https://www.modelscope.cn/")
        print(res)
        if res.status_code!=200:
            return False
    except Exception as e:
        return False
    return True


def convert_numbers_to_words(text):
    def replace_number(match):
        number = match.group(0)
        # 处理整数和小数部分
        if '.' in number:
            integer_part, fractional_part = number.split('.')
            return f"{num2words(integer_part)} point {' '.join(num2words(digit) for digit in fractional_part)}"
        else:
            return num2words(number)

    # 使用正则表达式匹配数字
    return re.sub(r'\b\d+(\.\d+)?\b', replace_number, text)


