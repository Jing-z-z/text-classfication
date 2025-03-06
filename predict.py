import regex
import jieba
import jieba.analyse
import fasttext
import pandas as pd

"""
用于预测数据的csv，xlsx文件第一行的前两列需要有title，content作为标题
格式为：
title      content
新闻标题1    新闻内容1
新闻标题2    新闻内容2
"""

label_to_category = {
    '__label__100': '其他',  # 包括'民生','文化','旅游','国际','证劵','农业'
    '__label__102': '娱乐',
    '__label__103': '体育',
    '__label__104': '财经',
    '__label__106': '房产',
    '__label__107': '汽车',
    '__label__108': '教育',
    '__label__109': '科技',
    '__label__110': '军事',
    '__label__116': '电竞',
}


# stopwords = set()
# with open('stopwords.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         stopwords.add(line.strip())


def preprocess(title, content):
    # 将缺失值或者非字符类型改为空字符
    if pd.isna(title) or not isinstance(title, str):
        title = ''
    if pd.isna(content) or not isinstance(content, str):
        content = ''

    # 如果content为空，则仅使用title
    if content.strip() == '':
        text = title
    else:
        # cleaned_content = regex.sub(r'[^\p{L}\p{N}\p{Han}\s]', '', content, flags=regex.UNICODE)
        # words = jieba.lcut(cleaned_content)
        # filtered_words = [word for word in words if word not in stopwords]
        # filtered_content = ' '.join(filtered_words)
        keywords = jieba.analyse.extract_tags(content, topK=10, allowPOS=('n', 'v', 'a'))
        text = title + '' + ' '.join(keywords)

    cleaned_content = regex.sub(r'[^\p{L}\p{N}\p{Han}\s]', '', text, flags=regex.UNICODE)
    words = jieba.lcut(cleaned_content)
    return ' '.join(words)


def predict(file_path):
    # 检测文件类型并读取数据
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8')
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError("不支持的文件格式，请上传CSV或XLSX文件！")

    model = fasttext.load_model('model_fold_1.bin')
    results = []

    # 批量预测
    for _, row in df.iterrows():
        title = row['title']
        content = row['content']
        text = preprocess(title, content)
        if pd.isna(title) or title.strip() == '':
            continue
        labels, probabilities = model.predict(text, k=3)
        # 将预测结果转换为可读格式
        predictions = [(label_to_category.get(label, '未知分类'), prob) for label, prob in zip(labels, probabilities)]
        results.append({'title': title, 'predictions': predictions})

    print("\n预测结果如下：")
    for result in results:
        print(f"新闻标题: {result['title']}")
        print("预测类别及置信度:")
        for category, prob in result['predictions']:
            print(f"  分类: {category}, 置信度: {prob:.4f}")
        print("-" * 40)


file_path = input("请输入csv或xlsx文件路径: ").strip()
try:
    predict(file_path)
except Exception as e:
    print(f"发生错误: {e}")
