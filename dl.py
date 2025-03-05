import pandas as pd
import regex
import jieba
import fasttext
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# 共382688个样本，部分样本没有关键词
# 样本示例：6552295326462509575_!_106_!_news_house_!_2018年的房价回是什么趋势？_!_房地产泡沫,海南岛,房价,大洗牌,房地产开发商
columns = ['item_id', 'news_id', 'news_category', 'title', 'keywords']
data = []


# 预处理
def preprocess(title, keywords):
    text = title + ' ' + keywords
    cleaned_text = regex.sub(r'[^\p{L}\p{N}\p{Han}\s]', '', text, flags=regex.UNICODE)
    cleaned_text = ' '.join(cleaned_text.split())
    words = jieba.lcut(cleaned_text)
    return ' '.join(words)


# 加载数据
with open('toutiao_cat_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line1 = line.strip().split('_!_')
        if len(line1) >= 3:
            row = line1 + [''] * (len(columns) - len(line1))
            row = row[:len(columns)]
            if row[1] in {'101', '112', '113', '114', '115'}:
                row[1] = '100'
            data.append(row)

# 转换格式，清理文本
df = pd.DataFrame(data, columns=columns)
df['text'] = df.apply(lambda x: preprocess(x['title'], x['keywords']), axis=1)
# df['text'] = [preprocess(title, keywords) for title, keywords in zip(df['title'], df['keywords'])]
df['label_ft'] = '__label__' + df['news_id'].astype(str)

# 5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# 用于存储每折的结果
accuracies = []
f1_scores = []

for fold, (train_index, test_index) in enumerate(skf.split(df, df['news_id']), 1):
    print(f'正在处理第{fold}折...')
    # 划分训练集和测试集
    train_df, test_df = df.iloc[train_index], df.iloc[test_index]

    # 训练模型
    train_df[['label_ft', 'text']].to_csv('train.csv', index=False, header=False, sep='\t', encoding='utf-8')

    model = fasttext.train_supervised(
        'train.csv', lr=0.8, wordNgrams=3, minCount=2, minn=2, maxn=6, epoch=40, loss='hs')

    # 测试模型
    predictions = [model.predict(text)[0][0] for text in test_df['text']]

    accuracy = accuracy_score(test_df['label_ft'], predictions)
    f1 = f1_score(test_df['label_ft'], predictions, average='macro')
    print(f'第{fold}折 - 准确率: {accuracy:.4f}, f1分数: {f1:.4f}')
    # model.save_model(f'model_fold_{fold}.bin')

    # 保存结果
    accuracies.append(accuracy)
    f1_scores.append(f1)

print(f'平均准确率: {sum(accuracies) / len(accuracies):.4f}')
print(f'平均f1分数: {sum(f1_scores) / len(f1_scores):.4f}')
