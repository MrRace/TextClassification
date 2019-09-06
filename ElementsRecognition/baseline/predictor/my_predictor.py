import json
import jieba
import numpy as np
from sklearn.externals import joblib
"""
在第2阶段数据(数据量较大）上训练后，用第1阶段的数据集进行评估，评估结果与线上比较接近。所以，第2阶段数据整体作为训练集，而dev set
采用第1阶段的数据集？？
"""

class Predictor(object):
    def __init__(self, model_dir):
        self.tfidf = joblib.load(model_dir + 'tfidf.model')
        self.tag = joblib.load(model_dir + 'tag.model')
        self.batch_size = 1

        self.cut = jieba

    def predict_tag(self, vec):
        y = self.tag.predict(vec)#此时的y的类型是csc_matrix
        # y = y.toarray().astype(np.int32)#此时可能需要做转换，也可能不需要。比如使用决策树时候，就不需要
        indexs = np.where(y == 1)
        # print(indexs)
        if len(indexs[0]) > 0:
            # print(indexs[0])
            temp = []
            for i in indexs[1]:
                temp.append(i + 1)
            return temp
        else:
            return []



    def predict(self, content):
        fact = ' '.join(self.cut.cut(content))
        vec = self.tfidf.transform([fact])
        ans = self.predict_tag(vec)#预测结果
        # print(ans)
        return ans


def generate_pred_file(tags_list, prd, inf_path, outf_path):
    with open(inf_path, 'r', encoding='utf-8') as inf, open(
            outf_path, 'w', encoding='utf-8') as outf:
        for line in inf.readlines():
            pre_doc = json.loads(line)
            predict_doc = []
            for ind in range(len(pre_doc)):
                pred_sent = pre_doc[ind]
                pre_content = pre_doc[ind]['sentence']
                pred_label = prd.predict(pre_content)#这里的label是纯数字
                label_names = []
                for label in pred_label:
                    label_names.append(tags_list[label - 1])
                pred_sent['labels'] = label_names
                predict_doc.append(pred_sent)
            json.dump(predict_doc, outf, ensure_ascii=False)
            outf.write('\n')

if __name__ == '__main__':
    # 生成labor领域的预测文件
    print('predict_labor...')
    tags_list = []
    with open('../../data/labor/tags.txt', 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())
    prd = Predictor('model_labor/')
    inf_path = '../../data/labor/data_small_selected.json'
    outf_path = '../../output/labor_output.json'
    generate_pred_file(tags_list, prd, inf_path, outf_path)

    # 生成divorce领域的预测文件
    print('predict_divorce...')
    tags_list = []
    with open('../../data/divorce/tags.txt', 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())
    prd = Predictor('model_divorce/')
    inf_path = '../../data/divorce/data_small_selected.json'
    outf_path = '../../output/divorce_output.json'
    generate_pred_file(tags_list, prd, inf_path, outf_path)

    # 生成loan领域的预测文件
    print('predict_loan...')
    tags_list = []
    with open('../../data/loan/tags.txt', 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())
    prd = Predictor('model_loan/')
    inf_path = '../../data/loan/data_small_selected.json'
    outf_path = '../../output/loan_output.json'
    generate_pred_file(tags_list, prd, inf_path, outf_path)
