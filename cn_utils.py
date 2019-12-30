import pandas as pd
import json
import os
import jieba
from tqdm import tqdm
import tencent_translate
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException


def prepare_flickr8k_cn():
    cn_captions = pd.read_csv('./data/flickr8k/flickr8kzhc.caption.txt', sep='[ #]', header=None,
                              names=['path', 'lang', 'cap_id', 'cap'])
    names = os.listdir('./data/flickr8k/Flickr_Data/Images')
    not_found = []
    # for name in names:
    #     if name not in list(cn_captions.path):
    #         not_found.append(name)
    # for name in cn_captions.path: # x.jpg is not found in files
    #     if name not in names:
    #         not_found.append(name)
    #
    # print(len(not_found))
    karpathy_json_path = './data/caption_datasets/dataset_flickr8k.json'
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    imgs = data['images']
    # list of dict of images with their captions, with keys (sentids, ingid, sentences, split, filename)
    # sentences is a list of dict of each caption with tokens ...
    for img in tqdm(imgs):  # about 5min for flickr8k
        sentences = img['sentences']
        for sentence in sentences:
            target = cn_captions.loc[(cn_captions.path == img['filename']) & (cn_captions.cap_id ==
                                                                              sentence['sentid'] % 5)].cap.to_numpy()[0]
            target = target if target[-1] != '。' else target[:-1]
            cut_target = jieba.lcut(target)
            sentence['raw'] = target
            sentence['tokens'] = cut_target

    with open('./data/caption_datasets/dataset_flickr8k_cn.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False)


def translate_flickr30k(start_sentid=0, karpathy_json_path='./data/caption_datasets/dataset_flickr30k.json'):
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    imgs = data['images']
    try:
        for img in tqdm(imgs):  # about 5min for flickr8k
            sentences = img['sentences']
            for sentence in sentences:
                if sentence['sentid'] < start_sentid:
                    continue
                raw = tencent_translate.translate(sentence['raw'])  # exception might happen here!
                target = raw[:-1] if raw[-1] in '。？，' else raw
                cut_target = jieba.lcut(target)
                sentence['raw'] = target
                sentence['tokens'] = cut_target
    except TencentCloudSDKException:
        print(f"\nOut of free amount, stop at sentence {sentence['sentid']}\n")

    with open(f"./data/caption_datasets/dataset_flickr30k_cn_at_sent_{sentence['sentid']}.json", 'w') as f:
        json.dump(data, f, ensure_ascii=False)
    pass


def check_translated_file(karpathy_json_path='./data/caption_datasets/dataset_flickr30k_cn_at_sent_155069.json'):
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    imgs = data['images']

    for img in tqdm(imgs):
        sentences = img['sentences']
        for sentence in sentences:
            raw = sentence['raw']
            if sentence['sentid'] == 82418:
                print(raw)

            # target = filter(lambda c: '\u4e00' <= c <= '\u9fa5' or c in '，：！', raw)
            # cut_target = jieba.lcut(target)


if __name__ == '__main__':
    # translate_flickr30k(start_sentid=155070,
    #                     karpathy_json_path='./data/caption_datasets/dataset_flickr30k_cn_at_sent_155069.json')
    check_translated_file()
