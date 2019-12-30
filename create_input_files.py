from utils import create_input_files
import argparse

if __name__ == '__main__':
    # Create input files (along with word map)
    parser = argparse.ArgumentParser(description='Prepare data for training imaging captioning model.')

    parser.add_argument('--dataset_name', '-n', default='flickr30k', help='数据集名称')
    parser.add_argument('--karpathy_json_path', '-p', default='./data/caption_datasets/dataset_flickr30k_cn.json',
                        help='字幕json文件，使用karpathy格式')
    parser.add_argument('--image_folder', '-i', default='./data/flickr30k/flickr30k_images', help='图片文件夹位置')
    parser.add_argument('--captions_per_image', '-c', default=5, type=int, help='每张图片抽取的字幕数量')
    parser.add_argument('--min_word_freq', '-f', default=5, type=int, help='加入词汇表的词的最小出现频率')
    parser.add_argument('--output_folder', '-o', default='./data/flickr30k_output_5_min_cn', help='预处理后输出文件的目录')
    parser.add_argument('--max_len', '-l', default=50, type=int, help='字幕最长单词数')

    args = parser.parse_args()

    create_input_files(dataset=args.dataset_name,
                       karpathy_json_path=args.karpathy_json_path,
                       image_folder=args.image_folder,
                       captions_per_image=args.captions_per_image,
                       min_word_freq=args.min_word_freq,
                       output_folder=args.output_folder,
                       max_len=args.max_len)
