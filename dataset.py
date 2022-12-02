import torch
import time
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler
from counting_utils import gen_counting_label


class HMERDataset(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True, use_aug=False):
        super(HMERDataset, self).__init__()
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = pkl.load(f)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params
        self.reverse_color = self.params['data_process']['reverse_color'] if 'data_process' in params else False
        self.equal_range = self.params['data_process']['equal_range'] if 'data_process' in params else False

        with open(self.params['matrix_path'], 'rb') as f:
            matrix = pkl.load(f)
        self.matrix = torch.Tensor(matrix)
        
    def __len__(self):
        # assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]

        image = torch.Tensor(image)
        if self.reverse_color:
            image = 255 - image
        if self.equal_range:
            image = (image / 255 - 0.5) * 2
        else:
            image = image / 255

        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        words = self.words.encode(labels) + [0]
        words = torch.LongTensor(words)
        return image, words
    
    def gen_matrix(self, labels):
        (B, L), device = labels.shape, labels.device
        matrix = []
        for i in range(B):
            _L = []
            label = labels[i]
            for x in range(L):
                _T = []
                for y in range(L):
                    if x == y:
                        _T.append(1.)
                    else:
                        if label[x] == label[y] or label[x] == 0 or label[y] == 0:
                            _T.append(0.)
                        else:
                            _T.append(self.matrix[label[x], label[y]])
                _L.append(_T)
            matrix.append(_L)
        matrix = torch.tensor(matrix).to(device)
        return matrix.detach()
    
    def collate_fn(self, batch_images):
        max_width, max_height, max_length = 0, 0, 0
        batch, channel = len(batch_images), batch_images[0][0].shape[0]
        proper_items = []
        for item in batch_images:
            if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
                continue
            max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
            max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
            max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
            proper_items.append(item)

        images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros(
            (len(proper_items), 1, max_height, max_width))
        labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros(
            (len(proper_items), max_length))

        for i in range(len(proper_items)):
            _, h, w = proper_items[i][0].shape
            images[i][:, :h, :w] = proper_items[i][0]
            image_masks[i][:, :h, :w] = 1
            l = proper_items[i][1].shape[0]
            labels[i][:l] = proper_items[i][1]
            labels_masks[i][:l] = 1
        matrix = self.gen_matrix(labels)
        counting_labels = gen_counting_label(labels, self.params['counting_decoder']['out_channel'], True)
        return images, image_masks, labels, labels_masks, matrix, counting_labels


def get_crohme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['words'] = words
    print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words)
    eval_dataset_2014 = HMERDataset(params, params['eval_image_path'],
                                                                   params['eval_label_path'], words, is_train=False)
    eval_dataset_2016 = HMERDataset(params, params['16_eval_image_path'],
                                                                   params['16_eval_label_path'], words, is_train=False)
    eval_dataset_2019 = HMERDataset(params, params['19_eval_image_path'],
                                                                   params['19_eval_label_path'], words, is_train=False)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler_2014 = RandomSampler(eval_dataset_2014)
    eval_sampler_2016 = RandomSampler(eval_dataset_2016)
    eval_sampler_2019 = RandomSampler(eval_dataset_2019)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=train_dataset.collate_fn, pin_memory=True)
    eval_loader_2014 = DataLoader(eval_dataset_2014, batch_size=1, sampler=eval_sampler_2014,
                              num_workers=params['workers'], collate_fn=eval_dataset_2014.collate_fn, pin_memory=True)
    eval_loader_2016 = DataLoader(eval_dataset_2016, batch_size=1, sampler=eval_sampler_2016,
                                  num_workers=params['workers'], collate_fn=eval_dataset_2016.collate_fn, pin_memory=True)
    eval_loader_2019 = DataLoader(eval_dataset_2019, batch_size=1, sampler=eval_sampler_2019,
                                  num_workers=params['workers'], collate_fn=eval_dataset_2019.collate_fn,
                                  pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'2014 eval dataset: {len(eval_dataset_2014)} eval steps: {len(eval_loader_2014)} '
          f'2016 eval dataset: {len(eval_dataset_2016)} eval steps: {len(eval_loader_2016)} '
          f'2019 eval dataset: {len(eval_dataset_2019)} eval steps: {len(eval_loader_2019)}')

    return train_loader, eval_loader_2014, eval_loader_2016, eval_loader_2019


class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label

