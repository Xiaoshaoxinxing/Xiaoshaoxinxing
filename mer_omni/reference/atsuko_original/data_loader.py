import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
__all__ = ['MMDataLoader']
logger = logging.getLogger('Atsuko')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'mintrec': self.__init_mintrec,
            'sims': self.__init_mosi,
            'sims_v2': self.__init_mosi,
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_mosi(self):
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)
        if 'use_bert' in self.args and self.args['use_bert']:
            self.text = data[self.mode]['text_bert'].astype(np.float32)  # BERT feature
        else:
            self.text = data[self.mode]['text'].astype(np.float32)  # GLOVE feature
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        if self.args['feature_T'] != "":
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if 'use_bert' in self.args and self.args['use_bert']:
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]
        if self.args['feature_A'] != "":
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]
        if self.args['feature_V'] != "":
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        self.labels = {
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args['need_data_aligned']:
            if self.args['feature_A'] != "":
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            else:
                self.audio_lengths = data[self.mode]['audio_lengths']
            if self.args['feature_V'] != "":
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __truncate(self):
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
        
        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    
    # def __init_mintrec(self):
    #     """Load MIntRec (multimodal intent classification) features from a processed pkl.

    #     Expected pkl structure:
    #         data['train'/'valid'/'test'] contains:
    #             - 'text_bert': (N, 3, seq_len)  # input_ids, input_mask, segment_ids
    #             - 'audio': (N, T, D_a)
    #             - 'vision': (N, T, D_v)
    #             - 'raw_text': list[str]
    #             - 'id': list[str] or list[int]
    #             - 'intent_labels': (N,) int64 in [0, num_classes-1]
    #     """
    #     with open(self.args['featurePath'], 'rb') as f:
    #         data = pickle.load(f)

    #     split = self.mode
    #     # Some datasets use 'dev' instead of 'valid'
    #     if split not in data and split == 'valid' and 'dev' in data:
    #         split = 'dev'

    #     if 'use_bert' in self.args and self.args['use_bert']:
    #         self.text = data[split]['text_bert'].astype(np.float32)
    #     else:
    #         # If you don't want BERT, store 'text' (e.g., GloVe features) in the pkl
    #         self.text = data[split]['text'].astype(np.float32)

    #     self.vision = data[split]['vision'].astype(np.float32)
    #     self.audio = data[split]['audio'].astype(np.float32)
    #     self.raw_text = data[split].get('raw_text', [''] * len(self.audio))
    #     self.ids = data[split].get('id', list(range(len(self.audio))))

    #     # Classification labels
    #     label_key = self.args.get('label_key', 'intent') if isinstance(self.args, dict) else getattr(self.args, 'label_key', 'intent')
    #     labels_arr = data[split].get('intent_labels', data[split].get('labels', None))
    #     if labels_arr is None:
    #         raise KeyError("MIntRec pkl must contain 'intent_labels' (or 'labels') under each split.")
    #     self.labels = {label_key: np.array(labels_arr).astype(np.int64)}

    #     logger.info(f"{self.mode} samples: {self.labels[label_key].shape}")
    def __init_mintrec(self):
        """Load MIntRec features from processed pkl.
    
        Expected under each split:
          - text_bert: (N,3,L)
          - audio: (N,T_a,D_a)
          - vision: (N,T_v,D_v)
          - intent_labels: (N,)
        Optional (for need_data_aligned=False):
          - audio_lengths: (N,)
          - vision_lengths: (N,)
        """
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)
    
        split = self.mode
        if split not in data and split == 'valid' and 'dev' in data:
            split = 'dev'
    
        if 'use_bert' in self.args and self.args['use_bert']:
            self.text = data[split]['text_bert'].astype(np.float32)
        else:
            self.text = data[split]['text'].astype(np.float32)
    
        self.vision = data[split]['vision'].astype(np.float32)
        self.audio = data[split]['audio'].astype(np.float32)
        self.raw_text = data[split].get('raw_text', [''] * len(self.audio))
        self.ids = data[split].get('id', list(range(len(self.audio))))
    
        label_key = self.args.get('label_key', 'intent') if isinstance(self.args, dict) else getattr(self.args, 'label_key', 'intent')
        labels_arr = data[split].get('intent_labels', data[split].get('labels', None))
        if labels_arr is None:
            raise KeyError("MIntRec pkl must contain 'intent_labels' (or 'labels') under each split.")
        self.labels = {label_key: np.array(labels_arr).astype(np.int64)}
    
        # If unaligned training is enabled, require lengths in pkl
        if (not self.args['need_data_aligned']):
            self.audio_lengths = data[split].get('audio_lengths', None)
            self.vision_lengths = data[split].get('vision_lengths', None)
            if self.audio_lengths is None or self.vision_lengths is None:
                raise KeyError(
                    "need_data_aligned=False for MIntRec requires 'audio_lengths' and 'vision_lengths' in the pkl. "
                    "Please generate unaligned_pad.pkl with --save_unaligned_pad."
                )
            self.audio_lengths = list(self.audio_lengths)
            self.vision_lengths = list(self.vision_lengths)
    
        logger.info(f"{self.mode} samples: {self.labels[label_key].shape}")

        # ===== 新增：支持 unaligned 模式 =====
        if not self.args['need_data_aligned']:
            # 若 pkl 中有 audio_lengths / vision_lengths，使用之；否则取实际长度
            if 'audio_lengths' in data[split]:
                self.audio_lengths = list(data[split]['audio_lengths'])
            else:
                self.audio_lengths = [self.audio.shape[1]] * len(self.audio)
            if 'vision_lengths' in data[split]:
                self.vision_lengths = list(data[split]['vision_lengths'])
            else:
                self.vision_lengths = [self.vision.shape[1]] * len(self.vision)
        # 清理无效值
        self.audio[self.audio == -np.inf] = 0
        self.audio[np.isnan(self.audio)] = 0
        self.vision[np.isnan(self.vision)] = 0
        # ===== 新增结束 =====


    def __len__(self):
        return len(next(iter(self.labels.values())))

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': self._get_label_tensors(index)
        }
        
        # 模态缺失处理：根据 modality_mask 配置对特征置零
        # 只有在存在被禁用的模态时才执行，全模态时完全跳过
        modality_mask = self.args.get('modality_mask', {})
        use_text = modality_mask.get('text', True) if isinstance(modality_mask, dict) else True
        use_audio = modality_mask.get('audio', True) if isinstance(modality_mask, dict) else True
        use_vision = modality_mask.get('vision', True) if isinstance(modality_mask, dict) else True
        has_masked_modality = not (use_text and use_audio and use_vision)
        
        if has_masked_modality:
            if not use_text:
                sample['text'] = torch.zeros_like(sample['text'])
            if not use_audio:
                sample['audio'] = torch.zeros_like(sample['audio'])
            if not use_vision:
                sample['vision'] = torch.zeros_like(sample['vision'])
        
        if (not self.args['need_data_aligned']) and hasattr(self, "audio_lengths") and hasattr(self, "vision_lengths"):
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample

    def _get_label_tensors(self, index):
        labels = {}
        for k, v in self.labels.items():
            arr = v[index].reshape(-1)
            # Preserve dtype for classification; otherwise use float
            if getattr(self.args, 'train_mode', 'regression') == 'classification':
                labels[k] = torch.from_numpy(arr).long() if isinstance(arr, np.ndarray) else torch.tensor(arr).long()
            else:
                labels[k] = torch.from_numpy(arr).float() if isinstance(arr, np.ndarray) else torch.tensor(arr).float()
        return labels


def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader
