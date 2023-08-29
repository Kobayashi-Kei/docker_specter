from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
import sys
sys.path.append("..")  # 親ディレクトリのパスを追加
from pretrain_average_pooling_label_entire import MyData, Specter
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import unittest
import json, glob, argparse

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"



batch_size = 2
num_workers = 1
max_length = 512




class MyTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        path = "/workspace/dataserver/axcell/large/specter/test_triple.json"
        with open(path, 'r') as f:
            data = json.load(f)

        path = "/workspace/dataserver/axcell/large/paperDict.json"
        with open(path, 'r') as f:
            paper_dict = json.load(f)

        path = "/workspace/dataserver/axcell/large/result_ssc.json"
        with open(path, 'r') as f:
            ssc_result_dict = json.load(f)

        self.label_dict = {
            "title": 1,
            "bg": 2,
            "obj": 3,
            "method": 4,
            "res": 5,
            "other": 6,
        }
        self.num_label_dict = {
            "1": "title",
            "2": "bg",
            "3": "obj",
            "4": "method",
            "5": "res",
            "6": "other"
        }

        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
        self.mydata = MyData(data, paper_dict, ssc_result_dict,
                             self.label_dict, self.tokenizer)
        self.loader = DataLoader(self.mydata, batch_size=batch_size, num_workers=num_workers,
                                 shuffle=False, pin_memory=False)
        args = {
            "checkpoint_path": "",
            "seqlen": "",
            "label_dict": self.label_dict,
            "num_label_dict": self.num_label_dict,
            "margin": 1
        }
        self.specter = Specter(args)
        self.specter.debug = True

    def setUp(self):
        pass

    def test_batch_form(self):
        """バッチが source, pos ,negを含んでいることのテスト
        """
        for batch in self.loader:
            self.assertEqual(len(batch), 3)

    def test_batch_size(self):
        """source, pos, negがそれぞれバッチサイズと一致することのテスト
        """
        for batch in self.loader:
            # print(batch)
            self.assertEqual(batch[0]['input']
                             ['input_ids'].size(), torch.Size([batch_size, max_length]))
            self.assertEqual(batch[1]['input']
                             ['input_ids'].size(), torch.Size([batch_size, max_length]))
            self.assertEqual(batch[2]['input']
                             ['input_ids'].size(), torch.Size([batch_size, max_length]))
            self.assertEqual(batch[0]['input']
                             ['attention_mask'].size(), torch.Size([batch_size, max_length]))
            self.assertEqual(batch[1]['input']
                             ['attention_mask'].size(), torch.Size([batch_size, max_length]))
            self.assertEqual(batch[2]['input']
                             ['attention_mask'].size(), torch.Size([batch_size, max_length]))

    def test_tokenize_with_label(self):
        t1 = "Reading Wikipedia to Answer Open-Domain Questions"
        a1 = "This paper proposes to tackle open- domain question answering using Wikipedia as the unique knowledge source: the answer to any factoid question is a text span in a Wikipedia article. This task of machine reading at scale combines the challenges of document retrieval (finding the relevant articles) with that of machine comprehension of text (identifying the answer spans from those articles). Our approach combines a search component based on bigram hashing and TF-IDF matching with a multi-layer recurrent neural network model trained to detect answers in Wikipedia paragraphs. Our experiments on multiple existing QA datasets indicate that (1) both modules are highly competitive with respect to existing counterparts and (2) multitask learning using distant supervision on their combination is an effective complete system on this challenging task."
        l1 = [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ta1 = t1 + self.tokenizer.sep_token + a1
        self.assertEqual(self.mydata.tokenize_with_label(
            ta1, self.mydata.ssc_result_dict[t1], self.tokenizer)[1], l1)

        t2 = "Towards Corner Case Detection for Autonomous Driving"
        a2 = "The progress in autonomous driving is also due to the increased availability of vast amounts of training data for the underlying machine learning approaches. Machine learning systems are generally known to lack robustness, e.g., if the training data did rarely or not at all cover critical situations. The challenging task of corner case detection in video, which is also somehow related to unusual event or anomaly detection, aims at detecting these unusual situations, which could become critical, and to communicate this to the autonomous driving system (online use case). Such a system, however, could be also used in offline mode to screen vast amounts of data and select only the relevant situations for storing and (re)training machine learning algorithms. So far, the approaches for corner case detection have been limited to videos recorded from a fixed camera, mostly for security surveillance. In this paper, we provide a formal definition of a corner case and propose a system framework for both the online and the offline use case that can handle video signals from front cameras of a naturally moving vehicle and can output a corner case score."
        l2 = [-1, 1, 1, 1, 1, 1, 1, 1, -1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ta2 = t2 + self.tokenizer.sep_token + a2
        self.assertEqual(self.mydata.tokenize_with_label(
            ta2, self.mydata.ssc_result_dict[t2], self.tokenizer)[1], l2)

        # label positionが一致しているか目視で検査するコード
        # self.validation_label_position(ta1, l1)
        # self.validation_label_position(ta2, l2)

    def validation_label_position(self, ta, l):
        ta_tokenize = self.tokenizer(ta,
                                     padding=True,
                                     truncation=True,
                                     return_tensors="pt",
                                     max_length=512
                                     )['input_ids'][0]
        for i, t in enumerate(ta_tokenize):
            print(self.tokenizer.decode(t), l[i])
    
    def test_label_pooling(self):
        batch_last_hidden_state = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                                                [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
        batch_position_label_list = torch.tensor([[2,3,3,2], [1,1,4,4]])
        label_pooling = self.specter.label_pooling(
            batch_last_hidden_state, batch_position_label_list)
        expected = [{'title': None, 'bg': torch.tensor([1., 2., 3.]), 'obj': torch.tensor([1., 1.5, 2.]), 'method': None, 'res': None, 'other': None}, {
            'title': torch.tensor([1., 1.5, 2.]), 'bg': None, 'obj': None, 'method': torch.tensor([1., 2., 3.]), 'res': None, 'other': None}]
        print(expected)
        print(label_pooling)
        for key in expected:
            if expected[key] is not None and isinstance(expected[key], torch.Tensor):
                self.assertTensorEqual(expected[key], label_pooling[key])
            else:
                self.assertEqual(expected[key], label_pooling[key])
        self.assertEqual(expected, label_pooling)
    
    def test_training_step(self):
        for batch in self.loader:
            loss = self.specter.training_step(batch, 0)
            print(loss)
            # self.assertEqual(1, loss['loss'].size())
            break
    
    def assertTensorEqual(self, tensor1, tensor2):
        self.assertTrue(torch.equal(tensor1, tensor2))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default=None,
                        help='path to the model (if not setting checkpoint)')
    parser.add_argument('--method')
    parser.add_argument('--margin', default=1)
    parser.add_argument('--version', default=0)
    parser.add_argument('--input_dir', default=None,
                        help='optionally provide a directory of the data and train/test/dev files will be automatically detected')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--seed', default=1918, type=int)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--test_checkpoint', default=None)
    parser.add_argument('--limit_test_batches', default=1.0, type=float)
    parser.add_argument('--limit_val_batches', default=1.0, type=float)
    parser.add_argument('--val_check_interval', default=1.0, type=float)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="kwarg passed to DataLoader")
    parser.add_argument("--adafactor", action="store_true")
    parser.add_argument('--save_dir', required=True)

    parser.add_argument('--num_samples', default=None, type=int)
    parser.add_argument("--lr_scheduler",
                        default="linear",
                        choices=arg_to_scheduler_choices,
                        metavar=arg_to_scheduler_metavar,
                        type=str,
                        help="Learning rate scheduler")
    args = parser.parse_args()

    if args.input_dir is not None:
        files = glob.glob(args.input_dir + '/*')
        for f in files:
            fname = f.split('/')[-1]
            if 'train' in fname:
                args.train_file = f
            elif 'dev' in fname or 'val' in fname:
                args.dev_file = f
            elif 'test' in fname:
                args.test_file = f

    return args

if __name__ == '__main__':
    unittest.main()
