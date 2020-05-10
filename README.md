# ProbeGrammarRobustness
Source code for our ACL2020 paper: **On the Robustness of Language Encoders against Grammatical Errors**

## Dependencies
Python >= 3.5

Download and install [berkeleyparser](https://github.com/slavpetrov/berkeleyparser).

Install python requirments via requirments file: <code> pip install -r requirements.txt </code>

## Preparation
### Download datasets
The [General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com) benchmark aims to analyze model ability in natural language understanding. We use some tasks of GLUE as downstream tasks.

You should follow the instructions in [this repo](https://github.com/nyu-mll/GLUE-baselines) to download GLUE benchmark and unpack it to your <code> $data_dir</code>. 

The [CoNLL-2014 Shared Task: Grammatical Error Correction](comp.nus.edu.sg/~nlp/conll14st.html) is where we collect error statistics.

Follow the instructions in [this page](comp.nus.edu.sg/~nlp/conll14st.html)
to download NUCLE Release3.2 and annotated test data.

Remember to change the file path in line 13 and 141 of <code> utils/statistic_all.py</code> to your own path.

### Download pre-trained models
For experiments regarding Infersent, you need to download fastText embeddings and the corresponding pre-trained Infersent model.

<pre><code>curl -Lo crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
curl -Lo examples/infersent2.pkl https://dl.fbaipublicfiles.com/senteval/infersent/infersent2.pkl
</code></pre>

## Usage
### Downstream task evaluations
The framework in this repo allows evaluating BERT, RoBERTa, and Infersent on MRPC, MNLI, QNLI, and SST-2. We will provide an example of evaluating <code>bert-base-uncased</code> on MRPC dataset.

The follows can be done with
<code> bash run_trans_case.sh </code>. But let's elaborate it step by step. The script for Infersent is <code> run_infer_case.sh </code>.

First train or fine-tune models on clean data (<code>${data_dir}</code> indicates where you store clean data):

<pre><code>python run_transformers.py --mode fine-tune --target_model bert --model_name_or_path bert-base-uncased --do_lower_case --data_dir ${data_dir}/MRPC --data_sign MRPC
</code></pre>

To inject grammatical errors using adversarial attack algorithms, you need to assign importance scores to each token (not necessary for genetic algorithm):
<pre><code>python run_transformers.py --mode score --target_model bert --model_name_or_path bert-base-uncased --do_lower_case --data_dir ${data_dir}/MRPC --data_sign MRPC
</code></pre>

then, run the attack algorithms, <code>--adv_type</code> can be <code>greedy</code>, <code>beam_search</code> or <code>genetic</code>:
<pre><code>python run_transformers.py --mode attack  --adv_type greedy --target_model bert --model_name_or_path bert-base-uncased --do_lower_case --data_dir ${data_dir}/MRPC --data_sign MRPC
</code></pre>

To inject grammatical errors based on [berkeleyparser](https://github.com/slavpetrov/berkeleyparser) (the probabilistic case in our paper), you need to first obtain syntactic parse tree for each sentence in the dataset. Then run:
<pre><code>python generate_error_sent_all.py csv --input_tsv ${data_dir}/MRPC/dev.tsv --parsed_sent1 ${data_dir}/MRPC/parsed_sent1 --parsed_sent2 ${data_dir}/MRPC/parsed_sent2 --output_tsv ${data_dir}/MRPC/mrpc.tsv --rate 0.15
</code></pre>

then, test the model under probabilistic case:
<pre><code>python run_transformers.py --mode attack --adv_type random --random_attack_file ${data_dir}/MRPC/mrpc.tsv --target_model bert --model_name_or_path bert-base-uncased --do_lower_case --data_dir ${data_dir}/MRPC --data_sign MRPC 
</code></pre>

Note that our framework is flexible. If you want to test new models, you can simply add a new class in <code> attack_agent.py </code> like what we did ( See <code> attack_agent.py </code> for detials, the new class mainly tells attack algorithms how to construct and forward a new instance):
<pre><code>class infersent_enc(object):
    def __init__(self, infersent, config):
        self.infersent = infersent
        self.config = config

    def make_instance(self, text_a, text_b, label, label_map):
        sent1s = [' '.join(text_a)]
        if isinstance(text_b, list):
            sent2s = [' '.join(text_b)]
        else:
            sent2s = [text_b]
        return [sent1s, sent2s, [label]]

    def model_forward(self, model, batch):
        sent1s, sent2s, label_ids = [list(item) for item in batch]
        sent1_tensor = self.infersent.encode(sent1s, tokenize=True)
        sent2_tensor = self.infersent.encode(sent2s, tokenize=True)
        ...
        return logits
</code></pre>


### Model layer evaluations
### BERT masked language model evaluations

## Acknowledgement
Our framework is developed based on [PyTorch](https://github.com/pytorch/pytorch) implementations of BERT and RoBERTa from [PyTorch-Transformers](https://github.com/huggingface/transformers), Infersent from [SentEval](https://github.com/facebookresearch/SentEval), and ELMo from [AllenNLP](https://github.com/allenai/allennlp) and [Jiant](https://github.com/nyu-mll/jiant).

We also borrowed and edited code from the following repos:
[nlp_adversarial_examples](https://github.com/nesl/nlp_adversarial_examples), 
[nmt_grammar_noise](https://bitbucket.org/antonis/nmt-grammar-noise), 
[interpret_bert](https://github.com/ganeshjawahar/interpret_bert).

We would like to thank the authors of these repos for their efforts.
## Citation
If you find our work useful, please cite our ACL2020 paper:
**On the Robustness of Language Encoders against Grammatical Errors**
<pre><code>
@inproceedings{yin2020robustnest,
  author = {Yin, Fan and Long, Quanyu and Meng, Tao and Chang, Kai-Wei},
  title = {On the Robustness of Language Encoders against Grammatical Errors},
  booktitle = {ACL},
  year = {2020}
}
</code></pre>
