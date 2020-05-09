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
<pre><code>
curl -Lo crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
curl -Lo examples/infersent2.pkl https://dl.fbaipublicfiles.com/senteval/infersent/infersent2.pkl
</code></pre>

## Usage
### Downstream task evaluations
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
