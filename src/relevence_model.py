import torch
import pandas as pd
import torch.nn.functional as F
from torch.nn import Linear, ReLU

from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


class QADataset(Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path, sep='\t')
        data = data.sample(frac=1, random_state=42).reset_index()
        pos_data = data[data['Label']==1]
        neg_data = data[data['Label']==0][:len(pos_data)]
        data = pd.concat([pos_data, neg_data], axis=0)
        data = data.sample(frac=1, random_state=42).reset_index()
        # print(data)
        questions = data['Question'].to_list()
        sentences = data['Sentence'].to_list()
        self.qa_pair = [*zip(questions, sentences)]
        self.label = data['Label'].to_list()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.qa_pair[index], self.label[index]


class QABert(LightningModule):
    def __init__(self):
        super(QABert, self).__init__()
        self.basic_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.basic_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sim_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.linear_layer1 = Linear(384, 64)
        self.linear_layer2 = Linear(64, 2)
        self.FC = Linear(7, 2)

    def get_sentence_embeddings(self, sentences):
        sentences = [sent for pair in sentences for sent in pair ]
        #Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


        # Tokenize sentences
        encoded_input = self.sim_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.sim_model(**encoded_input)
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


    def forward(self, x):
        x = [*zip(*x)]
        # context similarity
        basic_model_tokens = self.basic_tokenizer(text=x, padding=True, truncation=True, return_tensors='pt')
        basic_model_out = self.basic_model(**basic_model_tokens).logits
        # sentence embedding
        sentence_embeddings = self.get_sentence_embeddings(x)
        # cosine simlarity
        sentence_embedding_pairs = sentence_embeddings.reshape(-1, 2, 384).tolist()
        sentence_embedding_pair1 = torch.tensor([pair[0] for pair in sentence_embedding_pairs])
        sentence_embedding_pair2 = torch.tensor([pair[1] for pair in sentence_embedding_pairs])
        cos_sim = F.cosine_similarity(sentence_embedding_pair1, sentence_embedding_pair2).reshape(-1, 1)
        # compress sentence embedding
        sentence_embeddings = F.gelu(self.linear_layer1(sentence_embeddings))
        sentence_embeddings = F.gelu(self.linear_layer2(sentence_embeddings))
        sentence_embeddings = sentence_embeddings.reshape(-1,4)
        sentence_embeddings = F.softmax(sentence_embeddings)
        features = torch.concat((basic_model_out, cos_sim, sentence_embeddings), 1)
        out = F.softmax(self.FC(features))
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    # def training_epoch_end(self, outputs):
    #     # epoch_metric = torch.stack([x for x in outputs])
    #     # print(epoch_metric)
    #     print(outputs)

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, sync_dist=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, sync_dist=True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            dirpath='./ckpt_model',
            filename='sample-demo-epoch{epoch:02d}-val_loss{val_loss:.2f}',
            auto_insert_metric_name=False
        )
        return [early_stop, checkpoint]


class QAInference:
    def __init__(self, model_path) -> None:
        self.model = QABert.load_from_checkpoint(model_path)
        self.model.eval()
    
    def inference(self, x):
        with torch.no_grad():
            out = self.model(x)
            out = [i[1] for i in out]
            return out


if __name__ == "__main__":
    qaInference = QAInference('ckpt_model/sample-demo-epoch03-val_loss0.57.ckpt')
    x = [['how are glacier caves formed?', 'how are glacier caves formed?', "how a rocket engine works", "how are antibodies used in"], ["A partly submerged glacier cave on Perito Moreno Glacier", "A glacier cave is a cave formed within the ice of a glacier .", 'A rocket engine, or simply "rocket", is a jet engine that uses only stored propellant mass for forming its high speed propulsive jet .', 'The antibody recognizes a unique part of the foreign target, called an antigen .']]
    out = qaInference.inference(x)
    print(out)