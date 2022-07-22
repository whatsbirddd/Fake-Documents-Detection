from torch import nn
from transformers import ElectraModel,FunnelModel, BertModel, AlbertModel,AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from util import FocalLoss


class KRElectraClassificationModel(nn.Module):   
    def __init__(self, args, embed_dim=768, num_labels=2):
        super(KRElectraClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.max_len = args.maxlen
        self.embed_dim = embed_dim
        self.args = args
        
        self.model = ElectraModel.from_pretrained("snunlp/KR-ELECTRA-discriminator")
        self.classifier = nn.Linear(embed_dim,num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        #classifier
        x = outputs[0]
        #x = self.dropout(outputs[0]) #(batch_size, sequence_length, hidden_size)
        x = nn.AvgPool2d((self.max_len,1))(x).squeeze() #(batch_size, emb_size)
        x = self.classifier(x) #(batch_size,num_labels)
        return x
    
class BertClassificationModel(nn.Module):   
    def __init__(self, args, embed_dim=768, num_labels=2):
        super(BertClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.max_len = args.maxlen
        self.embed_dim = embed_dim
        self.args = args
        
        #tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        self.model = BertModel.from_pretrained("kykim/bert-kor-base")
        self.classifier = nn.Linear(embed_dim,num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #classifier
        x = outputs[0]
        #x = self.dropout(outputs[0]) #(batch_size, sequence_length, hidden_size)
        x = nn.AvgPool2d((self.max_len,1))(x).squeeze() #(batch_size, emb_size)
        x = self.classifier(x) #(batch_size,num_labels)
        return x
    
class KRElectraLstmClassifier(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional,maxlen):
        super(KRElectraLstmClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.maxlen = maxlen
        
        #tokenizer = ElectraTokenizer.from_pretrained("snunlp/KR-ELECTRA-discriminator")
        self.embeded = ElectraModel.from_pretrained("snunlp/KR-ELECTRA-discriminator") #(batch, maxlen, embed)
        self.lstm = nn.LSTM(input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional
            )
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*2),
            nn.Linear(self.hidden_size*2, self.output_size)
        )
        
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        x= self.embeded(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        z, _ = self.lstm(x) #(output) = (batch, max_len, hidden_size), (h_n,c_n) 
        #z = z[:,-1,:].squeeze(1) #last hidden state (batch, hidden_size)
        z = nn.AvgPool2d((self.maxlen,1))(z).squeeze() #(batch_size, hidden_size)
        y_pred = self.linear(z)
        return y_pred

class BertLstmClassifier(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional):
        super(BertLstmClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        #tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")
        self.embeded = BertModel.from_pretrained("kykim/bert-kor-base") #(batch, maxlen, embed)
        self.lstm = nn.LSTM(input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional
            )
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*2),
            nn.Linear(self.hidden_size*2, self.output_size)
        )  

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        x= self.embeded(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        z, _ = self.lstm(x) #(output), (h_n,c_n) 
        z = z[:,-1,:].squeeze(1) #last hidden state (batch, hidden_size)
        y_pred = self.linear(z)
        return y_pred
        
class BigBirdClassificationModel(nn.Module):   
    def __init__(self, args, embed_dim=768, num_labels=2):
        super(BigBirdClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.max_len = args.maxlen
        self.embed_dim = embed_dim
        self.args = args
        
        self.model = AutoModel.from_pretrained("monologg/kobigbird-bert-base")
        #tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")  # BertTokenizer
        self.classifier = nn.Linear(embed_dim,num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        #classifier
        x = outputs[0]
        x = nn.AvgPool2d((self.max_len,1))(x).squeeze() #(batch_size, emb_size)
        x = self.classifier(x) #(batch_size,num_labels)
        return x

class BigBirdLstmClassifier(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional):
        super(BigBirdLstmClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        self.embeded = AutoModel.from_pretrained("monologg/kobigbird-bert-base")
        #tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")  # BertTokenizer
        self.lstm = nn.LSTM(input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional
            )
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*2),
            nn.Linear(self.hidden_size*2, self.output_size)
        )

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        x= self.embeded(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        z, _ = self.lstm(x) #(output), (h_n,c_n) 
        z = z[:,-1,:].squeeze(1) #last hidden state (batch, hidden_size)
        y_pred = self.linear(z)
        return y_pred
    
class AlbertClassificationModel(nn.Module):   
    def __init__(self, args, embed_dim=768, num_labels=2):
        super(AlbertClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.max_len = args.maxlen
        self.embed_dim = embed_dim
        self.args = args
        
        self.model = BertModel.from_pretrained("kykim/bert-kor-base")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim,num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #classifier
        x = outputs[0] #(batch_size, sequence_length, hidden_size)
        x = self.dropout(x)
        x = nn.AvgPool2d((self.max_len,1))(x).squeeze() #(batch_size, emb_size)
        x = self.classifier(x) #(batch_size,num_labels)
        return x
        
class FunnelClassificationModel(nn.Module):   
    def __init__(self, args, embed_dim = 768, num_labels = 2):
        super(FunnelClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.max_len = args.maxlen
        self.embed_dim = embed_dim
        self.args = args
        
        self.model = FunnelModel.from_pretrained("kykim/funnel-kor-base")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim,num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask) #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        x = outputs[0]
        
        #classifier
        #x = self.dropout(outputs[0]) #(batch_size, sequence_length, hidden_size)
        x = nn.AvgPool2d((self.max_len,1))(x).squeeze() #(batch_size, emb_size)
        x = self.classifier(x) #calculate losses
      
        return x

