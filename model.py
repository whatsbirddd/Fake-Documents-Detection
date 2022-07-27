from torch import nn
from transformers import ElectraModel,FunnelModel, BertModel, AlbertModel,AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn.functional as F


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
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

class KRElectraLstmCnn2(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional,maxlen):
        super(KRElectraLstmCnn2, self).__init__()
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
        self.conv = ConvBlock()
        self.conv_linear = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.hidden_size*2)
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
        
        conv = self.conv(x) #(batch,out_channels*2) = (32,256)
        conv_linear = self.conv_linear(conv)#(batch, hidden_size*2)

        #z = z[:,-1,:].squeeze(1) #last hidden state (batch, hidden_size)
        z = nn.AvgPool2d((self.maxlen,1))(z).squeeze() #(batch_size, hidden_size*2)
        concat = torch.cat((conv_linear,z),1) #(batch, hidden_size*2)
        y_pred = self.linear(z)
        return y_pred
    
class KrelectraLstm2Classifier(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional):
        super(KrelectraLstm2Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        #tokenizer = ElectraTokenizer.from_pretrained("snunlp/KR-ELECTRA-discriminator")
        self.embeded = ElectraModel.from_pretrained("snunlp/KR-ELECTRA-discriminator") #(batch, maxlen, embed)
        self.lstm = nn.LSTM(input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional
            )
        self.linear1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear_out = nn.Linear(self.hidden_size*4, self.output_size)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        x= self.embeded(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        z, _ = self.lstm(x) #(output)=(batch, maxlen, hidden size*2), (h_n,c_n) 
        avg_pool = torch.mean(z,1) #(batch,hidden size*2)
        max_pool, _ = torch.max(z,1) #(batch, hidden size*2)
        
        concat = torch.cat((max_pool,avg_pool),1) #(batch, hidden size * 4)
        concat_linear1 = self.linear1(concat) #(batch, hidden_size * 4)
        concat_linear2 = self.linear2(concat) #(batch, hidden_size * 4)
        
        hidden = concat + concat_linear1 + concat_linear2 
        y_pred = self.linear_out(hidden) #(batch, 2)
        return y_pred

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.in_channels = 1
        self.out_channels = 128
        self.kernel_heights = 3
        self.stride = 3
        self.padding = 1
        self.embed_dim = 768
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels,(self.kernel_heights, self.embed_dim), self.stride, self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_heights, self.embed_dim), self.stride, self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        
    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input) # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out
    
    def forward(self, x):
        input = x.unsqueeze(1) #(Batch, 1, max_len ,embed_size)
        max_out1 = self.conv_block(input, self.conv1) #(batch, out_channels)
        max_out2 = self.conv_block(input, self.conv2) #(batch, out_channels)
        concat = torch.cat((max_out1,max_out2),1) #(batch, out_channels *2)
        return concat

class KrelectraLstmCNN(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional):
        super(KrelectraLstmCNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        #tokenizer = ElectraTokenizer.from_pretrained("snunlp/KR-ELECTRA-discriminator")
        self.embeded = ElectraModel.from_pretrained("snunlp/KR-ELECTRA-discriminator") #(batch, maxlen, embed)
        self.lstm = nn.LSTM(input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional
            )
        self.conv = ConvBlock()
        self.linear1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear_out = nn.Linear(self.hidden_size*4, self.output_size)
        

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        x= self.embeded(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        z, _ = self.lstm(x) #(output)=(batch, maxlen, hidden size*2), (h_n,c_n) 
        avg_pool = torch.mean(z,1) #(batch,hidden size*2)
        max_pool, _ = torch.max(z,1) #(batch, hidden size*2)
        
        conv = self.conv(x) #(batch,out_channels*2) = (32,256)
        
        concat_lstm = torch.cat((max_pool,avg_pool),1) #(batch, hidden size * 4)
        concat_linear1 = self.linear1(concat_lstm) #(batch, hidden_size * 4)
        concat_linear2 = self.linear2(concat_lstm) #(batch, hidden_size * 4)
        
        hidden = concat_lstm + concat_linear1 + concat_linear2 + conv #(batch, hidden_size * 4)
        y_pred = self.linear_out(hidden) #(batch, 2)
        return y_pred
    
class BertLstmCNN(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional):
        super(BertLstmCNN, self).__init__()
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
        self.conv = ConvBlock()
        self.linear1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear_out = nn.Linear(self.hidden_size*4, self.output_size)
        

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        x= self.embeded(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        z, _ = self.lstm(x) #(output)=(batch, maxlen, hidden size*2), (h_n,c_n) 
        avg_pool = torch.mean(z,1) #(batch,hidden size*2)
        max_pool, _ = torch.max(z,1) #(batch, hidden size*2)
        
        conv = self.conv(x) #(batch,out_channels*2) = (32,256)
        
        concat_lstm = torch.cat((max_pool,avg_pool),1) #(batch, hidden size * 4)
        concat_linear1 = self.linear1(concat_lstm) #(batch, hidden_size * 4)
        concat_linear2 = self.linear2(concat_lstm) #(batch, hidden_size * 4)
        
        hidden = concat_lstm + concat_linear1 + concat_linear2 + conv #(batch, hidden_size * 4)
        y_pred = self.linear_out(hidden) #(batch, 2)
        return y_pred

class BertLstmCNN2(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional):
        super(BertLstmCNN2, self).__init__()
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
        self.conv = ConvBlock()
        self.linear1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear_out = nn.Linear(self.hidden_size*4, self.output_size)
        

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        x= self.embeded(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        z, _ = self.lstm(x) #(output)=(batch, maxlen, hidden size*2), (h_n,c_n) 
        avg_pool = torch.mean(z,1) #(batch,hidden size*2)
        max_pool, _ = torch.max(z,1) #(batch, hidden size*2)
        
        conv = self.conv(x) #(batch,out_channels*2) = (32,256) = (batch, hidden_size * 4)
        
        concat_lstm = torch.cat((max_pool,avg_pool),1) #(batch, hidden size * 4)
        concat_linear1 = self.linear1(concat_lstm) #(batch, hidden_size * 4)
        #concat_linear2 = self.linear2(concat_lstm) #(batch, hidden_size * 4)
        
        hidden = concat_lstm + concat_linear1 + conv #(batch, hidden_size * 4)
        y_pred = self.linear_out(hidden) #(batch, 2)
        return y_pred

 
class BigBirdLstm2Classifier(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional):
        super(BigBirdLstm2Classifier, self).__init__()
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
        self.linear1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear_out = nn.Linear(self.hidden_size*4, self.output_size)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        x= self.embeded(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        z, _ = self.lstm(x) #(output)=(batch, maxlen, hidden size*2), (h_n,c_n) 
        avg_pool = torch.mean(z,1) #(batch,hidden size*2)
        max_pool, _ = torch.max(z,1) #(batch, hidden size*2)
        
        concat = torch.cat((max_pool,avg_pool),1) #(batch, hidden size * 4)
        concat_linear1 = self.linear1(concat) #(batch, hidden_size * 4)
        concat_linear2 = self.linear2(concat) #(batch, hidden_size * 4)
        
        hidden = concat + concat_linear1 + concat_linear2 
        y_pred = self.linear_out(hidden) #(batch, 2)
        return y_pred
    
class AlbertLstmClassifier(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional,maxlen):
        super(AlbertLstmClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.maxlen = maxlen
        
        #tokenizer = BertTokenizer.from_pretrained("kykim/albert-kor-base")
        self.embeded = AlbertModel.from_pretrained("kykim/albert-kor-base") #(batch, maxlen, embed)
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
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional,maxlen):
        super(BertLstmClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.maxlen = maxlen
        
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
        #z = z[:,-1,:].squeeze(1) #last hidden state (batch, hidden_size)
        z = nn.AvgPool2d((self.maxlen,1))(z).squeeze() #(batch_size, hidden_size)
        y_pred = self.linear(z)
        return y_pred

class BertLstm2Classifier(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional):
        super(BertLstm2Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        #tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")
        self.embeded = BertModel.from_pretrained("kykim/bert-kor-base") #(batch, maxlen, embed)
        self.lstm1 = nn.LSTM(input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional
            )
        self.linear1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear_out = nn.Linear(self.hidden_size*4, self.output_size)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        x= self.embeded(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        lstm, _ = self.lstm1(x) #(output)=(batch, maxlen, hidden size*2), (h_n,c_n) 
        avg_pool = torch.mean(lstm,1) #(batch,hidden size*2)
        max_pool, _ = torch.max(lstm,1) #(batch, hidden size*2)
        
        concat = torch.cat((max_pool,avg_pool),1) #(batch, hidden size * 4)
        concat_linear1 = self.linear1(concat) #(batch, hidden_size * 4)
        concat_linear2 = self.linear2(concat) #(batch, hidden_size * 4)
        
        hidden = concat + concat_linear1 + concat_linear2 
        y_pred = self.linear_out(hidden) #(batch, 2)
        return y_pred
        
class BertLstm3Classifier(nn.Module):   
    def __init__(self, hidden_size, output_size, embed_dim, num_layers, batch_first, bidirectional):
        super(BertLstm3Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        #tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")
        self.embeded = BertModel.from_pretrained("kykim/bert-kor-base") #(batch, maxlen, embed)
        self.embedding_dropout = SpatialDropout(0.3)
        self.lstm1 = nn.LSTM(input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional
            )
        self.lstm2 = nn.LSTM(input_size = self.hidden_size*2,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional
            )
        self.linear1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size*4)
        ) 
        self.linear_out = nn.Linear(self.hidden_size*4, self.output_size)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None):
        #(last_hiddne_state, hidden_states, attentions, cross_attentions)
        x= self.embeded(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = self.embedding_dropout(x[0])
        lstm1, _ = self.lstm1(x) #(output)=(batch, maxlen, hidden size*2), (h_n,c_n) 
        lstm2, _ = self.lstm2(lstm1)
        avg_pool = torch.mean(lstm2,1) #(batch,hidden size*2)
        max_pool, _ = torch.max(lstm2,1) #(batch, hidden size*2)
        
        concat = torch.cat((max_pool,avg_pool),1) #(batch, hidden size * 4)
        concat_linear1 = self.linear1(concat) #(batch, hidden_size * 4)
        concat_linear2 = self.linear2(concat) #(batch, hidden_size * 4)
        
        hidden = concat + concat_linear1 + concat_linear2 
        y_pred = self.linear_out(hidden) #(batch, 2)
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
        #z = z[:,-1,:].squeeze(1) #last hidden state (batch, hidden_size)
        z = nn.AvgPool2d((self.maxlen,1))(z).squeeze() #(batch_size, hidden_size)
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