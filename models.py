import torch
from torch import nn

class TourismModel(nn.Module):
    def __init__(self, bert, hidden_size, classes=None):
        super(TourismModel, self).__init__()
        self.bert = bert
        self.bert.gradient_checkpointing_enable()

        self.classifierInputSize = hidden_size#1152 + 1024
        #self.classifier = []
        if classes == None:
            self.num_classes = [6, 18, 128]
        else:
            self.num_classes = [len(_class) for _class in classes]
        #for _class in self.num_classes:
        self.classifier1 = ClassifierLayer(self.classifierInputSize, self.num_classes[0])
        self.classifier2 = ClassifierLayer(self.classifierInputSize, self.num_classes[1])
        self.classifier3 = ClassifierLayer(self.classifierInputSize, self.num_classes[2])

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, img, text, attention_mask):
        #attention_mask = self.gen_attention_mask(text, length)
        #img_feature = self.cnn_model(img)
        tmp = self.bert(input_ids=text, attention_mask=attention_mask)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, nhead=8).to(text.device)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(text.device)
        outputs = transformer_encoder(tmp.last_hidden_state)
        outputs = outputs[:,0]

        #pooler_output = tmp[1]
        #feature = torch.cat([img_feature, pooler_output], axis=1)
        #feature = pooler_output
        output1 = self.classifier1.to(text.device)(outputs)
        output2 = self.classifier2.to(text.device)(outputs)
        output3 = self.classifier3.to(text.device)(outputs)
        return [output1, output2, output3]
# class classifier(nn.Module):
#     def __init__(self, bert, hidden_size, classes=None):
#         super(classifier, self).__init__()
#         self.hidden_size = hidden_size
#         if classes == None:
#             self.num_classes = [6, 18, 128]
#         else:
#             self.num_classes = [len(_class) for _class in classes]
#         for _class in self.num_classes:
#             self.classifier.append(ClassifierLayer(self.classifierInputSize, _class))
#
#     def forward(self, output):
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8).to(output.device)
#         transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(output.device)
#         output = transformer_encoder(output.last_hidden_state)
#         output = output[:,0]
#
#         outputs = []
#         for i in range(3):
#             _output = self.classifier[i].to(output.device)(output)
#             outputs.append(_output)
#         return outputs

class ClassifierLayer(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(ClassifierLayer, self).__init__()
        dr_late = 0.1
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=dr_late),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, input):

        return self.cls(input)


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()

        self.cnn_extract = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            #1152
        )
    def forward(self, img):
        img_feature = self.cnn_extract(img)
        img_feature = torch.flatten(img_feature, start_dim=1)
        return img_feature

class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        input_size = 256
        self.hidden_size = 768
        self.embedding = nn.Embedding(8002, embedding_dim = input_size, padding_idx=1)
        self.nlp_layer = nn.LSTM(input_size, self.hidden_size, num_layers=4, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, 1024)
        self.relu = nn.ReLU()

    def forward(self, text_token):
        em = self.embedding(text_token)
        output, (final_hidden_state, final_cell_state) = self.nlp_layer(em)
        final_hidden_state = final_hidden_state[-1]
        final_hidden_state = final_hidden_state.view(-1, self.hidden_size)
        out = self.relu(final_hidden_state)
        text_feature = self.linear(out)
        return text_feature
