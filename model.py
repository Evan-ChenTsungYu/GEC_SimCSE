
import torch.nn as nn
import torch
from transformers import T5Config,T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoConfig

class SimCSE_model(nn.Module):
    def __init__(self, pretrain_name):
        super().__init__()

        config = AutoConfig.from_pretrained(pretrain_name)
        # input model 
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrain_name,
        config = config
        )
        self.droup = nn.Dropout(p = 0.2)
        self.similarity = nn.CosineSimilarity(dim = -1)

    def forward(self, correct_sent, correct_sent_mask, wrong_sent, wrong_sent_mask, train_embed = True):
        batch_size, sent_len = correct_sent.size()
        if train_embed == True:
            self.model.decoder.requires_grad = False
            self.model.lm_head.requires_grad = False

            w_output_1 =  self.model.encoder(input_ids = wrong_sent, attention_mask = wrong_sent_mask, output_hidden_states=True)
            w_output_2 =  self.model.encoder(input_ids = wrong_sent, attention_mask = wrong_sent_mask, output_hidden_states=True)
            c_output_1 =  self.model.encoder(input_ids = correct_sent, attention_mask = correct_sent_mask, output_hidden_states=True)
            c_output_2 =  self.model.encoder(input_ids = correct_sent, attention_mask = correct_sent_mask, output_hidden_states=True)

            w_output_1 = self.droup(w_output_1.last_hidden_state.view(batch_size, -1))
            w_output_2 = self.droup(w_output_2.last_hidden_state.view(batch_size, -1))
            c_output_1 = self.droup(c_output_1.last_hidden_state.view(batch_size, -1))
            c_output_2 = self.droup(c_output_2.last_hidden_state.view(batch_size, -1))

            w_similar = self.similarity(w_output_1, w_output_2)
            c_similar = self.similarity(c_output_1, c_output_2)
            different_1 = self.similarity(w_output_1, c_output_2)
            different_2 = self.similarity(w_output_2, c_output_1)
            
            return w_similar, c_similar, different_1, different_2
        else:
            self.model.decoder.requires_grad = True
            self.model.lm_head.requires_grad = True
            output = self.model(input_ids = wrong_sent, attention_mask = wrong_sent_mask, labels = correct_sent)
            return output.loss, output.logits
           
        
    def generate(self, input_data, max_len):
        return self.model.generate(input_data, max_length = max_len, num_beams = 8)
        
        
        
if __name__ == '__main__':
    DEVICE = 'cuda:2'
    test_model = SimCSE_model('t5-base')
    test_correct_input = torch.randint(0,255,(2,5))
    test_mask = torch.ones((2,5))
    test_wrong_input = torch.randint(0,255,(2,5))
    output = test_model(test_correct_input, test_mask, test_wrong_input, test_mask, train_embed = True)
    print(output)