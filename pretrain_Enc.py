import torch
import torch.nn as nn
from tqdm import tqdm
from loss_fn import Similar_sum
def pretrain_simCSE(model, TT_DL,Val_DL, epochs, result_save_path, tokenizer, show_epoch_result = True, DEVICE = 'cpu', joint_training = 'none'):
    print(f'Pretrain on {len(TT_DL)} sentence, with Validation set be {len(Val_DL)} sentence')
    start_epoch = 0 
    mini_val_loss = 1e10
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.2)
    for epoch in tqdm(range(start_epoch, epochs)):
        running_loss = 0.0

        for index, content in enumerate(TT_DL):
            tag_label = content['error_tag'].squeeze(1).to(DEVICE)
            type_label = content['error_type'].squeeze(1).to(DEVICE)
            text_label = content['text_label'].squeeze(1).to(DEVICE)
            text_mask = content['text_mask'].squeeze(1).to(DEVICE)
            input_data = content['Data'].squeeze(1).to(DEVICE)
            input_mask = content['Data_Mask'].to(DEVICE)
            optimizer.zero_grad()

            w_sim, c_sim, diff_1, diff_2 = model(text_label, text_mask, input_data, input_mask, train_embed = True)
            
            similar_loss = torch.mean(Similar_sum(w_sim, c_sim, diff_1, diff_2, 1))
            
            similar_loss.backward()
            optimizer.step()
            
            running_loss += similar_loss.item()/len(TT_DL)

            ## Val Dataset  ###############################################################################
        scheduler.step()
        val_loss = 0
        print(f'Test Validation Loss, set val loss = {val_loss}')
        with torch.no_grad():
            running_val_loss = 0.0
            for val_index, val_content in enumerate(Val_DL):
                # val_class_label = val_content['classify_label'].squeeze(1).to(DEVICE)
                val_tag_label = val_content['error_tag'].squeeze(1).to(DEVICE)
                val_type_label = val_content['error_type'].squeeze(1).to(DEVICE)
                val_text_label = val_content['text_label'].squeeze(1).to(DEVICE)
                val_text_mask = val_content['text_mask'].squeeze(1).to(DEVICE)
                val_input_data = val_content['Data'].squeeze(1).to(DEVICE)
                val_input_mask = val_content['Data_Mask'].to(DEVICE)

                val_w_sim, val_c_sim, val_diff_1, val_diff_2 = model(val_text_label, val_text_mask, val_input_data, val_input_mask, train_embed = True)


                val_similar_loss = torch.mean(Similar_sum(val_w_sim, val_c_sim, val_diff_1, val_diff_2, 1))
                running_val_loss = val_similar_loss.item()

            if running_val_loss <= mini_val_loss:
                print(f'renew val_loss: {running_val_loss/len(Val_DL)}')
                mini_val_loss = running_val_loss
                torch.save(model.state_dict(), result_save_path)
        
        if show_epoch_result :
            print(f'[{epoch}/ {epochs-1}], loss:{running_loss}/ val_loss:{running_val_loss/len(Val_DL)}') 
        
        