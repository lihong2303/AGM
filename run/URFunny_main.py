import time
import torch
import os
import math
import torch.nn as nn
import numpy as np
from utils.metric import Accuracy
from tasks.URFunny_task import Funny_Task
from utils.function_tools import save_config,get_logger,get_device,set_seed
from torch.utils.tensorboard import SummaryWriter
from sklearn import linear_model


def train(model,train_dataloader,optimizer,scheduler,cfgs,device,logger,epoch,writer,last_train_score_v,last_train_score_a,last_train_score_t,\
          visual_lr_ratio,audio_lr_ratio,text_lr_ratio):
    softmax = nn.Softmax(dim=1)
    model.train()
    model.mode = 'train'
    loss_fn = nn.CrossEntropyLoss()
    total_batch = len(train_dataloader)
    if cfgs.modality == "Audio":
        train_audio_acc = 0.
        train_score_a = last_train_score_a
    elif cfgs.modality == "Visual":
        train_visual_acc = 0.
        train_score_v = last_train_score_v
    elif cfgs.modality == "Text":
        train_text_acc = 0.
        train_score_t = last_train_score_t
    else:
        train_acc = 0.
        train_audio_acc = 0.
        train_visual_acc = 0.
        train_text_acc = 0.
        train_score_v = last_train_score_v
        train_score_a = last_train_score_a
        train_score_t = last_train_score_t

    start_time = time.time()
    for step,(feature,feature_length,index,label) in enumerate(train_dataloader):
        vision = feature[0].float().to(device)
        audio = feature[1].float().to(device)
        text = feature[2].float().to(device)
        label = label.squeeze(1)
        label = label.to(device)
        
        iteration = (epoch -1) * cfgs.batch_size + step + 1
        optimizer.zero_grad()
        
        if cfgs.modality == 'Audio':
            if step == 0:
                logger.info('Training for Audio-Only modality...')
            if cfgs.fusion_type == "early_fusion":
                m_a_out = model.net(vision,audio,text,feature_length,pad_audio=False,pad_visual=True,pad_text=True)
            elif cfgs.fusion_type == "late_fusion":
                m_a_out = model(audio,feature_length)
            score_audio = 0.
            
            for k in range(m_a_out.size(0)):
                score_audio += - torch.log(softmax(m_a_out)[k][label[k]]) / m_a_out.size(0)
            train_score_a = train_score_a * (iteration - 1) / iteration + score_audio.item() / iteration
            audio_preds = softmax(m_a_out)
            loss = loss_fn(m_a_out,label)
            loss.backward()
            audio_acc = Accuracy(audio_preds,label)
            train_audio_acc += audio_acc.item() / total_batch
        elif cfgs.modality == 'Visual':
            if step == 0:
                logger.info('Training for Visual-Only modality...')
            if cfgs.fusion_type == "early_fusion":
                m_v_out = model.net(vision,audio,text,feature_length,pad_audio=True,pad_visual=False,pad_text=True)
            elif cfgs.fusion_type == "late_fusion":
                m_v_out = model(vision,feature_length)
            score_visual = 0.
            
            for k in range(m_v_out.size(0)):
                score_visual += - torch.log(softmax(m_v_out)[k][label[k]]) / m_v_out.size(0)
            train_score_v = train_score_v * (iteration - 1) / iteration + score_visual.item() / iteration
            visual_preds = softmax(m_v_out)
            loss = loss_fn(m_v_out,label)
            loss.backward()
            visual_acc = Accuracy(visual_preds,label)
            train_visual_acc += visual_acc.item() / total_batch
        elif cfgs.modality == 'Text':
            if step == 0:
                logger.info('Training for Text-Only modality...')
            if cfgs.fusion_type == "early_fusion":
                m_t_out = model.net(vision,audio,text,feature_length,pad_audio=True,pad_visual=True,pad_text=False)
            elif cfgs.fusion_type == "late_fusion":
                m_t_out = model(text,feature_length)
            score_text = 0.
            
            for k in range(m_t_out.size(0)):
                score_text += - torch.log(softmax(m_t_out)[k][label[k]]) / m_t_out.size(0)
            train_score_t = train_score_t * (iteration - 1) / iteration + score_text.item() / iteration
            text_preds = softmax(m_t_out)
            loss = loss_fn(m_t_out,label)
            loss.backward()
            text_acc = Accuracy(text_preds,label)
            train_text_acc += text_acc.item() / total_batch
            
        else:
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                m_a_mc,m_v_mc,m_t_mc,m_a_out,m_v_out,m_t_out,out = model(vision,audio,text,feature_length)
            else:
                m_a_out,m_v_out,m_t_out,out = model(vision,audio,text,feature_length)
            loss = loss_fn(out,label)
            
            score_visual = 0.
            score_audio = 0.
            score_text = 0.
            for k in range(out.size(0)):
                score_visual += - torch.log(softmax(m_v_out)[k][label[k]]) / m_v_out.size(0)
                score_audio += - torch.log(softmax(m_a_out)[k][label[k]]) / m_a_out.size(0)
                score_text += - torch.log(softmax(m_t_out)[k][label[k]]) / m_t_out.size(0)
            
            mean_score = (score_visual.item() + score_audio.item() + score_text.item()) / 3
            
            ratio_v = math.exp((mean_score - score_visual.item())*3/2)
            ratio_a = math.exp((mean_score - score_audio.item())*3/2)
            ratio_t = math.exp((mean_score - score_text.item())*3/2)
            
            optimal_mean_score = (train_score_v + train_score_a + train_score_t) / 3
            optimal_ratio_v = math.exp((optimal_mean_score -train_score_v)*3/2)
            optimal_ratio_a = math.exp((optimal_mean_score - train_score_a)*3/2)
            optimal_ratio_t = math.exp((optimal_mean_score -train_score_t)*3/2)
            
            coeff_a = math.exp(cfgs.alpha*(min(optimal_ratio_a-ratio_a,7)))
            coeff_v = math.exp(cfgs.alpha*(min(optimal_ratio_v-ratio_v,7)))
            coeff_t = math.exp(cfgs.alpha*(min(optimal_ratio_t-ratio_t,7)))
            
            train_score_v = train_score_v * (iteration - 1) / iteration + score_visual.item() / iteration
            train_score_a = train_score_a * (iteration - 1) / iteration + score_audio.item() / iteration
            train_score_t = train_score_t * (iteration - 1) / iteration + score_text.item() / iteration

            if cfgs.methods == "AGM" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                
                if step == 0:
                    logger.info('Using AGM for training...')
                if cfgs.use_mgpu:
                    model.module.update_scale(coeff_a,coeff_v,coeff_t)
                else:
                    model.update_scale(coeff_a,coeff_v,coeff_t)
                
                loss.backward()
            elif cfgs.methods == "MSLR" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                audio_lr_init_coeff = 0.9
                visual_lr_init_coeff = 1.4
                text_lr_init_coeff = 0.7
                mslr_audio_coeff = audio_lr_init_coeff * audio_lr_ratio
                mslr_visual_coeff = visual_lr_init_coeff * visual_lr_ratio
                mslr_text_coeff = text_lr_init_coeff * text_lr_ratio
                if step == 0:
                    logger.info("Using MSLR methods.")
                    logger.info("Audio learning rate:{:.7f},Visual learning rate:{:.7f},Text learning rate:{:.4f}".format(mslr_audio_coeff,mslr_visual_coeff,mslr_text_coeff))
                loss.backward()
                for name,params in model.named_parameters():
                    layer = str(name).split('.')[0]
                    if 'audio_encoder' in layer or 'audio_cls' in layer:
                        params.grad = params.grad * mslr_audio_coeff
                    if 'visual_encoder' in layer or 'visual_cls' in layer:
                        params.grad = params.grad * mslr_visual_coeff
                    if 'text_encoder' in layer or 'text_cls' in layer:
                        params.grad = params.grad * mslr_text_coeff
            elif cfgs.methods == "MSES" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                audio_lr_init_coeff = 1.0
                visual_lr_init_coeff = 1.0
                text_lr_init_coeff = 1.0
                mses_audio_coeff = audio_lr_init_coeff * audio_lr_ratio
                mses_visual_coeff = visual_lr_init_coeff * visual_lr_ratio
                mses_text_coeff = text_lr_init_coeff * text_lr_ratio
                if step == 0:
                    logger.info("Using MSES methods")
                    logger.info("MSES audio coeff:{:.4f},MSES visual coeff:{:.4f},MSES text coeff:{:.4f}".format(mses_audio_coeff,mses_visual_coeff,mses_text_coeff))
                loss.backward()
                for name,params in model.named_parameters():
                    layer = str(name).split('.')[0]
                    if 'audio_encoder' in layer or 'audio_cls' in layer:
                        params.grad = params.grad * mses_audio_coeff
                    if 'visual_encoder' in layer or 'visual_cls' in layer:
                        params.grad = params.grad * mses_visual_coeff
                    if 'text_encoder' in layer or 'text_cls' in layer:
                        params.grad = params.grad * mses_text_coeff
            else:
                if step == 0:
                    logger.info('No modulation methods is used...')
                loss.backward()

            preds = softmax(out)
            audio_preds = softmax(m_a_out)
            visual_preds = softmax(m_v_out)
            text_preds = softmax(m_t_out)

            accuracy = Accuracy(preds,label)
            audio_acc = Accuracy(audio_preds,label)
            visual_acc = Accuracy(visual_preds,label)
            text_acc = Accuracy(text_preds,label)

            train_acc += accuracy.item() / total_batch
            train_audio_acc += audio_acc.item() / total_batch
            train_visual_acc += visual_acc.item() / total_batch
            train_text_acc += text_acc.item() / total_batch
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                grad_max = torch.max(model.net.head.fc.weight.grad)
                grad_min = torch.min(model.net.head.fc.weight.grad)
                if grad_max > 5. or grad_min < -5.:
                    nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
        optimizer.step()
        writer.add_scalar('Train (loss/epoch)',loss,iteration-1)
        if step % 100 ==0:
            logger.info('EPOCH:[{:3d}/{:3d}]--STEP:[{:5d}/{:5d}]--{}--loss:{:.4f}--lr:{}'.format(epoch,cfgs.EPOCHS,step,total_batch,'Train',loss.item(),[group['lr'] for group in optimizer.param_groups]))

    scheduler.step()

    end_time = time.time()
    elapse_time = end_time - start_time
    if cfgs.modality == 'Audio':
        writer.add_scalar('Accuracy(Train)',train_audio_acc,epoch)
        logger.info('EPOCH[{:03d}/{:03d}]-{}-Audio_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',train_audio_acc))
        return last_train_score_v,train_score_a,last_train_score_t
    elif cfgs.modality == 'Visual':
        writer.add_scalar('Accuracy(Train)',train_visual_acc,epoch)
        logger.info('EPOCH[{:03d}/{:03d}]-{}-Visual_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',train_visual_acc))
        return train_score_v,last_train_score_a,last_train_score_t
    elif cfgs.modality == 'Text':
        writer.add_scalar('Accuracy(Train)',train_text_acc,epoch)
        logger.info('EPOCH[{:03d}/{:03d}]-{}-Text_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',train_text_acc))
        return last_train_score_v,last_train_score_a,train_score_t
    else:
        writer.add_scalars('Accuracy(Train)',{'acc':train_acc,
                                                'audio_acc':train_audio_acc,
                                                'visual_acc':train_visual_acc,
                                                'text_acc':train_text_acc},epoch)
        logger.info('EPOCH:[{:03d}/{:03d}]-{}-elapse time:{:.2f}-train accuracy:{:.4f}-train audio acc:{:.4f}-train visual acc:{:.4f}-train text acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',elapse_time,train_acc,train_audio_acc,train_visual_acc,train_text_acc))
        return train_score_v,train_score_a,train_score_t


def validate(model,validate_dataloader,cfgs,device,logger,epoch,writer):
    softmax = nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()
    model.mode = 'eval'
    with torch.no_grad():
        if cfgs.fusion_type == "late_fusion" and cfgs.modality != "Multimodal":
            model.eval()
        else:
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                if cfgs.use_mgpu:
                    model.module.net.eval()
                else:
                    model.net.eval()
            else:
                model.eval()
        
        if cfgs.modality == "Audio":
            valid_score_a = 0.
            validate_audio_acc = 0.
        elif cfgs.modality == "Visual":
            valid_score_v = 0.
            validate_visual_acc = 0.
        elif cfgs.modality == "Text":
            valid_score_t = 0.
            validate_text_acc = 0.
        else:
            valid_score_a = 0.
            valid_score_v = 0.
            valid_score_t = 0.    
            
            validate_acc = 0.
            validate_visual_acc = 0.
            validate_audio_acc = 0.
            validate_text_acc = 0.
            validate_visual_loss = 0.
            validate_audio_loss = 0.
            validate_text_loss = 0.
        total_batch = len(validate_dataloader)
        start_time = time.time()
        for step,(feature,feature_length,index,label) in enumerate(validate_dataloader):
            vision = feature[0].float().to(device)
            audio = feature[1].float().to(device)
            text = feature[2].float().to(device)
            label = label.squeeze(1)
            label = label.to(device)
            
            if cfgs.modality == 'Audio':
                if step == 0:
                    logger.info("Validating for Audio-Only models...")
                if cfgs.fusion_type == "early_fusion":
                    m_a_out = model.net(vision,audio,text,feature_length,pad_audio=False,pad_visual=True,pad_text=True)
                elif cfgs.fusion_type == "late_fusion":
                    m_a_out = model(audio,feature_length)
                score_audio = 0.
                for k in range(m_a_out.size(0)):
                    score_audio += - torch.log(softmax(m_a_out)[k][label[k]]) / m_a_out.size(0)
                    
                loss = loss_fn(m_a_out,label)
                valid_score_a = valid_score_a * step / (step + 1) + score_audio.item() / (step + 1)
                audio_preds = softmax(m_a_out)
                audio_acc = Accuracy(audio_preds,label)
                validate_audio_acc += audio_acc.item() / total_batch
            elif cfgs.modality == 'Visual':
                if step == 0:
                    logger.info("Validating for Visual-Only models...")
                if cfgs.fusion_type == "early_fusion":
                    m_v_out = model.net(vision,audio,text,feature_length,pad_audio=True,pad_visual=False,pad_text=True)
                elif cfgs.fusion_type == "late_fusion":
                    m_v_out = model(vision,feature_length)
                score_visual = 0.
                for k in range(m_v_out.size(0)):
                    score_visual += - torch.log(softmax(m_v_out)[k][label[k]]) / m_v_out.size(0)
                    
                loss = loss_fn(m_v_out,label)
                valid_score_v = valid_score_v * step / (step + 1) + score_visual.item() / (step + 1)
                visual_preds = softmax(m_v_out)
                visual_acc = Accuracy(visual_preds,label)
                validate_visual_acc += visual_acc.item() / total_batch
            elif cfgs.modality == 'Text':
                if step == 0:
                    logger.info("Validating for Text-Only models...")
                if cfgs.fusion_type == "early_fusion":
                    m_t_out = model.net(vision,audio,text,feature_length,pad_audio=True,pad_visual=True,pad_text=False)
                elif cfgs.fusion_type == "late_fusion":
                    m_t_out = model(text,feature_length)
                score_text = 0.
                for k in range(m_t_out.size(0)):
                    score_text += - torch.log(softmax(m_t_out)[k][label[k]]) / m_t_out.size(0)
                    
                loss = loss_fn(m_t_out,label)
                valid_score_t = valid_score_t * step / (step + 1) + score_text.item() / (step + 1)
                text_preds = softmax(m_t_out)
                text_acc = Accuracy(text_preds,label)
                validate_text_acc += text_acc.item() / total_batch
            else:
                if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                    m_a_mc,m_v_mc,m_t_mc,m_a_out,m_v_out,m_t_out,out = model(vision,audio,text,feature_length)
                else:
                    m_a_out,m_v_out,m_t_out,out = model(vision,audio,text,feature_length)
                
                score_visual = 0.
                score_audio = 0.
                score_text = 0.
                for k in range(out.size(0)):
                    score_visual += - torch.log(softmax(m_v_out)[k][label[k]]) / m_v_out.size(0)
                    score_audio += - torch.log(softmax(m_a_out)[k][label[k]]) / m_a_out.size(0)
                    score_text += - torch.log(softmax(m_t_out)[k][label[k]]) / m_t_out.size(0)
                
                valid_score_v = valid_score_v * step / (step + 1) + score_visual.item() / (step + 1)
                valid_score_a = valid_score_a * step / (step + 1) + score_audio.item() / (step + 1)
                valid_score_t = valid_score_t * step / (step + 1) + score_text.item() / (step + 1)
                
                mean_score = (score_visual.item() + score_audio.item() + score_text.item()) / 3
                
                ratio_v = math.exp(mean_score - score_visual.item())
                ratio_a = math.exp(mean_score - score_audio.item())
                ratio_t = math.exp(mean_score - score_text.item())
                
                optimal_mean_score = (valid_score_v + valid_score_a + valid_score_t) / 3
                optimal_ratio_v = math.exp(optimal_mean_score - valid_score_v)
                optimal_ratio_a = math.exp(optimal_mean_score - valid_score_a)
                optimal_ratio_t = math.exp(optimal_mean_score - valid_score_t)

                loss = loss_fn(out,label)
                loss_v = loss_fn(m_v_out,label)
                loss_a = loss_fn(m_a_out,label)
                loss_t = loss_fn(m_t_out,label)
                validate_visual_loss += loss_v.item() / total_batch
                validate_audio_loss += loss_a.item() / total_batch
                validate_text_loss += loss_t.item() / total_batch
                preds = softmax(out)
                visual_preds = softmax(m_v_out)
                audio_preds = softmax(m_a_out)
                text_preds = softmax(m_t_out)
                accuracy = Accuracy(preds,label)
                visual_acc = Accuracy(visual_preds,label)
                audio_acc = Accuracy(audio_preds,label)
                text_acc = Accuracy(text_preds,label)
                validate_acc += accuracy.item()
                validate_visual_acc += visual_acc.item()
                validate_audio_acc += audio_acc.item()
                validate_text_acc += text_acc.item()

            iteration = (epoch-1)*total_batch + step
            writer.add_scalar('Validate loss/step',loss,iteration)

            if step % 20 == 0:
                logger.info('EPOCHS[{:02d}/{:02d}]--STEP[{:02d}/{:02d}]--{}--loss:{:.4f}'.format(epoch,cfgs.EPOCHS,step,total_batch,'Validate',loss))
        end_time = time.time()
        elapse_time = end_time - start_time
        if cfgs.modality == 'Audio':
            writer.add_scalar('Accuracy(valid)',validate_audio_acc,epoch)
            logger.info('EPOCH[{:03d}/{:03d}]-{}-Audio_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'Valid',validate_audio_acc))
            return validate_audio_acc
        elif cfgs.modality == 'Visual':
            writer.add_scalar('Accuracy(valid)',validate_visual_acc,epoch)
            logger.info('EPOCH[{:03d}/{:03d}]-{}-Visual_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'Valid',validate_visual_acc))
            return validate_visual_acc
        elif cfgs.modality == 'Text':
            writer.add_scalar('Accuracy(valid)',validate_text_acc,epoch)
            logger.info('EPOCH[{:03d}/{:03d}]-{}-Text_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'Valid',validate_text_acc))
            return validate_text_acc
        else:
            validate_mean_acc = validate_acc / total_batch
            validate_mean_visual_acc = validate_visual_acc / total_batch
            validate_mean_audio_acc = validate_audio_acc / total_batch
            validate_mean_text_acc = validate_text_acc / total_batch
            
            writer.add_scalars('Accuracy(Validate)',{'acc':validate_mean_acc,
                                                    'audio_acc':validate_mean_audio_acc,
                                                    'visual_acc':validate_mean_visual_acc,
                                                    'text_acc':validate_mean_text_acc},epoch)
            logger.info('EPOCH[{:02d}/{:02d}]-{}-elapse time:{:.2f}-validate acc:{:.4f}-validate_audio_acc:{:.4f}-validate_visual_acc:{:.4f}--validate_text_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'validate',elapse_time,validate_mean_acc,validate_mean_audio_acc,validate_mean_visual_acc,validate_mean_text_acc))

            model.mode = 'train'
            return validate_mean_acc,validate_mean_visual_acc,validate_mean_audio_acc,validate_mean_text_acc,validate_visual_loss,validate_audio_loss,validate_text_loss
def validate_compute_weight(model,validate_dataloader,cfgs,device,logger,epoch,writer,mm_to_audio_lr,mm_to_visual_lr,mm_to_text_lr,test_audio_out,test_visual_out,test_text_out):
    softmax = nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()
    model.mode = 'eval'
    with torch.no_grad():
        if cfgs.use_mgpu:
            model.module.net.eval()
        else:
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                model.net.eval()
            else:
                model.eval()
        ota = []
        otv = []
        ott = []
        valid_score_a = 0.
        valid_score_v = 0.
        valid_score_t = 0. 

        valid_batch_audio_loss = 0.
        valid_batch_visual_loss = 0.
        valid_batch_text_loss = 0.   
        
        validate_acc = 0.
        validate_visual_acc = 0.
        validate_audio_acc = 0.
        validate_text_acc = 0.
        total_batch = len(validate_dataloader)
        start_time = time.time()
        model.extract_mm_feature = True
        for step,(feature,feature_length,index,label) in enumerate(validate_dataloader):
            vision = feature[0].float().to(device)
            audio = feature[1].float().to(device)
            text = feature[2].float().to(device)
            label = label.squeeze(1)
            label = label.to(device)
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                m_a_mc,m_v_mc,m_t_mc,m_a_out,m_v_out,m_t_out,out,encoded_feature = model(vision,audio,text,feature_length)
            else:
                m_a_out,m_v_out,m_t_out,out,encoded_feature = model(vision,audio,text,feature_length)
            out_to_audio = mm_to_audio_lr.predict(encoded_feature.detach().cpu())
            out_to_visual = mm_to_visual_lr.predict(encoded_feature.detach().cpu())
            out_to_text = mm_to_text_lr.predict(encoded_feature.detach().cpu())
            ota.append(torch.from_numpy(out_to_audio))
            otv.append(torch.from_numpy(out_to_visual))
            ott.append(torch.from_numpy(out_to_text))
            
            score_visual = 0.
            score_audio = 0.
            score_text = 0.
            for k in range(out.size(0)):
                score_visual += - torch.log(softmax(m_v_out)[k][label[k]]) / m_v_out.size(0)
                score_audio += - torch.log(softmax(m_a_out)[k][label[k]]) / m_a_out.size(0)
                score_text += - torch.log(softmax(m_t_out)[k][label[k]]) / m_t_out.size(0)
            
            valid_score_v = valid_score_v * step / (step + 1) + score_visual.item() / (step + 1)
            valid_score_a = valid_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            valid_score_t = valid_score_t * step / (step + 1) + score_text.item() / (step + 1)
            
            mean_score = (score_visual.item() + score_audio.item() + score_text.item()) / 3
            
            ratio_v = math.exp(mean_score - score_visual.item())
            ratio_a = math.exp(mean_score - score_audio.item())
            ratio_t = math.exp(mean_score - score_text.item())
            
            optimal_mean_score = (valid_score_v + valid_score_a + valid_score_t) / 3
            optimal_ratio_v = math.exp(optimal_mean_score - valid_score_v)
            optimal_ratio_a = math.exp(optimal_mean_score - valid_score_a)
            optimal_ratio_t = math.exp(optimal_mean_score - valid_score_t)
            

            loss = loss_fn(out,label)
            loss_a = loss_fn(m_a_out,label)
            loss_v = loss_fn(m_v_out,label)
            loss_t = loss_fn(m_t_out,label)
            valid_batch_audio_loss += loss_a.item() / total_batch
            valid_batch_visual_loss += loss_v.item() / total_batch
            valid_batch_text_loss += loss_t.item() / total_batch

            preds = softmax(out)
            visual_preds = softmax(m_v_out)
            audio_preds = softmax(m_a_out)
            text_preds = softmax(m_t_out)
            accuracy = Accuracy(preds,label)
            visual_acc = Accuracy(visual_preds,label)
            audio_acc = Accuracy(audio_preds,label)
            text_acc = Accuracy(text_preds,label)
            validate_acc += accuracy.item()
            validate_visual_acc += visual_acc.item()
            validate_audio_acc += audio_acc.item()
            validate_text_acc += text_acc.item()

            iteration = (epoch-1)*total_batch + step
            writer.add_scalar('Validate loss/step',loss,iteration)

            if step % 20 == 0:
                logger.info('EPOCHS[{:02d}/{:02d}]--STEP[{:02d}/{:02d}]--{}--loss:{:.4f}'.format(epoch,cfgs.EPOCHS,step,total_batch,'Validate',loss))
        model.extract_mm_feature = False    
        ota = torch.cat(ota,dim=0).float()
        otv = torch.cat(otv,dim=0).float()
        ott = torch.cat(ott,dim=0).float()

        ota = ota - test_audio_out
        otv = otv - test_visual_out
        ott = ott - test_text_out
        ba = torch.cov(test_audio_out.T) * test_audio_out.size(0)
        bv = torch.cov(test_visual_out.T) * test_visual_out.size(0)
        bt = torch.cov(test_text_out.T) * test_text_out.size(0)

        ra = torch.sum(torch.multiply(ota @ torch.pinverse(ba),ota)) / test_audio_out.size(1)
        rv = torch.sum(torch.multiply(otv @ torch.pinverse(bv),otv)) / test_visual_out.size(1)
        rt = torch.sum(torch.multiply(ott @ torch.pinverse(bt),ott)) / test_text_out.size(1)
        end_time = time.time()
        elapse_time = end_time - start_time
        validate_mean_acc = validate_acc / total_batch
        validate_mean_visual_acc = validate_visual_acc / total_batch
        validate_mean_audio_acc = validate_audio_acc / total_batch
        validate_mean_text_acc = validate_text_acc / total_batch
        
        writer.add_scalars('Accuracy(Validate)',{'acc':validate_mean_acc,
                                                'audio_acc':validate_mean_audio_acc,
                                                'visual_acc':validate_mean_visual_acc,
                                                'text_acc':validate_mean_text_acc},epoch)
        logger.info('EPOCH[{:02d}/{:02d}]-{}-elapse time:{:.2f}-validate acc:{:.4f}-validate_audio_acc:{:.4f}-validate_visual_acc:{:.4f}--validate_text_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'validate',elapse_time,validate_mean_acc,validate_mean_audio_acc,validate_mean_visual_acc,validate_mean_text_acc))

        model.mode = 'train'
        return validate_mean_acc,validate_mean_visual_acc,validate_mean_audio_acc,validate_mean_text_acc,valid_batch_visual_loss,valid_batch_audio_loss,valid_batch_text_loss

def extract_mm_feature(model,dep_dataloader,device,cfgs):
    model.mode = 'eval'
    all_feature = []
    with torch.no_grad():
        model.eval()
        total_batch = len(dep_dataloader)
        for step,(feature,feature_length,index,label) in enumerate(dep_dataloader):
            vision = feature[0].float().to(device)
            audio = feature[1].float().to(device)
            text = feature[2].float().to(device)
            label = label.squeeze(1)
            label = label.to(device)
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                model.net.mode = 'feature'
                classify_out,out = model.net(vision,audio,text,feature_length)
                all_feature.append(out.detach().cpu())
                model.net.mode = 'classify'
            else:
                model.extract_mm_feature = True
                out_a,out_v,out_t,out,feature = model(vision,audio,text,feature_length)
                all_feature.append(feature.detach().cpu())
        all_feature = torch.cat(all_feature,dim=0)
        return all_feature

def URFunny_main(cfgs):
    set_seed(cfgs.random_seed)
    ts = time.strftime('%Y_%m_%d %H:%M:%S',time.localtime())
    save_dir = os.path.join(cfgs.expt_dir,f"{ts}_{cfgs.expt_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_config(cfgs,save_dir)
    if cfgs.use_mgpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.gpu_ids
        gpu_ids = list(map(int,cfgs.gpu_ids.split(",")))
        device = get_device(cfgs.device)
    else:
        device = get_device(cfgs.device)
    logger = get_logger("train_logger",logger_dir=save_dir)
    logger.info(vars(cfgs))
    logger.info(f"Processed ID:{os.getpid()},Device:{device},System Version:{os.uname()}")
    
    writer = SummaryWriter(os.path.join(save_dir,'tensorboard_out'))

    task = Funny_Task(cfgs)
    train_dataloader = task.train_dataloader
    validate_dataloader = task.valid_dataloader
    dep_dataloader = task.dep_dataloader

    model = task.model
    optimizer = task.optimizer
    scheduler = task.scheduler

    model.to(device)
    if cfgs.use_mgpu:
        model = torch.nn.DataParallel(model,device_ids=gpu_ids)
        model.cuda()

    best_epoch = {'epoch':0,'acc':0.}

    train_score_v = 0.
    train_score_a = 0.
    train_score_t = 0.

    audio_lr_ratio = 1.0
    visual_lr_ratio = 1.0
    text_lr_ratio = 1.0
    audio_lr_memory = []
    visual_lr_memory = []
    text_lr_memory = []
    min_test_audio_loss = 1000.
    min_test_visual_loss = 1000.
    min_test_text_loss = 1000.
    audio_argmax_loss_epoch = 0
    visual_argmax_loss_epoch = 0
    text_argmax_loss_epoch = 0

    for epoch in range(1,cfgs.EPOCHS+1):
        logger.info(f'Training for epoch:{epoch}...')
        train_score_v,train_score_a,train_score_t = train(model,train_dataloader,optimizer,scheduler,cfgs,device,logger,epoch,writer,\
                                                    train_score_v,train_score_a,train_score_t,visual_lr_ratio,audio_lr_ratio,text_lr_ratio)
        if cfgs.modality == "Multimodal":
            dep_audio_out_path = ""
            dep_visual_out_path = ""
            dep_text_out_path = ""
            test_audio_out_path = ""
            test_visual_out_path = ""
            test_text_out_path = ""
            if os.path.exists(dep_audio_out_path) and os.path.exists(dep_visual_out_path) and os.path.exists(dep_text_out_path) \
            and os.path.exists(test_audio_out_path) and os.path.exists(test_visual_out_path) and os.path.exists(test_text_out_path): 
                dep_mm_feature = extract_mm_feature(model,dep_dataloader,device,cfgs)
                if cfgs.fusion_type == "early_fusion":
                    dep_audio_out = torch.load(dep_audio_out_path,map_location="cpu")
                    dep_visual_out = torch.load(dep_visual_out_path,map_location="cpu")
                    dep_text_out = torch.load(dep_text_out_path,map_location="cpu")
                    test_audio_out = torch.load(test_audio_out_path,map_location="cpu")
                    test_visual_out = torch.load(test_visual_out_path,map_location="cpu")
                    test_text_out = torch.load(test_text_out_path,map_location="cpu")
                elif cfgs.fusion_type == "late_fusion":
                    dep_audio_out = torch.load(dep_audio_out_path,map_location="cpu")
                    dep_visual_out = torch.load(dep_visual_out_path,map_location="cpu")
                    dep_text_out = torch.load(dep_text_out_path,map_location="cpu")
                    test_audio_out = torch.load(test_audio_out_path,map_location="cpu")
                    test_visual_out = torch.load(test_visual_out_path,map_location="cpu")
                    test_text_out = torch.load(test_text_out_path,map_location="cpu")
                
                mm_to_audio_lr = linear_model.Ridge(alpha=120) 
                mm_to_visual_lr = linear_model.Ridge(alpha=120)
                mm_to_text_lr = linear_model.Ridge(alpha=120)
                mm_to_audio_lr.fit(dep_mm_feature.detach().cpu(),dep_audio_out)
                mm_to_visual_lr.fit(dep_mm_feature.detach().cpu(),dep_visual_out)
                mm_to_text_lr.fit(dep_mm_feature.detach().cpu(),dep_text_out)
                logger.info(f'Validating for epoch:{epoch}...')
                validate_acc,accuracy_v,accuracy_a,accuracy_t,validate_loss_v,validate_loss_a,validate_loss_t = validate_compute_weight(model,validate_dataloader,cfgs,device,logger,epoch,writer,mm_to_audio_lr,mm_to_visual_lr,mm_to_text_lr,test_audio_out,test_visual_out,test_text_out)
            else:
                validate_acc,accuracy_v,accuracy_a,accuracy_t,validate_loss_v,validate_loss_a,validate_loss_t = validate(model,validate_dataloader,cfgs,device,logger,epoch,writer)
        else:
            logger.info(f'Validating for epoch:{epoch}...')
            validate_acc = validate(model,validate_dataloader,cfgs,device,logger,epoch,writer)

        if cfgs.modality == "Multimodal" and cfgs.methods == "MSLR":
            if len(audio_lr_memory) < 5:
                audio_lr_memory.append(accuracy_a)
                visual_lr_memory.append(accuracy_v)
                text_lr_memory.append(accuracy_t)
            else:
                audio_lr_ratio = accuracy_a / np.mean(audio_lr_memory)
                visual_lr_ratio = accuracy_v / np.mean(visual_lr_memory)
                text_lr_ratio = accuracy_t / np.mean(text_lr_memory)

                audio_lr_memory = audio_lr_memory[1:]
                visual_lr_memory = visual_lr_memory[1:]
                text_lr_memory = text_lr_memory[1:]
                audio_lr_memory.append(accuracy_a)
                visual_lr_memory.append(accuracy_v)
                text_lr_memory.append(accuracy_t)
                if len(audio_lr_memory) != 5 or len(visual_lr_memory) != 5 or len(text_lr_memory) != 5:
                    raise ValueError
        elif cfgs.modality == "Multimodal" and cfgs.methods == "MSES":
            if epoch == 1:
                min_test_audio_loss = validate_loss_a
                min_test_visual_loss = validate_loss_v
                min_test_text_loss = validate_loss_t
                audio_argmax_loss_epoch = 1
                visual_argmax_loss_epoch = 1
                text_argmax_loss_epoch = 1
                audio_batch_loss = validate_loss_a
                visual_batch_loss = validate_loss_v
                text_batch_loss = validate_loss_t
            else:
                audio_batch_loss = 0.6 * bre_audio_batch_loss + 0.4 * validate_loss_a
                visual_batch_loss = 0.6 * bre_visual_batch_loss + 0.4 * validate_loss_v
                text_batch_loss = 0.6 * bre_text_batch_loss + 0.4 * validate_loss_t
            
            bre_audio_batch_loss = validate_loss_a
            bre_visual_batch_loss = validate_loss_v
            bre_text_batch_loss = validate_loss_t

            if min_test_audio_loss > audio_batch_loss:
                min_test_audio_loss = audio_batch_loss
                audio_argmax_loss_epoch = epoch
                torch.save({'epoch':best_epoch['epoch'],
                            'acc':best_epoch['acc'],
                            'optimizer':optimizer.state_dict(),
                            'state_dict':model.state_dict(),
                            'train_score_a':train_score_a,
                            'train_score_v':train_score_v,
                            'train_score_t':train_score_t},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))
            if min_test_visual_loss > visual_batch_loss:
                min_test_visual_loss = visual_batch_loss
                visual_argmax_loss_epoch = epoch
                torch.save({'epoch':best_epoch['epoch'],
                            'acc':best_epoch['acc'],
                            'optimizer':optimizer.state_dict(),
                            'state_dict':model.state_dict(),
                            'train_score_a':train_score_a,
                            'train_score_v':train_score_v,
                            'train_score_t':train_score_t},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))
            if min_test_text_loss > text_batch_loss:
                min_test_text_loss = text_batch_loss
                text_argmax_loss_epoch = epoch
                torch.save({'epoch':best_epoch['epoch'],
                            'acc':best_epoch['acc'],
                            'optimizer':optimizer.state_dict(),
                            'state_dict':model.state_dict(),
                            'train_score_a':train_score_a,
                            'train_score_v':train_score_v,
                            'train_score_t':train_score_t},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))
            
            if (epoch - audio_argmax_loss_epoch > 30) and (audio_lr_ratio != 0.0):
                audio_lr_ratio = 0.0
                logger.info("Audio modality training finished.")
                if audio_argmax_loss_epoch != 1:
                    ckpt = torch.load(os.path.join(save_dir,f'ckpt_full_epoch{audio_argmax_loss_epoch}.pth.tar'))
                    model.load_state_dict(ckpt["state_dict"])
                    optimizer = task.optimizer
                    optimizer.load_state_dict(ckpt["optimizer"])
                    best_epoch["epoch"] = ckpt["epoch"]
                    best_epoch['acc'] = ckpt["acc"]
                    train_score_a = ckpt["train_score_a"]
                    train_score_v = ckpt["train_score_v"]
                    train_score_t = ckpt["train_score_t"]
            
            if (epoch - visual_argmax_loss_epoch > 30) and (visual_lr_ratio != 0.0):
                visual_lr_ratio = 0.0
                logger.info("Visual modality training finished.")
                if visual_argmax_loss_epoch != 1:
                    ckpt = torch.load(os.path.join(save_dir,f'ckpt_full_epoch{visual_argmax_loss_epoch}.pth.tar'))
                    model.load_state_dict(ckpt["state_dict"])
                    optimizer = task.optimizer
                    optimizer.load_state_dict(ckpt["optimizer"])
                    best_epoch["epoch"] = ckpt["epoch"]
                    best_epoch['acc'] = ckpt["acc"]
                    train_score_a = ckpt["train_score_a"]
                    train_score_v = ckpt["train_score_v"]
                    train_score_t = ckpt["train_score_t"]
            
            if (epoch - text_argmax_loss_epoch > 30) and (text_lr_ratio != 0.0):
                text_lr_ratio = 0.0
                logger.info("Text modality training finished.")
                if text_argmax_loss_epoch != 1:
                    ckpt = torch.load(os.path.join(save_dir,f'ckpt_full_epoch{text_argmax_loss_epoch}.pth.tar'))
                    model.load_state_dict(ckpt["state_dict"])
                    optimizer = task.optimizer
                    optimizer.load_state_dict(ckpt["optimizer"])
                    best_epoch["epoch"] = ckpt["epoch"]
                    best_epoch['acc'] = ckpt["acc"]
                    train_score_a = ckpt["train_score_a"]
                    train_score_v = ckpt["train_score_v"]
                    train_score_t = ckpt["train_score_t"]
            logger.info("Audio min loss epoch:{:.4f},Visual min loss epoch:{:.4f},Text min loss epoch:{:.4f}".format(audio_argmax_loss_epoch,visual_argmax_loss_epoch,text_argmax_loss_epoch))
            if (audio_lr_ratio == 0.0) and (visual_lr_ratio == 0.0) and (text_lr_ratio == 0.0):
                logger.info("All modalities training finished.")
                exit()

        if validate_acc > best_epoch['acc']:
            best_epoch['acc'] = validate_acc
            best_epoch['epoch'] = epoch

            if cfgs.save_checkpoint:
                torch.save({'epoch':best_epoch['epoch'],
                            'acc':best_epoch['acc'],
                            'optimizer':optimizer.state_dict(),
                            'state_dict':model.state_dict()},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))
        
        logger.info(f'Best epoch{best_epoch["epoch"]},best accuracy{best_epoch["acc"]}')
