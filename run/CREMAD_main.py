import time
import torch
import os
import math
import torch.nn as nn
import numpy as np
from tasks.CREMAD_task import Cramed_Task
from utils.function_tools import save_config,get_device,get_logger,set_seed,weight_init
from torch.utils.tensorboard import SummaryWriter
from sklearn import linear_model


def train(model,train_dataloader,optimizer,scheduler,logger,cfgs,epoch,device,writer,last_score_a,last_score_v,audio_lr_ratio,visual_lr_ratio):
    criterion = nn.CrossEntropyLoss() 
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    model.train()
    total_batch = len(train_dataloader)

    if cfgs.dataset == 'CREMAD':
        n_classes = 6
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(cfgs.dataset))

    model.mode = 'train'
    if cfgs.modality == "Audio":
        train_score_a = last_score_a
        ra_score_a = 0.
    elif cfgs.modality == "Visual":
        train_score_v = last_score_v
        ra_score_v = 0.
    else:
        train_score_a = last_score_a
        train_score_v = last_score_v
        ra_score_a = 0.
        ra_score_v = 0.
    train_batch_loss = 0.
    
    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]
    acc_a = [0.0 for _ in range(n_classes)]
    acc_v  = [0.0 for _ in range(n_classes)]
    for step,(item,spec,image,label) in enumerate(train_dataloader):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        iteration = (epoch-1)*total_batch + step + 1

        if cfgs.modality == "Audio":
            if cfgs.fusion_type == "early_fusion":
                out_a = model.net(spec.unsqueeze(1).float(),image.float(),pad_audio = False,pad_visual=True)
            elif cfgs.fusion_type == "late_fusion":
                out_a = model(spec.unsqueeze(1))
            loss = criterion(out_a,label)
        elif cfgs.modality == "Visual":
            if cfgs.fusion_type == "early_fusion":
                out_v = model.net(spec.unsqueeze(1).float(),image.float(),pad_audio = True,pad_visual=False)
            elif cfgs.fusion_type == "late_fusion":
                out_v = model(image.float())
            loss = criterion(out_v,label)
        else:
            if cfgs.fusion_type == "early_fusion" or cfgs.methods == "AGM":
                total_out,pad_visual_out,pad_audio_out,zero_padding_out,out = model(spec.unsqueeze(1).float(),image.float())
                out_a = 0.5*(total_out-pad_audio_out+pad_visual_out)
                out_v = 0.5*(total_out-pad_visual_out+pad_audio_out)
            else:
                out_a,out_v,out = model(spec.unsqueeze(1).float(),image.float())
            loss = criterion(out,label)
        
        train_batch_loss += loss.item() / total_batch
        
        writer.add_scalar('loss/step',loss,(epoch-1)*total_batch + step)
        
        if cfgs.modality == 'Audio':
            pred_a = softmax(out_a)
        elif cfgs.modality == 'Visual':
            pred_v = softmax(out_v)
        else:
            prediction = softmax(out)
            pred_a = softmax(out_a)
            pred_v = softmax(out_v)

        for j in range(image.shape[0]):
            if cfgs.modality == 'Audio':
                a = np.argmax(pred_a[j].cpu().data.numpy())
                num[label[j]] += 1.0
                if np.asarray(label[j].cpu()) == a:
                    acc_a[label[j]] += 1.0
            elif cfgs.modality == 'Visual':
                v = np.argmax(pred_v[j].cpu().data.numpy())
                num[label[j]] += 1.0
                if np.asarray(label[j].cpu()) == v:
                    acc_v[label[j]] += 1.0
            else:
                ma = np.argmax(prediction[j].cpu().data.numpy())
                v = np.argmax(pred_v[j].cpu().data.numpy())
                a = np.argmax(pred_a[j].cpu().data.numpy())
                num[label[j]] += 1.0

                if np.asarray(label[j].cpu()) == ma:
                    acc[label[j]] += 1.0
                if np.asarray(label[j].cpu()) == v:
                    acc_v[label[j]] += 1.0
                if np.asarray(label[j].cpu()) == a:
                    acc_a[label[j]] += 1.0

        optimizer.zero_grad()
        
        if cfgs.modality == "Audio":
            score_audio = 0.
            for k in range(out_a.size(0)):
                score_audio += - torch.log(softmax(out_a)[k][label[k]])
                
            score_audio = score_audio / out_a.size(0)
            train_score_a = train_score_a * (iteration - 1) / iteration  + score_audio.item() / iteration \
            
            loss.backward()
        elif cfgs.modality == "Visual":
            score_visual = 0.
            for k in range(out_v.size(0)):
                score_visual +=  - torch.log(softmax(out_v)[k][label[k]])
            score_visual = score_visual / out_v.size(0)
            train_score_v = train_score_v * (iteration - 1) / iteration + score_visual.item() / iteration
            loss.backward()
        else:
            if torch.isnan(out_a).any() or torch.isnan(out_v).any():
                raise ValueError
            
            score_audio = 0.
            score_visual = 0.
            for k in range(out_a.size(0)):
                if torch.isinf(torch.log(softmax(out_a)[k][label[k]])) or softmax(out_a)[k][label[k]] < 1e-8:
                    score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.device))
                else:
                    score_audio += - torch.log(softmax(out_a)[k][label[k]])
                
                if torch.isinf(torch.log(softmax(out_v)[k][label[k]])) or softmax(out_v)[k][label[k]] < 1e-8:
                    score_visual += - torch.log(torch.tensor(1e-8,dtype=out_v.dtype,device=out_v.device))
                else:
                    score_visual += - torch.log(softmax(out_v)[k][label[k]])
            score_audio = score_audio / out_a.size(0)
            score_visual = score_visual / out_v.size(0)

            ratio_a = math.exp(score_visual.item() - score_audio.item())
            ratio_v = math.exp(score_audio.item() - score_visual.item())
            
            optimal_ratio_a = math.exp(train_score_v - train_score_a)
            optimal_ratio_v = math.exp(train_score_a - train_score_v)
            
            coeff_a = math.exp(cfgs.alpha*(min(optimal_ratio_a - ratio_a,10)))
            coeff_v = math.exp(cfgs.alpha*(min(optimal_ratio_v - ratio_v,10)))

            train_score_a = train_score_a * (iteration - 1) / iteration  + score_audio.item() / iteration 
            train_score_v = train_score_v * (iteration - 1) / iteration + score_visual.item() / iteration
            ra_score_a = ra_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            ra_score_v = ra_score_v * step / (step + 1) + score_visual.item() / (step + 1)

            if cfgs.methods=="AGM" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                if step ==0:
                    logger.info('Using AGM methods...')
                if cfgs.use_mgpu:
                    model.module.update_scale(coeff_a,coeff_v)
                else:
                    model.update_scale(coeff_a,coeff_v)
                loss.backward()

            elif cfgs.methods == "OGM-GE" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                if step ==0:
                    logger.info('Using OGM_GE methods...')
                # original OGM-GE Method.
                loss.backward()
                OGM_score_v = sum([softmax(out_v)[j][label[j]]for j in range(out_v.size(0))])
                OGM_score_a = sum([softmax(out_a)[j][label[j]]for j in range(out_a.size(0))])
                OGM_ratio_v = OGM_score_v / OGM_score_a
                OGM_ratio_a = 1 / OGM_ratio_v
                if OGM_ratio_v > 1:
                    OGM_coeff_v = 1 - tanh(cfgs.alpha * relu(OGM_ratio_v))
                    OGM_coeff_a = 1
                else:
                    OGM_coeff_a = 1 - tanh(cfgs.alpha * relu(OGM_ratio_a))
                    OGM_coeff_v = 1

                for name, parms in model.named_parameters():
                    layer = str(name).split('.')[0]
                    if 'audio_net' in layer or 'audio_cls' in layer:
                            # bug fixed
                        parms.grad = parms.grad * OGM_coeff_a + \
                                        torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    if 'visual_net' in layer or 'visual_cls' in layer:
                            # bug fixed
                        parms.grad = parms.grad * OGM_coeff_v + \
                                        torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            elif cfgs.methods == "MSLR" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                audio_lr_init_coeff = 0.9
                visual_lr_init_coeff = 1.1
                mslr_audio_coeff = audio_lr_init_coeff * audio_lr_ratio
                mslr_visual_coeff = visual_lr_init_coeff * visual_lr_ratio
                if step == 0:
                    logger.info("Using MSLR methods...")
                    logger.info("Audio learning rate:{:.4f},Visual learning rate:{:.4f}".format(mslr_audio_coeff*cfgs.learning_rate,mslr_visual_coeff*cfgs.learning_rate))
                loss.backward()
                for name,params in model.named_parameters():
                    layer = str(name).split('.')[0]
                    if 'audio_net' in layer or 'audio_cls' in layer:
                        params.grad = params.grad * mslr_audio_coeff
                    
                    if 'visual_net' in layer or 'visual_cls' in layer:
                        params.grad = params.grad * mslr_visual_coeff
                  
            elif cfgs.methods == "MSES" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                audio_lr_init_coeff = 1.0
                visual_lr_init_coeff = 1.0
                mses_audio_coeff = audio_lr_init_coeff * audio_lr_ratio
                mses_visual_coeff = visual_lr_init_coeff * visual_lr_ratio
                if step == 0:
                    logger.info("Using MSES methods...")
                    logger.info("mses audio coeff:{:.4f},MSES visual coeff:{:.4f}".format(mses_audio_coeff,mses_visual_coeff))
                loss.backward()
                for name,params in model.named_parameters():
                    layer = str(name).split('.')[0]
                    if 'audio_net' in layer or 'audio_cls' in layer:
                        params.grad = params.grad * mses_audio_coeff
                    
                    if 'visual_net' in layer or 'visual_cls' in layer:
                        params.grad = params.grad * mses_visual_coeff
            else:
                # normal
                if step ==0:
                    logger.info('Using normal methods...')
                
                loss.backward()
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                if cfgs.fusion_type == "late_fusion":
                    grad_max = torch.max(model.net.fusion_module.fc_out.weight.grad)
                    grad_min = torch.min(model.net.fusion_module.fc_out.weight.grad)
                    if grad_max > 1 or grad_min < -1:
                        nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                else:
                    grad_max = torch.max(model.net.head.fc.weight.grad)
                    grad_min = torch.min(model.net.head.fc.weight.grad)
                    if grad_max > 1 or grad_min < -1:
                        nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimizer.step()
            
        if step % 100 == 0:
            logger.info('EPOCH:[{:3d}/{:3d}]--STEP:[{:5d}/{:5d}]--{}--Loss:{:.4f}--lr:{}'.format(epoch,cfgs.EPOCHS,step,total_batch,'Train',loss.item(),[group['lr'] for group in optimizer.param_groups]))
    scheduler.step()
    if cfgs.modality == "Audio":
        accuracy_a = sum(acc_a) / sum(num)
        writer.add_scalar('Epoch Accuracy(Train)',accuracy_a,epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]-{}-Accuracy:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',accuracy_a))
        return train_score_a,last_score_v
    elif cfgs.modality == "Visual":
        accuracy_v = sum(acc_v) / sum(num)
        writer.add_scalar('Epoch Accuracy(Train)',accuracy_v,epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]-{}-Accuracy:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',accuracy_v))
        return last_score_a,train_score_v
    else:
        accuracy = sum(acc) / sum(num)
        accuracy_a = sum(acc_a) / sum(num)
        accuracy_v = sum(acc_v) / sum(num)
        
        writer.add_scalars('Epoch Accuracy(train)',{'accuracy':accuracy,
                                                   'accuracy audio':accuracy_a,
                                                   'accuracy visual':accuracy_v},epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]--{}--acc:{:.4f}--acc_a:{:.4f}--acc_v:{:.4f}-Alpha:{}'.format(epoch,cfgs.EPOCHS,'Train',accuracy,accuracy_a,accuracy_v,cfgs.alpha))
        return train_score_a,train_score_v
    

def test(model,test_dataloader,logger,cfgs,epoch,device,writer):
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    if cfgs.dataset == 'CREMAD':
        n_classes = 6
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(cfgs.dataset))
    start_time = time.time()
    with torch.no_grad():
        if cfgs.use_mgpu:
            model.module.eval()
        else:
            model.eval()
            
        if cfgs.modality == "Audio":
            num = [0.0 for _ in range(n_classes)]
            acc_a = [0.0 for _ in range(n_classes)]
            valid_score_a = 0.
        elif cfgs.modality == "Visual":
            num = [0.0 for _ in range(n_classes)]
            acc_v  = [0.0 for _ in range(n_classes)]
            valid_score_v = 0.
        else:
            model.mode = 'eval'
            num = [0.0 for _ in range(n_classes)]
            acc = [0.0 for _ in range(n_classes)]
            acc_a = [0.0 for _ in range(n_classes)]
            acc_v  = [0.0 for _ in range(n_classes)]
            valid_score_a = 0.
            valid_score_v = 0.
            test_loss = 0.
            test_audio_loss = 0.
            test_visual_loss = 0.
        total_batch = len(test_dataloader)

        for step,(item,spec,image,label) in enumerate(test_dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if cfgs.modality == "Audio":
                if cfgs.fusion_type == "early_fusion":
                    out_a = model.net(spec.unsqueeze(1).float(),image.float(),pad_audio = False,pad_visual=True)
                elif cfgs.fusion_type == "late_fusion":
                    out_a = model(spec.unsqueeze(1))
                loss = criterion(out_a,label)
            elif cfgs.modality == "Visual":
                if cfgs.fusion_type == "early_fusion":
                    out_v = model.net(spec.unsqueeze(1).float(),image.float(),pad_audio = True,pad_visual=False)
                elif cfgs.fusion_type == "late_fusion":
                    out_v = model(image.float())
                loss = criterion(out_v,label)
            else:
                if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                    total_out,pad_visual_out,pad_audio_out,zero_padding_out,out = model(spec.unsqueeze(1).float(),image.float())
                    out_a = 0.5*(total_out-pad_audio_out+pad_visual_out)
                    out_v = 0.5*(total_out-pad_visual_out+pad_audio_out)
                else:
                    out_a,out_v,out = model(spec.unsqueeze(1).float(),image.float())
                loss = criterion(out,label)
                loss_a = criterion(out_a,label)
                loss_v = criterion(out_v,label)
            
            if cfgs.modality == "Audio":
                score_audio = 0.
                for k in range(out_a.size(0)):
                    score_audio += - torch.log(softmax(out_a)[k][label[k]])
                score_audio = score_audio / out_a.size(0)
                valid_score_a = valid_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            elif cfgs.modality == "Visual":
                score_visual = 0.
                for k in range(out_v.size(0)):
                    score_visual += - torch.log(softmax(out_v)[k][label[k]])
                score_visual = score_visual / out_v.size(0)
                valid_score_v = valid_score_v * step / (step + 1) + score_visual.item() / (step + 1)
            else:
                score_audio = 0.
                score_visual = 0.
                for k in range(out_a.size(0)):
                    if torch.isinf(torch.log(softmax(out_a)[k][label[k]])) or softmax(out_a)[k][label[k]] < 1e-8:
                        score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.device))
                    else:
                        score_audio += - torch.log(softmax(out_a)[k][label[k]])
                    if torch.isinf(torch.log(softmax(out_v)[k][label[k]])) or softmax(out_v)[k][label[k]] < 1e-8:
                        score_visual += - torch.log(torch.tensor(1e-8,dtype=out_v.dtype,device=out_v.device))
                    else:
                        score_visual += - torch.log(softmax(out_v)[k][label[k]])
                score_audio = score_audio / out_a.size(0)
                score_visual = score_visual / out_v.size(0)
                
                valid_score_a = valid_score_a * step / (step + 1) + score_audio.item() / (step + 1)
                valid_score_v = valid_score_v * step / (step + 1) + score_visual.item() / (step + 1)
                
                ratio_a = math.exp(valid_score_v - valid_score_a)
                ratio_v = math.exp(valid_score_a - valid_score_v)

                test_loss += loss.item() / total_batch
                test_audio_loss += loss_a.item() / total_batch
                test_visual_loss += loss_v.item() / total_batch
            
            iteration = (epoch-1)*total_batch +step
            writer.add_scalar('test (loss/step)',loss,iteration)
            
            if cfgs.modality == 'Audio':
                pred_a = softmax(out_a)
            elif cfgs.modality == 'Visual':
                pred_v = softmax(out_v)
            else:
                prediction = softmax(out)
                pred_a = softmax(out_a)
                pred_v = softmax(out_v) 
                
            for j in range(image.shape[0]):
                if cfgs.modality == 'Audio':
                    a = np.argmax(pred_a[j].cpu().data.numpy())
                    num[label[j]] += 1.0
                    if np.asarray(label[j].cpu()) == a:
                        acc_a[label[j]] += 1.0
                elif cfgs.modality == 'Visual':
                    v = np.argmax(pred_v[j].cpu().data.numpy())
                    num[label[j]] += 1.0
                    if np.asarray(label[j].cpu()) == v:
                        acc_v[label[j]] += 1.0
                else:
                    ma = np.argmax(prediction[j].cpu().data.numpy())
                    v = np.argmax(pred_v[j].cpu().data.numpy())
                    a = np.argmax(pred_a[j].cpu().data.numpy())
                    num[label[j]] += 1.0

                    if np.asarray(label[j].cpu()) == ma:
                        acc[label[j]] += 1.0
                    if np.asarray(label[j].cpu()) == v:
                        acc_v[label[j]] += 1.0
                        acc_a[label[j]] += 1.0
                
            if step % 20 == 0:
                logger.info('EPOCH:[{:03d}/{:03d}]--STEP:[{:05d}/{:05d}]--{}--loss:{}'.format(epoch,cfgs.EPOCHS,step,total_batch,'Validate',loss))
    if cfgs.modality == "Audio":
        accuracy_a = sum(acc_a) / sum(num)
        writer.add_scalar('Epoch Accuracy(Valid)',accuracy_a,epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]-{}-Accuracy:{:.4f}'.format(epoch,cfgs.EPOCHS,'Valid',accuracy_a))
        return accuracy_a
    elif cfgs.modality == "Visual":
        accuracy_v = sum(acc_v) / sum(num)
        writer.add_scalar('Epoch Accuracy(Valid)',accuracy_v,epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]-{}-Accuracy:{:.4f}'.format(epoch,cfgs.EPOCHS,'Valid',accuracy_v))
        return accuracy_v
    else:
        accuracy = sum(acc) / sum(num)
        accuracy_a = sum(acc_a) / sum(num)
        accuracy_v = sum(acc_v) / sum(num)
        writer.add_scalars('Accuracy(Test)',{'accuracy':accuracy,
                                            'audio_accuracy':accuracy_a,
                                            'visual accuracy':accuracy_v},epoch)

        model.mode = 'train'
        end_time = time.time()
        elapse_time = end_time - start_time
        logger.info('EPOCH:[{:03d}/{:03d}]--{}--Elapse time:{:.2f}--Accuracy:{:.4f}--acc_a:{:.4f}--acc_v:{:.4f}'.format(epoch,cfgs.EPOCHS,'Validate',elapse_time,accuracy,accuracy_a,accuracy_v))
        return accuracy,accuracy_a,accuracy_v,test_audio_loss,test_visual_loss
    
def test_compute_weight(model,test_dataloader,logger,cfgs,epoch,device,writer,mm_to_audio_lr,mm_to_visual_lr,test_audio_out,test_visual_out):
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    if cfgs.dataset == 'CREMAD':
        n_classes = 6
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(cfgs.dataset))
    start_time = time.time()
    test_loss = 0.
    test_audio_loss = 0.
    test_visual_loss = 0.
    with torch.no_grad():
        if cfgs.use_mgpu:
            model.module.eval()
        else:
            model.eval()
        
        model.mode = 'eval'
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v  = [0.0 for _ in range(n_classes)]
        valid_score_a = 0.
        valid_score_v = 0.
        total_batch = len(test_dataloader)
        model.extract_mm_feature = True
        ota = []
        otv = []
        for step,(item,spec,image,label) in enumerate(test_dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if cfgs.methods == "AGM" or cfgs.fusion_type == "early fusion":
                total_out,pad_visual_out,pad_audio_out,zero_padding_out,out,encoded_feature = model(spec.unsqueeze(1).float(),image.float())
                out_to_audio = mm_to_audio_lr.predict(encoded_feature.detach().cpu())
                out_to_visual = mm_to_visual_lr.predict(encoded_feature.detach().cpu())
                ota.append(torch.from_numpy(out_to_audio))
                otv.append(torch.from_numpy(out_to_visual))
                out_a = 0.5*(total_out-pad_audio_out+pad_visual_out)
                out_v = 0.5*(total_out-pad_visual_out+pad_audio_out)
                loss = criterion(out,label)
                loss_a = criterion(out_a,label)
                loss_v = criterion(out_v,label)
            else:
                out_a,out_v,out,encoded_feature = model(spec.unsqueeze(1).float(),image.float())
                out_to_audio = mm_to_audio_lr.predict(encoded_feature.detach().cpu())
                out_to_visual = mm_to_visual_lr.predict(encoded_feature.detach().cpu())
                ota.append(torch.from_numpy(out_to_audio))
                otv.append(torch.from_numpy(out_to_visual))
                loss = criterion(out,label)
                loss_a = criterion(out_a,label)
                loss_v = criterion(out_v,label)
            
            score_audio = 0.
            score_visual = 0.
            for k in range(out_a.size(0)):
                if torch.isinf(torch.log(softmax(out_a)[k][label[k]])) or softmax(out_a)[k][label[k]] < 1e-8:
                    score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.device))
                else:
                    score_audio += - torch.log(softmax(out_a)[k][label[k]])
                if torch.isinf(torch.log(softmax(out_v)[k][label[k]])) or softmax(out_v)[k][label[k]] < 1e-8:
                    score_visual += - torch.log(torch.tensor(1e-8,dtype=out_v.dtype,device=out_v.device))
                else:
                    score_visual += - torch.log(softmax(out_v)[k][label[k]])
            score_audio = score_audio / out_a.size(0)
            score_visual = score_visual / out_v.size(0)
            
            valid_score_a = valid_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            valid_score_v = valid_score_v * step / (step + 1) + score_visual.item() / (step + 1)
            
            ratio_a = math.exp(valid_score_v - valid_score_a)
            ratio_v = math.exp(valid_score_a - valid_score_v)

            test_loss += loss.item() / total_batch
            test_audio_loss += loss_a.item() / total_batch
            test_visual_loss += loss_v.item() / total_batch
            
            iteration = (epoch-1)*total_batch +step
            writer.add_scalar('test (loss/step)',loss,iteration)
            
            prediction = softmax(out)
            pred_a = softmax(out_a)
            pred_v = softmax(out_v) 
                
            for j in range(image.shape[0]):
                ma = np.argmax(prediction[j].cpu().data.numpy())
                v = np.argmax(pred_v[j].cpu().data.numpy())
                a = np.argmax(pred_a[j].cpu().data.numpy())
                num[label[j]] += 1.0

                if np.asarray(label[j].cpu()) == ma:
                    acc[label[j]] += 1.0
                if np.asarray(label[j].cpu()) == v:
                    acc_v[label[j]] += 1.0
                if np.asarray(label[j].cpu()) == a:
                    acc_a[label[j]] += 1.0
                
            if step % 20 == 0:
                logger.info('EPOCH:[{:03d}/{:03d}]--STEP:[{:05d}/{:05d}]--{}--loss:{}'.format(epoch,cfgs.EPOCHS,step,total_batch,'Validate',loss))
        model.extract_mm_feature = False
        ota = torch.cat(ota,dim=0).float()
        otv = torch.cat(otv,dim=0).float()
        ota = ota - test_audio_out
        otv = otv - test_visual_out
        
        ba = torch.cov(test_audio_out.T) * test_audio_out.size(0)
        bv = torch.cov(test_visual_out.T) * test_visual_out.size(0)
        
        ra = torch.trace((ota @ torch.pinverse(ba)).T @ ota) / test_audio_out.size(1)
        rv = torch.trace((otv @ torch.pinverse(bv)).T @ otv) / test_visual_out.size(1)
        accuracy = sum(acc) / sum(num)
        accuracy_a = sum(acc_a) / sum(num)
        accuracy_v = sum(acc_v) / sum(num)
        writer.add_scalars('Accuracy(Test)',{'accuracy':accuracy,
                                            'audio_accuracy':accuracy_a,
                                            'visual accuracy':accuracy_v},epoch)

        model.mode = 'train'
        end_time = time.time()
        elapse_time = end_time - start_time
        logger.info('EPOCH:[{:03d}/{:03d}]--{}--Elapse time:{:.2f}--Accuracy:{:.4f}--acc_a:{:.4f}--acc_v:{:.4f}'.format(epoch,cfgs.EPOCHS,'Validate',elapse_time,accuracy,accuracy_a,accuracy_v))
        return accuracy,accuracy_a,accuracy_v,test_audio_loss,test_visual_loss
def extract_mm_feature(model,dep_dataloader,device,cfgs):
    all_feature = []
    model.eval()
    if cfgs.methods=="AGM" or cfgs.fusion_type == "early_fusion":
        model.mode = 'eval'
    with torch.no_grad():
        total_batch = len(dep_dataloader)
        for step,(item,spec,image,label) in enumerate(dep_dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                model.net.mode = 'feature'
                classify_out,out = model.net(spec.unsqueeze(1).float(),image.float(),pad_audio = False,pad_visual=False)
                all_feature.append(out)
                model.net.mode = "classify"
            else:
                model.extract_mm_feature = True
                out_a,out_v,out,feature = model(spec.unsqueeze(1).float(),image.float())
                all_feature.append(feature)
                model.extract_mm_feature = False
        all_feature = torch.cat(all_feature,dim=0)
        return all_feature

def CREMAD_main(cfgs):
    set_seed(cfgs.random_seed)
    ts = time.strftime('%Y_%m_%d %H:%M:%S',time.localtime())
    save_dir = os.path.join(cfgs.expt_dir,f"{ts}_{cfgs.expt_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_config(cfgs,save_dir)
    logger = get_logger("train_logger",logger_dir=save_dir)
    if cfgs.use_mgpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.gpu_ids
        gpu_ids = list(map(int,cfgs.gpu_ids.split(",")))
        device = get_device(cfgs.device)
    else:
        device = get_device(cfgs.device)
    writer = SummaryWriter(os.path.join(save_dir,'tensorboard_out'))
    logger.info(vars(cfgs))
    logger.info(f"Processing ID:{os.getpid()},Device:{device},System-dependence Version:{os.uname()}")
    task = Cramed_Task(cfgs)
    train_dataloader = task.train_dataloader
    test_dataloader = task.test_dataloader
    dep_dataloader = task.dep_dataloader
    model = task.model
    optimizer = task.optimizer
    scheduler = task.scheduler
    model.apply(weight_init)
    model.to(device)
    
    if cfgs.use_mgpu:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        model.to(device)
        
    best_epoch = {'epoch':0,'acc':0.}  
    start_epoch = 1
    each_epoch_score_a = 0.
    each_epoch_score_v = 0.
    if cfgs.breakpoint_path is not None:
        ckpt = torch.load(cfgs.breakpoint_path)
        task.last_epoch = ckpt["epoch"]
        start_epoch = ckpt["epoch"]+1
        model.load_state_dict(ckpt["state_dict"])
        optimizer = task.optimizer
        optimizer.load_state_dict(ckpt["optimizer"])
        best_epoch["epoch"] = ckpt["epoch"]
        best_epoch['acc'] = ckpt["best_acc"]
        each_epoch_score_a = ckpt["train_score_a"]
        each_epoch_score_v = ckpt["train_score_v"]
    
    
    audio_lr_ratio = 1.0
    visual_lr_ratio = 1.0
    audio_lr_memory = []
    visual_lr_memory = []
    min_test_audio_loss = 1000
    min_test_visual_loss = 1000
    audio_argmax_loss_epoch = 0.
    visual_argmax_loss_epoch = 0.
    for epoch in range(start_epoch,cfgs.EPOCHS+1):
        logger.info(f'Training for epoch {epoch}...')
        each_epoch_score_a,each_epoch_score_v = train(model,train_dataloader,\
                optimizer,scheduler,logger,cfgs,epoch,device,writer,each_epoch_score_a,each_epoch_score_v,audio_lr_ratio,visual_lr_ratio)

        if cfgs.modality == "Multimodal":
            # Path for saved mono-modal feature
            dep_audio_out_path = ""
            dep_visual_out_path = ""
            test_audio_out_path = ""
            test_visual_out_path = ""
            logger.info(f"Validating for epoch {epoch}...")
            if os.path.exists(dep_audio_out_path) and os.path.exists(dep_visual_out_path) and os.path.exists(test_audio_out_path) and os.path.exists(test_visual_out_path):
                dep_mm_feature = extract_mm_feature(model,dep_dataloader,device,cfgs)
                if cfgs.fusion_type == "late_fusion":
                    dep_audio_out = torch.load(dep_audio_out_path,map_location="cpu")
                    dep_visual_out = torch.load(dep_visual_out_path,map_location="cpu")
                    test_audio_out = torch.load(test_audio_out_path,map_location="cpu")
                    test_visual_out = torch.load(test_visual_out_path,map_location="cpu")
                else:
                    dep_audio_out = torch.load(dep_audio_out_path,map_location="cpu")
                    dep_visual_out = torch.load(dep_visual_out_path,map_location="cpu")
                    test_audio_out = torch.load(test_audio_out_path,map_location="cpu")
                    test_visual_out = torch.load(test_visual_out_path,map_location="cpu")
                
                mm_to_audio_lr = linear_model.Ridge(alpha=120)
                mm_to_visual_lr = linear_model.Ridge(alpha=120)
                mm_to_audio_lr.fit(dep_mm_feature.detach().cpu(),dep_audio_out)
                mm_to_visual_lr.fit(dep_mm_feature.detach().cpu(),dep_visual_out)

                validate_accuracy,accuracy_a,accuracy_v,validate_audio_batch_loss,validate_visual_batch_loss = test_compute_weight(model,test_dataloader,logger,cfgs,epoch,device,writer,mm_to_audio_lr,mm_to_visual_lr,test_audio_out,test_visual_out)
            else:
                validate_accuracy,accuracy_a,accuracy_v,validate_audio_batch_loss,validate_visual_batch_loss = test(model,test_dataloader,logger,cfgs,epoch,device,writer)
        else:
            logger.info(f'Validating for epoch {epoch}...')
            validate_accuracy = test(model,test_dataloader,logger,cfgs,epoch,device,writer)
        if cfgs.modality == "Multimodal" and cfgs.methods == "MSLR":
            if len(audio_lr_memory) < 5:
                audio_lr_memory.append(accuracy_a)
                visual_lr_memory.append(accuracy_v)
            else:
                audio_lr_ratio = accuracy_a / np.mean(audio_lr_memory)
                visual_lr_ratio = accuracy_v / np.mean(visual_lr_memory)

                audio_lr_memory = audio_lr_memory[1:]
                visual_lr_memory = visual_lr_memory[1:]
                audio_lr_memory.append(accuracy_a)
                visual_lr_memory.append(accuracy_v)
                if len(audio_lr_memory) != 5 or len(visual_lr_memory) != 5:
                    raise ValueError
        elif cfgs.modality == "Multimodal" and cfgs.methods == "MSES":
            if epoch == 1:
                min_test_audio_loss = validate_audio_batch_loss
                min_test_visual_loss = validate_visual_batch_loss
                audio_argmax_loss_epoch = 1.0
                visual_argmax_loss_epoch = 1.0 
                audio_batch_loss = validate_audio_batch_loss
                visual_batch_loss = validate_visual_batch_loss
            else:
                audio_batch_loss = 0.6 * bre_audio_batch_loss + 0.4 * validate_audio_batch_loss
                visual_batch_loss = 0.6 * bre_visual_batch_loss + 0.4 * validate_visual_batch_loss
            
            bre_audio_batch_loss = validate_audio_batch_loss
            bre_visual_batch_loss = validate_visual_batch_loss
            if min_test_audio_loss > audio_batch_loss:
                min_test_audio_loss = audio_batch_loss
                audio_argmax_loss_epoch = epoch
                torch.save({'epoch':best_epoch['epoch'],
                            'state_dict':model.state_dict(),
                            'best_acc':best_epoch['acc'],
                            'optimizer':optimizer.state_dict(),
                            "train_score_a":each_epoch_score_a,
                            "train_score_v":each_epoch_score_v},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))

            if min_test_visual_loss > visual_batch_loss:
                min_test_visual_loss = visual_batch_loss
                visual_argmax_loss_epoch = epoch
                torch.save({'epoch':best_epoch['epoch'],
                            'state_dict':model.state_dict(),
                            'best_acc':best_epoch['acc'],
                            'optimizer':optimizer.state_dict(),
                            "train_score_a":each_epoch_score_a,
                            "train_score_v":each_epoch_score_v},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))
            
            if (epoch - audio_argmax_loss_epoch > 80) and (audio_lr_ratio != 0.0):
                audio_lr_ratio = 0.0
                logger.info("Audio modality training finished.")
                ckpt = torch.load(os.path.join(save_dir,f'ckpt_full_epoch{audio_argmax_loss_epoch}.pth.tar'))
                model.load_state_dict(ckpt["state_dict"])
                optimizer = task.optimizer
                optimizer.load_state_dict(ckpt["optimizer"])
                best_epoch["epoch"] = ckpt["epoch"]
                best_epoch['acc'] = ckpt["best_acc"]
                each_epoch_score_a = ckpt["train_score_a"]
                each_epoch_score_v = ckpt["train_score_v"]
            
            if (epoch - visual_argmax_loss_epoch > 80) and (visual_lr_ratio != 0.0):
                visual_lr_ratio = 0.0
                logger.info("Visual modality training finished.")
                ckpt = torch.load(os.path.join(save_dir,f'ckpt_full_epoch{visual_argmax_loss_epoch}.pth.tar'))
                model.load_state_dict(ckpt["state_dict"])
                optimizer = task.optimizer
                optimizer.load_state_dict(ckpt["optimizer"])
                best_epoch["epoch"] = ckpt["epoch"]
                best_epoch['acc'] = ckpt["best_acc"]
                each_epoch_score_a = ckpt["train_score_a"]
                each_epoch_score_v = ckpt["train_score_v"]
            logger.info("audio_min_loss_epoch:{},visual_min_loss_epoch:{}".format(audio_argmax_loss_epoch,visual_argmax_loss_epoch))
            if (audio_lr_ratio == 0.0) and (visual_lr_ratio == 0.0):
                logger.info("All modality training finished.")
                exit()

        if validate_accuracy > best_epoch['acc']:
    
            best_epoch['acc'] = validate_accuracy
            best_epoch['epoch'] = epoch
            
            logger.info(f'need to compute OGR...')

            if cfgs.save_checkpoint:
                torch.save({'epoch':best_epoch['epoch'],
                            'state_dict':model.state_dict(),
                            'best_acc':best_epoch['acc'],
                            'optimizer':optimizer.state_dict(),
                            "train_score_a":each_epoch_score_a,
                            "train_score_v":each_epoch_score_v},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))


        logger.info('Best epoch{},best accuracy{:.4f}'.format(best_epoch["epoch"],best_epoch["acc"]))        
