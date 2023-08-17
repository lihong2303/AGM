import time
import torch
import os
import math
import torch.nn as nn
import numpy as np
from tasks.AVMNIST_task import AVMNIST_Task
from utils.function_tools import save_config,get_device,get_logger,set_seed
from torch.utils.tensorboard import SummaryWriter
from utils.metric import Accuracy
from sklearn import linear_model

def train(model,train_dataloader,optimizer,scheduler,epoch,cfgs,device,writer,logger,last_score_a,last_score_v, \
            audio_lr_ratio,visual_lr_ratio):
    loss_fn = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    model.train()
    train_loss = 0.
    if cfgs.modality == 'Audio' or cfgs.modality == 'Multimodal':
        train_audio_acc = 0.
        train_score_a = last_score_a
    if cfgs.modality == 'Visual' or cfgs.modality == 'Multimodal':
        train_visual_acc = 0.
        train_score_v = last_score_v
    if cfgs.modality == 'Multimodal':
        model.mode = 'train'
        train_acc = 0.
        train_score_a = last_score_a
        train_score_v = last_score_v
    num_batch = len(train_dataloader)
    for step,(audio,image,label) in enumerate(train_dataloader):
        audio = audio.float().to(device)
        image = image.float().to(device)
        label = label.long().to(device)
        
        iteration = (epoch-1)*num_batch + step
        optimizer.zero_grad()
        if cfgs.modality == 'Audio':
            if cfgs.fusion_type == "late_fusion":
                out_a = model(audio)
            elif cfgs.fusion_type == "early_fusion":
                out_a = model.net(audio,image,pad_audio=False,pad_visual=True)
            loss = loss_fn(out_a,label)
            batch_audio_acc = Accuracy(softmax(out_a),label)
            train_audio_acc += batch_audio_acc.item() / num_batch
        elif cfgs.modality == 'Visual':
            if cfgs.fusion_type == "late_fusion":
                out_v = model(image)
            elif cfgs.fusion_type == "early_fusion":
                out_v = model.net(audio,image,pad_audio=True,pad_visual=False)
            loss = loss_fn(out_v,label)
            batch_visual_acc = Accuracy(softmax(out_v),label)
            train_visual_acc += batch_visual_acc.item() / num_batch
        elif cfgs.modality == 'Multimodal':
            if cfgs.methods=='AGM' or cfgs.fusion_type == "early_fusion":
                if step == 0:
                    logger.info("Using SHAPE to compute individual output.")
                out_a,out_v,out = model(audio,image)
                loss = loss_fn(out,label)
                batch_acc = Accuracy(softmax(out),label)
                batch_audio_acc = Accuracy(softmax(out_a),label)
                batch_visual_acc = Accuracy(softmax(out_v),label)
                
                train_audio_acc += batch_audio_acc.item() / num_batch
                train_visual_acc += batch_visual_acc.item() / num_batch
                train_acc += batch_acc.item() / num_batch
            else:
                if step == 0:
                    logger.info("Training with addition-fusion model.")
                out_a,out_v,out = model(audio,image)
                loss = loss_fn(out,label)
                batch_acc = Accuracy(softmax(out),label)
                batch_audio_acc = Accuracy(softmax(out_a),label)
                batch_visual_acc = Accuracy(softmax(out_v),label)
                
                train_audio_acc += batch_audio_acc.item() / num_batch
                train_visual_acc += batch_visual_acc.item() / num_batch
                train_acc += batch_acc.item() / num_batch
            
        train_loss += loss.item() / num_batch
        
        
        writer.add_scalar('Loss(Train)',loss.item(),iteration)
        
        if cfgs.modality == 'Multimodal':
            if torch.isnan(out_a).any() or torch.isnan(out_v).any():
                raise ValueError
            
            score_audio = 0.
            score_visual = 0.
            for k in range(out_a.size(0)):
                if torch.isinf(softmax(out_a)[k][label[k]]) or softmax(out_a)[k][label[k]] < 1e-8:
                    score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.device))
                else:
                    score_audio +=  - torch.log(softmax(out_a)[k][label[k]])
                if torch.isinf(softmax(out_v)[k][label[k]]) or softmax(out_v)[k][label[k]] < 1e-8:
                    score_visual += - torch.log(torch.tensor(1e-8,dtype=out_v.dtype,device=out_v.device))
                else:
                    score_visual += - torch.log(softmax(out_v)[k][label[k]])
            
            score_audio = score_audio / out_a.size(0)
            score_visual = score_visual / out_v.size(0)

            mean_ratio = (score_visual.item() + score_audio.item()) / 2
            
            ratio_a = math.exp((mean_ratio - score_audio.item())*2)
            ratio_v = math.exp((mean_ratio - score_visual.item())*2)
            
            mean_optimal_ratio = (train_score_a + train_score_v) / 2

            optimal_ratio_a = math.exp((mean_optimal_ratio - train_score_a)*2)
            optimal_ratio_v = math.exp((mean_optimal_ratio - train_score_v)*2)

            coeff_a = math.exp(cfgs.alpha * (optimal_ratio_a - ratio_a))
            coeff_v = math.exp(cfgs.alpha * (optimal_ratio_v - ratio_v))
            
            train_score_a = train_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
            train_score_v = train_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)
            
            if cfgs.methods == "AGM" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                if step == 0:
                    logger.info('Using Adaptive Gradient Modulation methods...')
                if cfgs.use_mgpu:
                    model.module.update_scale(coeff_a,coeff_v)
                else:
                    model.update_scale(coeff_a,coeff_v)
                loss.backward()
            elif cfgs.methods == "OGM-GE" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                if step == 0:
                    logger.info("Using OGM-GE methods...")
                loss.backward()
                OGM_score_a = sum([softmax(out_a)[k][label[k]] for k in range(out_a.size(0))])
                OGM_score_v = sum([softmax(out_v)[k][label[k]] for k in range(out_v.size(0))])
                OGM_ratio_v = OGM_score_v / OGM_score_a
                OGM_ratio_a = 1 / OGM_ratio_v
                if OGM_ratio_v > 1:
                    OGM_coeff_v = 1 - tanh(cfgs.alpha * relu(OGM_ratio_v))
                    OGM_coeff_a = 1
                else:
                    OGM_coeff_a = 1 - tanh(cfgs.alpha * relu(OGM_ratio_a))
                    OGM_coeff_v = 1
                for name,params in model.named_parameters():
                    layer = str(name).split('.')[1]
                    if 'audio_encoder' in layer and len(params.grad.size()) == 4:
                        params.grad = params.grad*OGM_coeff_a + \
                            torch.zeros_like(params.grad).normal_(0,params.grad.std().item() + 1e-8)
                    if 'visual_encoder' in layer and len(params.grad.size()) == 4:
                        params.grad = params.grad * OGM_coeff_v + \
                            torch.zeros_like(params.grad).normal_(0,params.grad.std().item() + 1e-8)
            elif cfgs.methods == "MSLR" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                audio_lr_init_coeff = 1.3
                visual_lr_init_coeff = 0.7
                mslr_audio_coeff = audio_lr_init_coeff * audio_lr_ratio
                mslr_visual_coeff = visual_lr_init_coeff * visual_lr_ratio
                if step == 0:
                    logger.info("Using MSLR methods.")
                    logger.info("Audio learning rate:{:.7f},Visual learning rate:{:.7f}".format(mslr_audio_coeff*cfgs.learning_rate,mslr_visual_coeff*cfgs.learning_rate))
                loss.backward()
                for name,params in model.named_parameters():
                    layer = str(name).split('.')[0]
                    if 'audio_encoder' in layer or 'audio_cls' in layer:
                        params.grad = params.grad * mslr_audio_coeff
                    if 'visual_encoder' in layer or 'visual_cls' in layer:
                        params.grad = params.grad * mslr_visual_coeff
            elif cfgs.methods == "MSES" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                audio_lr_init_coeff = 1.0
                visual_lr_init_coeff = 1.0
                mses_audio_coeff = audio_lr_init_coeff * audio_lr_ratio
                mses_visual_coeff = visual_lr_init_coeff * visual_lr_ratio
                if step == 0:
                    logger.info("Using MSES methods.")
                    logger.info("MSES audio coeff:{:.7f},MSES visual coeff:{:.7f}".format(mses_audio_coeff,mses_visual_coeff))   
                loss.backward()
                for name,params in model.named_parameters():
                    layer = str(name).split('.')[0]
                    if 'audio_encoder' in layer or 'audio_cls' in layer:
                        params.grad = params.grad * mses_audio_coeff
                    if 'visual_encoder' in layer or 'visual_cls' in layer:
                        params.grad = params.grad * mses_visual_coeff
            else:
                if step == 0:
                    logger.info("Using Normal Methods...")
                loss.backward()
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                grad_max = torch.max(model.net.head.fc.weight.grad)
                grad_min = torch.min(model.net.head.fc.weight.grad)
                if grad_max > 1.0 or grad_min < -1.0:
                    nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            
        elif cfgs.modality == 'Audio':
            score_audio = 0.
            for k in range(out_a.size(0)):
                score_audio +=  - torch.log(softmax(out_a)[k][label[k]])
            score_audio = score_audio / out_a.size(0)
            train_score_a = train_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
            
            loss.backward()
        elif cfgs.modality == 'Visual':
            score_visual = 0.
            for k in range(out_v.size(0)):
                score_visual += - torch.log(softmax(out_v)[k][label[k]])
            train_score_v = train_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)
            loss.backward()
        
        optimizer.step()
        
        if step % 200 == 0:
            logger.info('[{:03d}/{:03d}]-[{:04d}/{:04d}]-{}-loss:{:.4f}-lr:{}'.format(epoch,cfgs.EPOCHS,step,num_batch,'Train',loss,[group['lr'] for group in optimizer.param_groups]))
        
    scheduler.step()
    
    if cfgs.modality == 'Audio':
        writer.add_scalar('Accuracy(Train)',train_audio_acc,epoch)
        logger.info('[{:03d}/{:03d}]-{}-Audio_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',train_audio_acc))
        return train_score_a,last_score_v
    elif cfgs.modality == 'Visual':
        writer.add_scalar('Accuracy(Train)',train_visual_acc,epoch)
        logger.info('[{:03d}/{:03d}]-{}-Visual_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',train_visual_acc))
        return last_score_a,train_score_v
    elif cfgs.modality == 'Multimodal':
        writer.add_scalars('Accuracy(Train)',{'acc':train_acc,
                                                'audio acc':train_audio_acc,
                                                'visual acc':train_visual_acc},epoch)
        
        logger.info('[{:03d}/{:03d}]-{}-Acc:{:.4f}-Acc_a:{:.4f}-Acc_v:{:.4f}-Alpha:{}'.format(epoch,cfgs.EPOCHS,'Train',train_acc,train_audio_acc,train_visual_acc,cfgs.alpha))
        return train_score_a,train_score_v
              
        
def test(model,test_dataloader,epoch,cfgs,device,writer,logger):
    loss_fn = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    if cfgs.modality == 'Multimodal':
        model.mode = 'eval'
    model.eval()
    with torch.no_grad():
        if cfgs.modality == 'Audio':
            test_audio_acc = 0.
            test_score_a = 0.
        elif cfgs.modality == 'Visual':
            test_visual_acc = 0.
            test_score_v = 0.
        elif cfgs.modality == 'Multimodal':
            test_acc = 0.
            test_audio_acc = 0.
            test_visual_acc = 0.
            test_score_a = 0.
            test_score_v = 0.
            test_batch_audio_loss = 0.
            test_batch_visual_loss = 0.
        
        test_loss = 0.
        num_batch = len(test_dataloader)
        for step,(audio,image,label) in enumerate(test_dataloader):
            audio = audio.float().to(device)
            image = image.float().to(device)
            label = label.long().to(device)
            
            iteration = (epoch-1)*cfgs.batch_size + step
            if cfgs.modality == 'Audio':
                if cfgs.fusion_type == "late_fusion":
                    out_a = model(audio)
                elif cfgs.fusion_type == "early_fusion":
                    out_a = model.net(audio,image,pad_audio=False,pad_visual=True)
                loss = loss_fn(out_a,label)
                batch_audio_acc = Accuracy(softmax(out_a),label)
                test_audio_acc += batch_audio_acc.item() / num_batch
            elif cfgs.modality == 'Visual':
                if cfgs.fusion_type == "late_fusion":
                    out_v = model(image)
                elif cfgs.fusion_type == "early_fusion":
                    out_v = model.net(audio,image,pad_audio=True,pad_visual=False)
                loss = loss_fn(out_v,label)
                batch_visual_acc = Accuracy(softmax(out_v),label)
                test_visual_acc += batch_visual_acc.item() / num_batch
            elif cfgs.modality == 'Multimodal':
                out_a,out_v,out = model(audio,image)
                loss = loss_fn(out,label)
                batch_acc = Accuracy(softmax(out),label)
                batch_audio_acc = Accuracy(softmax(out_a),label)
                batch_visual_acc = Accuracy(softmax(out_v),label)
                
                test_acc += batch_acc.item() / num_batch
                test_audio_acc += batch_audio_acc.item() / num_batch
                test_visual_acc += batch_visual_acc.item() / num_batch

                loss_a = loss_fn(out_a,label)
                loss_v = loss_fn(out_v,label)
                test_batch_audio_loss += loss_a.item() / num_batch
                test_batch_visual_loss += loss_v.item() / num_batch
                test_loss += loss.item() / num_batch            
            
            writer.add_scalar('Loss(Test)',loss.item(),iteration)
             
            if cfgs.modality == 'Multimodal':
                if torch.isnan(out_a).any() or torch.isnan(out_v).any():
                    raise ValueError
                
                score_audio = 0.
                score_visual = 0.
                for k in range(out_a.size(0)):
                    score_audio += - torch.log(softmax(out_a)[k][label[k]]) / out_a.size(0)
                    score_visual += - torch.log(softmax(out_v)[k][label[k]]) / out_v.size(0)
                        
                ratio_a = math.exp(score_visual.item() - score_audio.item())
                ratio_v = math.exp(score_audio.item() - score_visual.item())
                
                test_score_a = test_score_a * step / (step + 1) + score_audio.item() / (step + 1)
                test_score_v = test_score_v * step / (step + 1) + score_visual.item() / (step + 1)
                
                optimal_ratio_a = math.exp(test_score_v - test_score_a)
                optimal_ratio_v = math.exp(test_score_a - test_score_v)
            elif cfgs.modality == 'Audio':
                score_audio = 0.
                for k in range(out_a.size(0)):
                    score_audio += - torch.log(softmax(out_a)[k][label[k]])
                score_audio = score_audio / out_a.size(0)
                test_score_a = test_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            elif cfgs.modality == 'Visual':
                score_visual = 0.
                for k in range(out_v.size(0)):
                    score_visual += - torch.log(softmax(out_v)[k][label[k]]) / out_v.size(0)
                test_score_v = test_score_v * step / (step + 1) + score_visual.item() / (step + 1)
            else:
                NotImplementedError
           
            if step % 40 == 0:
                logger.info('[{:03d}/{:03d}]-[{:03d}/{:03d}]-{}-loss:{:.4f}'.format(epoch,cfgs.EPOCHS,step,num_batch,'Test',loss))
            
        if cfgs.modality == 'Audio':
            writer.add_scalar('Accuracy(Test)',test_audio_acc,epoch)
            logger.info('[{:03d}/{:03d}]-{}-Audio_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'Test',test_audio_acc))
            return test_audio_acc
        elif cfgs.modality == 'Visual':
            writer.add_scalar('Accuracy(Test)',test_visual_acc,epoch)
            logger.info('[{:03d}/{:03d}]-{}-Visual_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'Test',test_visual_acc))
            return test_visual_acc
        elif cfgs.modality == 'Multimodal':
            writer.add_scalars('Accuracy(Test)',{'Acc':test_acc,
                                                 'audio acc':test_audio_acc,
                                                 'visual acc':test_visual_acc},epoch)
            model.mode = 'train'
    
            logger.info('[{:03d}/{:03d}]-{}-Acc:{:.4f}-Acc_a:{:.4f}-Acc_v:{:.4f}'.format(epoch,cfgs.EPOCHS,'Test',test_acc,test_audio_acc,test_visual_acc))
            return test_acc,test_audio_acc,test_visual_acc,test_batch_audio_loss,test_batch_visual_loss

def test_compute_weight(model,test_dataloader,epoch,cfgs,device,writer,logger,mm_to_audio_lr,mm_to_visual_lr,test_audio_out,test_visual_out):
    loss_fn = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    model.mode = 'eval'
    model.eval()
    ota = []
    otv = []
    with torch.no_grad():
       
        test_acc = 0.
        test_audio_acc = 0.
        test_visual_acc = 0.
        test_score_a = 0.
        test_score_v = 0.
        
        test_batch_audio_loss = 0.
        test_batch_visual_loss = 0.
        test_loss = 0.
        num_batch = len(test_dataloader)
        model.extract_mm_feature = True
        for step,(audio,image,label) in enumerate(test_dataloader):
            audio = audio.float().to(device)
            image = image.float().to(device)
            label = label.long().to(device)
            
            iteration = (epoch-1)*cfgs.batch_size + step
            
            out_a,out_v,out,feature = model(audio,image)

            out_to_audio = mm_to_audio_lr.predict(feature.detach().cpu())
            out_to_visual = mm_to_visual_lr.predict(feature.detach().cpu())
            ota.append(torch.from_numpy(out_to_audio))
            otv.append(torch.from_numpy(out_to_visual))

            loss = loss_fn(out,label)
            loss_a = loss_fn(out_a,label)
            loss_v = loss_fn(out_v,label)
            test_batch_audio_loss += loss_a.item() / num_batch
            test_batch_visual_loss += loss_v.item() / num_batch

            batch_acc = Accuracy(softmax(out),label)
            batch_audio_acc = Accuracy(softmax(out_a),label)
            batch_visual_acc = Accuracy(softmax(out_v),label)
            
            test_acc += batch_acc.item() / num_batch
            test_audio_acc += batch_audio_acc.item() / num_batch
            test_visual_acc += batch_visual_acc.item() / num_batch
            test_loss += loss.item() / num_batch
            
            writer.add_scalar('Loss(Test)',loss.item(),iteration)
             
            if torch.isnan(out_a).any() or torch.isnan(out_v).any():
                raise ValueError
            
            score_audio = 0.
            score_visual = 0.
            for k in range(out_a.size(0)):
                score_audio += - torch.log(softmax(out_a)[k][label[k]]) / out_a.size(0)
                score_visual += - torch.log(softmax(out_v)[k][label[k]]) / out_v.size(0)
                    
            ratio_a = math.exp(score_visual.item() - score_audio.item())
            ratio_v = math.exp(score_audio.item() - score_visual.item())
            
            test_score_a = test_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            test_score_v = test_score_v * step / (step + 1) + score_visual.item() / (step + 1)
            
            optimal_ratio_a = math.exp(test_score_v - test_score_a)
            optimal_ratio_v = math.exp(test_score_a - test_score_v)
            
            if step % 40 == 0:
                logger.info('[{:03d}/{:03d}]-[{:03d}/{:03d}]-{}-loss:{:.4f}'.format(epoch,cfgs.EPOCHS,step,num_batch,'Test',loss))
        model.extract_mm_feature = False
        ota = torch.cat(ota,dim=0).float()
        otv = torch.cat(otv,dim=0).float()
        ota = ota - test_audio_out
        otv = otv - test_visual_out
        ba = torch.cov(test_audio_out.T) * test_audio_out.size(0)
        bv = torch.cov(test_visual_out.T) * test_visual_out.size(0)
        ra = torch.trace((ota @ torch.pinverse(ba)).T @ ota) / test_audio_out.size(1)
        rv = torch.trace((otv @ torch.pinverse(bv)).T @ otv) / test_visual_out.size(1)

        writer.add_scalars('Accuracy(Test)',{'Acc':test_acc,
                                                'audio acc':test_audio_acc,
                                                'visual acc':test_visual_acc},epoch)
        model.mode = 'train'

        logger.info('[{:03d}/{:03d}]-{}-Acc:{:.4f}-Acc_a:{:.4f}-Acc_v:{:.4f}-da:{:.4f}-dv:{:.4f}'.format(epoch,cfgs.EPOCHS,'Test',test_acc,test_audio_acc,test_visual_acc,ra,rv))
        return test_acc,test_audio_acc,test_visual_acc,test_batch_audio_loss,test_batch_visual_loss

def extract_mm_feature(model,dep_dataloader,device,cfgs):
    model.mode = 'eval'
    model.eval()
    all_feature = []
    with torch.no_grad():
        num_batch = len(dep_dataloader)
        for step,(audio,image,label) in enumerate(dep_dataloader):
            audio = audio.float().to(device)
            image = image.float().to(device)
            label = label.long().to(device)
            if cfgs.methods=="AGM" or cfgs.fusion_type == "early_fusion":
                model.net.mode = 'feature'
                cls_out,out = model.net(audio,image,pad_audio = False,pad_visual=False)
                all_feature.append(out.detach().cpu())
                model.net.mode = 'classify'
            else:
                model.extract_mm_feature = True
                out_a,out_v,out,feature = model(audio,image)
                all_feature.append(feature)
                model.extract_mm_feature = False
    all_feature = torch.cat(all_feature,dim=0)
    return all_feature

def AVMNIST_main(cfgs):
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
    logger.info(f"Process ID:{os.getpid()},System Version:{os.uname()}")
    task = AVMNIST_Task(cfgs)
    train_dataloader = task.train_dataloader
    test_dataloader = task.test_dataloader
    dep_dataloader = task.dep_dataloader
    model = task.model
    optimizer = task.optimizer
    scheduler = task.scheduler
    model.to(device)
    
    if cfgs.use_mgpu:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        model.to(device)
    epoch_score_a = 0.
    epoch_score_v = 0.
    best_epoch = {'epoch':0,'acc':0.}

    audio_lr_ratio = 1.0
    visual_lr_ratio = 1.0
    audio_lr_memory = []
    visual_lr_memory = []
    min_test_audio_loss = 1000
    min_test_visual_loss = 1000
    audio_argmax_loss_epoch = 0.
    visual_argmax_loss_epoch = 0.

    for epoch in range(1,cfgs.EPOCHS):
        logger.info(f'Training for epoch {epoch}...')
        epoch_score_a,epoch_score_v = train(model,train_dataloader,optimizer,scheduler,epoch,cfgs,device,writer,logger,\
                                                                                                    epoch_score_a,epoch_score_v,\
                                                                                                        audio_lr_ratio,visual_lr_ratio)
        logger.info(f'Test for epoch {epoch}...')
        if cfgs.modality == "Multimodal":
            # path for saved feature
            dep_audio_out_path = ""
            dep_visual_out_path = ""
            test_audio_out_path = ""
            test_visual_out_path = ""
            if os.path.exists(dep_audio_out_path) and os.path.exists(dep_visual_out_path) and os.path.exists(test_audio_out_path) and os.path.exists(test_visual_out_path):
                dep_mm_feature = extract_mm_feature(model,dep_dataloader,device,cfgs)
                if cfgs.fusion_type == "early_fusion":
                    dep_audio_out = torch.load(dep_audio_out_path,map_location="cpu")
                    dep_visual_out = torch.load(dep_visual_out_path,map_location="cpu")
                    test_audio_out = torch.load(test_audio_out_path,map_location="cpu")
                    test_visual_out = torch.load(test_visual_out_path,map_location="cpu")
                elif cfgs.fusion_type == "late_fusion":
                    dep_audio_out = torch.load(dep_audio_out_path,map_location="cpu")
                    dep_visual_out = torch.load(dep_visual_out_path,map_location="cpu")
                    test_audio_out = torch.load(test_audio_out_path,map_location="cpu")
                    test_visual_out = torch.load(test_visual_out_path,map_location="cpu")
                mm_to_audio_lr = linear_model.Ridge(alpha=120)
                mm_to_visual_lr = linear_model.Ridge(alpha=120)
                mm_to_audio_lr.fit(dep_mm_feature.detach().cpu(),dep_audio_out)
                mm_to_visual_lr.fit(dep_mm_feature.detach().cpu(),dep_visual_out)
                test_acc,accuracy_a,accuracy_v,validate_audio_batch_loss,validate_visual_batch_loss = test_compute_weight(model,test_dataloader,epoch,cfgs,device,writer,logger,mm_to_audio_lr,mm_to_visual_lr,test_audio_out,test_visual_out)
            else:
                test_acc,accuracy_a,accuracy_v,validate_audio_batch_loss,validate_visual_batch_loss = test(model,test_dataloader,epoch,cfgs,device,writer,logger)
        else:
            test_acc = test(model,test_dataloader,epoch,cfgs,device,writer,logger)

        if cfgs.modality =="Multimodal" and cfgs.methods == "MSLR":
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
                            'train_score_a':epoch_score_a,
                            'train_score_v':epoch_score_v},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))
            
            if min_test_visual_loss > visual_batch_loss:
                min_test_visual_loss = visual_batch_loss
                visual_argmax_loss_epoch = epoch
                torch.save({'epoch':best_epoch['epoch'],
                            'state_dict':model.state_dict(),
                            'best_acc':best_epoch['acc'],
                            'optimizer':optimizer.state_dict(),
                            'train_score_a':epoch_score_a,
                            'train_score_v':epoch_score_v},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))
            if (epoch - audio_argmax_loss_epoch > 10) and (audio_lr_ratio != 0.0):
                audio_lr_ratio = 0.0
                logger.info("Audio modality training finished.")
                ckpt = torch.load(os.path.join(save_dir,f'ckpt_full_epoch{audio_argmax_loss_epoch}.pth.tar'))
                model.load_state_dict(ckpt["state_dict"])
                optimizer = task.optimizer
                optimizer.load_state_dict(ckpt["optimizer"])
                best_epoch["epoch"] = ckpt["epoch"]
                best_epoch['acc'] = ckpt["best_acc"]
                epoch_score_a = ckpt["train_score_a"]
                epoch_score_v = ckpt["train_score_v"]
            
            if (epoch - visual_argmax_loss_epoch > 10) and (visual_lr_ratio != 0.0):
                visual_lr_ratio = 0.0
                logger.info("Visual modality training finished.")
                logger.info("Audio modality training finished.")
                ckpt = torch.load(os.path.join(save_dir,f'ckpt_full_epoch{visual_argmax_loss_epoch}.pth.tar'))
                model.load_state_dict(ckpt["state_dict"])
                optimizer = task.optimizer
                optimizer.load_state_dict(ckpt["optimizer"])
                best_epoch["epoch"] = ckpt["epoch"]
                best_epoch['acc'] = ckpt["best_acc"]
                epoch_score_a = ckpt["train_score_a"]
                epoch_score_v = ckpt["train_score_v"]
            
            if (audio_lr_ratio == 0.0) and (visual_lr_ratio == 0.0):
                logger.info("All modalities training finished.")
                exit()

        if test_acc > best_epoch['acc']:
            best_epoch['acc'] = test_acc
            best_epoch['epoch'] = epoch
            
            if cfgs.save_checkpoint:
                torch.save({'epoch':best_epoch['epoch'],
                            'state_dict':model.state_dict(),
                            'best_acc':best_epoch['acc'],
                            'optimizer':optimizer.state_dict()},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))
        logger.info('[{:03d}/{:03d}]-best_epoch:{}-best_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,best_epoch['epoch'],best_epoch['acc']))
                
