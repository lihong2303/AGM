import warnings
import time
import torch
import os
import math
import torch.nn as nn
import numpy as np
from utils.metric import Accuracy
from tasks.MOSEI_task import Mosei_Task
from utils.function_tools import save_config,get_logger,get_device,set_seed
from torch.utils.tensorboard import SummaryWriter


def train(model,train_dataloader,optimizer,scheduler,cfgs,device,logger,epoch,writer,last_epoch_score_t,last_epoch_score_a):
    softmax = nn.Softmax(dim=1)
    model.train()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_batch = len(train_dataloader)
    if cfgs.modality == "Audio" or cfgs.modality == "Multimodal":
        train_audio_acc = 0.
        train_epoch_score_a = last_epoch_score_a

    if cfgs.modality == "Text" or cfgs.modality == "Multimodal":
        train_text_acc = 0.
        train_epoch_score_t = last_epoch_score_t
    if cfgs.modality == "Multimodal":
        train_acc = 0.
        model.mode = 'train'
    
    start_time = time.time()
    for step,(id,x,y,z,ans) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        ans = ans.to(device)
        optimizer.zero_grad()
        iteration = (epoch-1)*total_batch +step +1
        if cfgs.modality == "Audio":
            out_a = model.net(x,y,pad_x = True,pad_y = False)
            loss = loss_fn(out_a,ans)
            pred_a = softmax(out_a)
            audio_accuracy = Accuracy(pred_a,ans)
            train_audio_acc += audio_accuracy.item() / total_batch

            score_audio = 0.
            for k in range(out_a.size(0)):
                if torch.isinf(torch.log(softmax(out_a)[k][ans[k]])) or softmax(out_a)[k][ans[k]] < 1e-8:
                    score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.device))
                else:
                    score_audio += - torch.log(softmax(out_a)[k][ans[k]])
            score_audio = score_audio / out_a.size(0)

            train_epoch_score_a = train_epoch_score_a * (iteration - 1) / iteration + score_audio.item() / iteration
        elif cfgs.modality == "Text":
            out_t = model.net(x,y,pad_x=False,pad_y=True)
            loss = loss_fn(out_t,ans)
            pred_t = softmax(out_t)
            text_accuracy = Accuracy(pred_t,ans)
            train_text_acc += text_accuracy.item() / total_batch

            score_text = 0.
            for k in range(out_t.size(0)):
                if torch.isinf(torch.log(softmax(out_t)[k][ans[k]])) or softmax(out_t)[k][ans[k]] < 1e-8:
                    score_text += - torch.log(torch.tensor(1e-8,dtype=out_t.dtype,device=out_t.device))
                else:
                    score_text += - torch.log(softmax(out_t)[k][ans[k]])
            score_text = score_text / out_t.size(0)

            train_epoch_score_t = train_epoch_score_t * (iteration - 1) / iteration + score_text.item() / iteration
        elif cfgs.modality == "Multimodal":
            out_t,out_a,C,out = model(x,y,z)
            loss = loss_fn(out,ans)
            pred = softmax(out)
            pred_t = softmax(out_t)
            pred_a = softmax(out_a)

            accuracy = Accuracy(pred,ans)
            text_accuracy = Accuracy(pred_t,ans)
            audio_accuracy = Accuracy(pred_a,ans)

            train_acc += accuracy.item() / total_batch
            train_text_acc += text_accuracy.item() / total_batch
            train_audio_acc += audio_accuracy.item() / total_batch
            if torch.isnan(out_t).any() or torch.isnan(out_a).any():
                raise ValueError

            score_text = 0.
            score_audio = 0.
            for k in range(out_t.size(0)):
                if torch.isinf(torch.log(softmax(out_t)[k][ans[k]])) or softmax(out_t)[k][ans[k]] < 1e-8:
                    score_text += - torch.log(torch.tensor(1e-8,dtype=out_t.dtype,device=out_t.device))
                else:
                    score_text += - torch.log(softmax(out_t)[k][ans[k]])
                    
                if torch.isinf(torch.log(softmax(out_a)[k][ans[k]])) or softmax(out_a)[k][ans[k]] < 1e-8:
                    score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.device))
                else:
                    score_audio += - torch.log(softmax(out_a)[k][ans[k]])
            score_text = score_text / out_t.size(0)
            score_audio = score_audio / out_a.size(0)
            mean_ratio = (score_audio.item() + score_text.item()) / 2
            ratio_t = math.exp((mean_ratio - score_text.item())*2)
            ratio_a = math.exp((mean_ratio - score_audio.item())*2)
            
            optimal_mean_score = (train_epoch_score_a + train_epoch_score_t) / 2
            optimal_ratio_a = math.exp((optimal_mean_score - train_epoch_score_a)*2)
            optimal_ratio_t = math.exp((optimal_mean_score - train_epoch_score_t)*2)
            
            coeff_t = math.exp(cfgs.alpha*(optimal_ratio_t - ratio_t))
            coeff_a = math.exp(cfgs.alpha*(optimal_ratio_a - ratio_a))

        writer.add_scalar('loss/step',loss,iteration-1)        
        
        if cfgs.modality == "Multimodal":
            if cfgs.methods == "AGM" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                if step ==0:
                    logger.info('Using AGM methods...')
                
                if cfgs.use_mgpu:
                    model.module.update_scale(coeff_a,coeff_t)
                else:
                    model.update_scale(coeff_a,coeff_t)

            else:
                if step ==0:
                    logger.info('Using normal methods...')

        loss.backward()
        if cfgs.modality == "Multimodal":
            train_epoch_score_t = train_epoch_score_t * (iteration - 1) / iteration + score_text.item() / iteration
            train_epoch_score_a = train_epoch_score_a * (iteration - 1) / iteration + score_audio.item() / iteration
            if cfgs.use_mgpu:
                grad_max = torch.max(model.module.net.proj.weight.grad)
                grad_min = torch.min(model.module.net.proj.weight.grad)
            else:
                grad_max = torch.max(model.net.proj.weight.grad)
                grad_min = torch.min(model.net.proj.weight.grad)

            if grad_max > 1.0 and grad_min < 1.0:
                nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

        optimizer.step()
        if step % 100 ==0:
            logger.info('EPOCH:[{:3d}/{:3d}]--STEP:[{:5d}/{:5d}]--{}--loss:{:.4f}-lr:{}'.format(epoch,cfgs.EPOCHS,step,total_batch,'Train',loss,[group['lr'] for group in optimizer.param_groups]))

    scheduler.step()
    
    end_time = time.time()
    elapse_time = end_time - start_time

    if cfgs.modality == "Audio":
        writer.add_scalars("Accuracy",{'train_audio_accuracy':train_audio_acc},epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]--{}-audio_accuracy:{:.4f}-elapse_time:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',train_audio_acc,elapse_time))
        return last_epoch_score_t,train_epoch_score_a
    elif cfgs.modality == "Text":
        writer.add_scalars("Accuracy",{'train_text_accuracy':train_text_acc},epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]--{}-text_accuracy:{:.4f}-elapse_time:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',train_text_acc,elapse_time))
        return train_epoch_score_t,last_epoch_score_a
    elif cfgs.modality == "Multimodal":
        writer.add_scalars("Accuracy(Train)",{'accuracy':train_acc,
                                              'text accuracy':train_text_acc,
                                              'audio accuracy':train_audio_acc},epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]--{}--acc:{:.4f}--acc_t:{:.4f}--acc_a:{:.4f}-elapse time:{:.4f}-Alpha:{}'.format(epoch,cfgs.EPOCHS,'Train',train_acc,train_text_acc,train_audio_acc,elapse_time,cfgs.alpha))
        return train_epoch_score_t,train_epoch_score_a

def validate(model,validate_dataloader,cfgs,device,logger,epoch,writer):
    softmax = nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        if cfgs.use_mgpu:
            model.module.eval()
        else:
            model.eval()
        if cfgs.modality == "Audio":
            validate_audio_acc = 0.
            validate_score_a = 0.
        elif cfgs.modality == "Text":
            validate_text_acc = 0.
            validate_score_t = 0.
        elif cfgs.modality == "Multimodal":
            model.mode = "eval"
            validate_acc = 0.
            validate_text_acc = 0.
            validate_audio_acc = 0.
            validate_score_t = 0.
            validate_score_a = 0.
        total_batch = len(validate_dataloader)
        start_time = time.time()
        for step,(id,x,y,z,ans) in enumerate(validate_dataloader):
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            ans = ans.to(device)
            if cfgs.modality == "Audio":
                out_a = model.net(x,y,pad_x = True,pad_y = False)
                loss = loss_fn(out_a,ans)
                pred_a = softmax(out_a)
                audio_accuracy = Accuracy(pred_a,ans)
                validate_audio_acc += audio_accuracy.item() / total_batch

                score_audio = 0.
                for k in range(out_a.size(0)):   
                    if torch.isinf(torch.log(softmax(out_a)[k][ans[k]])) or softmax(out_a)[k][ans[k]] < 1e-8:
                        score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.devcie))
                    else:
                        score_audio += - torch.log(softmax(out_a)[k][ans[k]])
                score_audio = score_audio / out_a.size(0)
                validate_score_a = validate_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            elif cfgs.modality == "Text":
                out_t = model.net(x,y,pad_x = False,pad_y = True)
                pred_t = softmax(out_t)
                text_accuracy = Accuracy(pred_t,ans)
                loss = loss_fn(out_t,ans)
                validate_text_acc += text_accuracy.item() / total_batch

                score_text = 0.
                for k in range(out_t.size(0)):
                    if torch.isinf(torch.log(softmax(out_t)[k][ans[k]])) or softmax(out_t)[k][ans[k]] < 1e-8:
                        score_text += - torch.log(torch.tensor(1e-8,dtype=out_t.dtype,device=out_t.device))
                    else:
                        score_text += - torch.log(softmax(out_t)[k][ans[k]])
                score_text = score_text / out_t.size(0)
                validate_score_t = validate_score_t * step / (step + 1) + score_text.item() / (step + 1)
            elif cfgs.modality == "Multimodal":
                out_t,out_a,C,out = model(x,y,z)
                loss = loss_fn(out,ans)
                pred = softmax(out)
                pred_t = softmax(out_t)
                pred_a = softmax(out_a)

                accuracy = Accuracy(pred,ans)
                text_accuracy = Accuracy(pred_t,ans)
                audio_accuracy = Accuracy(pred_a,ans)

                validate_acc += accuracy.item() / total_batch
                validate_text_acc += text_accuracy.item() / total_batch
                validate_audio_acc += audio_accuracy.item() / total_batch

                if torch.isnan(out_t).any() or torch.isnan(out_a).any():
                    raise ValueError

                score_text = 0.
                score_audio = 0.
                for k in range(out_t.size(0)):
                    if torch.isinf(torch.log(softmax(out_t)[k][ans[k]])) or softmax(out_t)[k][ans[k]] < 1e-8:
                        score_text += - torch.log(torch.tensor(1e-8,dtype=out_t.dtype,device=out_t.device))
                    else:
                        score_text += - torch.log(softmax(out_t)[k][ans[k]])
                        
                    if torch.isinf(torch.log(softmax(out_a)[k][ans[k]])) or softmax(out_a)[k][ans[k]] < 1e-8:
                        score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.devcie))
                    else:
                        score_audio += - torch.log(softmax(out_a)[k][ans[k]])
                score_text = score_text / out_t.size(0)
                score_audio = score_audio / out_a.size(0)
                ratio_t = math.exp(score_audio.item() - score_text.item())
                ratio_a = math.exp(score_text.item() - score_audio.item())
                
                validate_score_t = validate_score_t * step / (step + 1) + score_text.item() / (step + 1)
                validate_score_a = validate_score_a * step / (step + 1) + score_audio.item() / (step + 1)
                
                optimal_ratio_a = math.exp(validate_score_t - validate_score_a)
                optimal_ratio_t = math.exp(validate_score_a - validate_score_t)

            iteration = (epoch-1)*total_batch +step
            writer.add_scalar('test loss/step',loss,iteration)

            if step % 20 == 0:
                logger.info('EPOCHS[{:02d}/{:02d}]--STEP[{:02d}/{:02d}]--{}--loss:{:.4f}'.format(epoch,cfgs.EPOCHS,step,total_batch,'Validate',loss))
        model.mode = "train"
        end_time = time.time()
        elapse_time = end_time - start_time
        if cfgs.modality == "Audio":
            writer.add_scalars("Accuracy",{'validate_audio_accuracy':validate_audio_acc},epoch)
            logger.info('EPOCH:[{:3d}/{:3d}]--{}-audio_accuracy:{:.4f}-elapse_time:{:.4f}'.format(epoch,cfgs.EPOCHS,'Validate',validate_audio_acc,elapse_time))
            return validate_audio_acc
        elif cfgs.modality == "Text":
            writer.add_scalars("Accuracy",{'validate_text_accuracy':validate_text_acc},epoch)
            logger.info('EPOCH:[{:3d}/{:3d}]--{}-text_accuracy:{:.4f}-elapse_time:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',validate_text_acc,elapse_time))
            return validate_text_acc
        elif cfgs.modality == "Multimodal":
            writer.add_scalars("Accuracy(Validate)",{"accuracy":validate_acc,
                                                     "text accuracy":validate_text_acc,
                                                     "audio accuracy":validate_audio_acc},epoch)

            logger.info('EPOCH:[{:03d}/{:03d}]--{}--Elapse time:{:.2f}--Accuracy:{:.4f}--acc_t:{:.4f}--acc_a:{:.4f}'.format(epoch,cfgs.EPOCHS,'Validate',elapse_time,validate_acc,validate_text_acc,validate_audio_acc))
            return validate_acc

def MOSEI_main(cfgs):
    warnings.filterwarnings('ignore')
    set_seed(cfgs.random_seed)
    ts = time.strftime('%Y_%m_%d %H:%M:%S',time.localtime())
    save_dir = os.path.join(cfgs.expt_dir,f"{ts}_{cfgs.expt_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_config(cfgs,save_dir)
    logger = get_logger("train_logger",logger_dir=save_dir)
    logger.info(vars(cfgs))
    logger.info(f"Process ID:{os.getpid()},System Version:{os.uname()}")
    if cfgs.use_mgpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.gpu_ids
        gpu_ids = list(map(int,cfgs.gpu_ids.split(",")))
        device = get_device(cfgs.device)
    else:
        device = get_device(cfgs.device)
    writer = SummaryWriter(os.path.join(save_dir,'tensorboard_out'))
    task =Mosei_Task(cfgs)
    train_dataloader = task.train_dataloader
    validate_dataloader = task.valid_dataloader
    test_dataloader = task.test_dataloader

    model = task.model
    optimizer = task.optimizer
    scheduler = task.scheduler

    model.to(device)
    if cfgs.use_mgpu:
        model = torch.nn.DataParallel(model,device_ids=gpu_ids)
        model.to(device)
    
    best_epoch = {'epoch':0,'acc':0.}
    epoch_score_t = 0.
    epoch_score_a = 0.
    start_epoch = 1    
    if cfgs.breakpoint_path is not None:
        ckpt = torch.load(cfgs.breakpoint_path)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_epoch['acc'] = ckpt['acc']
        best_epoch['epoch'] = ckpt['epoch']
        epoch_score_t = ckpt['epoch_score_t']
        epoch_score_a = ckpt['epoch_score_a']
    

    for epoch in range(start_epoch,cfgs.EPOCHS+1):
        logger.info(f'Training for epoch:{epoch}...')
        epoch_score_t,epoch_score_a = train(model,train_dataloader,optimizer,scheduler,cfgs,device,logger,epoch,writer,epoch_score_t,epoch_score_a)
        
        logger.info(f'Validating for epoch:{epoch}...')
        validate_acc = validate(model,validate_dataloader,cfgs,device,logger,epoch,writer)
        if validate_acc > best_epoch['acc']:
            best_epoch['acc'] = validate_acc
            best_epoch['epoch'] = epoch
            if cfgs.save_checkpoint:
                torch.save({'epoch':epoch,
                            'state_dict':model.state_dict(),
                            'optimizer':optimizer.state_dict(),
                            'acc':best_epoch['acc'],
                            'epoch_score_t':epoch_score_t,
                            'epoch_score_a':epoch_score_a},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))
        
        logger.info(f'Best epoch{best_epoch["epoch"]},best accuracy{best_epoch["acc"]}')
